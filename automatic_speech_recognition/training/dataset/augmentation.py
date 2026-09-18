import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asr.utils import load_audio, to_rel_path
from src.utils import log_and_print, setup_logging

from ..utils import write_jsonl

logger = logging.getLogger("training")

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)

def make_noise(noise_type: str, length: int) -> np.ndarray:
    if noise_type == "white":
        return np.random.randn(length).astype(np.float32)
    if noise_type == "pink":
        white = np.random.randn(length).astype(np.float32)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length)
        freqs[0] = 1.0
        pink = np.fft.irfft(spec / np.sqrt(freqs), n=length).astype(np.float32)
        return pink / (_rms(pink) + 1e-12)
    if noise_type == "brown":
        brown = np.cumsum(np.random.randn(length)).astype(np.float32)
        return brown / (_rms(brown) + 1e-12)
    raise ValueError("unsupported noise_type: " + str(noise_type))

def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    noise = noise[: len(speech)]
    alpha = 10.0 ** (-snr_db / 20.0) * _rms(speech) / _rms(noise)
    mixed = speech + alpha * noise
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)

def speed_perturb(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    if rate == 1.0 or audio.size == 0:
        return audio
    return librosa.resample(audio, orig_sr=sr, target_sr=int(sr / rate)).astype(np.float32)

def volume_perturb(audio: np.ndarray, db_range: float = 3.0) -> np.ndarray:
    gain = 10 ** (random.uniform(-db_range, db_range) / 20.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

class NoiseAugmenter:
    def __init__(
        self, 
        noise_types: Optional[List[str]] = None,
        snr_db_list: Optional[List[float]] = None,
        speed_rates: Optional[List[float]] = None,
        volume_db_range: float = 3.0,
        real_noise_dir: Optional[str] = None,
        seed: int = 42
    ):
        self.noise_types = noise_types or ["white", "pink", "brown"]
        self.snr_db_list = snr_db_list or [20.0, 10.0, 5.0]
        self.speed_rates = speed_rates or [1.0]
        self.volume_db_range = volume_db_range
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.real_files: List[str] = []
        if real_noise_dir and os.path.isdir(real_noise_dir):
            self.real_files = [os.path.join(real_noise_dir, n) for n in sorted(os.listdir(real_noise_dir)) if n.lower().endswith(".wav")]

    def augment_one(self, audio_path: str, text: str, output_dir: str, sr: int = 16000) -> List[Dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)
        speech = load_audio(audio_path, sr)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        rows: List[Dict[str, Any]] = []

        for noise_type in self.noise_types:
            noise = make_noise(noise_type, len(speech))
            for snr in self.snr_db_list:
                mixed = volume_perturb(mix_at_snr(speech, noise, snr), self.volume_db_range)
                path = os.path.join(output_dir, f"{base}_{noise_type}_{int(snr)}db.wav")
                sf.write(path, mixed, sr)
                rows.append({
                    "audio": path, 
                    "text": text, 
                    "condition": "noise",
                    "noise_type": noise_type, 
                    "snr_db": float(snr)
                })

        for nf in self.real_files:
            noise, n_sr = sf.read(nf, dtype="float32", always_2d=False)
            if n_sr != sr:
                noise = librosa.resample(noise, orig_sr=n_sr, target_sr=sr)
            snr = self.snr_db_list[0]
            mixed = mix_at_snr(speech, noise, snr)
            path = os.path.join(output_dir, f"{base}_real_{int(snr)}db.wav")
            sf.write(path, mixed, sr)
            rows.append({
                "audio": path, 
                "text": text, 
                "condition": "noise",
                "noise_type": "real", 
                "snr_db": float(snr)
            })

        for rate in self.speed_rates:
            if rate == 1.0:
                continue
            sped = speed_perturb(speech, sr, rate)
            path = os.path.join(output_dir, f"{base}_speed{rate}.wav")
            sf.write(path, sped, sr)
            rows.append({
                "audio": path, 
                "text": text, 
                "condition": "speed",
                "rate": float(rate)
            })
        return rows

def build_augmented_manifests(
    records: List[Dict[str, Any]], 
    output_dir: str,
    train_manifest: str, 
    clean_manifest: str,
    eval_manifest: Optional[str] = None,
    eval_ratio: float = 0.1,
    augmenter: Optional[NoiseAugmenter] = None,
    sr: int = 16000
) -> Dict[str, int]:
    augmenter = augmenter or NoiseAugmenter()
    clean_rows: List[Dict[str, Any]] = []
    for rec in records:
        audio = rec.get("audio_file") or rec.get("audio")
        text = rec.get("text") or rec.get("translation") or ""
        if not audio or not os.path.exists(audio):
            logger.warning("skip missing audio: %s", audio)
            continue
        clean_rows.append({"audio": audio, "text": text, "condition": "clean"})

    random.shuffle(clean_rows)
    n_eval = int(len(clean_rows) * eval_ratio)
    eval_rows = clean_rows[:n_eval]
    replay_rows = clean_rows[n_eval:]

    train_rows: List[Dict[str, Any]] = list(replay_rows)
    for row in replay_rows:
        train_rows.extend(augmenter.augment_one(row["audio"], row["text"], output_dir, sr))

    for rows in (train_rows, replay_rows, eval_rows):
        for row in rows:
            if row.get("audio"):
                row["audio"] = to_rel_path(row["audio"])

    write_jsonl(train_rows, train_manifest)
    write_jsonl(replay_rows, clean_manifest)
    if eval_manifest:
        write_jsonl(eval_rows, eval_manifest)
    logger.info("manifests: train=%d clean=%d eval=%d", len(train_rows), len(replay_rows), len(eval_rows))
    return {"train": len(train_rows), "clean": len(replay_rows), "eval": len(eval_rows)}


def _resolve_audio(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.join(str(ROOT), path)


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("records", [])
    if not isinstance(data, list):
        raise ValueError("unsupported input manifest format: " + path)
    for item in data:
        if isinstance(item, dict):
            key = "audio" if "audio" in item else ("audio_file" if "audio_file" in item else None)
            if key:
                item[key] = _resolve_audio(item[key])
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build augmented ASR fine-tuning manifests")
    parser.add_argument("--config", default="configs/training/augmentation.yaml")
    parser.add_argument("--input-manifest", default=None, help="override input_manifest")
    parser.add_argument("--log-file", default=None, help="log file path; empty string disables it")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    log_file = args.log_file if args.log_file is not None else cfg.get("log_file", "logs/augmentation.log")
    setup_logging(log_file, level=getattr(logging, args.log_level))

    input_manifest = args.input_manifest or cfg["input_manifest"]
    records = _load_records(input_manifest)
    log_and_print(f"[augmentation] loaded {len(records)} records from {input_manifest}")

    augmenter = NoiseAugmenter(
        noise_types=cfg.get("noise_types"),
        snr_db_list=cfg.get("snr_db_list"),
        speed_rates=cfg.get("speed_rates"),
        volume_db_range=float(cfg.get("volume_db_range", 3.0)),
        real_noise_dir=cfg.get("real_noise_dir") or None,
        seed=int(cfg.get("seed", 42)),
    )
    summary = build_augmented_manifests(
        records,
        output_dir=cfg["output_dir"],
        train_manifest=cfg["manifest_path"],
        clean_manifest=cfg["clean_manifest_path"],
        eval_manifest=cfg.get("eval_manifest_path"),
        eval_ratio=float(cfg.get("eval_ratio", 0.1)),
        augmenter=augmenter,
        sr=int(cfg.get("sample_rate", 16000)),
    )
    log_and_print(f"[augmentation] done: {summary}")


if __name__ == "__main__":
    main()