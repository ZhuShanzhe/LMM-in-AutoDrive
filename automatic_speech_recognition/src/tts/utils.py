import json
import os
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]

def load_commands(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return [{"index": i, "text": t.strip()} for i, t in enumerate(data, 1) if t and t.strip()]
    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        out = []
        for i, item in enumerate(data, 1):
            text = item.get("text") or item.get("translation") or item.get("original") or ""
            text = str(text).strip()
            if text: out.append({"index": item.get("index", i), "text": text})
        return out
    raise ValueError("commands file must be a list of strings or a list of dicts")

def to_rel_path(path: str) -> str:
    if not path:
        return path
    abs_path = path if os.path.isabs(path) else os.path.join(str(ROOT), path)
    abs_path = os.path.abspath(abs_path)
    try:
        rel = os.path.relpath(abs_path, str(ROOT))
    except ValueError:
        return abs_path
    if rel.startswith(".."):
        return abs_path
    return rel.replace(os.sep, "/")

def resolve_path(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.join(str(ROOT), path)

def load_mapping(path: str, resolve: bool = True) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if resolve and isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                key = "audio_file" if "audio_file" in item else ("audio" if "audio" in item else None)
                if key:
                    item[key] = resolve_path(item[key])
    return data

def write_json(obj: Any, path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)

def to_mono_16k(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio.astype(np.float32)

def save_wav_16k(audio: np.ndarray, orig_sr: int, path: str, target_sr: int = 16000) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    sf.write(path, to_mono_16k(audio, orig_sr, target_sr), target_sr)

def read_wav(path: str, target_sr: int = 16000):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return to_mono_16k(audio, sr, target_sr), target_sr

def rms(x: np.ndarray) -> float:
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
        return pink / (rms(pink) + 1e-12)
    if noise_type == "brown":
        brown = np.cumsum(np.random.randn(length)).astype(np.float32)
        return brown / (rms(brown) + 1e-12)
    raise ValueError("unsupported noise_type: " + str(noise_type))

def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) > len(speech):
        noise = noise[: len(speech)]
    elif len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))[: len(speech)]
    alpha = 10.0 ** (-snr_db / 20.0) * rms(speech) / (rms(noise) + 1e-12)
    mixed = speech + alpha * noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)

def add_noise(speech: np.ndarray, noise_type: str = "white", snr_db: float = 20.0,
              real_noise_file=None, target_sr: int = 16000) -> np.ndarray:
    if real_noise_file:
        noise, n_sr = sf.read(real_noise_file, dtype="float32", always_2d=False)
        noise = to_mono_16k(noise, n_sr, target_sr)
    else:
        noise = make_noise(noise_type, len(speech))
    return mix_at_snr(speech, noise, snr_db)