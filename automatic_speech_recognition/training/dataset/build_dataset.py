import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tts.utils import (add_noise, load_commands, load_mapping, read_wav, save_wav_16k, to_rel_path, write_json)
from src.utils import log_and_print, setup_logging


def build_standard(config_path: str) -> None:
    from src import Qwen3TTSService

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    commands_file = os.path.join(ROOT, cfg["commands_file"])
    output_dir = os.path.join(ROOT, cfg["output_dir"])
    mapping_file = os.path.join(ROOT, cfg["mapping_file"])

    commands = load_commands(commands_file)
    log_and_print(f"[build] loaded {len(commands)} commands from {commands_file}")
    if not commands:
        raise SystemExit("no commands loaded")

    service = Qwen3TTSService(
        model_id_or_path=cfg["model_id_or_path"],
        device_map=cfg.get("device_map"),
        num_gpus=int(cfg.get("num_gpus", 1)),
        dtype=cfg.get("dtype", "bfloat16"),
        attn_implementation=cfg.get("attn_implementation", ""),
        gen_kwargs=cfg.get("gen_kwargs") or {},
        output_dir=output_dir,
        target_sample_rate=int(cfg.get("target_sample_rate", 16000)),
    )

    speakers = cfg["speakers"]
    language = cfg.get("language", "Chinese")
    batch_size = int(cfg.get("batch_size", 4))

    records = []
    for start in range(0, len(commands), batch_size):
        chunk = commands[start:start + batch_size]
        texts = [c["text"] for c in chunk]
        spk = [speakers[(c["index"] - 1) % len(speakers)] for c in chunk]
        names = [f"std_{c['index']:05d}.wav" for c in chunk]

        batch_records = service.generate_batch(
            texts=texts,
            output_dir=output_dir,
            file_names=names,
            speakers=spk,
            language=language,
        )

        for c, speaker, rec in zip(chunk, spk, batch_records):
            records.append({
                "index": c["index"],
                "text": c["text"],
                "audio_file": rec["audio_file"],
                "sample_rate": rec["sample_rate"],
                "speaker": speaker,
                "dataset": "standard",
            })
        log_and_print(f"[build] standard progress {len(records)}/{len(commands)}")

    for rec in records:
        rec["audio_file"] = to_rel_path(rec["audio_file"])
    write_json(records, mapping_file)
    log_and_print(f"[build] standard dataset done: {len(records)} files -> {mapping_file}")


def build_dialect(config_path: str) -> None:
    from src import Qwen3TTSService

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    commands = load_commands(os.path.join(ROOT, cfg["commands_file"]))
    if not commands:
        raise SystemExit("no commands loaded")
    output_dir = os.path.join(ROOT, cfg["output_dir"])
    mapping_file = os.path.join(ROOT, cfg["mapping_file"])
    profiles = cfg["profiles"]
    per_profile_all = bool(cfg.get("per_profile_all", True))
    batch_size = int(cfg.get("batch_size", 4))
    language = cfg.get("language", "Chinese")

    service = Qwen3TTSService(
        model_id_or_path=cfg["model_id_or_path"],
        device_map=cfg.get("device_map"),
        num_gpus=int(cfg.get("num_gpus", 1)),
        dtype=cfg.get("dtype", "bfloat16"),
        attn_implementation=cfg.get("attn_implementation", ""),
        gen_kwargs=cfg.get("gen_kwargs") or {},
        output_dir=output_dir,
        target_sample_rate=int(cfg.get("target_sample_rate", 16000)),
    )

    resolved = []
    for profile in profiles:
        preset = Qwen3TTSService.DIALECT_SPEAKERS.get(profile.get("name"), {})
        speaker = profile.get("speaker") or preset.get("speaker")
        instruct = profile.get("instruct") or preset.get("instruct")
        if not speaker:
            raise SystemExit(
                f"profile '{profile.get('name')}' has no speaker; set 'speaker' "
                f"or use a built-in dialect name from {Qwen3TTSService.supported_dialects()}")
        resolved.append({"name": profile["name"], "speaker": speaker, "instruct": instruct})

    plan = []
    if per_profile_all:
        for profile in resolved:
            for cmd in commands:
                plan.append({**cmd, "profile": profile})
    else:
        for i, cmd in enumerate(commands):
            plan.append({**cmd, "profile": resolved[i % len(resolved)]})

    records = []
    pending = []
    reused = 0
    for item in plan:
        path = os.path.join(output_dir, f"{item['profile']['name']}_{item['index']:05d}.wav")
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            records.append({
                "index": item["index"],
                "text": item["text"],
                "audio_file": path,
                "sample_rate": service.target_sample_rate,
                "dialect": item["profile"]["name"],
                "speaker": item["profile"].get("speaker"),
                "dataset": "dialect",
            })
            reused += 1
            log_and_print(f"[build] dialect reuse {reused}/{len(plan)}: {to_rel_path(path)}")
        else:
            pending.append(item)
    if reused:
        log_and_print(f"[build] dialect resume: {reused}/{len(plan)} files already exist, {len(pending)} to generate")

    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        wavs = service.generate_batch(
            texts=[c["text"] for c in chunk],
            output_dir=output_dir,
            file_names=[f"{c['profile']['name']}_{c['index']:05d}.wav" for c in chunk],
            speakers=[c["profile"].get("speaker") for c in chunk],
            instructs=[c["profile"].get("instruct") for c in chunk],
            language=language,
        )
        for c, rec in zip(chunk, wavs):
            records.append({
                "index": c["index"],
                "text": c["text"],
                "audio_file": rec["audio_file"],
                "sample_rate": rec["sample_rate"],
                "dialect": c["profile"]["name"],
                "speaker": c["profile"].get("speaker"),
                "dataset": "dialect",
            })
        log_and_print(f"[build] dialect progress {len(records)}/{len(plan)}")

    for rec in records:
        rec["audio_file"] = to_rel_path(rec["audio_file"])
    write_json(records, mapping_file)
    log_and_print(f"[build] dialect dataset done: {len(records)} files -> {mapping_file}")


def build_noise(config_path: str, log_file: Optional[str] = None, log_level: str = "INFO") -> None:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    sr = int(cfg.get("sample_rate", 16000))
    np.random.seed(int(cfg.get("seed", 42)))
    noise_types = cfg.get("noise_types", ["white"])
    snr_list = [float(s) for s in cfg.get("snr_db_list", [20])]
    real_noise_dir = cfg.get("real_noise_dir") or ""
    real_files = ([os.path.join(real_noise_dir, n) for n in sorted(os.listdir(real_noise_dir)) if n.lower().endswith(".wav")] if real_noise_dir and os.path.isdir(real_noise_dir) else [])
    combos = [(nt, snr) for nt in noise_types for snr in snr_list]
    variants_per_file = int(cfg.get("variants_per_file") or len(combos))
    progress_every = max(1, int(cfg.get("progress_every", 50)))

    for task in cfg["tasks"]:
        task_log = log_file if log_file is not None else task.get("log_file", f"logs/tts/build_{task['name']}.log")
        setup_logging(task_log, level=getattr(logging, log_level))
        log_and_print(f"[build] noise task '{task.get('name')}' started (log: {task_log})")
        clean_records = load_mapping(os.path.join(ROOT, task["input_mapping"]))
        output_dir = os.path.join(ROOT, task["output_dir"])
        mapping_file = os.path.join(ROOT, task["mapping_file"])
        os.makedirs(output_dir, exist_ok=True)
        name = task.get("name", "noise")
        total = len(clean_records)
        expected = total * variants_per_file + (total if real_files else 0)
        log_and_print(
            f"[build] noise task '{name}' config: noise_types={noise_types} snr_db_list={[int(s) for s in snr_list]} "
            f"combos={len(combos)} variants_per_file={variants_per_file} clean_records={total} "
            f"real_noise_dir={real_noise_dir or 'none'} real_noise_files={len(real_files)} expected_outputs={expected}"
        )

        records = []
        skipped = 0
        for done, item in enumerate(clean_records, start=1):
            src = item.get("audio_file") or item.get("audio")
            if not src or not os.path.exists(src):
                skipped += 1
                log_and_print(f"[build] {name} skip {done}/{total}: missing audio {src}", logging.WARNING)
                continue
            speech, _ = read_wav(src, sr)
            base = os.path.splitext(os.path.basename(src))[0]
            for variant in range(variants_per_file):
                noise_type, snr = combos[(len(records) + variant) % len(combos)]
                noisy = add_noise(speech, noise_type, snr, None, sr)
                out = os.path.join(output_dir, f"{base}_{noise_type}_{int(snr)}db.wav")
                save_wav_16k(noisy, sr, out, sr)
                records.append({
                    "index": item.get("index"),
                    "text": item.get("text", ""),
                    "audio_file": out,
                    "noise_type": noise_type,
                    "snr_db": snr, "dataset": task.get("name", "noise")
                })
            if done % progress_every == 0 or done == total:
                log_and_print(f"[build] {name} progress {done}/{total} (audio_files={len(records)})")
            if real_files:
                nf = real_files[len(base) % len(real_files)]
                snr = snr_list[0]
                noisy = add_noise(speech, "real", snr, nf, sr)
                out = os.path.join(output_dir, f"{base}_real_{int(snr)}db.wav")
                save_wav_16k(noisy, sr, out, sr)
                records.append({
                    "index": item.get("index"), 
                    "text": item.get("text", ""),
                    "audio_file": out, 
                    "noise_type": "real",
                    "snr_db": snr, "dataset": task.get("name", "noise")
                })
        for rec in records:
            rec["audio_file"] = to_rel_path(rec["audio_file"])
        write_json(records, mapping_file)
        dist: dict = {}
        for rec in records:
            dist[rec["noise_type"]] = dist.get(rec["noise_type"], 0) + 1
        log_and_print(f"[build] noise task '{name}' distribution: {dist}")
        if skipped:
            log_and_print(f"[build] noise task '{name}' skipped {skipped}/{total} records with missing audio", logging.WARNING)
        log_and_print(f"[build] noise task '{name}' done: {len(records)} files -> {to_rel_path(mapping_file)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build speech datasets with Qwen3-TTS")
    parser.add_argument("--kind", default="standard", choices=["standard", "dialect", "noise"])
    parser.add_argument("--config", default="configs/tts/standard.yaml", help="path to tts yaml config (relative to repo root)")
    parser.add_argument("--log-file", default=None, help="log file path; empty string disables it")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if args.kind == "noise":
        build_noise(config_path, log_file=args.log_file, log_level=args.log_level)
        return

    log_file = args.log_file if args.log_file is not None else cfg.get("log_file", "logs/build_dataset.log")
    setup_logging(log_file, level=getattr(logging, args.log_level))

    log_and_print(f"[build] using config: {args.config} kind={args.kind}")
    if args.kind == "standard":
        build_standard(config_path)
    elif args.kind == "dialect":
        build_dialect(config_path)
    else:
        raise SystemExit("unsupported kind: " + args.kind)


if __name__ == "__main__":
    main()
