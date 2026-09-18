import os
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS: Dict[str, Any] = {
    "device_map": "cuda:0",
    "dtype": "bfloat16",
    "sample_rate": 16000,
    "prompt": "Transcribe the audio into Chinese text.",
    "max_length": 2048,
    "method": "lora",
    "replay_ratio": 0.3,
    "output_dir": "outputs/asr_finetune",
    "batch_size": 2,
    "grad_accum": 8,
    "learning_rate": 1e-4,
    "epochs": 3,
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "save_steps": 200,
    "gradient_checkpointing": True,
    "bf16": True,
    "eval_batch_size": 1,
    "eval_max_new_tokens": 256,
}

def load_config(path: str, defaults: Dict[str, Any] = DEFAULTS) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    merged = dict(defaults)
    merged.update(cfg)
    return merged

def resolve_path(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return str(REPO_ROOT / path)

def resolve_paths(cfg: Dict[str, Any], keys) -> Dict[str, Any]:
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, str) and value:
            cfg[key] = resolve_path(value)
    return cfg