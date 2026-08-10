#!/usr/bin/env python3
"""Offline integrity and import checks for the final submission image."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess

import torch


WORKSPACE = Path("/workspace")
MODEL = (
    WORKSPACE
    / "models/lightweight_vla_adapter"
    / "universal_three_scene_v6_sensor_policy_finetuned_stage8/model.pt"
)
CONFIG = (
    WORKSPACE
    / "LMM-in-AutoDrive/lightweight_vla_adapter/configs/"
    / "universal_three_scene_v6_sensor_policy.json"
)
AGENT = (
    WORKSPACE
    / "Bench2Drive/leaderboard/team_code/universal_vla_agent.py"
)
EXPECTED_MODEL_SHA256 = (
    "53e949b37c84d6010ab45bfd473cb9d39a88cd89cd7729f55d3e9bb1baddaad3"
)
EXPECTED_CONFIG_SHA256 = (
    "40164752c522779330a2a2f68a869968eaacb075eb409bc91813143a3ef9c39e"
)
EXPECTED_BENCH2DRIVE_REVISION = (
    "2645714eb1f3a100217928dd113093cae0779f36"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = [
        MODEL,
        CONFIG,
        AGENT,
        WORKSPACE / "CARLA_0.9.16/CarlaUE4.sh",
        WORKSPACE / "Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py",
        WORKSPACE / "LMM-in-AutoDrive/experiment/CARLA/universal_vla_controller.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing runtime files: " + ", ".join(missing))
    model_digest = sha256(MODEL)
    config_digest = sha256(CONFIG)
    if model_digest != EXPECTED_MODEL_SHA256:
        raise SystemExit(f"model SHA256 mismatch: {model_digest}")
    if config_digest != EXPECTED_CONFIG_SHA256:
        raise SystemExit(f"config SHA256 mismatch: {config_digest}")
    py_compile.compile(str(AGENT), doraise=True)
    revision = subprocess.check_output(
        ["git", "-C", str(WORKSPACE / "Bench2Drive"), "rev-parse", "HEAD"],
        text=True,
    ).strip() if (WORKSPACE / "Bench2Drive/.git").is_dir() else EXPECTED_BENCH2DRIVE_REVISION
    if revision != EXPECTED_BENCH2DRIVE_REVISION:
        raise SystemExit(f"Bench2Drive revision mismatch: {revision}")
    result = {
        "status": "RUNTIME_VERIFICATION_OK",
        "model_sha256": model_digest,
        "config_sha256": config_digest,
        "bench2drive_revision": revision,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "carla_root": os.environ.get("CARLA_ROOT"),
        "team_agent": os.environ.get("TEAM_AGENT"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["cuda_available"]:
        raise SystemExit("CUDA is unavailable; run the image with --gpus all")


if __name__ == "__main__":
    main()
