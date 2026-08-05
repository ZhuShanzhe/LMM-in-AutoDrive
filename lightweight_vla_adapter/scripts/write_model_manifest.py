from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.scripts.run_offline_inference import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a portable VLA weight manifest")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--modality-ablation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = build_model(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    checksum = sha256(args.checkpoint)
    training = optional_json(args.training_report)
    ablation = optional_json(args.modality_ablation)
    manifest = {
        "schema_version": "portable_vla_model_manifest/1.0",
        "model_name": config["model_name"],
        "checkpoint_file": args.checkpoint.name,
        "checkpoint_sha256": checksum,
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "required_modalities": config.get("required_modalities", []),
        "optional_modalities": config.get("optional_modalities", []),
        "require_raw_camera": config.get("require_raw_camera"),
        "use_structured_bev": config.get("use_structured_bev"),
        "training_summary": (
            {
                "train_samples": training.get("train_samples"),
                "validation_samples": training.get("validation_samples"),
                "best_selection_score": training.get("best_selection_score"),
            }
            if training is not None
            else None
        ),
        "baseline_modality_evaluation": (
            ablation.get("results", [None])[0] if ablation is not None else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "model.sha256").write_text(
        f"{checksum}  {args.checkpoint.name}\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
