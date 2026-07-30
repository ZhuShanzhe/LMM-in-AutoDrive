from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "models" / "lightweight_vla_adapter" / "v10" / "model.pt"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.src.contracts import ACTION_LABELS
from lightweight_vla_adapter.scripts.train_student import (
    build_model,
    evaluate,
    load_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained VLA student")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    device = torch.device(args.device)
    model = build_model(config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    data = load_dataset(args.dataset)
    dtype = {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    metrics = evaluate(
        model,
        data,
        batch_size=args.batch_size,
        device=device,
        autocast_dtype=dtype,
    )
    safety_slices = {}
    if "safety_targets" in data:
        safety = data["safety_targets"].bool()
        for label, mask in (("safe", safety), ("unsafe", ~safety)):
            if not bool(mask.any()):
                continue
            subset = {
                key: value[mask]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and int(value.shape[0]) == int(safety.shape[0])
                else value
                for key, value in data.items()
            }
            safety_slices[label] = evaluate(
                model,
                subset,
                batch_size=args.batch_size,
                device=device,
                autocast_dtype=dtype,
            )
    confusion = metrics["confusion_matrix"]
    per_class = {}
    for index, label in enumerate(ACTION_LABELS):
        row = confusion[index]
        support = sum(row)
        per_class[label] = {
            "support": support,
            "recall": 0.0 if support == 0 else row[index] / support,
        }
    result = {
        **metrics,
        "per_class": per_class,
        "safety_slices": safety_slices,
        "checkpoint": str(Path(args.checkpoint)),
        "dataset": str(Path(args.dataset)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
