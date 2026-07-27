from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.src.decision_adapter import LightweightDecisionAdapter
from lightweight_vla_adapter.src.distillation import DistillationLoss


INPUT_KEYS = (
    "camera_bev",
    "lidar_bev",
    "ego_features",
    "candidate_features",
    "candidate_mask",
    "intent_tokens",
    "intent_mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight VLA student")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, help="Trusted batched torch tensor file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_model(config: dict) -> LightweightDecisionAdapter:
    return LightweightDecisionAdapter(
        camera_channels=config["camera_channels"],
        lidar_channels=config["lidar_channels"],
        candidate_dim=config["candidate_dim"],
        ego_dim=config["ego_dim"],
        intent_dim=config["intent_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        bev_grid=tuple(config["bev_grid"]),
    )


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    data = torch.load(args.dataset, map_location="cpu", weights_only=True)
    required = set(INPUT_KEYS) | {"action_targets", "speed_targets"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError("dataset is missing tensors: " + ", ".join(missing))
    sample_count = int(data["action_targets"].shape[0])
    if any(int(data[key].shape[0]) != sample_count for key in required):
        raise ValueError("all dataset tensors must share the first dimension")

    device = torch.device(args.device)
    model = build_model(config).to(device)
    criterion = DistillationLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    generator = torch.Generator().manual_seed(2026)
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(sample_count, generator=generator)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, sample_count, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            inputs = {key: data[key][indices].to(device) for key in INPUT_KEYS}
            output = model(**inputs)
            loss, _ = criterion(
                output,
                action_targets=data["action_targets"][indices].to(device),
                speed_targets=data["speed_targets"][indices].to(device),
                teacher_action_logits=(
                    data["teacher_action_logits"][indices].to(device)
                    if "teacher_action_logits" in data
                    else None
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "mean_loss": round(epoch_loss / max(batches, 1), 6),
                }
            )
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)


if __name__ == "__main__":
    main()
