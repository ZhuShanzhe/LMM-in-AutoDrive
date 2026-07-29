from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.src.contracts import ACTION_LABELS
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
TARGET_KEYS = (
    "action_targets",
    "speed_targets",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight VLA student")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, help="Trusted training tensor file")
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--class-balance",
        choices=("none", "sqrt"),
        default="sqrt",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    return parser.parse_args()


def build_model(config: dict[str, Any]) -> LightweightDecisionAdapter:
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


def load_dataset(path: str | Path) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("dataset must be a tensor dictionary")
    required = set(INPUT_KEYS) | set(TARGET_KEYS)
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError("dataset is missing tensors: " + ", ".join(missing))
    sample_count = int(data["action_targets"].shape[0])
    checked_keys = required | {
        key
        for key in ("teacher_action_logits", "lane_targets", "pointer_targets")
        if key in data
    }
    if any(int(data[key].shape[0]) != sample_count for key in checked_keys):
        raise ValueError("all dataset tensors must share the first dimension")
    return data


def move_batch(
    data: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs: dict[str, torch.Tensor] = {}
    for key in INPUT_KEYS:
        value = data[key][indices].to(device, non_blocking=True)
        if value.is_floating_point():
            value = value.float()
        inputs[key] = value
    targets = {
        key: data[key][indices].to(device, non_blocking=True)
        for key in TARGET_KEYS
    }
    for key in ("teacher_action_logits", "lane_targets", "pointer_targets"):
        if key in data:
            targets[key] = data[key][indices].to(device, non_blocking=True)
    return inputs, targets


def macro_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    *,
    supported_only: bool = False,
) -> float:
    scores = []
    for label in range(num_classes):
        predicted = predictions == label
        expected = targets == label
        if supported_only and not bool(expected.any()):
            continue
        true_positive = int((predicted & expected).sum())
        false_positive = int((predicted & ~expected).sum())
        false_negative = int((~predicted & expected).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return 0.0 if not scores else sum(scores) / len(scores)


def evaluate(
    model: LightweightDecisionAdapter,
    data: dict[str, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, Any]:
    model.eval()
    predictions = []
    targets = []
    speed_errors = []
    lane_correct = 0
    lane_total = 0
    with torch.inference_mode():
        for start in range(0, len(data["action_targets"]), batch_size):
            indices = torch.arange(
                start,
                min(start + batch_size, len(data["action_targets"])),
            )
            inputs, batch_targets = move_batch(data, indices, device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                output = model(**inputs)
            predictions.append(output.action_logits.argmax(dim=-1).cpu())
            targets.append(batch_targets["action_targets"].long().cpu())
            speed_errors.append(
                (
                    output.target_speed_kmh.float()
                    - batch_targets["speed_targets"].float()
                )
                .abs()
                .cpu()
            )
            if "lane_targets" in batch_targets:
                lane_prediction = output.target_lane_logits.argmax(dim=-1)
                lane_target = batch_targets["lane_targets"].long()
                lane_correct += int((lane_prediction == lane_target).sum())
                lane_total += int(lane_target.numel())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    confusion = torch.zeros(
        len(ACTION_LABELS),
        len(ACTION_LABELS),
        dtype=torch.int64,
    )
    for expected, predicted in zip(target.tolist(), prediction.tolist()):
        confusion[expected, predicted] += 1
    return {
        "samples": int(target.numel()),
        "action_accuracy": float((prediction == target).float().mean()),
        "action_macro_f1": macro_f1(prediction, target, len(ACTION_LABELS)),
        "action_macro_f1_supported": macro_f1(
            prediction,
            target,
            len(ACTION_LABELS),
            supported_only=True,
        ),
        "speed_mae_kmh": float(torch.cat(speed_errors).mean()),
        "lane_accuracy": (
            None if lane_total == 0 else float(lane_correct / lane_total)
        ),
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    train_data = load_dataset(args.dataset)
    validation_data = load_dataset(args.validation_dataset)

    device = torch.device(args.device)
    model = build_model(config).to(device)
    if args.init_checkpoint:
        initial_state = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(initial_state)
    class_counts = torch.bincount(
        train_data["action_targets"].long(),
        minlength=len(ACTION_LABELS),
    )
    class_weights = torch.ones(len(ACTION_LABELS), dtype=torch.float32)
    if args.class_balance == "sqrt":
        present = class_counts > 0
        class_weights.zero_()
        class_weights[present] = torch.sqrt(
            class_counts[present].max().float()
            / class_counts[present].float()
        )
        class_weights[present] /= class_weights[present].mean()
    criterion = DistillationLoss(
        action_class_weights=class_weights.to(device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    autocast_dtype = {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output
        else output.with_suffix(".metrics.json")
    )

    generator = torch.Generator().manual_seed(args.seed)
    best_score = -1.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(
            len(train_data["action_targets"]),
            generator=generator,
        )
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            inputs, targets = move_batch(train_data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                prediction = model(**inputs)
                loss, _ = criterion(
                    prediction,
                    action_targets=targets["action_targets"],
                    speed_targets=targets["speed_targets"],
                    teacher_action_logits=targets.get("teacher_action_logits"),
                    lane_targets=targets.get("lane_targets"),
                    pointer_targets=targets.get("pointer_targets"),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach())
            batches += 1

        metrics = evaluate(
            model,
            validation_data,
            batch_size=args.batch_size,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        record = {
            "epoch": epoch,
            "mean_train_loss": epoch_loss / max(batches, 1),
            **metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        score = float(metrics["action_macro_f1_supported"])
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output)
        else:
            epochs_without_improvement += 1
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "best_validation_macro_f1": best_score,
                    "best_checkpoint": str(output),
                    "class_counts": class_counts.tolist(),
                    "class_weights": class_weights.tolist(),
                    "class_balance": args.class_balance,
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if epochs_without_improvement >= args.patience:
            break


if __name__ == "__main__":
    main()
