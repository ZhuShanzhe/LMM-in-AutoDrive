from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from structured_command_parser.scripts.train_modernbert_parser import (
    BatchCollator,
    CommandDataset,
    move_batch,
)
from structured_command_parser.src.modernbert_labels import ACTION_LABELS, DIRECTION_LABELS
from structured_command_parser.src.modernbert_model import ModernBertDrivingModel


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = MODULE_ROOT / "data" / "processed" / "english_pseudolabels" / "validation.jsonl"
DEFAULT_MODEL = Path("/root/autodl-tmp/models/modernbert-drive-command-base")


def binary_f1(predicted: torch.Tensor, gold: torch.Tensor) -> float:
    true_positive = int((predicted & gold).sum())
    false_positive = int((predicted & ~gold).sum())
    false_negative = int((~predicted & gold).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return 2 * precision * recall / max(1e-12, precision + recall)


def calibrate(probabilities: torch.Tensor, gold: torch.Tensor, labels: tuple[str, ...]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    grid = [value / 100 for value in range(10, 96, 5)]
    for index, label in enumerate(labels):
        class_gold = gold[:, index]
        if not bool(class_gold.any()):
            thresholds[label] = 0.5
            continue
        scored = [
            (binary_f1(probabilities[:, index] >= threshold, class_gold), threshold)
            for threshold in grid
        ]
        best_score = max(score for score, _ in scored)
        near_best = [
            threshold for score, threshold in scored if score >= best_score - 0.001
        ]
        thresholds[label] = max(near_best)
    return thresholds


def multilabel_metrics(
    probabilities: torch.Tensor,
    gold: torch.Tensor,
    labels: tuple[str, ...],
    thresholds: dict[str, float],
) -> dict[str, float]:
    threshold_tensor = torch.tensor([thresholds[label] for label in labels])
    predicted = probabilities >= threshold_tensor
    true_positive = int((predicted & gold).sum())
    false_positive = int((predicted & ~gold).sum())
    false_negative = int((~predicted & gold).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "exact_match": float((predicted == gold).all(dim=1).float().mean()),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ModernBERT multilabel thresholds")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda")
    dataset = CommandDataset([args.dataset])
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=BatchCollator(tokenizer, args.max_length),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = ModernBertDrivingModel.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    action_probabilities: list[torch.Tensor] = []
    action_gold: list[torch.Tensor] = []
    direction_probabilities: list[torch.Tensor] = []
    direction_gold: list[torch.Tensor] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**batch["model_inputs"])
        action_probabilities.append(torch.sigmoid(logits["actions"]).float().cpu())
        action_gold.append(batch["actions"].bool().cpu())
        direction_probabilities.append(torch.sigmoid(logits["directions"]).float().cpu())
        direction_gold.append(batch["directions"].bool().cpu())

    action_probs = torch.cat(action_probabilities)
    actions = torch.cat(action_gold)
    direction_probs = torch.cat(direction_probabilities)
    directions = torch.cat(direction_gold)
    action_thresholds = calibrate(action_probs, actions, ACTION_LABELS)
    direction_thresholds = calibrate(direction_probs, directions, DIRECTION_LABELS)
    default_actions = {label: 0.5 for label in ACTION_LABELS}
    default_directions = {label: 0.5 for label in DIRECTION_LABELS}
    report: dict[str, Any] = {
        "schema": "modernbert-driving-thresholds-v1",
        "calibration_dataset": str(args.dataset),
        "calibration_rows": len(dataset),
        "action_thresholds": action_thresholds,
        "direction_thresholds": direction_thresholds,
        "before": {
            "actions": multilabel_metrics(action_probs, actions, ACTION_LABELS, default_actions),
            "directions": multilabel_metrics(
                direction_probs, directions, DIRECTION_LABELS, default_directions
            ),
        },
        "after": {
            "actions": multilabel_metrics(action_probs, actions, ACTION_LABELS, action_thresholds),
            "directions": multilabel_metrics(
                direction_probs, directions, DIRECTION_LABELS, direction_thresholds
            ),
        },
    }
    output = args.model / "inference_config.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
