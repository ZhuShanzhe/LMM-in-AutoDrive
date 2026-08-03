from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from structured_command_parser.src.modernbert_labels import (
    ACTION_LABELS,
    CATEGORY_LABELS,
    CHANGE_LABELS,
    DIRECTION_LABELS,
    STATUS_LABELS,
    URGENCY_LABELS,
    label_schema,
)
from structured_command_parser.src.modernbert_model import ModernBertDrivingModel


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = MODULE_ROOT / "data" / "processed" / "english_pseudolabels"
DEFAULT_BASE_MODEL = Path("/root/autodl-tmp/models/ModernBERT-base")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/models/modernbert-drive-command-base")


class CommandDataset(Dataset[dict[str, Any]]):
    def __init__(self, paths: list[Path], limit: int | None = None) -> None:
        self.rows: list[dict[str, Any]] = []
        for path in paths:
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.rows.append(
                            {
                                "sample_id": row["sample_id"],
                                "text_en": row["text_en"],
                                "expected": row["expected"],
                                "weight": float(row.get("pseudo_label", {}).get("weight", 1.0)),
                            }
                        )
                        if limit is not None and len(self.rows) >= limit:
                            return

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class BatchCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.action_index = {value: index for index, value in enumerate(ACTION_LABELS)}
        self.status_index = {value: index for index, value in enumerate(STATUS_LABELS)}
        self.category_index = {value: index for index, value in enumerate(CATEGORY_LABELS)}
        self.urgency_index = {value: index for index, value in enumerate(URGENCY_LABELS)}
        self.direction_index = {value: index for index, value in enumerate(DIRECTION_LABELS)}
        self.change_index = {value: index for index, value in enumerate(CHANGE_LABELS)}

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [row["text_en"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        actions = torch.zeros((len(rows), len(ACTION_LABELS)), dtype=torch.float32)
        directions = torch.zeros((len(rows), len(DIRECTION_LABELS)), dtype=torch.float32)
        status: list[int] = []
        category: list[int] = []
        urgency: list[int] = []
        change: list[int] = []
        for row_index, row in enumerate(rows):
            expected = row["expected"]
            for action in expected.get("actions", []):
                if action in self.action_index:
                    actions[row_index, self.action_index[action]] = 1.0
            for direction in expected.get("directions", []):
                if direction in self.direction_index:
                    directions[row_index, self.direction_index[direction]] = 1.0
            status.append(self.status_index.get(expected.get("status"), 1))
            category.append(self.category_index.get(expected.get("category"), 0))
            urgency.append(self.urgency_index.get(expected.get("urgency"), 0))
            change.append(self.change_index.get(expected.get("change"), 0))
        return {
            "model_inputs": encoded,
            "actions": actions,
            "directions": directions,
            "status": torch.tensor(status, dtype=torch.long),
            "category": torch.tensor(category, dtype=torch.long),
            "urgency": torch.tensor(urgency, dtype=torch.long),
            "change": torch.tensor(change, dtype=torch.long),
            "weights": torch.tensor([row["weight"] for row in rows], dtype=torch.float32),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_counts(dataset: CommandDataset) -> dict[str, Counter[str]]:
    counts = {
        "actions": Counter(),
        "status": Counter(),
        "category": Counter(),
        "urgency": Counter(),
        "directions": Counter(),
        "change": Counter(),
    }
    for row in dataset.rows:
        expected = row["expected"]
        counts["actions"].update(expected.get("actions", []))
        counts["status"][expected.get("status", "NEEDS_CLARIFICATION")] += 1
        counts["category"][expected.get("category", "BASIC_CONTROL")] += 1
        counts["urgency"][expected.get("urgency", "NORMAL")] += 1
        counts["directions"].update(expected.get("directions", []))
        counts["change"][expected.get("change", "NONE")] += 1
    return counts


def ce_weights(labels: tuple[str, ...], counts: Counter[str], device: torch.device) -> torch.Tensor:
    total = max(1, sum(counts.values()))
    raw = [math.sqrt(total / max(1, counts[label])) if counts[label] else 0.0 for label in labels]
    nonzero = [value for value in raw if value > 0]
    scale = sum(nonzero) / len(nonzero) if nonzero else 1.0
    return torch.tensor([min(10.0, value / scale) if value else 0.0 for value in raw], device=device)


def pos_weights(labels: tuple[str, ...], counts: Counter[str], total: int, device: torch.device) -> torch.Tensor:
    values = []
    for label in labels:
        positive = counts[label]
        if positive == 0:
            values.append(1.0)
        else:
            values.append(min(20.0, math.sqrt(max(1, total - positive) / positive)))
    return torch.tensor(values, dtype=torch.float32, device=device)


class MultiTaskLoss:
    def __init__(self, dataset: CommandDataset, device: torch.device) -> None:
        counts = class_counts(dataset)
        self.actions = nn.BCEWithLogitsLoss(
            pos_weight=pos_weights(ACTION_LABELS, counts["actions"], len(dataset), device),
            reduction="none",
        )
        self.directions = nn.BCEWithLogitsLoss(
            pos_weight=pos_weights(DIRECTION_LABELS, counts["directions"], len(dataset), device),
            reduction="none",
        )
        self.status = nn.CrossEntropyLoss(
            weight=ce_weights(STATUS_LABELS, counts["status"], device), reduction="none"
        )
        self.category = nn.CrossEntropyLoss(
            weight=ce_weights(CATEGORY_LABELS, counts["category"], device), reduction="none"
        )
        self.urgency = nn.CrossEntropyLoss(
            weight=ce_weights(URGENCY_LABELS, counts["urgency"], device), reduction="none"
        )
        self.change = nn.CrossEntropyLoss(
            weight=ce_weights(CHANGE_LABELS, counts["change"], device), reduction="none"
        )

    def __call__(self, logits: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        weights = batch["weights"]
        per_row = (
            2.0 * self.actions(logits["actions"], batch["actions"]).mean(dim=1)
            + 0.5 * self.status(logits["status"], batch["status"])
            + 0.5 * self.category(logits["category"], batch["category"])
            + 0.5 * self.urgency(logits["urgency"], batch["urgency"])
            + 0.5 * self.directions(logits["directions"], batch["directions"]).mean(dim=1)
            + 0.5 * self.change(logits["change"], batch["change"])
        )
        return (per_row * weights).sum() / weights.sum().clamp_min(1e-6)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            {name: tensor.to(device, non_blocking=True) for name, tensor in value.items()}
            if key == "model_inputs"
            else value.to(device, non_blocking=True)
        )
        for key, value in batch.items()
    }


@torch.inference_mode()
def evaluate(
    model: ModernBertDrivingModel,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    action_thresholds: dict[str, float] | None = None,
    direction_thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    model.eval()
    total = 0
    action_exact = 0
    action_tp = action_fp = action_fn = 0
    correct = Counter()
    action_cutoffs = torch.tensor(
        [(action_thresholds or {}).get(label, 0.5) for label in ACTION_LABELS],
        device=device,
    )
    direction_cutoffs = torch.tensor(
        [
            (direction_thresholds or {}).get(label, 0.5)
            for label in DIRECTION_LABELS
        ],
        device=device,
    )
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**batch["model_inputs"])
        predicted_actions = torch.sigmoid(logits["actions"]) >= action_cutoffs
        gold_actions = batch["actions"].bool()
        action_exact += int((predicted_actions == gold_actions).all(dim=1).sum())
        action_tp += int((predicted_actions & gold_actions).sum())
        action_fp += int((predicted_actions & ~gold_actions).sum())
        action_fn += int((~predicted_actions & gold_actions).sum())
        for name in ("status", "category", "urgency", "change"):
            correct[name] += int((logits[name].argmax(dim=1) == batch[name]).sum())
        predicted_directions = torch.sigmoid(logits["directions"]) >= direction_cutoffs
        correct["directions"] += int((predicted_directions == batch["directions"].bool()).all(dim=1).sum())
        total += gold_actions.shape[0]
    precision = action_tp / max(1, action_tp + action_fp)
    recall = action_tp / max(1, action_tp + action_fn)
    return {
        "samples": total,
        "action_exact_match": action_exact / max(1, total),
        "action_micro_precision": precision,
        "action_micro_recall": recall,
        "action_micro_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "status_accuracy": correct["status"] / max(1, total),
        "category_accuracy": correct["category"] / max(1, total),
        "urgency_accuracy": correct["urgency"] / max(1, total),
        "direction_exact_match": correct["directions"] / max(1, total),
        "change_accuracy": correct["change"] / max(1, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ModernBERT for driving commands")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ModernBERT training")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    train_paths = [args.data / "train.jsonl", args.data / "train_sparse_augmentation.jsonl"]
    train_dataset = CommandDataset(train_paths, limit=args.train_limit)
    validation_dataset = CommandDataset(
        [args.data / "validation.jsonl"], limit=args.validation_limit
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("Training and validation datasets must not be empty")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    collator = BatchCollator(tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = ModernBertDrivingModel.from_pretrained(
        args.base_model, dtype=torch.bfloat16
    ).to(device)
    criterion = MultiTaskLoss(train_dataset, device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = -1.0
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_started = perf_counter()
        for step, raw_batch in enumerate(train_loader, start=1):
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch["model_inputs"])
                loss = criterion(logits, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach())
            if step % 100 == 0 or step == len(train_loader):
                elapsed = perf_counter() - epoch_started
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={running_loss / step:.5f} "
                    f"samples_per_second={step * args.batch_size / max(elapsed, 1e-6):.1f}",
                    flush=True,
                )
        metrics = evaluate(model, validation_loader, device)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(1, len(train_loader)),
                "epoch_seconds": perf_counter() - epoch_started,
            }
        )
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        score = metrics["action_micro_f1"] + metrics["status_accuracy"]
        if score > best_score:
            best_score = score
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)

    summary = {
        "schema": "modernbert-driving-command-training-v1",
        "base_model": str(args.base_model),
        "output": str(args.output),
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "elapsed_seconds": perf_counter() - started,
        "label_schema": label_schema(),
        "history": history,
        "metric_scope": "PSEUDO_LABEL_TEACHER_AGREEMENT_NOT_HUMAN_GOLD_ACCURACY",
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "label_schema.json").write_text(
        json.dumps(label_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
