from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from structured_command_parser.src.compositional_frame import (
    SEMANTIC_TAG_LABELS,
)
from structured_command_parser.src.modernbert_model import (
    ModernBertDrivingModel,
)


LABEL_TO_ID = {
    label: index for index, label in enumerate(SEMANTIC_TAG_LABELS)
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SpanDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = _read_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class SpanCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [row["text_en"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        labels = torch.full(offsets.shape[:2], -100, dtype=torch.long)
        for row_index, row in enumerate(rows):
            spans = row["spans"]
            for token_index, (start, end) in enumerate(offsets[row_index].tolist()):
                if start == end:
                    continue
                labels[row_index, token_index] = LABEL_TO_ID["O"]
                for span in spans:
                    if end <= span["start"] or start >= span["end"]:
                        continue
                    prefix = "B" if start <= span["start"] < end else "I"
                    labels[row_index, token_index] = LABEL_TO_ID[
                        f"{prefix}_{span['role']}"
                    ]
                    break
        return {"model_inputs": encoded, "labels": labels}


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    valid = labels >= 0
    positive_gold = (labels != LABEL_TO_ID["O"]) & valid
    positive_pred = (predictions != LABEL_TO_ID["O"]) & valid
    correct_positive = (predictions == labels) & positive_gold
    tp = int(correct_positive.sum())
    fp = int((positive_pred & ~correct_positive).sum())
    fn = int((positive_gold & ~correct_positive).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "token_accuracy": float((predictions[valid] == labels[valid]).float().mean()),
    }


@torch.inference_mode()
def evaluate(
    model: ModernBertDrivingModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    logits_parts = []
    label_parts = []
    for batch in loader:
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch["model_inputs"].items()
        }
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**inputs)["semantic_tags"]
        valid = labels >= 0
        logits_parts.append(logits[valid].float().cpu())
        label_parts.append(labels[valid].cpu())
    return _metrics(torch.cat(logits_parts), torch.cat(label_parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    collator = SpanCollator(tokenizer, args.max_length)
    train_loader = DataLoader(
        SpanDataset(args.data / "train.jsonl"),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        SpanDataset(args.data / "validation.jsonl"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    model = ModernBertDrivingModel.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.semantic_token_head.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        model.semantic_token_head.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    class_weights = torch.tensor(
        [0.15, 2.0, 1.0, 2.0, 1.0],
        device=device,
    )

    best_f1 = -1.0
    best_state = None
    history = []
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["model_inputs"].items()
            }
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs)["semantic_tags"]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    weight=class_weights,
                    ignore_index=-100,
                )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
        metrics = evaluate(model, validation_loader, device)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(1, len(train_loader)),
            }
        )
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["token_f1"] > best_f1:
            best_f1 = metrics["token_f1"]
            best_state = {
                key: value.detach().cpu()
                for key, value in model.semantic_token_head.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No semantic-head checkpoint was produced")
    model.semantic_token_head.load_state_dict(best_state)
    model.semantic_head_loaded = True
    model.save_semantic_head(args.model)
    report = {
        "schema": "modernbert-semantic-token-head-v1",
        "labels": list(SEMANTIC_TAG_LABELS),
        "train_rows": len(train_loader.dataset),
        "validation_rows": len(validation_loader.dataset),
        "best_validation_token_f1": best_f1,
        "frozen_backbone": True,
        "elapsed_seconds": perf_counter() - started,
        "history": history,
    }
    (args.model / "semantic_token_head_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
