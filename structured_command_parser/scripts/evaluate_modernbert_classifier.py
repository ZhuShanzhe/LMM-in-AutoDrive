from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from structured_command_parser.scripts.train_modernbert_parser import (
    BatchCollator,
    CommandDataset,
    evaluate,
)
from structured_command_parser.src.modernbert_model import ModernBertDrivingModel


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST = MODULE_ROOT / "data" / "processed" / "english_pseudolabels" / "test.jsonl"
DEFAULT_MODEL = Path("/root/autodl-tmp/models/modernbert-drive-command-base")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ModernBERT classifier in batches")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    device = torch.device("cuda")
    dataset = CommandDataset([args.dataset], limit=args.limit)
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
    inference_config_path = args.model / "inference_config.json"
    inference_config = (
        json.loads(inference_config_path.read_text(encoding="utf-8"))
        if inference_config_path.is_file()
        else {}
    )
    metrics = evaluate(
        model,
        loader,
        device,
        action_thresholds=inference_config.get("action_thresholds"),
        direction_thresholds=inference_config.get("direction_thresholds"),
    )
    report = {
        "schema": "modernbert-driving-command-test-v1",
        "model": str(args.model),
        "dataset": str(args.dataset),
        "metrics": metrics,
        "metric_scope": "PSEUDO_LABEL_TEACHER_AGREEMENT_NOT_HUMAN_GOLD_ACCURACY",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
