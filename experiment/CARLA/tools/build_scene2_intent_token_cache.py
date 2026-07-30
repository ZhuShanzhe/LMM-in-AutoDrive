"""Precompute ModernBERT intent tokens and unload the language model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightweight_vla_adapter.src.intent_encoder import ModernBertIntentEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.intents.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    encoder = ModernBertIntentEncoder.from_pretrained(
        args.model,
        dtype=dtype,
    ).to(args.device).eval()
    records = {}
    latencies = {}
    with torch.inference_mode():
        for intent in payload["driving_intents"]:
            request_id = intent["request_id"]
            text = (
                intent["input"].get("translated_text")
                or intent["input"]["normalized_text"]
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
            )
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            tokens, mask = encoder(**inputs)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies[request_id] = round(
                (time.perf_counter() - started) * 1000.0, 3
            )
            records[request_id] = {
                "text": text,
                "tokens": tokens.float().cpu(),
                "mask": mask.cpu(),
            }
    output = {
        "schema_version": "scene2_intent_token_cache/v1",
        "model_path": str(args.model.resolve()),
        "max_length": args.max_length,
        "latency_ms": latencies,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(records),
                "mean_latency_ms": round(
                    sum(latencies.values()) / len(latencies), 3
                ),
                "max_latency_ms": max(latencies.values()),
                "cuda_allocated_mib": round(
                    torch.cuda.memory_allocated() / 2**20, 3
                )
                if args.device.startswith("cuda")
                else 0.0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
