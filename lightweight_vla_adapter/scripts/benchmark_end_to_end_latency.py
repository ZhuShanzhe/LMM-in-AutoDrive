from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "models" / "lightweight_vla_adapter" / "v10" / "model.pt"
)
DEFAULT_LANGUAGE_MODEL = (
    REPO_ROOT / "models" / "modernbert-drive-command-compositional"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.scripts.train_student import build_model


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 4),
        "p50": round(percentile(values, 0.50), 4),
        "p95": round(percentile(values, 0.95), 4),
        "max": round(max(values), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ModernBERT plus the trained VLA adapter"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--language-model", default=str(DEFAULT_LANGUAGE_MODEL))
    parser.add_argument(
        "--text",
        default="Move to the left lane when it is safe.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--output")
    args = parser.parse_args()

    from transformers import AutoModel, AutoTokenizer

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    language_model = AutoModel.from_pretrained(
        args.language_model,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    if int(language_model.config.hidden_size) != int(config["intent_dim"]):
        raise ValueError("language model hidden size does not match adapter intent_dim")
    adapter = build_model(config)
    adapter.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    adapter.to(device=device, dtype=dtype).eval()

    static_inputs = {
        "camera_bev": torch.randn(
            1, config["camera_channels"], 64, 64, device=device, dtype=dtype
        ),
        "lidar_bev": torch.randn(
            1, config["lidar_channels"], 64, 64, device=device, dtype=dtype
        ),
        "ego_features": torch.randn(
            1, config["ego_dim"], device=device, dtype=dtype
        ),
        "candidate_features": torch.randn(
            1,
            config["max_candidates"],
            config["candidate_dim"],
            device=device,
            dtype=dtype,
        ),
        "candidate_mask": torch.ones(
            1, config["max_candidates"], device=device, dtype=torch.bool
        ),
    }

    def tokenize() -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            args.text,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in encoded.items()}

    with torch.inference_mode():
        for _ in range(args.warmup):
            encoded = tokenize()
            hidden = language_model(**encoded).last_hidden_state
            adapter(
                **static_inputs,
                intent_tokens=hidden,
                intent_mask=encoded["attention_mask"].bool(),
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        tokenize_ms: list[float] = []
        language_adapter_ms: list[float] = []
        total_ms: list[float] = []
        for _ in range(args.runs):
            total_started = time.perf_counter()
            token_started = time.perf_counter()
            encoded = tokenize()
            tokenize_ms.append((time.perf_counter() - token_started) * 1000.0)
            gpu_started = time.perf_counter()
            hidden = language_model(**encoded).last_hidden_state
            adapter(
                **static_inputs,
                intent_tokens=hidden,
                intent_mask=encoded["attention_mask"].bool(),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            language_adapter_ms.append(
                (time.perf_counter() - gpu_started) * 1000.0
            )
            total_ms.append((time.perf_counter() - total_started) * 1000.0)

    result = {
        "device": str(device),
        "precision": args.precision,
        "runs": args.runs,
        "max_length": args.max_length,
        "checkpoint": args.checkpoint,
        "language_model": args.language_model,
        "tokenize_ms": summary(tokenize_ms),
        "modernbert_plus_adapter_ms": summary(language_adapter_ms),
        "text_to_proposal_features_ms": summary(total_ms),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
