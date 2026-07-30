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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.src.decision_adapter import LightweightDecisionAdapter


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark student adapter latency")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    device = torch.device(args.device)
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    model = LightweightDecisionAdapter(
        camera_channels=config["camera_channels"],
        lidar_channels=config["lidar_channels"],
        candidate_dim=config["candidate_dim"],
        ego_dim=config["ego_dim"],
        intent_dim=config["intent_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=0.0,
        bev_grid=tuple(config["bev_grid"]),
    )
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.to(device=device, dtype=dtype).eval()
    inputs = {
        "camera_bev": torch.randn(
            1, config["camera_channels"], 64, 64, device=device, dtype=dtype
        ),
        "lidar_bev": torch.randn(
            1, config["lidar_channels"], 64, 64, device=device, dtype=dtype
        ),
        "ego_features": torch.randn(1, config["ego_dim"], device=device, dtype=dtype),
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
        "intent_tokens": torch.randn(
            1, 32, config["intent_dim"], device=device, dtype=dtype
        ),
        "intent_mask": torch.ones(1, 32, device=device, dtype=torch.bool),
    }
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies: list[float] = []
        for _ in range(args.runs):
            started = time.perf_counter()
            model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "device": str(device),
        "precision": args.precision,
        "runs": args.runs,
        "checkpoint": args.checkpoint,
        "checkpoint_loaded": bool(args.checkpoint),
        "parameters": parameter_count,
        "latency_ms_mean": round(statistics.fmean(latencies), 4),
        "latency_ms_p50": round(percentile(latencies, 0.50), 4),
        "latency_ms_p95": round(percentile(latencies, 0.95), 4),
        "latency_ms_max": round(max(latencies), 4),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
