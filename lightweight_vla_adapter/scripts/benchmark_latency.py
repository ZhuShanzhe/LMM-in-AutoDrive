from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.inference_optimization import fuse_inference_conv_bn


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def load_config_and_state(
    config_path: Path, checkpoint_path: Path | None
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None, str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if checkpoint_path is None:
        return config, None, "none"
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("checkpoint must contain a state-dict object")
    if isinstance(artifact.get("base"), dict):
        embedded_config = artifact.get("config")
        if not isinstance(embedded_config, dict):
            raise ValueError("challenge checkpoint is missing its embedded config")
        return embedded_config, artifact["base"], "challenge_sequence_artifact"
    return config, artifact, "plain_adapter_state_dict"


def configured_inputs(
    config: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    height, width = (int(value) for value in config.get("bev_input_size", (64, 64)))
    max_candidates = int(config["max_candidates"])
    candidate_dim = int(config["candidate_dim"])
    intent_length = int(config.get("intent_max_length", 32))
    num_views = int(config.get("num_camera_views", 4))
    use_candidates = bool(config.get("use_candidate_entities", True))
    inputs: dict[str, torch.Tensor] = {
        "camera_bev": torch.randn(
            1, int(config["camera_channels"]), height, width,
            device=device, dtype=dtype,
        ),
        "lidar_bev": torch.randn(
            1, int(config["lidar_channels"]), height, width,
            device=device, dtype=dtype,
        ),
        "ego_features": torch.randn(
            1, int(config["ego_dim"]), device=device, dtype=dtype,
        ),
        "candidate_features": torch.randn(
            1, max_candidates, candidate_dim, device=device, dtype=dtype,
        ),
        "candidate_mask": torch.full(
            (1, max_candidates), use_candidates, device=device, dtype=torch.bool,
        ),
        "intent_tokens": torch.randn(
            1, intent_length, int(config["intent_dim"]),
            device=device, dtype=dtype,
        ),
        "intent_mask": torch.ones(
            1, intent_length, device=device, dtype=torch.bool,
        ),
    }
    if bool(config.get("use_environment", True)):
        inputs["environment_features"] = torch.randn(
            1, int(config.get("environment_dim", 12)),
            device=device, dtype=dtype,
        )
    if bool(config.get("use_raw_camera", True)):
        camera_height = int(config.get("camera_input_height", 224))
        camera_width = int(config.get("camera_input_width", 224))
        inputs["camera_images"] = torch.rand(
            1, num_views, 3, camera_height, camera_width,
            device=device, dtype=dtype,
        )
        inputs["camera_view_mask"] = torch.ones(
            1, num_views, device=device, dtype=torch.bool,
        )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all adapter modalities enabled by a model config"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="fp16"
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("warmup must be non-negative and runs must be positive")

    config, state, checkpoint_format = load_config_and_state(
        args.config, args.checkpoint
    )
    device = torch.device(args.device)
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
    model = build_model(config).eval()
    if state is not None:
        model.load_state_dict(state, strict=True)
    fused_pairs = 0
    if bool(config.get("fuse_conv_bn", False)):
        model, fused_pairs = fuse_inference_conv_bn(model)
    model.to(device=device, dtype=dtype).eval()
    inputs = configured_inputs(config, device, dtype)

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

    p95 = percentile(latencies, 0.95)
    budget = config.get("deployment", {}).get("adapter_latency_budget_ms")
    result = {
        "benchmark_scope": "configured adapter forward; excludes language model, sequence head, safety gate and CARLA",
        "model_name": config.get("model_name"),
        "device": str(device),
        "precision": args.precision,
        "runs": args.runs,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_loaded": state is not None,
        "checkpoint_format": checkpoint_format,
        "raw_camera_included": "camera_images" in inputs,
        "raw_camera_input_size": (
            list(model.raw_camera_encoder.input_size)
            if "camera_images" in inputs
            else None
        ),
        "structured_bev_included": bool(config.get("use_structured_bev", True)),
        "candidate_entities_included": bool(config.get("use_candidate_entities", True)),
        "conv_bn_pairs_fused": fused_pairs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "latency_ms_mean": round(statistics.fmean(latencies), 4),
        "latency_ms_p50": round(percentile(latencies, 0.50), 4),
        "latency_ms_p95": round(p95, 4),
        "latency_ms_max": round(max(latencies), 4),
        "adapter_latency_budget_ms": budget,
        "p95_within_budget": bool(
            isinstance(budget, (int, float)) and p95 <= float(budget)
        ),
        "evidence_boundary": "synthetic tensors and random weights unless checkpoint_loaded is true",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
