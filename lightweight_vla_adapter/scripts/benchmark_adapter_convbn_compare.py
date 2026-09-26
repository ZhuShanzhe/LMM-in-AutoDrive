from dataclasses import fields
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.scripts.benchmark_latency import (
    configured_inputs,
)
from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.inference_optimization import fuse_inference_conv_bn


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, round((len(ordered) - 1) * fraction)),
    )
    return ordered[index]


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def load_challenge_checkpoint(checkpoint):
    """Load the self-describing challenge artifact without a sidecar config."""
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise ValueError("checkpoint must contain a dictionary artifact")
    config = artifact.get("config")
    state = artifact.get("base")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError(
            "checkpoint must provide embedded 'config' and 'base' state_dict entries"
        )
    return config, state, "challenge_sequence_artifact"


def measure(model, inputs, device, warmup, runs):
    with torch.inference_mode():
        for _ in range(warmup):
            model(**inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            started = time.perf_counter()
            model(**inputs)

            if device.type == "cuda":
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - started) * 1000.0)

    return {
        "mean_ms": round(statistics.fmean(latencies), 4),
        "p50_ms": round(percentile(latencies, 0.50), 4),
        "p95_ms": round(percentile(latencies, 0.95), 4),
        "max_ms": round(max(latencies), 4),
    }


def compare_outputs(reference, candidate, rtol, atol):
    max_abs_difference = 0.0

    for field in fields(type(reference)):
        expected = getattr(reference, field.name)
        actual = getattr(candidate, field.name)

        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            difference = float(
                (actual - expected).abs().max().detach().cpu()
            )
            max_abs_difference = max(max_abs_difference, difference)
        elif expected != actual:
            raise ValueError(f"Non-tensor output changed: {field.name}")

    return max_abs_difference


parser = argparse.ArgumentParser(
    description="Compare original and Conv-BN-folded adapter inference."
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--runs", type=int, default=200)
args = parser.parse_args()

if args.warmup < 0 or args.runs <= 0:
    raise ValueError("warmup must be non-negative and runs must be positive")

device = torch.device(args.device)
if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is unavailable")

torch.set_num_threads(2)
torch.manual_seed(925)

if device.type == "cuda":
    torch.cuda.manual_seed_all(925)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parity_rtol = 3e-3
    parity_atol = 1e-3
else:
    parity_rtol = 3e-4
    parity_atol = 3e-5

config, state, checkpoint_format = load_challenge_checkpoint(args.checkpoint)

inputs = configured_inputs(config, device, torch.float32)

baseline = build_model(config).eval()
baseline.load_state_dict(state, strict=True)

optimized, fused_pairs = fuse_inference_conv_bn(baseline)

baseline = baseline.to(device=device, dtype=torch.float32).eval()
optimized = optimized.to(device=device, dtype=torch.float32).eval()

with torch.inference_mode():
    baseline_output = baseline(**inputs)
    optimized_output = optimized(**inputs)

max_abs_difference = compare_outputs(
    baseline_output,
    optimized_output,
    rtol=parity_rtol,
    atol=parity_atol,
)

report = {
    "status": "passed",
    "scope": "Actual checkpoint, synthetic fixed inputs, local adapter forward only",
    "device": str(device),
    "checkpoint": str(args.checkpoint),
    "checkpoint_format": checkpoint_format,
    "raw_camera_input_size": [
        config.get("camera_input_height"),
        config.get("camera_input_width"),
    ],
    "candidate_entities_included": bool(
        config.get("use_candidate_entities", True)
    ),
    "conv_bn_pairs_fused": fused_pairs,
    "parity_tolerance": {
        "rtol": parity_rtol,
        "atol": parity_atol,
        "tf32_disabled": device.type == "cuda",
    },
    "max_abs_output_difference": max_abs_difference,
    "baseline": {
        "parameters": parameter_count(baseline),
        "latency": measure(
            baseline,
            inputs,
            device,
            args.warmup,
            args.runs,
        ),
    },
    "conv_bn_folded": {
        "parameters": parameter_count(optimized),
        "latency": measure(
            optimized,
            inputs,
            device,
            args.warmup,
            args.runs,
        ),
    },
    "not_validated": [
        "J6P performance",
        "full-chain latency",
        "CARLA closed-loop",
        "ASR/ModernBERT",
        "power, memory, utilization",
    ],
}

print(json.dumps(report, indent=2))
