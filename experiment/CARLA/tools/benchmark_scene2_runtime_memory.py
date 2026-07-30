"""Measure the real Scene 2 perception and VLA model memory footprint."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolop-root", type=Path, required=True)
    parser.add_argument("--yolo11-weights", type=Path, required=True)
    parser.add_argument("--vla-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def memory_snapshot(label: str) -> dict[str, float | str]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "label": label,
        "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 3),
        "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 3),
        "device_used_mib": round((total_bytes - free_bytes) / 2**20, 3),
        "device_free_mib": round(free_bytes / 2**20, 3),
    }


def synchronize() -> None:
    torch.cuda.synchronize()
    time.sleep(0.1)


def build_vla(config: dict):
    from lightweight_vla_adapter.src.decision_adapter import (
        LightweightDecisionAdapter,
    )

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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    snapshots = [memory_snapshot("baseline")]
    timings: dict[str, float] = {}
    image = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)

    from scene_understanding.realtime_perception.yolop_backend import (
        YolopPanopticBackend,
    )

    started = time.perf_counter()
    yolop = YolopPanopticBackend(
        args.yolop_root, device="cuda", image_size=args.image_size
    )
    synchronize()
    timings["yolop_load_warmup_ms"] = round(
        (time.perf_counter() - started) * 1000.0, 3
    )
    snapshots.append(memory_snapshot("yolop_loaded"))

    from scene_understanding.realtime_perception.ultralytics_backend import (
        UltralyticsTrafficDetector,
    )

    started = time.perf_counter()
    yolo11 = UltralyticsTrafficDetector(
        args.yolo11_weights,
        device="cuda",
        image_size=args.image_size,
    )
    synchronize()
    timings["yolo11_load_warmup_ms"] = round(
        (time.perf_counter() - started) * 1000.0, 3
    )
    snapshots.append(memory_snapshot("yolo11_loaded"))

    config = json.loads((args.vla_dir / "student_base.json").read_text("utf-8"))
    from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline

    started = time.perf_counter()
    vla = LightweightVLAPipeline.from_checkpoint(
        build_vla(config),
        str(args.vla_dir / "model.pt"),
        model_name=config["model_name"],
        device="cuda",
        dtype=torch.float16,
    )
    synchronize()
    timings["vla_load_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    snapshots.append(memory_snapshot("vla_loaded"))

    from lightweight_vla_adapter.src.contracts import SensorTensorBatch

    batch = SensorTensorBatch(
        camera_bev=torch.zeros(1, 8, 64, 64),
        lidar_bev=torch.zeros(1, 4, 64, 64),
        ego_features=torch.zeros(1, 8),
        candidate_features=torch.zeros(1, 32, 12),
        candidate_mask=torch.ones(1, 32, dtype=torch.bool),
        intent_tokens=torch.zeros(1, 64, config["intent_dim"]),
        intent_mask=torch.ones(1, 64, dtype=torch.bool),
    )
    vla.warmup(batch, iterations=3)
    synchronize()
    snapshots.append(memory_snapshot("all_models_warmed"))

    stage_latencies: dict[str, float] = {}
    _, stage_latencies["yolop_ms"] = yolop.detect(image)
    _, stage_latencies["yolo11_ms"] = yolo11.detect(image)
    proposal = vla.predict_proposal(
        batch,
        request_id="memory-benchmark",
        frame_id="frame-1",
        candidate_entity_ids=[[f"entity-{index}" for index in range(32)]],
    )
    stage_latencies["vla_ms"] = proposal["latency_ms"]
    synchronize()
    snapshots.append(memory_snapshot("after_combined_inference"))

    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "image_size": args.image_size,
        "snapshots": snapshots,
        "load_timings_ms": timings,
        "inference_latencies_ms": {
            key: round(float(value), 3) for key, value in stage_latencies.items()
        },
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 2**20, 3
        ),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    del vla, yolo11, yolop, batch
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
