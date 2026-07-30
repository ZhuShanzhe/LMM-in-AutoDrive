"""Run low-latency perception on one or more CARLA capture indexes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from .detector import TorchvisionTrafficDetector
from .pipeline import RealtimePerceptionPipeline
from .tracker import ByteTrackAdapter
from .ultralytics_backend import load_category_thresholds

MODEL_ROOT = Path(__file__).resolve().parents[2] / "models"
DEFAULT_YOLOP_ROOT = MODEL_ROOT / "external" / "YOLOP"
DEFAULT_YOLO11_WEIGHTS = (
    MODEL_ROOT
    / "scene_understanding"
    / "yolo11s_specialized_carla_v1"
    / "weights"
    / "best.pt"
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend", choices=("ssdlite", "yolop", "yolop_yolo11"), default="yolop"
    )
    parser.add_argument("--yolop-root", type=Path, default=DEFAULT_YOLOP_ROOT)
    parser.add_argument(
        "--yolo11-weights",
        type=Path,
        default=DEFAULT_YOLO11_WEIGHTS,
    )
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--road-score-threshold",
        type=float,
        help="Optional YOLOP threshold when the object detector uses a lower floor",
    )
    parser.add_argument(
        "--tracker-threshold",
        type=float,
        help="Optional ByteTrack activation threshold",
    )
    parser.add_argument(
        "--class-thresholds",
        type=Path,
        help="JSON containing a by_category confidence-threshold mapping",
    )
    parser.add_argument("--image-size", type=int, choices=(320, 640), default=640)
    parser.add_argument(
        "--object-image-size",
        type=int,
        choices=(320, 640, 768, 960),
        help="Optional object-detector resolution independent of YOLOP",
    )
    parser.add_argument(
        "--infrastructure-tiles",
        action="store_true",
        help="Add two upper-scene tiles for small traffic lights and signs",
    )
    parser.add_argument("--frame-rate", type=int, default=10)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    category_thresholds = load_category_thresholds(args.class_thresholds)
    if args.backend in {"yolop", "yolop_yolo11"}:
        from .yolop_backend import YolopPanopticBackend

        road_detector = YolopPanopticBackend(
            args.yolop_root,
            device=args.device,
            image_size=args.image_size,
            score_threshold=args.road_score_threshold
            if args.road_score_threshold is not None
            else args.score_threshold,
        )
        if args.backend == "yolop_yolo11":
            from .composite_backend import CompositePanopticBackend
            from .ultralytics_backend import UltralyticsTrafficDetector

            object_detector = UltralyticsTrafficDetector(
                args.yolo11_weights,
                device=args.device,
                image_size=args.object_image_size or args.image_size,
                score_threshold=args.score_threshold,
                category_thresholds=category_thresholds,
                infrastructure_tiles=args.infrastructure_tiles,
            )
            detector = CompositePanopticBackend(road_detector, object_detector)
        else:
            detector = road_detector
    else:
        detector = TorchvisionTrafficDetector(device=args.device, score_threshold=args.score_threshold)
    tracker_threshold = (
        args.tracker_threshold
        if args.tracker_threshold is not None
        else min(category_thresholds.values(), default=args.score_threshold)
    )
    tracker = ByteTrackAdapter(
        frame_rate=args.frame_rate,
        track_activation_threshold=tracker_threshold,
    )
    pipeline = RealtimePerceptionPipeline(detector, tracker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    detector_latencies: list[float] = []
    track_counts: dict[str, int] = {}
    frame_count = 0

    with args.output.open("w", encoding="utf-8") as output:
        for index_path in args.capture_index:
            pipeline.reset()
            for line in index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                world_state_path = Path(record["world_state_path"])
                world_state = json.loads(world_state_path.read_text(encoding="utf-8"))
                result = pipeline.process(
                    image_path=Path(record["image_path"]),
                    frame_id=record["frame_id"],
                    source="carla",
                    camera_name=record["camera_name"],
                    timestamp_s=record.get("timestamp_s"),
                    world_state=world_state,
                )
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                latencies.append(result["latency_ms"]["total"])
                detector_latencies.append(result["latency_ms"]["detector"])
                for track in result["tracks"]:
                    track_counts[track["category"]] = track_counts.get(track["category"], 0) + 1
                frame_count += 1
                if args.limit and frame_count >= args.limit:
                    break
            if args.limit and frame_count >= args.limit:
                break

    steady_latencies = latencies[1:]
    summary = {
        "frames": frame_count,
        "track_observations_by_category": track_counts,
        "latency_ms": {
            "total_mean": round(mean(latencies), 3) if latencies else 0.0,
            "total_median": round(median(latencies), 3) if latencies else 0.0,
            "total_p95": round(percentile(latencies, 0.95), 3),
            "total_max": round(max(latencies), 3) if latencies else 0.0,
            "detector_mean": round(mean(detector_latencies), 3) if detector_latencies else 0.0,
            "cold_start_first_frame": round(latencies[0], 3) if latencies else 0.0,
            "steady_mean": round(mean(steady_latencies), 3) if steady_latencies else 0.0,
            "steady_p95": round(percentile(steady_latencies, 0.95), 3),
            "steady_max": round(max(steady_latencies), 3) if steady_latencies else 0.0,
        },
        "competition_parse_budget_ms": 50,
        "detector_config": {
            "backend": args.backend,
            "weights": str(args.yolo11_weights),
            "inference_floor": args.score_threshold,
            "road_score_threshold": args.road_score_threshold
            if args.road_score_threshold is not None
            else args.score_threshold,
            "class_thresholds": str(args.class_thresholds)
            if args.class_thresholds
            else None,
            "tracker_threshold": tracker_threshold,
            "image_size": args.image_size,
            "road_image_size": args.image_size,
            "object_image_size": args.object_image_size or args.image_size,
            "infrastructure_tiles": args.infrastructure_tiles,
        },
        "meets_50ms_steady_frame_budget_p95": bool(
            steady_latencies and percentile(steady_latencies, 0.95) <= 50
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
