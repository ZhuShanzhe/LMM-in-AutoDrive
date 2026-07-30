"""Run the realtime perception stack on a generic image JSONL manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .detector import TorchvisionTrafficDetector
from .pipeline import RealtimePerceptionPipeline
from .run import DEFAULT_YOLO11_WEIGHTS, DEFAULT_YOLOP_ROOT, percentile
from .tracker import ByteTrackAdapter
from .ultralytics_backend import load_category_thresholds

ALLOWED_SOURCES = {"carla", "nuscenes", "waymo", "other"}


def normalize_source(value: object) -> str:
    source = str(value or "other").lower()
    return source if source in ALLOWED_SOURCES else "other"


def build_detector(args: argparse.Namespace):
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
                category_thresholds=load_category_thresholds(
                    args.class_thresholds
                ),
                infrastructure_tiles=args.infrastructure_tiles,
            )
            return CompositePanopticBackend(road_detector, object_detector)
        return road_detector
    return TorchvisionTrafficDetector(
        device=args.device, score_threshold=args.score_threshold
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend", choices=("ssdlite", "yolop", "yolop_yolo11"), default="yolop_yolo11"
    )
    parser.add_argument(
        "--yolop-root",
        type=Path,
        default=DEFAULT_YOLOP_ROOT,
    )
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
    detector = build_detector(args)
    category_thresholds = load_category_thresholds(args.class_thresholds)
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
    category_counts: Counter[str] = Counter()

    with args.manifest.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as output:
        for index, line in enumerate(source):
            if args.limit is not None and index >= args.limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            pipeline.reset()
            frame_id = str(record.get("frame_id") or record.get("image_id"))
            result = pipeline.process(
                image_path=Path(record["image_path"]),
                frame_id=frame_id,
                source=normalize_source(record.get("source") or record.get("dataset")),
                camera_name=str(record.get("camera_name") or "front_rgb"),
                timestamp_s=record.get("timestamp_s"),
            )
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            latencies.append(result["latency_ms"]["total"])
            category_counts.update(track["category"] for track in result["tracks"])

    steady = latencies[1:]
    summary = {
        "manifest": str(args.manifest),
        "frames": len(latencies),
        "track_observations_by_category": dict(category_counts),
        "latency_ms": {
            "total_mean": round(mean(latencies), 3) if latencies else 0.0,
            "total_p95": round(percentile(latencies, 0.95), 3),
            "cold_start_first_frame": round(latencies[0], 3) if latencies else 0.0,
            "steady_mean": round(mean(steady), 3) if steady else 0.0,
            "steady_p95": round(percentile(steady, 0.95), 3),
            "steady_max": round(max(steady), 3) if steady else 0.0,
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
            steady and percentile(steady, 0.95) <= 50
        ),
        "note": "Tracker is reset for each independent dataset image.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
