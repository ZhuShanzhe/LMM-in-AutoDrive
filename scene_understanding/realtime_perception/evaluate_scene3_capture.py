"""Evaluate the independent visual detector against Scene 3 CARLA truth proxies."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from .pipeline import RealtimePerceptionPipeline
from .run import percentile
from .tracker import ByteTrackAdapter
from .ultralytics_backend import UltralyticsTrafficDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--truth-tolerance-frames", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl_by_frame(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    return {int(row["simulation_frame"]): row for row in rows}


def nearest_truth(
    rows: dict[int, dict[str, Any]], frame: int, tolerance: int
) -> tuple[dict[str, Any] | None, int | None]:
    for offset in range(tolerance + 1):
        candidates = (frame,) if offset == 0 else (frame - offset, frame + offset)
        for candidate in candidates:
            if candidate in rows:
                return rows[candidate], candidate - frame
    return None, None


def truth_category(actor: dict[str, Any]) -> str | None:
    type_id = str(actor.get("type_id", "")).lower()
    if "walker" in type_id or "pedestrian" in type_id:
        return "pedestrian"
    if "vehicle" in type_id:
        return "vehicle"
    if "traffic_light" in type_id:
        return "traffic_light"
    if "traffic" in type_id and "sign" in type_id:
        return "traffic_sign"
    return None


def front_relevant_truth(truth: dict[str, Any]) -> Counter[str]:
    categories: Counter[str] = Counter()
    for actor in truth.get("actors", {}).values():
        relation = actor.get("relation_to_ego", {})
        longitudinal = float(relation.get("longitudinal_m", -1.0))
        lateral = abs(float(relation.get("lateral_m", 1e9)))
        distance = float(relation.get("euclidean_distance_m", 1e9))
        category = truth_category(actor)
        if (
            category is not None
            and 0.0 <= longitudinal <= 100.0
            and lateral <= 15.0
            and distance <= 110.0
        ):
            categories[category] += 1
    return categories


def relative_or_absolute(path: Path, anchor: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> int:
    args = parse_args()
    capture = args.capture_dir.resolve()
    images = sorted(
        (capture / "rgb" / "front_rgb").glob("*.png"),
        key=lambda path: int(path.stem),
    )[:: max(1, args.stride)]
    if args.limit is not None:
        images = images[: args.limit]
    truth_rows = read_jsonl_by_frame(capture / "frame_ground_truth.jsonl")
    detector = UltralyticsTrafficDetector(
        args.weights.resolve(),
        device=args.device,
        image_size=args.image_size,
        score_threshold=args.score_threshold,
    )
    pipeline = RealtimePerceptionPipeline(
        detector,
        ByteTrackAdapter(
            frame_rate=1,
            track_activation_threshold=args.score_threshold,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluated = 0
    skipped = 0
    truth_presence = Counter()
    matched_presence = Counter()
    detected_presence = Counter()
    frame_offsets = Counter()
    latencies: list[float] = []
    with args.output.open("w", encoding="utf-8") as output:
        for image_path in images:
            frame = int(image_path.stem)
            truth, frame_delta = nearest_truth(
                truth_rows, frame, args.truth_tolerance_frames
            )
            if truth is None:
                skipped += 1
                continue
            result = pipeline.process(
                image_path=image_path,
                frame_id=str(frame),
                source="carla",
                camera_name="front_rgb",
                timestamp_s=truth.get("timestamp_s"),
            )
            expected = front_relevant_truth(truth)
            detected = Counter(track["category"] for track in result["tracks"])
            for category in expected:
                truth_presence[category] += 1
                if detected[category] > 0:
                    matched_presence[category] += 1
            for category in detected:
                detected_presence[category] += 1
            frame_offsets[int(frame_delta)] += 1
            latencies.append(float(result["latency_ms"]["total"]))
            result["scene3_truth_proxy"] = {
                "truth_frame_delta": int(frame_delta),
                "front_relevant_categories": dict(expected),
                "matched_categories": sorted(set(expected) & set(detected)),
                "note": (
                    "Presence recall proxy only; CARLA actor transforms do not "
                    "provide a post-capture 2D-box ground truth benchmark."
                ),
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            evaluated += 1
    per_category = {
        category: {
            "truth_positive_frames": truth_presence[category],
            "matched_frames": matched_presence[category],
            "presence_recall": (
                matched_presence[category] / truth_presence[category]
            ),
        }
        for category in sorted(truth_presence)
    }
    summary = {
        "schema_version": "scene3_perception_capture_audit/1.0",
        "capture_dir": relative_or_absolute(capture, Path.cwd()),
        "weights": relative_or_absolute(args.weights, Path.cwd()),
        "frames_evaluated": evaluated,
        "frames_skipped_without_truth": skipped,
        "truth_frame_deltas": {
            str(key): value for key, value in sorted(frame_offsets.items())
        },
        "front_presence_recall_by_category": per_category,
        "detected_presence_frames_by_category": dict(detected_presence),
        "latency_ms": {
            "mean": round(fmean(latencies), 3) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "meets_50ms_visual_budget_p95": bool(
            latencies and percentile(latencies, 0.95) <= 50.0
        ),
        "metric_scope": "front-camera category-presence recall proxy",
        "not_claimed": ["2D mAP", "3D detection accuracy", "distance accuracy"],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
