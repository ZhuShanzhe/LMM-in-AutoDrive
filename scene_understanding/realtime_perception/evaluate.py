"""Evaluate PerceptionFrame JSONL against same-frame CARLA projections."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def load_projection_paths(index_paths: list[Path]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for index_path in index_paths:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                output[record["frame_id"]] = Path(record["projection_path"])
    return output


def evaluate(results_path: Path, index_paths: list[Path], *, iou_threshold: float) -> dict:
    projections = load_projection_paths(index_paths)
    truth_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    matched_truth: Counter[str] = Counter()
    matched_predictions: Counter[str] = Counter()
    frame_count = 0
    lane_frames = 0
    drivable_frames = 0

    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        frame_count += 1
        road = result["road_structure"]
        lane_frames += bool(road["lane_boundaries"])
        drivable_frames += road["drivable_area"] is not None
        truth = json.loads(projections[result["frame_id"]].read_text(encoding="utf-8"))["objects"]
        predictions = result["tracks"]
        truth_counts.update(item["category"] for item in truth)
        prediction_counts.update(item["category"] for item in predictions)

        candidates: list[tuple[float, int, int]] = []
        for truth_index, truth_item in enumerate(truth):
            for prediction_index, prediction in enumerate(predictions):
                if truth_item["category"] != prediction["category"]:
                    continue
                overlap = iou(truth_item["bbox_2d"], prediction["bbox_2d"])
                if overlap >= iou_threshold:
                    candidates.append((overlap, truth_index, prediction_index))
        used_truth: set[int] = set()
        used_predictions: set[int] = set()
        for _, truth_index, prediction_index in sorted(candidates, reverse=True):
            if truth_index in used_truth or prediction_index in used_predictions:
                continue
            used_truth.add(truth_index)
            used_predictions.add(prediction_index)
            category = truth[truth_index]["category"]
            matched_truth[category] += 1
            matched_predictions[category] += 1

    categories = sorted(set(truth_counts) | set(prediction_counts))
    per_category = {}
    for category in categories:
        truth_count = truth_counts[category]
        prediction_count = prediction_counts[category]
        per_category[category] = {
            "truth": truth_count,
            "predictions": prediction_count,
            "matched": matched_truth[category],
            "truth_recall": round(matched_truth[category] / truth_count, 4) if truth_count else None,
            "prediction_support": round(matched_predictions[category] / prediction_count, 4)
            if prediction_count
            else None,
        }
    total_truth = sum(truth_counts.values())
    total_predictions = sum(prediction_counts.values())
    total_matched = sum(matched_truth.values())
    return {
        "frames": frame_count,
        "iou_threshold": iou_threshold,
        "per_category": per_category,
        "overall": {
            "truth": total_truth,
            "predictions": total_predictions,
            "matched": total_matched,
            "truth_recall": round(total_matched / total_truth, 4) if total_truth else None,
            "prediction_support": round(total_matched / total_predictions, 4) if total_predictions else None,
        },
        "road_evidence": {
            "lane_boundary_frame_coverage": round(lane_frames / frame_count, 4) if frame_count else 0.0,
            "drivable_area_frame_coverage": round(drivable_frames / frame_count, 4) if frame_count else 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--capture-index", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    args = parser.parse_args()
    metrics = evaluate(args.results, args.capture_index, iou_threshold=args.iou_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
