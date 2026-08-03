"""Evaluate PerceptionFrame JSONL against normalized or pixel-space 2D truth."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .evaluate import iou

BDD_TO_DRIVING = {
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "train": "vehicle",
    "person": "pedestrian",
    "rider": "cyclist",
    "bike": "cyclist",
    "bicycle": "cyclist",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
}
DRIVING_CATEGORIES = {
    "vehicle",
    "pedestrian",
    "cyclist",
    "motorcycle",
    "traffic_light",
    "traffic_sign",
    "road_barrier",
    "traffic_cone",
    "obstacle",
}


def normalized_box(box: list[float], width: int, height: int) -> list[float]:
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]


def normalize_category(value: object) -> str | None:
    category = str(value).lower()
    return BDD_TO_DRIVING.get(
        category, category if category in DRIVING_CATEGORIES else None
    )


def load_truth(
    manifest: Path, limit: int | None
) -> tuple[dict[str, list[dict]], str]:
    truth: dict[str, list[dict]] = {}
    dataset = "unknown"
    with manifest.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            dataset = str(record.get("dataset") or record.get("source") or dataset)
            frame_id = str(record.get("frame_id") or record.get("image_id"))
            items = []
            if "ground_truth_objects" in record:
                for annotation in record["ground_truth_objects"]:
                    category = normalize_category(annotation["category"])
                    if category is not None:
                        items.append(
                            {
                                "category": category,
                                "bbox_2d": annotation["bbox_2d"],
                            }
                        )
            else:
                width, height = int(record["width"]), int(record["height"])
                for annotation in record.get("annotations", []):
                    category = normalize_category(annotation["category"])
                    if category is None:
                        continue
                    items.append(
                        {
                            "category": category,
                            "bbox_2d": normalized_box(
                                annotation["bbox_xyxy"], width=width, height=height
                            ),
                        }
                    )
            truth[frame_id] = items
    return truth, dataset


def evaluate(
    results: Path,
    manifest: Path,
    *,
    iou_threshold: float,
    limit: int | None,
) -> dict:
    truth_by_frame, dataset = load_truth(manifest, limit)
    truth_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    matched_counts: Counter[str] = Counter()
    frames = 0

    with results.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            result = json.loads(line)
            frame_id = result["frame_id"]
            truth = truth_by_frame[frame_id]
            predictions = result["tracks"]
            frames += 1
            truth_counts.update(item["category"] for item in truth)
            prediction_counts.update(item["category"] for item in predictions)
            candidates = []
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
                matched_counts[truth[truth_index]["category"]] += 1

    categories = sorted(set(truth_counts) | set(prediction_counts))
    per_category = {}
    for category in categories:
        truth_count = truth_counts[category]
        prediction_count = prediction_counts[category]
        matched = matched_counts[category]
        per_category[category] = {
            "truth": truth_count,
            "predictions": prediction_count,
            "matched": matched,
            "recall": round(matched / truth_count, 4) if truth_count else None,
            "precision": round(matched / prediction_count, 4) if prediction_count else None,
        }
    total_truth = sum(truth_counts.values())
    total_predictions = sum(prediction_counts.values())
    total_matched = sum(matched_counts.values())
    return {
        "dataset": dataset,
        "frames": frames,
        "iou_threshold": iou_threshold,
        "per_category": per_category,
        "overall": {
            "truth": total_truth,
            "predictions": total_predictions,
            "matched": total_matched,
            "recall": round(total_matched / total_truth, 4) if total_truth else None,
            "precision": round(total_matched / total_predictions, 4)
            if total_predictions
            else None,
        },
        "limitations": (
            [
                "BDD100K rider boxes are mapped to cyclist for the shared driving taxonomy.",
                "The mirror has no lane or drivable-area ground truth.",
            ]
            if dataset.lower() == "bdd100k"
            else [
                "CARLA boxes are simulator actor projections into the front camera.",
                "This 2D evaluation does not score hidden actors or map-only facts.",
            ]
            if dataset.lower() == "carla"
            else [
                "nuScenes boxes are official 3D annotations projected into each camera.",
                "This 2D evaluation does not score depth, velocity or map topology.",
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    metrics = evaluate(
        args.results,
        args.manifest,
        iou_threshold=args.iou_threshold,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
