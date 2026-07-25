"""Evaluate structured scene outputs against DriveLM key-object annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scene_understanding.core.validate_scene_output import validate_scene_output


def read_indexed_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Read JSONL records into a unique frame-ID index."""

    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            frame_id = record.get("frame_id")
            if not isinstance(frame_id, str) or not frame_id:
                raise ValueError(f"{path}:{line_number}: missing frame_id")
            if frame_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate frame_id {frame_id}")
            records[frame_id] = record
    return records


def bbox_iou(first: list[float], second: list[float]) -> float:
    """Return intersection over union for two normalized xyxy boxes."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def center_inside(inner: list[float], outer: list[float]) -> bool:
    """Return whether the center of the first box lies in the second box."""

    center_x = (inner[0] + inner[2]) / 2
    center_y = (inner[1] + inner[3]) / 2
    return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]


def truth_traffic_light_state(truth_objects: list[dict[str, Any]]) -> str | None:
    """Extract an explicitly annotated traffic-light color when available."""

    states: set[str] = set()
    for truth in truth_objects:
        if truth.get("category") != "traffic_light":
            continue
        text = " ".join(
            str(truth.get(field) or "").lower()
            for field in ("status_raw", "visual_description")
        )
        for state in ("red", "yellow", "green"):
            if state in text:
                states.add(state)
    return next(iter(states)) if len(states) == 1 else None


def evaluate_frame(
    manifest_record: dict[str, Any],
    inference_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate one frame using grouping-tolerant center containment and IoU."""

    frame_id = manifest_record["frame_id"]
    truths = manifest_record.get("ground_truth_objects") or []
    if inference_record is None:
        return {"frame_id": frame_id, "inference_found": False, "truth_count": len(truths)}

    output = inference_record.get("parsed_output")
    if not isinstance(output, dict):
        return {
            "frame_id": frame_id,
            "inference_found": True,
            "schema_valid": False,
            "schema_errors": ["parsed_output is unavailable"],
            "truth_count": len(truths),
            "prediction_count": 0,
        }

    schema_errors = validate_scene_output(
        output,
        expected_frame_id=frame_id,
        expected_source=manifest_record.get("source"),
        expected_camera_name=manifest_record.get("camera_name"),
    )
    predictions = output.get("objects") if isinstance(output.get("objects"), list) else []

    truth_hits: list[bool] = []
    best_ious: list[float] = []
    for truth in truths:
        candidates = [
            prediction
            for prediction in predictions
            if prediction.get("category") == truth.get("category")
            and isinstance(prediction.get("bbox_2d"), list)
        ]
        truth_bbox = truth["bbox_2d"]
        truth_hits.append(any(center_inside(candidate["bbox_2d"], truth_bbox) for candidate in candidates))
        best_ious.append(
            max((bbox_iou(candidate["bbox_2d"], truth_bbox) for candidate in candidates), default=0.0)
        )

    prediction_supported: list[bool] = []
    for prediction in predictions:
        prediction_bbox = prediction.get("bbox_2d")
        candidates = [truth for truth in truths if truth.get("category") == prediction.get("category")]
        prediction_supported.append(
            isinstance(prediction_bbox, list)
            and any(center_inside(prediction_bbox, truth["bbox_2d"]) for truth in candidates)
        )

    truth_state = truth_traffic_light_state(truths)
    predicted_state = (output.get("scene") or {}).get("traffic_light_state")
    category_metrics: dict[str, dict[str, int]] = {}
    categories = sorted(
        {str(truth.get("category")) for truth in truths}
        | {str(prediction.get("category")) for prediction in predictions}
    )
    for category in categories:
        category_truths = [truth for truth in truths if truth.get("category") == category]
        category_predictions = [
            prediction for prediction in predictions if prediction.get("category") == category
        ]
        category_truth_hits = sum(
            any(
                isinstance(prediction.get("bbox_2d"), list)
                and center_inside(prediction["bbox_2d"], truth["bbox_2d"])
                for prediction in category_predictions
            )
            for truth in category_truths
        )
        category_supported_predictions = sum(
            isinstance(prediction.get("bbox_2d"), list)
            and any(
                center_inside(prediction["bbox_2d"], truth["bbox_2d"])
                for truth in category_truths
            )
            for prediction in category_predictions
        )
        category_metrics[category] = {
            "truth_objects": len(category_truths),
            "truth_center_hits": category_truth_hits,
            "predicted_objects": len(category_predictions),
            "supported_predictions": category_supported_predictions,
        }
    return {
        "frame_id": frame_id,
        "inference_found": True,
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "truth_count": len(truths),
        "prediction_count": len(predictions),
        "truth_center_hits": sum(truth_hits),
        "supported_predictions": sum(prediction_supported),
        "best_iou_per_truth": [round(value, 6) for value in best_ious],
        "traffic_light_truth_state": truth_state,
        "traffic_light_predicted_state": predicted_state,
        "traffic_light_state_correct": predicted_state == truth_state if truth_state is not None else None,
        "category_metrics": category_metrics,
    }


def summarize(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate transparent counts and rates across evaluated frames."""

    found = [result for result in frame_results if result.get("inference_found")]
    truth_count = sum(result.get("truth_count", 0) for result in found)
    prediction_count = sum(result.get("prediction_count", 0) for result in found)
    truth_hits = sum(result.get("truth_center_hits", 0) for result in found)
    supported = sum(result.get("supported_predictions", 0) for result in found)
    all_ious = [value for result in found for value in result.get("best_iou_per_truth", [])]
    state_results = [
        result["traffic_light_state_correct"]
        for result in found
        if result.get("traffic_light_state_correct") is not None
    ]
    schema_valid_count = sum(bool(result.get("schema_valid")) for result in found)
    category_totals: dict[str, dict[str, int | float | None]] = {}
    for result in found:
        for category, metrics in result.get("category_metrics", {}).items():
            totals = category_totals.setdefault(
                category,
                {
                    "truth_objects": 0,
                    "truth_center_hits": 0,
                    "predicted_objects": 0,
                    "supported_predictions": 0,
                },
            )
            for key in (
                "truth_objects",
                "truth_center_hits",
                "predicted_objects",
                "supported_predictions",
            ):
                totals[key] = int(totals[key]) + metrics[key]
    for totals in category_totals.values():
        category_truths = int(totals["truth_objects"])
        category_predictions = int(totals["predicted_objects"])
        totals["truth_recall"] = (
            round(int(totals["truth_center_hits"]) / category_truths, 6)
            if category_truths
            else None
        )
        totals["supported_prediction_rate"] = (
            round(int(totals["supported_predictions"]) / category_predictions, 6)
            if category_predictions
            else None
        )
    return {
        "manifest_frames": len(frame_results),
        "inference_frames": len(found),
        "schema_valid_frames": schema_valid_count,
        "schema_valid_rate": round(schema_valid_count / len(found), 6) if found else None,
        "truth_objects": truth_count,
        "predicted_objects": prediction_count,
        "category_center_hit_truth_recall": round(truth_hits / truth_count, 6) if truth_count else None,
        "category_center_supported_prediction_rate": round(supported / prediction_count, 6)
        if prediction_count
        else None,
        "mean_best_iou": round(sum(all_ious) / len(all_ious), 6) if all_ious else None,
        "traffic_light_state_examples": len(state_results),
        "traffic_light_state_accuracy": round(sum(state_results) / len(state_results), 6)
        if state_results
        else None,
        "categories": category_totals,
    }


def evaluate_files(manifest_path: Path, inference_path: Path) -> dict[str, Any]:
    """Evaluate all manifest frames for which inference may or may not exist."""

    manifest = read_indexed_jsonl(manifest_path)
    inference = read_indexed_jsonl(inference_path)
    frames = [evaluate_frame(record, inference.get(frame_id)) for frame_id, record in manifest.items()]
    return {"summary": summarize(frames), "frames": frames}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate_files(args.manifest, args.inference)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote alignment report to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
