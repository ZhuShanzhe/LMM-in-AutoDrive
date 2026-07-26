"""Fuse validated keyframe semantics into metric CARLA WorldState objects."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from scene_understanding.core.evaluate_scene_alignment import bbox_iou, center_inside
from scene_understanding.core.prepare_carla_samples import validate_projection_record
from scene_understanding.core.validate_scene_output import validate_scene_output
from scene_understanding.core.world_state import validate_world_state


def _description(obj: Mapping[str, Any]) -> str:
    parts = [str(obj.get("category", "object")).replace("_", " ")]
    for field in (
        "subtype",
        "color",
        "relative_position",
        "lane_relation",
        "motion_state",
        "distance_level",
    ):
        value = obj.get(field)
        if isinstance(value, str) and value not in {"", "unknown"}:
            parts.append(value.replace("_", " "))
    return "; ".join(parts)


def _scene_output(inference_record: Mapping[str, Any]) -> dict[str, Any]:
    if "parsed_output" in inference_record:
        if inference_record.get("status") != "valid":
            raise ValueError("inference record must have status='valid'")
        output = inference_record.get("parsed_output")
    else:
        output = inference_record
    if not isinstance(output, dict):
        raise ValueError("validated scene output is unavailable")
    return output


def _semantic_source(inference_record: Mapping[str, Any], explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    config = inference_record.get("inference_config")
    if isinstance(config, Mapping):
        model_path = config.get("model_path")
        if isinstance(model_path, str) and model_path.strip():
            return Path(model_path).name
    return "Qwen2.5-VL"


def _candidate_pairs(
    predictions: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    *,
    min_iou: float,
    min_confidence: float,
) -> list[tuple[float, int, float, str, str, dict[str, Any], dict[str, Any]]]:
    candidates = []
    for prediction in predictions:
        confidence = prediction.get("confidence")
        numeric_confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
        if confidence is not None and numeric_confidence < min_confidence:
            continue
        prediction_bbox = prediction.get("bbox_2d")
        if not isinstance(prediction_bbox, list):
            continue
        for truth in truths:
            if prediction.get("category") != truth.get("category"):
                continue
            truth_bbox = truth.get("bbox_2d")
            if not isinstance(truth_bbox, list):
                continue
            iou = bbox_iou(prediction_bbox, truth_bbox)
            center_supported = center_inside(prediction_bbox, truth_bbox) or center_inside(
                truth_bbox, prediction_bbox
            )
            if iou < min_iou and not center_supported:
                continue
            candidates.append(
                (
                    iou,
                    int(center_supported),
                    numeric_confidence,
                    str(prediction["object_id"]),
                    str(truth["world_object_id"]),
                    prediction,
                    truth,
                )
            )
    candidates.sort(
        key=lambda item: (-item[1], -item[0], -item[2], item[3], item[4])
    )
    return candidates


def fuse_visual_semantics(
    world_state: dict[str, Any],
    inference_record: Mapping[str, Any],
    projection_record: dict[str, Any],
    *,
    min_iou: float = 0.05,
    min_confidence: float = 0.0,
    semantic_source: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an enriched WorldState and a transparent one-to-one match audit."""

    if not 0 <= min_iou <= 1:
        raise ValueError("min_iou must be between 0 and 1")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    world_errors = validate_world_state(world_state)
    if world_errors:
        raise ValueError("invalid WorldState: " + "; ".join(world_errors))
    frame_id = str(world_state["frame_id"])
    camera_name = str(projection_record.get("camera_name", ""))
    projection_errors = validate_projection_record(
        projection_record,
        frame_id=frame_id,
        camera_name=camera_name,
    )
    if projection_errors:
        raise ValueError("invalid projection record: " + "; ".join(projection_errors))

    world_truth = {
        str(obj["object_id"]): obj
        for obj in world_state["objects"]
        if isinstance(obj, dict)
    }
    for truth in projection_record["objects"]:
        truth_id = str(truth["world_object_id"])
        parent_id = str(truth.get("parent_world_object_id") or truth_id)
        world_object = world_truth.get(parent_id)
        if world_object is None:
            raise ValueError(
                f"projection references unknown WorldState object: {parent_id}"
            )
        if truth["category"] != world_object["category"]:
            raise ValueError(
                f"projection category mismatch for {parent_id}: "
                f"{truth['category']} != {world_object['category']}"
            )

    output = _scene_output(inference_record)
    scene_errors = validate_scene_output(
        output,
        expected_frame_id=frame_id,
        expected_source="carla",
        expected_camera_name=camera_name,
    )
    if scene_errors:
        raise ValueError("invalid scene output: " + "; ".join(scene_errors))

    predictions = [obj for obj in output["objects"] if isinstance(obj, dict)]
    truths = [obj for obj in projection_record["objects"] if isinstance(obj, dict)]
    candidates = _candidate_pairs(
        predictions,
        truths,
        min_iou=min_iou,
        min_confidence=min_confidence,
    )
    used_predictions: set[str] = set()
    used_truths: set[str] = set()
    selected: list[dict[str, Any]] = []
    for iou, center_supported, _, prediction_id, truth_id, prediction, truth in candidates:
        if prediction_id in used_predictions or truth_id in used_truths:
            continue
        used_predictions.add(prediction_id)
        used_truths.add(truth_id)
        selected.append(
            {
                "world_object_id": str(
                    truth.get("parent_world_object_id") or truth_id
                ),
                "projection_object_id": truth_id,
                "visual_object_id": prediction_id,
                "category": prediction["category"],
                "iou": round(iou, 6),
                "center_supported": bool(center_supported),
                "semantic_match": {
                    "camera_name": camera_name,
                    "visual_object_id": prediction_id,
                    "bbox_2d": list(prediction["bbox_2d"]),
                    "description": _description(prediction),
                    "confidence": prediction.get("confidence"),
                },
            }
        )

    enriched = deepcopy(world_state)
    world_objects = {
        str(obj["object_id"]): obj
        for obj in enriched["objects"]
        if isinstance(obj, dict)
    }
    for obj in world_objects.values():
        obj["semantic_matches"] = [
            match
            for match in obj.get("semantic_matches", [])
            if match.get("camera_name") != camera_name
        ]
    applied: list[dict[str, Any]] = []
    for match in selected:
        world_object = world_objects.get(match["world_object_id"])
        if world_object is None:
            continue
        world_object["semantic_matches"].append(match["semantic_match"])
        applied.append(
            {key: value for key, value in match.items() if key != "semantic_match"}
        )

    scene_summary = output.get("scene", {}).get("summary")
    if isinstance(scene_summary, str) and scene_summary.strip():
        enriched["environment"]["scene_summary"] = scene_summary.strip()
    source_name = _semantic_source(inference_record, semantic_source)
    enriched["provenance"]["semantic_source"] = source_name
    enriched["provenance"]["camera_names"] = sorted(
        set(enriched["provenance"].get("camera_names", [])) | {camera_name}
    )
    enriched_errors = validate_world_state(enriched)
    if enriched_errors:
        raise ValueError("invalid enriched WorldState: " + "; ".join(enriched_errors))

    projected_ids = {str(truth["world_object_id"]) for truth in truths}
    prediction_ids = {str(prediction["object_id"]) for prediction in predictions}
    audit = {
        "schema_version": "1.0",
        "frame_id": frame_id,
        "camera_name": camera_name,
        "semantic_source": source_name,
        "matched_count": len(applied),
        "matches": applied,
        "unmatched_world_object_ids": sorted(projected_ids - used_truths),
        "unmatched_visual_object_ids": sorted(prediction_ids - used_predictions),
        "min_iou": min_iou,
        "min_confidence": min_confidence,
    }
    return enriched, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-state", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()
    world_state = json.loads(args.world_state.read_text(encoding="utf-8"))
    inference = json.loads(args.inference.read_text(encoding="utf-8"))
    projection = json.loads(args.projection.read_text(encoding="utf-8"))
    enriched, audit = fuse_visual_semantics(
        world_state,
        inference,
        projection,
        min_iou=args.min_iou,
        min_confidence=args.min_confidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Fused {audit['matched_count']} visual objects into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
