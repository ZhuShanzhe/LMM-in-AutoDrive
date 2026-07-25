"""Merge focused traffic-control grounding into general scene results."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from scene_understanding.core.evaluate_scene_alignment import bbox_iou, center_inside
from scene_understanding.core.normalize_scene_output import normalize_scene_output
from scene_understanding.core.validate_scene_output import validate_scene_output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def _is_duplicate(candidate: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
    bbox = candidate["bbox_2d"]
    for obj in existing:
        other_bbox = obj.get("bbox_2d")
        if obj.get("category") != candidate.get("category") or not isinstance(other_bbox, list):
            continue
        if (
            bbox_iou(bbox, other_bbox) >= 0.3
            or center_inside(bbox, other_bbox)
            or center_inside(other_bbox, bbox)
        ):
            return True
    return False


def merge_record(
    base_record: dict[str, Any], grounding_record: dict[str, Any] | None
) -> dict[str, Any]:
    """Return a merged inference record with a complete audit trail."""

    updated = deepcopy(base_record)
    updated["pre_traffic_control_status"] = base_record.get("status")
    actions: list[str] = []
    scene_output = updated.get("parsed_output")
    if not isinstance(scene_output, dict):
        updated["traffic_control_merge_actions"] = ["base parsed_output unavailable"]
        return updated

    if grounding_record is None:
        actions.append("no focused grounding record for this frame")
    elif grounding_record.get("status") != "valid":
        actions.append(f"focused grounding status is {grounding_record.get('status')}; not merged")
    else:
        grounding = grounding_record.get("parsed_output") or {}
        objects = scene_output.get("objects")
        if not isinstance(objects, list):
            objects = []
            scene_output["objects"] = objects

        detected_states: set[str] = set()
        for light in grounding.get("traffic_lights", []):
            state = light.get("state", "unknown")
            if state in {"red", "yellow", "green"}:
                detected_states.add(state)
            candidate = {
                "object_id": light.get("grounding_id", "tc_light"),
                "category": "traffic_light",
                "subtype": "signal_head",
                "color": state if state in {"red", "yellow", "green"} else "unknown",
                "bbox_2d": light.get("bbox_2d"),
                "relative_position": "unknown",
                "lane_relation": "unknown",
                "motion_state": "unknown",
                "distance_level": "unknown",
                "occlusion": "unknown",
                "confidence": light.get("confidence"),
            }
            if _is_duplicate(candidate, objects):
                actions.append(f"skipped duplicate {light.get('grounding_id')}")
            else:
                objects.append(candidate)
                actions.append(f"added {light.get('grounding_id')} as traffic_light")

        for sign in grounding.get("traffic_signs", []):
            candidate = {
                "object_id": sign.get("grounding_id", "tc_sign"),
                "category": "traffic_sign",
                "subtype": sign.get("sign_type", "unknown"),
                "color": "unknown",
                "bbox_2d": sign.get("bbox_2d"),
                "relative_position": "unknown",
                "lane_relation": "unknown",
                "motion_state": "unknown",
                "distance_level": "unknown",
                "occlusion": "unknown",
                "confidence": sign.get("confidence"),
            }
            if _is_duplicate(candidate, objects):
                actions.append(f"skipped duplicate {sign.get('grounding_id')}")
            else:
                objects.append(candidate)
                actions.append(f"added {sign.get('grounding_id')} as traffic_sign")

        if len(detected_states) == 1 and isinstance(scene_output.get("scene"), dict):
            state = next(iter(detected_states))
            previous = scene_output["scene"].get("traffic_light_state")
            scene_output["scene"]["traffic_light_state"] = state
            if previous != state:
                actions.append(f"updated scene traffic_light_state from {previous} to {state}")

    normalized, normalization_actions = normalize_scene_output(scene_output)
    actions.extend(f"scene adapter: {action}" for action in normalization_actions)
    errors = validate_scene_output(
        normalized,
        expected_frame_id=base_record.get("frame_id"),
        expected_source=base_record.get("source"),
        expected_camera_name=base_record.get("camera_name"),
    )
    updated["parsed_output"] = normalized
    updated["traffic_control_grounding"] = (
        {
            "status": grounding_record.get("status"),
            "prompt_sha256": grounding_record.get("prompt_sha256"),
            "processed_image_size": grounding_record.get("processed_image_size"),
        }
        if grounding_record is not None
        else None
    )
    updated["traffic_control_merge_actions"] = actions
    updated["validation_errors"] = errors
    updated["status"] = "valid" if not errors else "invalid"
    return updated


def merge_files(base_path: Path, grounding_path: Path, output_path: Path) -> tuple[int, int]:
    """Merge indexed focused results into every base record without overwriting files."""

    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    base_records = read_jsonl(base_path)
    grounding_records = read_jsonl(grounding_path)
    grounding_by_id = {record["frame_id"]: record for record in grounding_records}
    if len(grounding_by_id) != len(grounding_records):
        raise ValueError("duplicate frame_id in focused grounding results")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = 0
    with output_path.open("x", encoding="utf-8") as handle:
        for base_record in base_records:
            updated = merge_record(base_record, grounding_by_id.get(base_record.get("frame_id")))
            handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
            valid += int(updated.get("status") == "valid")
    return len(base_records), valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    total, valid = merge_files(args.base, args.grounding, args.output)
    print(f"Merged {total} frames; schema-valid outputs: {valid}/{total}")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
