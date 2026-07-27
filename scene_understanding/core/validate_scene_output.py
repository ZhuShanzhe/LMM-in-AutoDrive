"""Validate structured VLM scene-understanding output using the standard library."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


TOP_LEVEL_KEYS = {
    "schema_version",
    "frame_id",
    "source",
    "camera_name",
    "scene",
    "objects",
    "potential_hazards",
}

SCENE_KEYS = {
    "summary",
    "road_type",
    "is_intersection",
    "weather",
    "visibility",
    "traffic_light_state",
    "left_lane_marking",
    "right_lane_marking",
}

OBJECT_KEYS = {
    "object_id",
    "category",
    "subtype",
    "color",
    "bbox_2d",
    "relative_position",
    "lane_relation",
    "motion_state",
    "distance_level",
    "occlusion",
    "confidence",
}

HAZARD_KEYS = {
    "hazard_id",
    "related_object_ids",
    "description",
    "confidence",
}

ENUMS = {
    "source": {"nuscenes", "waymo", "carla", "other"},
    "road_type": {"urban", "residential", "highway", "rural", "parking", "unknown"},
    "weather": {"clear", "rain", "fog", "snow", "unknown"},
    "visibility": {"good", "reduced", "poor", "unknown"},
    "traffic_light_state": {"red", "yellow", "green", "off", "not_visible", "unknown"},
    "lane_marking": {"solid", "dashed", "double", "none", "unknown"},
    "category": {
        "vehicle",
        "pedestrian",
        "cyclist",
        "motorcycle",
        "traffic_light",
        "traffic_sign",
        "road_barrier",
        "traffic_cone",
        "animal",
        "other",
        "unknown",
    },
    "relative_position": {
        "front",
        "front_left",
        "front_right",
        "left",
        "right",
        "rear_left",
        "rear",
        "rear_right",
        "unknown",
    },
    "lane_relation": {
        "ego_lane",
        "left_adjacent_lane",
        "right_adjacent_lane",
        "oncoming_lane",
        "crossing_ego_path",
        "roadside",
        "unknown",
    },
    "motion_state": {
        "stopped",
        "parked",
        "moving_same_direction",
        "moving_toward_ego",
        "moving_away",
        "crossing",
        "unknown",
    },
    "distance_level": {"near", "medium", "far", "unknown"},
    "occlusion": {"none", "partial", "heavy", "unknown"},
}

OBJECT_ID_PATTERN = re.compile(r"^vlm_obj_[0-9]{3,}$")
HAZARD_ID_PATTERN = re.compile(r"^hazard_[0-9]{3,}$")


def _check_exact_keys(value: Any, required: set[str], path: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected an object")
        return False

    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required)
    if missing:
        errors.append(f"{path}: missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{path}: unexpected fields: {', '.join(extra)}")
    return not missing and not extra


def _check_nonempty_string(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: expected a non-empty string")


def _check_enum(value: Any, enum_name: str, path: str, errors: list[str]) -> None:
    if value not in ENUMS[enum_name]:
        allowed = ", ".join(sorted(ENUMS[enum_name]))
        errors.append(f"{path}: invalid value {value!r}; allowed: {allowed}")


def _check_confidence(
    value: Any, path: str, errors: list[str], *, allow_null: bool = False
) -> None:
    if allow_null and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        suffix = " or null" if allow_null else ""
        errors.append(f"{path}: expected a number greater than 0 and at most 1{suffix}")
    elif not 0 < value <= 1:
        errors.append(f"{path}: must be greater than 0 and at most 1")


def _validate_scene(scene: Any, errors: list[str]) -> None:
    if not _check_exact_keys(scene, SCENE_KEYS, "scene", errors):
        if not isinstance(scene, dict):
            return

    _check_nonempty_string(scene.get("summary"), "scene.summary", errors)
    _check_enum(scene.get("road_type"), "road_type", "scene.road_type", errors)

    is_intersection = scene.get("is_intersection")
    if is_intersection is not None and not isinstance(is_intersection, bool):
        errors.append("scene.is_intersection: expected true, false, or null")

    _check_enum(scene.get("weather"), "weather", "scene.weather", errors)
    _check_enum(scene.get("visibility"), "visibility", "scene.visibility", errors)
    _check_enum(
        scene.get("traffic_light_state"),
        "traffic_light_state",
        "scene.traffic_light_state",
        errors,
    )
    _check_enum(scene.get("left_lane_marking"), "lane_marking", "scene.left_lane_marking", errors)
    _check_enum(scene.get("right_lane_marking"), "lane_marking", "scene.right_lane_marking", errors)


def _validate_bbox(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != 4:
        errors.append(f"{path}: expected four normalized coordinates")
        return

    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        errors.append(f"{path}: every coordinate must be a number")
        return

    if any(item < 0 or item > 1 for item in value):
        errors.append(f"{path}: every coordinate must be between 0 and 1")
        return

    x_min, y_min, x_max, y_max = value
    if x_min >= x_max:
        errors.append(f"{path}: x_min must be smaller than x_max")
    if y_min >= y_max:
        errors.append(f"{path}: y_min must be smaller than y_max")


def _validate_objects(objects: Any, errors: list[str]) -> set[str]:
    if not isinstance(objects, list):
        errors.append("objects: expected an array")
        return set()

    object_ids: set[str] = set()
    for index, obj in enumerate(objects):
        path = f"objects[{index}]"
        if not _check_exact_keys(obj, OBJECT_KEYS, path, errors):
            if not isinstance(obj, dict):
                continue

        object_id = obj.get("object_id")
        if not isinstance(object_id, str) or not OBJECT_ID_PATTERN.fullmatch(object_id):
            errors.append(f"{path}.object_id: expected vlm_obj_ followed by at least three digits")
        elif object_id in object_ids:
            errors.append(f"{path}.object_id: duplicate ID {object_id}")
        else:
            object_ids.add(object_id)

        _check_enum(obj.get("category"), "category", f"{path}.category", errors)
        _check_nonempty_string(obj.get("subtype"), f"{path}.subtype", errors)
        _check_nonempty_string(obj.get("color"), f"{path}.color", errors)
        _validate_bbox(obj.get("bbox_2d"), f"{path}.bbox_2d", errors)
        _check_enum(
            obj.get("relative_position"),
            "relative_position",
            f"{path}.relative_position",
            errors,
        )
        _check_enum(obj.get("lane_relation"), "lane_relation", f"{path}.lane_relation", errors)
        _check_enum(obj.get("motion_state"), "motion_state", f"{path}.motion_state", errors)
        if obj.get("category") in {"traffic_light", "traffic_sign"} and obj.get(
            "motion_state"
        ) != "unknown":
            errors.append(f"{path}.motion_state: fixed infrastructure must use 'unknown'")
        _check_enum(obj.get("distance_level"), "distance_level", f"{path}.distance_level", errors)
        _check_enum(obj.get("occlusion"), "occlusion", f"{path}.occlusion", errors)
        _check_confidence(
            obj.get("confidence"), f"{path}.confidence", errors, allow_null=True
        )

    return object_ids


def _validate_hazards(hazards: Any, object_ids: set[str], errors: list[str]) -> None:
    if not isinstance(hazards, list):
        errors.append("potential_hazards: expected an array")
        return

    hazard_ids: set[str] = set()
    for index, hazard in enumerate(hazards):
        path = f"potential_hazards[{index}]"
        if not _check_exact_keys(hazard, HAZARD_KEYS, path, errors):
            if not isinstance(hazard, dict):
                continue

        hazard_id = hazard.get("hazard_id")
        if not isinstance(hazard_id, str) or not HAZARD_ID_PATTERN.fullmatch(hazard_id):
            errors.append(f"{path}.hazard_id: expected hazard_ followed by at least three digits")
        elif hazard_id in hazard_ids:
            errors.append(f"{path}.hazard_id: duplicate ID {hazard_id}")
        else:
            hazard_ids.add(hazard_id)

        related_ids = hazard.get("related_object_ids")
        if not isinstance(related_ids, list) or any(not isinstance(item, str) for item in related_ids):
            errors.append(f"{path}.related_object_ids: expected an array of object IDs")
        else:
            if len(related_ids) != len(set(related_ids)):
                errors.append(f"{path}.related_object_ids: duplicate object IDs")
            unknown_ids = sorted(set(related_ids) - object_ids)
            if unknown_ids:
                errors.append(f"{path}.related_object_ids: unknown IDs: {', '.join(unknown_ids)}")

        _check_nonempty_string(hazard.get("description"), f"{path}.description", errors)
        _check_confidence(hazard.get("confidence"), f"{path}.confidence", errors)


def validate_scene_output(
    data: Any,
    *,
    expected_frame_id: str | None = None,
    expected_source: str | None = None,
    expected_camera_name: str | None = None,
) -> list[str]:
    """Return validation errors. An empty list means that the record is valid."""

    errors: list[str] = []
    if not _check_exact_keys(data, TOP_LEVEL_KEYS, "root", errors):
        if not isinstance(data, dict):
            return errors

    if data.get("schema_version") != "1.0":
        errors.append("schema_version: expected '1.0'")

    _check_nonempty_string(data.get("frame_id"), "frame_id", errors)
    _check_enum(data.get("source"), "source", "source", errors)
    _check_nonempty_string(data.get("camera_name"), "camera_name", errors)

    if expected_frame_id is not None and data.get("frame_id") != expected_frame_id:
        errors.append(f"frame_id: expected {expected_frame_id!r}, got {data.get('frame_id')!r}")
    if expected_source is not None and data.get("source") != expected_source:
        errors.append(f"source: expected {expected_source!r}, got {data.get('source')!r}")
    if expected_camera_name is not None and data.get("camera_name") != expected_camera_name:
        errors.append(
            f"camera_name: expected {expected_camera_name!r}, got {data.get('camera_name')!r}"
        )

    _validate_scene(data.get("scene"), errors)
    object_ids = _validate_objects(data.get("objects"), errors)
    scene = data.get("scene")
    objects = data.get("objects")
    if isinstance(scene, dict) and isinstance(objects, list):
        light_state = scene.get("traffic_light_state")
        if light_state in {"red", "yellow", "green"} and not any(
            isinstance(obj, dict) and obj.get("category") == "traffic_light" for obj in objects
        ):
            errors.append(
                "objects: a visible traffic_light_state must be grounded by a traffic_light object"
            )
    _validate_hazards(data.get("potential_hazards"), object_ids, errors)
    return errors


def parse_json_text(text: str) -> Any:
    """Parse strict JSON, while tolerating one outer Markdown JSON fence."""

    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


def load_json(path: Path) -> Any:
    """Load structured scene output from a UTF-8 file."""

    return parse_json_text(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file produced by the VLM")
    parser.add_argument("--frame-id", help="Expected frame ID")
    parser.add_argument("--source", choices=sorted(ENUMS["source"]), help="Expected data source")
    parser.add_argument("--camera-name", help="Expected camera name")
    args = parser.parse_args()

    try:
        data = load_json(args.input)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID JSON: {exc}", file=sys.stderr)
        return 1

    errors = validate_scene_output(
        data,
        expected_frame_id=args.frame_id,
        expected_source=args.source,
        expected_camera_name=args.camera_name,
    )
    if errors:
        print("INVALID SCENE OUTPUT", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("VALID SCENE OUTPUT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
