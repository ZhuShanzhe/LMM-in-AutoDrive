"""Deterministically adapt compact Qwen detections to the scene schema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


OBJECT_DEFAULTS = {
    "subtype": "unknown",
    "color": "unknown",
    "relative_position": "unknown",
    "lane_relation": "unknown",
    "motion_state": "unknown",
    "distance_level": "unknown",
    "occlusion": "unknown",
}

STATIC_CATEGORIES = {"traffic_light", "traffic_sign"}
HAZARD_KEYS = {"hazard_id", "related_object_ids", "description", "confidence"}

CATEGORY_ALIASES = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "van": "vehicle",
    "vehicle": "vehicle",
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "bicycle": "cyclist",
    "bike": "cyclist",
    "cyclist": "cyclist",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "traffic signal": "traffic_light",
    "traffic sign": "traffic_sign",
    "traffic_sign": "traffic_sign",
    "barrier": "road_barrier",
    "road barrier": "road_barrier",
    "road_barrier": "road_barrier",
    "cone": "traffic_cone",
    "traffic cone": "traffic_cone",
    "traffic_cone": "traffic_cone",
    "animal": "animal",
}


def _category_from_label(label: Any) -> str:
    """Map a common compact detection label into the schema vocabulary."""

    if not isinstance(label, str) or not label.strip():
        return "unknown"
    normalized = label.strip().lower().replace("-", " ")
    return CATEGORY_ALIASES.get(normalized, "other")


def normalize_bbox_coordinates(
    bbox: Any,
    *,
    processed_width: int | None,
    processed_height: int | None,
) -> list[float] | None:
    """Convert Qwen processed-image coordinates to normalized coordinates."""

    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        return None

    values = [float(value) for value in bbox]
    if all(0 <= value <= 1 for value in values):
        return [round(value, 6) for value in values]

    if not processed_width or not processed_height:
        return None

    x_min, y_min, x_max, y_max = values
    if not (0 <= x_min < x_max <= processed_width):
        return None
    if not (0 <= y_min < y_max <= processed_height):
        return None
    return [
        round(x_min / processed_width, 6),
        round(y_min / processed_height, 6),
        round(x_max / processed_width, 6),
        round(y_max / processed_height, 6),
    ]


def normalize_scene_output(
    data: Any,
    *,
    processed_width: int | None = None,
    processed_height: int | None = None,
) -> tuple[Any, list[str]]:
    """Return a schema-oriented copy and an audit trail of deterministic changes.

    The adapter only fills mechanical fields. It never invents a visible object,
    changes an existing semantic attribute, or fabricates a confidence score.
    """

    normalized = deepcopy(data)
    actions: list[str] = []
    if not isinstance(normalized, dict) or not isinstance(normalized.get("objects"), list):
        return normalized, actions

    object_id_map: dict[str, str] = {}
    for index, obj in enumerate(normalized["objects"]):
        if not isinstance(obj, dict):
            continue
        path = f"objects[{index}]"

        if "category" not in obj and "label" in obj:
            obj["category"] = _category_from_label(obj["label"])
            actions.append(f"{path}: mapped label to category")
        if "label" in obj:
            del obj["label"]
            actions.append(f"{path}: removed compact label field")

        expected_object_id = f"vlm_obj_{index + 1:03d}"
        previous_object_id = obj.get("object_id")
        if previous_object_id != expected_object_id:
            if isinstance(previous_object_id, str) and previous_object_id:
                object_id_map[previous_object_id] = expected_object_id
            obj["object_id"] = expected_object_id
            actions.append(f"{path}: assigned deterministic object_id {expected_object_id}")

        for field, default in OBJECT_DEFAULTS.items():
            if field not in obj:
                obj[field] = default
                actions.append(f"{path}: filled missing {field} with unknown")

        if obj.get("category") in STATIC_CATEGORIES and obj.get("motion_state") != "unknown":
            obj["motion_state"] = "unknown"
            actions.append(f"{path}: reset fixed-infrastructure motion_state to unknown")

        converted_bbox = normalize_bbox_coordinates(
            obj.get("bbox_2d"),
            processed_width=processed_width,
            processed_height=processed_height,
        )
        if converted_bbox is not None and converted_bbox != obj.get("bbox_2d"):
            obj["bbox_2d"] = converted_bbox
            actions.append(
                f"{path}: normalized bbox from processed image "
                f"{processed_width}x{processed_height}"
            )

    hazards = normalized.get("potential_hazards")
    if isinstance(hazards, list):
        retained_hazards: list[dict[str, Any]] = []
        for index, hazard in enumerate(hazards):
            path = f"potential_hazards[{index}]"
            if not isinstance(hazard, dict) or set(hazard) != HAZARD_KEYS:
                actions.append(f"{path}: dropped incomplete or non-schema visual hazard")
                continue
            related_ids = hazard.get("related_object_ids")
            if isinstance(related_ids, list):
                remapped_ids = [object_id_map.get(object_id, object_id) for object_id in related_ids]
                if remapped_ids != related_ids:
                    hazard["related_object_ids"] = remapped_ids
                    actions.append(f"{path}: remapped related object IDs")
            retained_hazards.append(hazard)
        normalized["potential_hazards"] = retained_hazards

    scene = normalized.get("scene")
    if isinstance(scene, dict):
        traffic_light_state = scene.get("traffic_light_state")
        has_traffic_light = any(
            isinstance(obj, dict) and obj.get("category") == "traffic_light"
            for obj in normalized["objects"]
        )
        if traffic_light_state not in {None, "unknown", "not_visible"} and not has_traffic_light:
            scene["traffic_light_state"] = "unknown"
            actions.append(
                "scene: reset ungrounded traffic_light_state to unknown"
            )

    return normalized, actions
