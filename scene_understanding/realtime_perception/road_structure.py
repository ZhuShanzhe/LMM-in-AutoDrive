"""Authoritative road topology adapter for CARLA and annotated datasets."""

from __future__ import annotations

from typing import Any, Mapping


def _adjacent_lane(value: Any, *, direction: str, legal_change: str) -> dict[str, Any]:
    exists = isinstance(value, Mapping) and value.get("lane_type") == "driving"
    return {
        "direction": direction,
        "exists": exists,
        "road_id": value.get("road_id") if exists else None,
        "lane_id": value.get("lane_id") if exists else None,
        "lane_type": value.get("lane_type") if exists else "unknown",
        "change_legal": bool(exists and legal_change in {direction, "both"}),
        "dynamic_safe": "unknown",
        "confidence": 1.0,
    }


def road_structure_from_world_state(world_state: Mapping[str, Any]) -> dict[str, Any]:
    """Build lane-alignment facts without letting image semantics invent geometry."""

    ego = world_state["ego"]
    environment = world_state["environment"]
    adjacent = ego["adjacent_lanes"]
    legal_change = str(ego.get("lane_change", "unknown"))
    source = str(world_state.get("source", "other"))
    source_name = "carla_map" if source == "carla" else f"{source}_map_or_annotation"
    return {
        "source": source_name,
        "safety_eligible": source in {"carla", "nuscenes", "waymo"},
        "ego_lane": {
            "road_id": ego.get("road_id"),
            "section_id": ego.get("section_id"),
            "lane_id": ego.get("lane_id"),
            "lane_type": ego.get("lane_type", "unknown"),
        },
        "adjacent_lanes": {
            "left": _adjacent_lane(adjacent.get("left"), direction="left", legal_change=legal_change),
            "right": _adjacent_lane(adjacent.get("right"), direction="right", legal_change=legal_change),
        },
        "is_junction": bool(ego.get("is_junction") or environment.get("is_intersection")),
        "lane_boundaries": [],
        "drivable_area": None,
        "crosswalks": [],
        "stop_lines": [],
        "curbs": [],
        "parking_areas": [],
        "limitations": [
            "dynamic_safe_requires_tracked_objects_and_metric_risk_assessment",
            "pixel_lane_boundaries_require_semantic_camera_or_segmentation_backend",
        ],
    }


def unavailable_road_structure() -> dict[str, Any]:
    lane = {
        "direction": "unknown",
        "exists": False,
        "road_id": None,
        "lane_id": None,
        "lane_type": "unknown",
        "change_legal": False,
        "dynamic_safe": "unknown",
        "confidence": 0.0,
    }
    return {
        "source": "unavailable",
        "safety_eligible": False,
        "ego_lane": {"road_id": None, "section_id": None, "lane_id": None, "lane_type": "unknown"},
        "adjacent_lanes": {"left": {**lane, "direction": "left"}, "right": {**lane, "direction": "right"}},
        "is_junction": False,
        "lane_boundaries": [],
        "drivable_area": None,
        "crosswalks": [],
        "stop_lines": [],
        "curbs": [],
        "parking_areas": [],
        "limitations": ["no_authoritative_map_or_dataset_road_structure"],
    }
