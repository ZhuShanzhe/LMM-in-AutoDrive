"""Shared perception vocabulary and DrivingIntent coverage."""

from __future__ import annotations

from typing import Any


COCO_TO_DRIVING: dict[str, tuple[str, str] | None] = {
    "person": ("pedestrian", "person"),
    "bicycle": ("cyclist", "bicycle"),
    "car": ("vehicle", "car"),
    "motorcycle": ("motorcycle", "motorcycle"),
    "bus": ("vehicle", "bus"),
    "truck": ("vehicle", "truck"),
    "traffic light": ("traffic_light", "traffic_light"),
    "stop sign": ("traffic_sign", "stop_sign"),
}

TRACKED_CATEGORIES = {
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

# Every DrivingIntent 1.1 target has an explicit primary source. Targets with
# no reliable camera detector are intentionally delegated to map or async VLM.
INTENT_TARGET_COVERAGE: dict[str, dict[str, Any]] = {
    "VEHICLE": {"source": "detector_tracker", "realtime": True},
    "SLOW_VEHICLE": {"source": "detector_tracker_metric_fusion", "realtime": True},
    "PEDESTRIAN": {"source": "detector_tracker", "realtime": True},
    "CYCLIST": {"source": "detector_tracker", "realtime": True},
    "OBSTACLE": {"source": "detector_or_dataset_annotation", "realtime": True},
    "TRAFFIC_CONE": {"source": "detector_or_dataset_annotation", "realtime": True},
    "CONSTRUCTION_ZONE": {"source": "map_detector_async_vlm", "realtime": False},
    "TRAFFIC_LIGHT": {"source": "detector_and_signal_api", "realtime": True},
    "TRAFFIC_SIGN": {"source": "detector_and_map", "realtime": True},
    "CROSSWALK": {"source": "map_or_segmentation", "realtime": True},
    "STOP_LINE": {"source": "map_or_segmentation", "realtime": True},
    "JUNCTION": {"source": "map", "realtime": True},
    "LANE": {"source": "map_and_road_structure", "realtime": True},
    "ROAD": {"source": "map_and_drivable_area", "realtime": True},
    "AREA": {"source": "map_or_async_vlm", "realtime": False},
    "PARKING_AREA": {"source": "map_or_async_vlm", "realtime": False},
    "PARKING_SPACE": {"source": "map_or_specialized_detector", "realtime": False},
    "CURB": {"source": "segmentation_or_map", "realtime": True},
    "LANDMARK": {"source": "map_or_async_vlm", "realtime": False},
    "DESTINATION": {"source": "route_map", "realtime": True},
    "PICKUP_POINT": {"source": "route_map", "realtime": True},
    "DROPOFF_POINT": {"source": "route_map", "realtime": True},
    "ROAD_HAZARD": {"source": "detector_metric_fusion", "realtime": True},
    "COORDINATE": {"source": "route_map", "realtime": True},
    "UNKNOWN": {"source": "unsupported", "realtime": False},
}

ROAD_FEATURE_TO_ACTIONS = {
    "lane_topology": ["KEEP_LANE", "CHANGE_LANE", "MERGE", "OVERTAKE", "PULL_OVER"],
    "drivable_area": ["KEEP_LANE", "CHANGE_LANE", "TURN", "AVOID", "PARK"],
    "crosswalk": ["YIELD", "STOP", "PROCEED"],
    "stop_line": ["STOP", "PROCEED"],
    "junction": ["TURN", "U_TURN", "PROCEED", "NAVIGATE_TO"],
    "curb": ["PULL_OVER", "PARK", "AVOID"],
    "parking": ["PARK", "NAVIGATE_TO"],
}


def map_coco_label(label: str) -> tuple[str, str] | None:
    return COCO_TO_DRIVING.get(label.strip().lower())


def map_detector_label(label: str) -> tuple[str, str] | None:
    """Accept either COCO names or the module's specialized detector names."""
    normalized = label.strip().lower()
    mapped = COCO_TO_DRIVING.get(normalized)
    if mapped is not None:
        return mapped
    if normalized in TRACKED_CATEGORIES:
        return normalized, normalized
    return None
