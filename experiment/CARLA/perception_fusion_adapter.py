"""Adapt synchronous detector tracks to the existing scene fusion contract."""

from __future__ import annotations

from typing import Any, Mapping

from scene_understanding.core.visual_semantic_fusion import (
    fuse_visual_semantics,
)


def _scene_output_from_perception(
    perception_frame: Mapping[str, Any],
) -> dict[str, Any]:
    tracks = perception_frame.get("tracks", [])
    if not isinstance(tracks, list):
        raise ValueError("PerceptionFrame tracks must be an array")
    objects = []
    for index, track in enumerate(tracks, start=1):
        if not isinstance(track, Mapping):
            continue
        objects.append(
            {
                "object_id": f"vlm_obj_{index:03d}",
                "category": str(track.get("category") or "unknown"),
                "subtype": str(track.get("subtype") or "unknown"),
                "color": "unknown",
                "bbox_2d": list(track.get("bbox_2d") or []),
                "relative_position": "unknown",
                "lane_relation": "unknown",
                "motion_state": "unknown",
                "distance_level": "unknown",
                "occlusion": "unknown",
                "confidence": track.get("confidence"),
            }
        )
    road = perception_frame.get("road_structure", {})
    road = road if isinstance(road, Mapping) else {}
    return {
        "schema_version": "1.0",
        "frame_id": str(perception_frame["frame_id"]),
        "source": str(perception_frame["source"]),
        "camera_name": str(perception_frame["camera_name"]),
        "scene": {
            "summary": (
                f"Synchronous detector produced {len(objects)} tracked objects."
            ),
            "road_type": "unknown",
            "is_intersection": road.get("is_junction"),
            "weather": "unknown",
            "visibility": "unknown",
            "traffic_light_state": "unknown",
            "left_lane_marking": "unknown",
            "right_lane_marking": "unknown",
        },
        "objects": objects,
        "potential_hazards": [],
    }


def fuse_perception_frame(
    world_state: dict[str, Any],
    perception_frame: Mapping[str, Any],
    projection_record: dict[str, Any],
    *,
    min_iou: float = 0.05,
    min_confidence: float = 0.25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fuse YOLO/ByteTrack evidence without changing metric CARLA fields."""

    if str(perception_frame.get("frame_id")) != str(world_state["frame_id"]):
        raise ValueError("PerceptionFrame and WorldState frame_id must match")
    inference_record = {
        "frame_id": str(world_state["frame_id"]),
        "source": str(perception_frame["source"]),
        "camera_name": str(perception_frame["camera_name"]),
        "status": "valid",
        "parsed_output": _scene_output_from_perception(perception_frame),
    }
    enriched, audit = fuse_visual_semantics(
        world_state,
        inference_record,
        projection_record,
        min_iou=min_iou,
        min_confidence=min_confidence,
        semantic_source="YOLOP+YOLO11s+ByteTrack",
    )
    audit["adapter"] = "experiment.CARLA.perception_fusion_adapter"
    audit["metric_fields_changed"] = False
    return enriched, audit
