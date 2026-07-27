"""Run synchronous camera perception and attach its evidence to WorldState.

The bridge intentionally keeps CARLA metric state authoritative for TTC and
lane legality.  Camera perception provides same-frame visual evidence that is
matched to CARLA actors before the existing scene-decision policy runs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scene_understanding.core.prepare_carla_samples import append_jsonl
from scene_understanding.core.visual_semantic_fusion import fuse_visual_semantics
from scene_understanding.realtime_perception.composite_backend import CompositePanopticBackend
from scene_understanding.realtime_perception.pipeline import RealtimePerceptionPipeline
from scene_understanding.realtime_perception.tracker import ByteTrackAdapter
from scene_understanding.realtime_perception.ultralytics_backend import UltralyticsTrafficDetector
from scene_understanding.realtime_perception.yolop_backend import YolopPanopticBackend


def _scene_weather(world_state: Mapping[str, Any]) -> str:
    weather = str(world_state.get("environment", {}).get("weather", "unknown")).lower()
    if "rain" in weather:
        return "rain"
    if "fog" in weather:
        return "fog"
    if "clear" in weather or "sun" in weather:
        return "clear"
    return "unknown"


def _scene_output(perception: Mapping[str, Any], world_state: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt detector tracks to the strict scene-output contract used by fusion."""

    supported_categories = {
        "vehicle", "pedestrian", "cyclist", "motorcycle", "traffic_light",
        "traffic_sign", "road_barrier", "traffic_cone",
    }
    objects = []
    for index, track in enumerate(perception.get("tracks", []), start=1):
        category = str(track.get("category", "unknown"))
        if category not in supported_categories:
            category = "other"
        objects.append({
            "object_id": "vlm_obj_{0:03d}".format(index),
            "category": category,
            "subtype": str(track.get("subtype") or category),
            "color": "unknown",
            "bbox_2d": list(track["bbox_2d"]),
            "relative_position": "unknown",
            "lane_relation": "unknown",
            "motion_state": "unknown",
            "distance_level": "unknown",
            "occlusion": "unknown",
            "confidence": float(track.get("confidence", 0.01)),
        })
    environment = world_state.get("environment", {})
    return {
        "schema_version": "1.0",
        "frame_id": perception["frame_id"],
        "source": "carla",
        "camera_name": perception["camera_name"],
        "scene": {
            "summary": "Synchronous detector/tracker scene evidence.",
            "road_type": str(environment.get("road_type") or "unknown"),
            "is_intersection": environment.get("is_intersection"),
            "weather": _scene_weather(world_state),
            "visibility": str(environment.get("visibility") or "unknown"),
            "traffic_light_state": "unknown",
            "left_lane_marking": "unknown",
            "right_lane_marking": "unknown",
        },
        "objects": objects,
        "potential_hazards": [],
    }


class LivePerceptionBridge:
    """Own the detector/tracker and persist frame-aligned perception evidence."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        yolop_root: str | Path,
        yolo11_weights: str | Path,
        device: str = "cuda",
        image_size: int = 640,
        object_image_size: int | None = None,
        score_threshold: float = 0.10,
        frame_rate: int = 10,
        min_iou: float = 0.05,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        road_detector = YolopPanopticBackend(
            Path(yolop_root), device=device, image_size=image_size,
            score_threshold=score_threshold,
        )
        object_detector = UltralyticsTrafficDetector(
            Path(yolo11_weights), device=device,
            image_size=object_image_size or image_size,
            score_threshold=score_threshold,
        )
        self.pipeline = RealtimePerceptionPipeline(
            CompositePanopticBackend(road_detector, object_detector),
            ByteTrackAdapter(frame_rate=frame_rate, track_activation_threshold=score_threshold),
        )
        self.min_iou = min_iou
        self.perception_path = self.output_dir / "perception_frames.jsonl"
        self.audit_path = self.output_dir / "visual_semantic_fusion_audits.jsonl"

    def process_capture(self, capture: Mapping[str, Any]) -> dict[str, Any]:
        """Process one capture record and return an enriched WorldState or error."""

        if capture.get("status") != "captured":
            return {"status": str(capture.get("status", "not_captured"))}
        started = time.perf_counter()
        try:
            world_state = json.loads(Path(capture["world_state_path"]).read_text(encoding="utf-8"))
            projection = json.loads(Path(capture["projection_path"]).read_text(encoding="utf-8"))
            perception = self.pipeline.process(
                image_path=Path(capture["image_path"]),
                frame_id=str(capture["frame_id"]), source="carla",
                camera_name=str(capture["camera_name"]),
                timestamp_s=capture.get("timestamp_s"), world_state=world_state,
            )
            inference = _scene_output(perception, world_state)
            enriched, audit = fuse_visual_semantics(
                world_state, inference, projection,
                min_iou=self.min_iou, min_confidence=0.0,
                semantic_source="YOLOP+YOLO11s+ByteTrack",
            )
            append_jsonl(self.perception_path, perception)
            append_jsonl(self.audit_path, audit)
            return {
                "status": "accepted",
                "frame_id": perception["frame_id"],
                "world_state": enriched,
                "perception": perception,
                "fusion_audit": audit,
                "latency_ms": {
                    "perception": perception["latency_ms"]["total"],
                    "fusion": round((time.perf_counter() - started) * 1000 - perception["latency_ms"]["total"], 3),
                    "total": round((time.perf_counter() - started) * 1000, 3),
                },
            }
        except Exception as exc:  # Keep the CARLA-truth safety path available.
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "latency_ms": {"total": round((time.perf_counter() - started) * 1000, 3)},
            }
