"""Compose detection, tracking and road topology into PerceptionFrame JSON."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from PIL import Image

from .road_structure import road_structure_from_world_state, unavailable_road_structure


class RealtimePerceptionPipeline:
    def __init__(self, detector: Any, tracker: Any) -> None:
        self.detector = detector
        self.tracker = tracker

    def reset(self) -> None:
        self.tracker.reset()

    def process(
        self,
        *,
        image_path: Path,
        frame_id: str,
        source: str,
        camera_name: str,
        timestamp_s: float | None = None,
        world_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = perf_counter()
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            width, height = image.size
            detections, detector_latency_ms = self.detector.detect(image)
        tracks = self.tracker.update(detections)
        for track in tracks:
            x1, y1, x2, y2 = track.pop("bbox_xyxy")
            track["bbox_2d"] = [x1 / width, y1 / height, x2 / width, y2 / height]

        road_structure = (
            road_structure_from_world_state(world_state)
            if world_state is not None
            else unavailable_road_structure()
        )
        visual_road = getattr(self.detector, "latest_road_evidence", None)
        if visual_road:
            road_structure["lane_boundaries"] = visual_road.get("lane_boundaries", [])
            road_structure["drivable_area"] = visual_road.get("drivable_area")
            road_structure["source"] += "+visual_road"
        total_latency_ms = (perf_counter() - started) * 1000
        return {
            "schema_version": "1.0",
            "frame_id": frame_id,
            "source": source,
            "camera_name": camera_name,
            "timestamp_s": timestamp_s,
            "image_size": {"width": width, "height": height},
            "tracks": tracks,
            "road_structure": road_structure,
            "latency_ms": {
                "detector": round(detector_latency_ms, 3),
                "total": round(total_latency_ms, 3),
            },
            "provenance": {
                "detector": self.detector.__class__.__name__,
                "tracker": self.tracker.__class__.__name__,
                "metric_fields_from_vlm": False,
            },
        }
