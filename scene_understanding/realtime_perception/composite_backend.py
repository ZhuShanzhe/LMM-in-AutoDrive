"""Fuse a road-structure model with a multi-class traffic detector."""

from __future__ import annotations

from typing import Any

from .detector import Detection


def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_nms(detections: list[Detection], threshold: float) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
        duplicate = any(
            current.category == candidate.category
            and box_iou(current.bbox_xyxy, candidate.bbox_xyxy) >= threshold
            for current in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


class CompositePanopticBackend:
    def __init__(self, road_backend: Any, object_backend: Any, *, merge_iou: float = 0.55) -> None:
        self.road_backend = road_backend
        self.object_backend = object_backend
        self.merge_iou = merge_iou
        self.latest_road_evidence = None

    def detect(self, image: Any) -> tuple[list[Detection], float]:
        road_detections, road_latency = self.road_backend.detect(image)
        object_detections, object_latency = self.object_backend.detect(image)
        self.latest_road_evidence = self.road_backend.latest_road_evidence
        detections = class_aware_nms(
            [*road_detections, *object_detections], threshold=self.merge_iou
        )
        return detections, road_latency + object_latency
