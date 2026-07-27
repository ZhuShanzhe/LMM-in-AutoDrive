"""Torchvision detector backend for the synchronous perception path."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .taxonomy import map_coco_label


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    category: str
    subtype: str
    class_id: int


class TorchvisionTrafficDetector:
    """SSDLite320-MobileNetV3 detector with a small, permissive dependency set."""

    def __init__(self, *, device: str = "cuda", score_threshold: float = 0.25) -> None:
        import torch
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        if not 0 < score_threshold < 1:
            raise ValueError("score_threshold must be between 0 and 1")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.categories = self.weights.meta["categories"]
        self.transforms = self.weights.transforms()
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights).to(self.device).eval()

    def detect(self, image: Any) -> tuple[list[Detection], float]:
        import torch

        tensor = self.transforms(image).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = perf_counter()
        with torch.inference_mode():
            prediction = self.model([tensor])[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (perf_counter() - started) * 1000

        detections: list[Detection] = []
        for box, score, class_id in zip(
            prediction["boxes"].detach().cpu().tolist(),
            prediction["scores"].detach().cpu().tolist(),
            prediction["labels"].detach().cpu().tolist(),
        ):
            if score < self.score_threshold:
                continue
            label = self.categories[int(class_id)]
            mapped = map_coco_label(label)
            if mapped is None:
                continue
            category, subtype = mapped
            detections.append(
                Detection(
                    bbox_xyxy=tuple(float(value) for value in box),
                    confidence=float(score),
                    category=category,
                    subtype=subtype,
                    class_id=int(class_id),
                )
            )
        return detections, latency_ms
