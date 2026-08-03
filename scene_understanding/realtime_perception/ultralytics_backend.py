"""Ultralytics COCO detector adapter for safety-relevant traffic classes."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from .detector import Detection
from .taxonomy import map_detector_label

INFRASTRUCTURE_CATEGORIES = frozenset({"traffic_light", "traffic_sign"})


def infrastructure_tile_regions(
    width: int, height: int
) -> list[tuple[int, int, int, int]]:
    """Return two overlapping crops covering the upper road scene."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    crop_width = max(1, round(width * 0.6))
    crop_height = max(1, round(height * 0.7))
    return [
        (0, 0, crop_width, crop_height),
        (width - crop_width, 0, width, crop_height),
    ]


def offset_box(
    box: list[float] | tuple[float, ...], x_offset: int, y_offset: int
) -> tuple[float, float, float, float]:
    return (
        float(box[0]) + x_offset,
        float(box[1]) + y_offset,
        float(box[2]) + x_offset,
        float(box[3]) + y_offset,
    )


def load_category_thresholds(path: Path | None) -> dict[str, float]:
    import json

    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("by_category", payload)
    thresholds = {str(key): float(value) for key, value in values.items()}
    invalid = {
        key: value for key, value in thresholds.items() if not 0.0 <= value <= 1.0
    }
    if invalid:
        raise ValueError(f"invalid category thresholds: {invalid}")
    return thresholds


class UltralyticsTrafficDetector:
    def __init__(
        self,
        weights: Path,
        *,
        device: str = "cuda",
        image_size: int = 640,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        category_thresholds: dict[str, float] | None = None,
        infrastructure_tiles: bool = False,
    ) -> None:
        import torch
        from ultralytics import YOLO

        if not weights.is_file():
            raise FileNotFoundError(f"Ultralytics weights not found: {weights}")
        self.torch = torch
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.category_thresholds = dict(category_thresholds or {})
        self.infrastructure_tiles = bool(infrastructure_tiles)
        self.model = YOLO(str(weights))
        self.latest_road_evidence = None
        self._warmup()

    def _warmup(self) -> None:
        import numpy as np

        self.model.predict(
            source=np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8),
            imgsz=self.image_size,
            conf=self.score_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

    def _predict(self, source: Any) -> Any:
        result = self.model.predict(
            source=source,
            imgsz=self.image_size,
            conf=self.score_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]
        return result

    def _decode(
        self,
        result: Any,
        *,
        x_offset: int = 0,
        y_offset: int = 0,
        allowed_categories: frozenset[str] | None = None,
    ) -> list[Detection]:
        output: list[Detection] = []
        if result.boxes is None:
            return output
        for box, score, class_id in zip(
            result.boxes.xyxy.detach().cpu().tolist(),
            result.boxes.conf.detach().cpu().tolist(),
            result.boxes.cls.detach().cpu().int().tolist(),
        ):
            label = str(result.names[int(class_id)])
            mapped = map_detector_label(label)
            if mapped is None:
                continue
            category, subtype = mapped
            if allowed_categories is not None and category not in allowed_categories:
                continue
            threshold = self.category_thresholds.get(
                category, self.score_threshold
            )
            if float(score) < threshold:
                continue
            output.append(
                Detection(
                    bbox_xyxy=offset_box(box, x_offset, y_offset),
                    confidence=float(score),
                    category=category,
                    subtype=subtype,
                    class_id=int(class_id),
                )
            )
        return output

    def detect(self, image: Any) -> tuple[list[Detection], float]:
        import numpy as np

        source = np.asarray(image)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        started = perf_counter()
        output = self._decode(self._predict(source))
        if self.infrastructure_tiles:
            height, width = source.shape[:2]
            for x1, y1, x2, y2 in infrastructure_tile_regions(width, height):
                output.extend(
                    self._decode(
                        self._predict(source[y1:y2, x1:x2]),
                        x_offset=x1,
                        y_offset=y1,
                        allowed_categories=INFRASTRUCTURE_CATEGORIES,
                    )
                )
            from .composite_backend import class_aware_nms

            output = class_aware_nms(output, threshold=0.55)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        latency_ms = (perf_counter() - started) * 1000
        return output, latency_ms
