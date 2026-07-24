"""Adapter for the official MIT-licensed YOLOP panoptic driving model."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from .detector import Detection


class YolopPanopticBackend:
    """Detect vehicles and extract drivable-area/lane evidence in one forward pass."""

    def __init__(
        self,
        model_root: Path,
        *,
        device: str = "cuda",
        image_size: int = 640,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        import torch

        model_root = model_root.resolve()
        if not (model_root / "weights" / "End-to-end.pth").is_file():
            raise FileNotFoundError(f"YOLOP weights not found under {model_root}")
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))
        from lib.core.general import non_max_suppression, scale_coords
        from lib.models import get_net
        from lib.config import cfg

        self.torch = torch
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.model = get_net(cfg)
        checkpoint = torch.load(model_root / "weights" / "End-to-end.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device).eval()
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.half()
        self.latest_road_evidence: dict[str, Any] | None = None
        self._warmup()

    def _warmup(self) -> None:
        torch = self.torch
        sample = torch.zeros((1, 3, self.image_size, self.image_size), device=self.device)
        sample = sample.half() if self.half else sample.float()
        with torch.inference_mode():
            self.model(sample)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @staticmethod
    def _letterbox(image: Any, size: int) -> tuple[Any, float, tuple[float, float]]:
        import cv2
        import numpy as np

        height, width = image.shape[:2]
        ratio = min(size / height, size / width)
        resized = (int(round(width * ratio)), int(round(height * ratio)))
        pad_w = (size - resized[0]) / 2
        pad_h = (size - resized[1]) / 2
        if (width, height) != resized:
            image = cv2.resize(image, resized, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return np.ascontiguousarray(image), ratio, (pad_w, pad_h)

    @staticmethod
    def _mask_to_road_evidence(drivable: Any, lanes: Any) -> dict[str, Any]:
        import numpy as np

        height, width = drivable.shape
        center = width // 2
        row_fractions = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
        left_edge: list[list[float]] = []
        right_edge: list[list[float]] = []
        lane_left: list[list[float]] = []
        lane_right: list[list[float]] = []

        for fraction in row_fractions:
            y = min(height - 1, int(height * fraction))
            road_x = np.flatnonzero(drivable[y] > 0)
            if road_x.size:
                splits = np.split(road_x, np.where(np.diff(road_x) > 1)[0] + 1)
                segment = next((part for part in splits if part[0] <= center <= part[-1]), max(splits, key=len))
                left_edge.append([round(float(segment[0]) / width, 6), round(y / height, 6)])
                right_edge.append([round(float(segment[-1]) / width, 6), round(y / height, 6)])

            lane_x = np.flatnonzero(lanes[y] > 0)
            left_candidates = lane_x[lane_x < center]
            right_candidates = lane_x[lane_x > center]
            if left_candidates.size:
                lane_left.append([round(float(left_candidates[-1]) / width, 6), round(y / height, 6)])
            if right_candidates.size:
                lane_right.append([round(float(right_candidates[0]) / width, 6), round(y / height, 6)])

        boundaries = []
        for boundary_id, side, points in (
            ("visual_lane_left", "left", lane_left),
            ("visual_lane_right", "right", lane_right),
        ):
            if len(points) >= 2:
                boundaries.append(
                    {
                        "boundary_id": boundary_id,
                        "side": side,
                        "role": "lane_marking",
                        "points": points,
                        "confidence": 0.6,
                    }
                )
        polygon = None
        if len(left_edge) >= 2 and len(right_edge) >= 2:
            polygon = {
                "polygon": left_edge + list(reversed(right_edge)),
                "confidence": 0.7,
                "source": "yolop_bdd100k",
            }
        return {"lane_boundaries": boundaries, "drivable_area": polygon}

    def detect(self, image: Any) -> tuple[list[Detection], float]:
        import cv2
        import numpy as np
        import torch.nn.functional as functional

        torch = self.torch
        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        original_height, original_width = bgr.shape[:2]
        padded, ratio, (pad_w, pad_h) = self._letterbox(bgr, self.image_size)
        tensor = torch.from_numpy(padded).to(self.device).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0)
        tensor = tensor.half() if self.half else tensor.float()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = perf_counter()
        with torch.inference_mode():
            det_out, drivable_out, lane_out = self.model(tensor)
            detections = self.non_max_suppression(
                det_out[0], conf_thres=self.score_threshold, iou_thres=self.iou_threshold
            )[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (perf_counter() - started) * 1000

        output: list[Detection] = []
        if len(detections):
            for x1, y1, x2, y2, confidence, _ in detections.detach().cpu().tolist():
                x1 = min(original_width, max(0.0, (x1 - pad_w) / ratio))
                x2 = min(original_width, max(0.0, (x2 - pad_w) / ratio))
                y1 = min(original_height, max(0.0, (y1 - pad_h) / ratio))
                y2 = min(original_height, max(0.0, (y2 - pad_h) / ratio))
                output.append(
                    Detection(
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=float(confidence),
                        category="vehicle",
                        subtype="vehicle",
                        class_id=2,
                    )
                )

        top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
        resized_height = int(round(original_height * ratio))
        resized_width = int(round(original_width * ratio))
        drivable_crop = drivable_out[:, :, top : top + resized_height, left : left + resized_width]
        lane_crop = lane_out[:, :, top : top + resized_height, left : left + resized_width]
        drivable_mask = functional.interpolate(
            drivable_crop, size=(original_height, original_width), mode="bilinear", align_corners=False
        ).argmax(1)[0].int().cpu().numpy()
        lane_mask = functional.interpolate(
            lane_crop, size=(original_height, original_width), mode="bilinear", align_corners=False
        ).argmax(1)[0].int().cpu().numpy()
        self.latest_road_evidence = self._mask_to_road_evidence(drivable_mask, lane_mask)
        return output, latency_ms
