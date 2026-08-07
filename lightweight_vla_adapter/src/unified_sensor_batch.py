"""Fixed unified input superset shared by every CARLA scene.

All scenes construct the exact same ``UnifiedSensorBatch`` field and tensor
structure.  Scene differences are expressed only through ``modality_mask`` and
``camera_view_mask``; unavailable modalities are represented as zero tensors
with the corresponding mask entry set to ``False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .contracts import SensorTensorBatch


UNIFIED_SENSOR_BATCH_SCHEMA_VERSION = "unified_sensor_batch/1.0"

CAMERA_VIEW_NAMES = ("front", "left", "right", "rear")

MODALITY_KEYS = (
    "text",
    "front_rgb",
    "left_rgb",
    "right_rgb",
    "rear_rgb",
    "lidar_bev",
    "vehicle_state",
    "environment_state",
)


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


@dataclass
class UnifiedSensorBatch:
    """Tensor superset consumed by the universal VLA pipeline.

    Tensor shapes (batch dimension ``B``):
      text_tokens        [B, L, D]
      front/left/right/rear_rgb [B, 3, H, W] (uint8 or float)
      lidar_bev          [B, 4, H, W]
      vehicle_state      [B, E]
      environment_state  [B, E]
      camera_view_mask   [B, 4] (bool)
      modality_mask      mapping from modality name to bool availability
    """

    schema_version: str = UNIFIED_SENSOR_BATCH_SCHEMA_VERSION
    text_tokens: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    front_rgb: torch.Tensor | None = None
    left_rgb: torch.Tensor | None = None
    right_rgb: torch.Tensor | None = None
    rear_rgb: torch.Tensor | None = None
    lidar_bev: torch.Tensor | None = None
    vehicle_state: torch.Tensor | None = None
    environment_state: torch.Tensor | None = None
    camera_view_mask: torch.Tensor | None = None
    modality_mask: dict[str, bool] = field(default_factory=dict)
    frame_id: str = ""
    timestamp_s: float = 0.0

    def modality(self, key: str) -> bool:
        return bool(self.modality_mask.get(key, False))

    def validate(self) -> None:
        if self.schema_version != UNIFIED_SENSOR_BATCH_SCHEMA_VERSION:
            raise ValueError(
                "UnifiedSensorBatch schema_version must be "
                f"{UNIFIED_SENSOR_BATCH_SCHEMA_VERSION!r}"
            )
        for name in MODALITY_KEYS:
            if name not in self.modality_mask:
                raise ValueError(f"modality_mask is missing key {name!r}")
        batch = (
            int(self.text_tokens.shape[0])
            if self.text_tokens is not None
            else None
        )
        if self.text_tokens is not None and self.text_tokens.ndim != 3:
            raise ValueError("text_tokens must have shape [B,L,D]")
        if self.text_mask is not None:
            if self.text_mask.ndim != 2 or (
                self.text_tokens is not None
                and self.text_mask.shape != self.text_tokens.shape[:2]
            ):
                raise ValueError("text_mask must have shape [B,L]")
        if self.camera_view_mask is not None:
            if (
                self.camera_view_mask.ndim != 2
                or self.camera_view_mask.shape[1] != len(CAMERA_VIEW_NAMES)
            ):
                raise ValueError("camera_view_mask must have shape [B,4]")
            batch = int(self.camera_view_mask.shape[0])
        if batch is None:
            raise ValueError("UnifiedSensorBatch requires at least one tensor")
        for view_name in CAMERA_VIEW_NAMES:
            tensor = getattr(self, f"{view_name}_rgb")
            if tensor is not None:
                if tensor.ndim != 4 or tensor.shape[0] != batch or tensor.shape[1] != 3:
                    raise ValueError(f"{view_name}_rgb must have shape [B,3,H,W]")
        if self.lidar_bev is not None:
            if self.lidar_bev.ndim != 4 or self.lidar_bev.shape[0] != batch:
                raise ValueError("lidar_bev must have shape [B,C,H,W]")
        for name in ("vehicle_state", "environment_state"):
            tensor = getattr(self, name)
            if tensor is not None:
                if tensor.ndim != 2 or tensor.shape[0] != batch:
                    raise ValueError(f"{name} must have shape [B,E]")
        if not _finite(self.timestamp_s) or float(self.timestamp_s) < 0:
            raise ValueError("timestamp_s must be a finite non-negative number")
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty string")

    def to_sensor_batch(self) -> SensorTensorBatch:
        """Convert to the model adapter's fixed tensor contract."""

        self.validate()
        batch = (
            int(self.text_tokens.shape[0])
            if self.text_tokens is not None
            else int(self.camera_view_mask.shape[0])
        )
        if self.camera_view_mask is None:
            raise ValueError("camera_view_mask is required")
        device = (
            self.text_tokens.device
            if self.text_tokens is not None
            else torch.device("cpu")
        )

        height = width = None
        for view_name in CAMERA_VIEW_NAMES:
            tensor = getattr(self, f"{view_name}_rgb")
            if tensor is not None:
                height, width = int(tensor.shape[2]), int(tensor.shape[3])
                break
        if height is None or width is None:
            raise ValueError("at least one RGB view tensor is required")

        rgb_views = []
        for view_name in CAMERA_VIEW_NAMES:
            tensor = getattr(self, f"{view_name}_rgb")
            if tensor is None or not self.modality(f"{view_name}_rgb"):
                tensor = torch.zeros(
                    batch, 3, height, width, dtype=torch.uint8, device=device
                )
            else:
                tensor = tensor.to(device=device)
            rgb_views.append(tensor)
        camera_images = torch.stack(rgb_views, dim=1)

        lidar = self.lidar_bev
        if lidar is None or not self.modality("lidar_bev"):
            channels = 4
            if self.lidar_bev is not None:
                channels = int(self.lidar_bev.shape[1])
            lidar = torch.zeros(batch, channels, 64, 64, device=device)
        else:
            lidar = lidar.to(device=device)

        vehicle_state = self.vehicle_state
        if vehicle_state is None or not self.modality("vehicle_state"):
            dim = (
                int(self.vehicle_state.shape[1])
                if self.vehicle_state is not None
                else 8
            )
            vehicle_state = torch.zeros(batch, dim, device=device)
        else:
            vehicle_state = vehicle_state.to(device=device)

        environment_state = self.environment_state
        if environment_state is None or not self.modality("environment_state"):
            dim = (
                int(self.environment_state.shape[1])
                if self.environment_state is not None
                else 14
            )
            environment_state = torch.zeros(batch, dim, device=device)
        else:
            environment_state = environment_state.to(device=device)

        return SensorTensorBatch(
            camera_bev=torch.zeros(
                batch, 8, 64, 64, dtype=torch.float32, device=device
            ),
            lidar_bev=lidar.to(dtype=torch.float32),
            ego_features=vehicle_state.to(dtype=torch.float32),
            candidate_features=torch.zeros(
                batch, 32, 12, dtype=torch.float32, device=device
            ),
            candidate_mask=torch.zeros(
                batch, 32, dtype=torch.bool, device=device
            ),
            intent_tokens=self.text_tokens.to(
                device=device, dtype=torch.float32
            ),
            intent_mask=torch.ones(
                batch,
                int(self.text_tokens.shape[1]),
                dtype=torch.bool,
                device=device,
            )
            if self.text_mask is None
            else self.text_mask.to(device=device),
            camera_images=camera_images,
            camera_view_mask=self.camera_view_mask.to(device=device),
            environment_features=environment_state.to(dtype=torch.float32),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "timestamp_s": round(float(self.timestamp_s), 3),
            "modality_mask": {
                key: bool(value) for key, value in self.modality_mask.items()
            },
            "camera_view_mask": (
                self.camera_view_mask[0].tolist()
                if self.camera_view_mask is not None
                else None
            ),
            "text_tokens": (
                list(self.text_tokens.shape) if self.text_tokens is not None else None
            ),
            "rgb_shapes": {
                name: (
                    list(getattr(self, f"{name}_rgb").shape)
                    if getattr(self, f"{name}_rgb") is not None
                    else None
                )
                for name in CAMERA_VIEW_NAMES
            },
            "lidar_bev": (
                list(self.lidar_bev.shape) if self.lidar_bev is not None else None
            ),
            "vehicle_state": (
                list(self.vehicle_state.shape)
                if self.vehicle_state is not None
                else None
            ),
            "environment_state": (
                list(self.environment_state.shape)
                if self.environment_state is not None
                else None
            ),
        }


def default_modality_mask(
    *,
    text: bool = True,
    front_rgb: bool = True,
    left_rgb: bool = False,
    right_rgb: bool = False,
    rear_rgb: bool = False,
    lidar_bev: bool = False,
    vehicle_state: bool = True,
    environment_state: bool = True,
) -> dict[str, bool]:
    return {
        "text": bool(text),
        "front_rgb": bool(front_rgb),
        "left_rgb": bool(left_rgb),
        "right_rgb": bool(right_rgb),
        "rear_rgb": bool(rear_rgb),
        "lidar_bev": bool(lidar_bev),
        "vehicle_state": bool(vehicle_state),
        "environment_state": bool(environment_state),
    }
