from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


SENSOR_BUNDLE_SCHEMA_VERSION = "1.0.0"
VLA_PROPOSAL_SCHEMA_VERSION = "1.0.0"
ACTION_LABELS = (
    "keep_lane",
    "accelerate",
    "decelerate",
    "stop",
    "emergency_brake",
    "lane_change_left",
    "lane_change_right",
    "turn_left",
    "turn_right",
)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_sensor_bundle(data: Any, *, max_skew_s: float = 0.1) -> list[str]:
    if not isinstance(data, dict):
        return ["root: expected an object"]
    required = {
        "schema_version",
        "frame_id",
        "timestamp_s",
        "cameras",
        "lidar",
        "ego_state",
        "candidate_entities",
        "feature_refs",
    }
    errors: list[str] = []
    missing = sorted(required - data.keys())
    if missing:
        errors.append("root: missing fields: " + ", ".join(missing))
        return errors
    if data.get("schema_version") != SENSOR_BUNDLE_SCHEMA_VERSION:
        errors.append("schema_version: expected '1.0.0'")
    if not isinstance(data.get("frame_id"), str) or not data["frame_id"]:
        errors.append("frame_id: expected a non-empty string")
    timestamp = data.get("timestamp_s")
    if not _finite_number(timestamp) or float(timestamp) < 0:
        errors.append("timestamp_s: expected a finite non-negative number")

    cameras = data.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        errors.append("cameras: expected a non-empty array")
    else:
        names: set[str] = set()
        for index, camera in enumerate(cameras):
            path = f"cameras[{index}]"
            if not isinstance(camera, dict):
                errors.append(f"{path}: expected an object")
                continue
            name = camera.get("name")
            if not isinstance(name, str) or not name:
                errors.append(f"{path}.name: expected a non-empty string")
            elif name in names:
                errors.append(f"{path}.name: duplicate value {name!r}")
            else:
                names.add(name)
            sensor_timestamp = camera.get("timestamp_s")
            if not _finite_number(sensor_timestamp):
                errors.append(f"{path}.timestamp_s: expected a finite number")
            elif _finite_number(timestamp) and abs(
                float(sensor_timestamp) - float(timestamp)
            ) > max_skew_s:
                errors.append(f"{path}.timestamp_s: exceeds max_skew_s={max_skew_s}")
            if not isinstance(camera.get("image_path"), str) or not camera["image_path"]:
                errors.append(f"{path}.image_path: expected a non-empty string")

    lidar = data.get("lidar")
    if not isinstance(lidar, dict):
        errors.append("lidar: expected an object")
    else:
        lidar_timestamp = lidar.get("timestamp_s")
        if not _finite_number(lidar_timestamp):
            errors.append("lidar.timestamp_s: expected a finite number")
        elif _finite_number(timestamp) and abs(
            float(lidar_timestamp) - float(timestamp)
        ) > max_skew_s:
            errors.append(f"lidar.timestamp_s: exceeds max_skew_s={max_skew_s}")
        if not isinstance(lidar.get("points_path"), str) or not lidar["points_path"]:
            errors.append("lidar.points_path: expected a non-empty string")

    ego = data.get("ego_state")
    if not isinstance(ego, dict):
        errors.append("ego_state: expected an object")
    else:
        for key in ("speed_mps", "acceleration_mps2", "yaw_rate_rps"):
            if not _finite_number(ego.get(key)):
                errors.append(f"ego_state.{key}: expected a finite number")

    candidates = data.get("candidate_entities")
    if not isinstance(candidates, list):
        errors.append("candidate_entities: expected an array")
    else:
        entity_ids: set[str] = set()
        for index, entity in enumerate(candidates):
            path = f"candidate_entities[{index}]"
            if not isinstance(entity, dict):
                errors.append(f"{path}: expected an object")
                continue
            entity_id = entity.get("entity_id")
            if not isinstance(entity_id, str) or not entity_id:
                errors.append(f"{path}.entity_id: expected a non-empty string")
            elif entity_id in entity_ids:
                errors.append(f"{path}.entity_id: duplicate value {entity_id!r}")
            else:
                entity_ids.add(entity_id)

    feature_refs = data.get("feature_refs")
    if not isinstance(feature_refs, dict):
        errors.append("feature_refs: expected an object")
    else:
        for key in ("camera_bev", "lidar_bev"):
            value = feature_refs.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                errors.append(f"feature_refs.{key}: expected null or a path string")
    return errors


def validate_vla_proposal(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return ["root: expected an object"]
    required = {
        "schema_version",
        "request_id",
        "frame_id",
        "action",
        "target_speed_kmh",
        "target_lane",
        "target_location",
        "target_entity_id",
        "confidence",
        "model",
        "latency_ms",
    }
    errors: list[str] = []
    missing = sorted(required - data.keys())
    extra = sorted(data.keys() - required)
    if missing:
        errors.append("root: missing fields: " + ", ".join(missing))
    if extra:
        errors.append("root: unexpected fields: " + ", ".join(extra))
    if data.get("schema_version") != VLA_PROPOSAL_SCHEMA_VERSION:
        errors.append("schema_version: expected '1.0.0'")
    for key in ("request_id", "frame_id", "model"):
        if not isinstance(data.get(key), str) or not data[key]:
            errors.append(f"{key}: expected a non-empty string")
    if data.get("action") not in ACTION_LABELS:
        errors.append("action: invalid value")
    speed = data.get("target_speed_kmh")
    if not _finite_number(speed) or not 0 <= float(speed) <= 100:
        errors.append("target_speed_kmh: expected a number between 0 and 100")
    if data.get("target_lane") not in {None, "left", "right"}:
        errors.append("target_lane: expected null, 'left', or 'right'")
    location = data.get("target_location")
    if location is not None and (
        not isinstance(location, dict)
        or set(location) != {"x", "y", "z"}
        or any(not _finite_number(location[key]) for key in ("x", "y", "z"))
    ):
        errors.append("target_location: expected null or a finite x/y/z object")
    target_entity_id = data.get("target_entity_id")
    if target_entity_id is not None and (
        not isinstance(target_entity_id, str) or not target_entity_id
    ):
        errors.append("target_entity_id: expected null or a non-empty string")
    confidence = data.get("confidence")
    if not _finite_number(confidence) or not 0 <= float(confidence) <= 1:
        errors.append("confidence: expected a number between 0 and 1")
    latency = data.get("latency_ms")
    if not _finite_number(latency) or float(latency) < 0:
        errors.append("latency_ms: expected a finite non-negative number")
    return errors


@dataclass
class SensorTensorBatch:
    camera_bev: torch.Tensor
    lidar_bev: torch.Tensor
    ego_features: torch.Tensor
    candidate_features: torch.Tensor
    candidate_mask: torch.Tensor
    intent_tokens: torch.Tensor
    intent_mask: torch.Tensor
    camera_images: torch.Tensor | None = None
    camera_view_mask: torch.Tensor | None = None
    environment_features: torch.Tensor | None = None

    def validate(self) -> None:
        tensors = {
            "camera_bev": self.camera_bev,
            "lidar_bev": self.lidar_bev,
            "ego_features": self.ego_features,
            "candidate_features": self.candidate_features,
            "candidate_mask": self.candidate_mask,
            "intent_tokens": self.intent_tokens,
            "intent_mask": self.intent_mask,
        }
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("all SensorTensorBatch fields must be torch tensors")
        batch = self.camera_bev.shape[0]
        if self.camera_bev.ndim != 4 or self.lidar_bev.ndim != 4:
            raise ValueError("camera_bev and lidar_bev must have shape [B,C,H,W]")
        if self.lidar_bev.shape[0] != batch:
            raise ValueError("camera_bev and lidar_bev batch sizes differ")
        if self.ego_features.ndim != 2 or self.ego_features.shape[0] != batch:
            raise ValueError("ego_features must have shape [B,E]")
        if self.candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [B,N,C]")
        if self.candidate_features.shape[0] != batch:
            raise ValueError("candidate_features batch size differs")
        if self.candidate_mask.shape != self.candidate_features.shape[:2]:
            raise ValueError("candidate_mask must have shape [B,N]")
        if self.intent_tokens.ndim != 3 or self.intent_tokens.shape[0] != batch:
            raise ValueError("intent_tokens must have shape [B,L,D]")
        if self.intent_mask.shape != self.intent_tokens.shape[:2]:
            raise ValueError("intent_mask must have shape [B,L]")
        if self.camera_images is not None:
            if not isinstance(self.camera_images, torch.Tensor):
                raise TypeError("camera_images must be a torch tensor")
            if (
                self.camera_images.ndim != 5
                or self.camera_images.shape[0] != batch
                or self.camera_images.shape[2] != 3
            ):
                raise ValueError(
                    "camera_images must have shape [B,V,3,H,W]"
                )
            if self.camera_view_mask is None:
                raise ValueError("camera_view_mask is required with camera_images")
            if self.camera_view_mask.shape != self.camera_images.shape[:2]:
                raise ValueError("camera_view_mask must have shape [B,V]")
        elif self.camera_view_mask is not None:
            raise ValueError("camera_images is required with camera_view_mask")
        if self.environment_features is not None:
            if not isinstance(self.environment_features, torch.Tensor):
                raise TypeError("environment_features must be a torch tensor")
            if (
                self.environment_features.ndim != 2
                or self.environment_features.shape[0] != batch
            ):
                raise ValueError(
                    "environment_features must have shape [B,E]"
                )
