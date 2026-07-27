from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from .contracts import SensorTensorBatch


class StructuredBEVRasterizer:
    """Build a deterministic CARLA-truth BEV proxy for integration testing.

    This is not a replacement for camera/LiDAR perception. It lets the adapter
    run against the current WorldState contract before a BEVFusion checkpoint
    is connected.
    """

    camera_channels = 8
    lidar_channels = 4
    candidate_dim = 12
    ego_dim = 8

    def __init__(
        self,
        *,
        height: int = 64,
        width: int = 64,
        x_range_m: tuple[float, float] = (-20.0, 60.0),
        y_range_m: tuple[float, float] = (-30.0, 30.0),
        max_candidates: int = 32,
    ) -> None:
        if min(height, width, max_candidates) <= 0:
            raise ValueError("grid dimensions and max_candidates must be positive")
        self.height = int(height)
        self.width = int(width)
        self.x_range_m = x_range_m
        self.y_range_m = y_range_m
        self.max_candidates = int(max_candidates)

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    @classmethod
    def _relative_xyz(cls, entity: Mapping[str, Any]) -> tuple[float, float, float]:
        position = (
            entity.get("relative_position_m")
            or entity.get("relative_position")
            or entity.get("position")
            or {}
        )
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            return (
                cls._number(position[0]),
                cls._number(position[1]),
                cls._number(position[2]) if len(position) > 2 else 0.0,
            )
        if isinstance(position, dict):
            return (
                cls._number(position.get("x")),
                cls._number(position.get("y")),
                cls._number(position.get("z")),
            )
        distance = cls._number(entity.get("distance_m"))
        return distance, 0.0, 0.0

    def _cell(self, x: float, y: float) -> tuple[int, int] | None:
        if not (
            self.x_range_m[0] <= x <= self.x_range_m[1]
            and self.y_range_m[0] <= y <= self.y_range_m[1]
        ):
            return None
        row = round(
            (self.x_range_m[1] - x)
            / (self.x_range_m[1] - self.x_range_m[0])
            * (self.height - 1)
        )
        column = round(
            (y - self.y_range_m[0])
            / (self.y_range_m[1] - self.y_range_m[0])
            * (self.width - 1)
        )
        return int(row), int(column)

    @staticmethod
    def _kind(entity: Mapping[str, Any]) -> tuple[bool, bool, bool]:
        label = str(
            entity.get("class")
            or entity.get("type")
            or entity.get("category")
            or ""
        ).lower()
        vehicle = any(token in label for token in ("vehicle", "car", "truck", "bus"))
        vru = any(token in label for token in ("pedestrian", "cyclist", "bicycle"))
        static = not vehicle and not vru
        return vehicle, vru, static

    def build(
        self,
        world_state: Mapping[str, Any],
        *,
        intent_tokens: torch.Tensor,
        intent_mask: torch.Tensor,
    ) -> tuple[SensorTensorBatch, list[list[str]]]:
        objects = world_state.get("objects", [])
        if not isinstance(objects, list):
            raise ValueError("WorldState objects must be an array")
        camera = torch.zeros(
            1, self.camera_channels, self.height, self.width, dtype=torch.float32
        )
        lidar = torch.zeros(
            1, self.lidar_channels, self.height, self.width, dtype=torch.float32
        )
        candidates = torch.zeros(
            1, self.max_candidates, self.candidate_dim, dtype=torch.float32
        )
        candidate_mask = torch.zeros(
            1, self.max_candidates, dtype=torch.bool
        )
        entity_ids: list[str] = []
        for entity in objects:
            if len(entity_ids) >= self.max_candidates:
                break
            if not isinstance(entity, dict):
                continue
            index = len(entity_ids)
            x, y, z = self._relative_xyz(entity)
            distance = math.sqrt(x * x + y * y + z * z)
            velocity = entity.get("relative_velocity_mps") or entity.get("velocity") or {}
            vx = self._number(velocity.get("x")) if isinstance(velocity, dict) else 0.0
            vy = self._number(velocity.get("y")) if isinstance(velocity, dict) else 0.0
            confidence = self._number(entity.get("confidence"), 1.0)
            lane_relation = str(entity.get("lane_relation") or "same").lower()
            vehicle, vru, static = self._kind(entity)
            candidates[0, index] = torch.tensor(
                [
                    x,
                    y,
                    z,
                    distance,
                    vx,
                    vy,
                    confidence,
                    float("left" in lane_relation),
                    float("same" in lane_relation or "ego" in lane_relation),
                    float("right" in lane_relation),
                    float(vru),
                    float(vehicle),
                ]
            )
            candidate_mask[0, index] = True
            entity_ids.append(str(entity.get("entity_id") or f"candidate_{index}"))
            cell = self._cell(x, y)
            if cell is None:
                continue
            row, column = cell
            camera[0, 0, row, column] = float(vehicle)
            camera[0, 1, row, column] = float(vru)
            camera[0, 2, row, column] = float(static)
            camera[0, 3, row, column] = confidence
            camera[0, 4, row, column] = float("left" in lane_relation)
            camera[0, 5, row, column] = float(
                "same" in lane_relation or "ego" in lane_relation
            )
            camera[0, 6, row, column] = float("right" in lane_relation)
            camera[0, 7, row, column] = min(distance / 100.0, 1.0)
            lidar[0, 0, row, column] = 1.0
            lidar[0, 1, row, column] = max(-1.0, min(z / 5.0, 1.0))
            lidar[0, 2, row, column] = min(distance / 100.0, 1.0)
            lidar[0, 3, row, column] = float(abs(vx) + abs(vy) > 0.1)

        ego = world_state.get("ego", {})
        if not isinstance(ego, dict):
            raise ValueError("WorldState ego must be an object")
        controls = ego.get("control") if isinstance(ego.get("control"), dict) else {}
        environment = world_state.get("environment", {})
        ego_features = torch.tensor(
            [
                [
                    self._number(ego.get("speed_mps")),
                    self._number(ego.get("acceleration_mps2")),
                    self._number(ego.get("yaw_rate_rps")),
                    self._number(controls.get("steer")),
                    self._number(controls.get("throttle")),
                    self._number(controls.get("brake")),
                    self._number(ego.get("speed_limit_mps")),
                    float(bool(environment.get("at_junction", False)))
                    if isinstance(environment, dict)
                    else 0.0,
                ]
            ],
            dtype=torch.float32,
        )
        batch = SensorTensorBatch(
            camera_bev=camera,
            lidar_bev=lidar,
            ego_features=ego_features,
            candidate_features=candidates,
            candidate_mask=candidate_mask,
            intent_tokens=intent_tokens,
            intent_mask=intent_mask,
        )
        batch.validate()
        return batch, [entity_ids]
