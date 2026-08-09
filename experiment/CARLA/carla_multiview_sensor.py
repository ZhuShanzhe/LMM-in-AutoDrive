"""Frame-synchronized physical sensor rig for online VLA inference."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np
import torch


CAMERA_ORDER = ("front", "left", "right", "rear")
CAMERA_TRANSFORMS = {
    "front": (1.45, 0.0, 1.55, -3.0, 0.0),
    "left": (0.15, -0.65, 1.50, -2.0, -90.0),
    "right": (0.15, 0.65, 1.50, -2.0, 90.0),
    "rear": (-1.35, 0.0, 1.50, -2.0, 180.0),
}


class SynchronizedMultiviewCameraRig:
    """Own CARLA RGB sensors and expose only complete frame bundles.

    The rig always returns the fixed four-view tensor structure defined by
    ``CAMERA_ORDER``.  Views that are not installed on the ego are returned as
    zero tensors with ``camera_view_mask=False``, so every scene produces the
    same input schema.
    """

    def __init__(
        self,
        world: Any,
        ego: Any,
        *,
        width: int = 224,
        height: int = 224,
        fov: float = 100.0,
        sensor_tick: float = 0.05,
        camera_attributes: dict[str, Any] | None = None,
        retained_frames: int = 12,
        enable_lidar: bool = False,
        lidar_range_m: float = 80.0,
        enable_radar: bool = True,
        radar_range_m: float = 80.0,
        lidar_channels: int = 32,
        available_cameras: Sequence[str] | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self._condition = threading.Condition()
        self._frames: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._latest_complete_frame = -1
        self._latest_multisensor_frame = -1
        self._latest_returned_frame = -1
        self._retained_frames = int(retained_frames)
        self.sensors: list[Any] = []
        self._radar_observations: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.enable_lidar = bool(enable_lidar)
        self.enable_radar = bool(enable_radar)
        if available_cameras is None:
            available_cameras = CAMERA_ORDER
        unknown = set(available_cameras) - set(CAMERA_ORDER)
        if unknown:
            raise ValueError(f"unknown camera views: {sorted(unknown)}")
        self.available_cameras = tuple(available_cameras)
        library = world.get_blueprint_library()
        carla = __import__("carla")
        attributes = dict(camera_attributes or {})
        for name in CAMERA_ORDER:
            if name not in self.available_cameras:
                continue
            blueprint = library.find("sensor.camera.rgb")
            standard = {
                "image_size_x": str(self.width),
                "image_size_y": str(self.height),
                "fov": str(float(fov)),
                "sensor_tick": str(float(sensor_tick)),
            }
            for key, value in {**attributes, **standard}.items():
                if key != "enabled" and blueprint.has_attribute(key):
                    blueprint.set_attribute(key, str(value))
            x, y, z, pitch, yaw = CAMERA_TRANSFORMS[name]
            sensor = world.spawn_actor(
                blueprint,
                carla.Transform(
                    carla.Location(x=x, y=y, z=z),
                    carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0),
                ),
                attach_to=ego,
                attachment_type=carla.AttachmentType.Rigid,
            )
            sensor.listen(self._callback(name))
            self.sensors.append(sensor)
        if self.enable_lidar:
            blueprint = library.find("sensor.lidar.ray_cast")
            lidar_attributes = {
                "sensor_tick": str(float(sensor_tick)),
                "channels": str(int(lidar_channels)),
                "range": str(float(lidar_range_m)),
                "points_per_second": "56000",
                "rotation_frequency": str(1.0 / max(float(sensor_tick), 1e-3)),
            }
            for key, value in lidar_attributes.items():
                if blueprint.has_attribute(key):
                    blueprint.set_attribute(key, value)
            lidar = world.spawn_actor(
                blueprint,
                carla.Transform(carla.Location(z=2.6)),
                attach_to=ego,
                attachment_type=carla.AttachmentType.Rigid,
            )
            lidar.listen(self._lidar_callback)
            self.sensors.append(lidar)
        if self.enable_radar:
            blueprint = library.find("sensor.other.radar")
            radar_attributes = {
                "sensor_tick": str(float(sensor_tick)),
                "horizontal_fov": "30.0",
                "vertical_fov": "10.0",
                "range": str(float(radar_range_m)),
                "points_per_second": "3000",
            }
            for key, value in radar_attributes.items():
                if blueprint.has_attribute(key):
                    blueprint.set_attribute(key, value)
            radar = world.spawn_actor(
                blueprint,
                carla.Transform(
                    carla.Location(x=2.0, z=1.0),
                    carla.Rotation(),
                ),
                attach_to=ego,
                attachment_type=carla.AttachmentType.Rigid,
            )
            radar.listen(self._radar_callback)
            self.sensors.append(radar)

    def view_available(self, name: str) -> bool:
        return name in self.available_cameras

    @staticmethod
    def _rasterize_lidar(measurement: Any) -> torch.Tensor:
        points = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
        x, y, z, intensity = (points[:, index] for index in range(4))
        valid = (
            (x >= -20.0)
            & (x <= 60.0)
            & (y >= -30.0)
            & (y <= 30.0)
        )
        x, y, z, intensity = x[valid], y[valid], z[valid], intensity[valid]
        rows = np.rint((60.0 - x) / 80.0 * 63.0).astype(np.int64)
        columns = np.rint((y + 30.0) / 60.0 * 63.0).astype(np.int64)
        counts = np.zeros((64, 64), dtype=np.float32)
        max_height = np.full((64, 64), -5.0, dtype=np.float32)
        max_intensity = np.zeros((64, 64), dtype=np.float32)
        np.add.at(counts, (rows, columns), 1.0)
        np.maximum.at(max_height, (rows, columns), z)
        np.maximum.at(max_intensity, (rows, columns), intensity)
        occupied = counts > 0
        distance = np.zeros((64, 64), dtype=np.float32)
        distance[rows, columns] = np.minimum(
            np.sqrt(x * x + y * y) / 100.0, 1.0
        )
        channels = np.stack(
            [
                occupied.astype(np.float32),
                np.where(
                    occupied,
                    np.clip(max_height / 5.0, -1.0, 1.0),
                    0.0,
                ),
                distance,
                np.minimum(np.log1p(counts) / np.log(17.0), 1.0)
                * np.clip(max_intensity, 0.0, 1.0),
            ]
        )
        return torch.from_numpy(channels)

    def _lidar_callback(self, measurement: Any) -> None:
        frame = int(measurement.frame)
        tensor = self._rasterize_lidar(measurement)
        with self._condition:
            bundle = self._frames.setdefault(frame, {})
            bundle["lidar_bev"] = tensor
            if all(name in bundle for name in self.available_cameras):
                self._latest_multisensor_frame = max(
                    self._latest_multisensor_frame, frame
                )
                self._condition.notify_all()
            while len(self._frames) > self._retained_frames:
                self._frames.popitem(last=False)

    def _radar_callback(self, measurement: Any) -> None:
        """Keep a narrow, sensor-derived forward obstacle observation.

        Only physical radar detections are used.  No CARLA actor enumeration,
        role names, scenario ids or event schedules enter this observation.
        """

        frame = int(measurement.frame)
        candidates: list[dict[str, float]] = []
        for detection in measurement:
            depth_m = float(detection.depth)
            azimuth_deg = float(np.degrees(detection.azimuth))
            altitude_deg = float(np.degrees(detection.altitude))
            if (
                0.5 <= depth_m <= 80.0
                and abs(azimuth_deg) <= 8.0
                and abs(altitude_deg) <= 8.0
            ):
                candidates.append(
                    {
                        "distance_m": depth_m,
                        "relative_velocity_mps": float(detection.velocity),
                        "azimuth_deg": azimuth_deg,
                        "altitude_deg": altitude_deg,
                    }
                )
        nearest = min(candidates, key=lambda item: item["distance_m"], default=None)
        observation = {
            "schema_version": "physical_forward_radar/1.0",
            "sensor_frame": frame,
            "candidate_count": len(candidates),
            "nearest_distance_m": (
                round(float(nearest["distance_m"]), 3) if nearest else None
            ),
            "nearest_relative_velocity_mps": (
                round(float(nearest["relative_velocity_mps"]), 3)
                if nearest
                else None
            ),
            "nearest_azimuth_deg": (
                round(float(nearest["azimuth_deg"]), 3) if nearest else None
            ),
        }
        with self._condition:
            self._radar_observations[frame] = observation
            while len(self._radar_observations) > self._retained_frames:
                self._radar_observations.popitem(last=False)
            self._condition.notify_all()

    def _callback(self, name: str):
        def receive(image: Any) -> None:
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                image.height, image.width, 4
            )
            rgb = np.ascontiguousarray(bgra[:, :, 2::-1])
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)
            frame = int(image.frame)
            with self._condition:
                bundle = self._frames.setdefault(frame, {})
                bundle[name] = tensor
                if all(
                    camera_name in bundle for camera_name in self.available_cameras
                ):
                    self._latest_complete_frame = max(
                        self._latest_complete_frame, frame
                    )
                    self._condition.notify_all()
                if self.enable_lidar and "lidar_bev" in bundle and all(
                    camera_name in bundle
                    for camera_name in self.available_cameras
                ):
                    self._latest_multisensor_frame = max(
                        self._latest_multisensor_frame, frame
                    )
                    self._condition.notify_all()
                while len(self._frames) > self._retained_frames:
                    self._frames.popitem(last=False)

        return receive

    def _stack_views(
        self,
        bundle: dict[str, torch.Tensor],
        frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        template = None
        for name in self.available_cameras:
            if name in bundle:
                template = bundle[name]
                break
        if template is None:
            raise RuntimeError("no camera frame is available")
        zeros = torch.zeros_like(template)
        views = []
        mask = torch.zeros((1, len(CAMERA_ORDER)), dtype=torch.bool)
        for index, name in enumerate(CAMERA_ORDER):
            if name in bundle:
                views.append(bundle[name])
                mask[0, index] = True
            else:
                views.append(zeros)
        images = torch.stack(views).unsqueeze(0)
        return images, mask

    def latest(
        self,
        *,
        minimum_frame: int | None = None,
        timeout_s: float = 0.08,
    ) -> tuple[int, torch.Tensor, torch.Tensor, float]:
        """Return ``[1,4,3,H,W]`` uint8 RGB from one exact CARLA frame."""

        started = time.monotonic()
        required = -1 if minimum_frame is None else int(minimum_frame)
        with self._condition:
            while self._latest_complete_frame < required:
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            frame = self._latest_complete_frame
            if frame < 0 or frame not in self._frames:
                raise RuntimeError("no synchronized multiview RGB frame is available")
            bundle = self._frames[frame]
            if any(name not in bundle for name in self.available_cameras):
                raise RuntimeError("latest camera frame bundle is incomplete")
            images, mask = self._stack_views(bundle, frame)
            self._latest_returned_frame = frame
        return frame, images, mask, (time.monotonic() - started) * 1000.0

    def latest_multisensor(
        self,
        *,
        minimum_frame: int | None = None,
        timeout_s: float = 0.08,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Return exact-frame four-view RGB and a real LiDAR point-cloud BEV."""

        if not self.enable_lidar:
            raise RuntimeError("LiDAR was not enabled for this sensor rig")
        started = time.monotonic()
        required = -1 if minimum_frame is None else int(minimum_frame)
        with self._condition:
            while self._latest_multisensor_frame < required:
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            frame = self._latest_multisensor_frame
            if frame < 0 or frame not in self._frames:
                raise RuntimeError("no synchronized RGB/LiDAR frame is available")
            bundle = self._frames[frame]
            if "lidar_bev" not in bundle or any(
                name not in bundle for name in self.available_cameras
            ):
                raise RuntimeError("latest RGB/LiDAR frame bundle is incomplete")
            images, mask = self._stack_views(bundle, frame)
            lidar_bev = bundle["lidar_bev"].unsqueeze(0)
            self._latest_returned_frame = frame
        return (
            frame,
            images,
            mask,
            lidar_bev,
            (time.monotonic() - started) * 1000.0,
        )

    def latest_radar(self, *, maximum_frame: int | None = None) -> dict[str, Any]:
        """Return the newest forward radar observation at/before a frame."""

        if not self.enable_radar:
            return {
                "schema_version": "physical_forward_radar/1.0",
                "sensor_frame": -1,
                "candidate_count": 0,
                "nearest_distance_m": None,
                "nearest_relative_velocity_mps": None,
                "nearest_azimuth_deg": None,
            }
        limit = math.inf if maximum_frame is None else int(maximum_frame)
        with self._condition:
            for frame, observation in reversed(self._radar_observations.items()):
                if frame <= limit:
                    return dict(observation)
        return {
            "schema_version": "physical_forward_radar/1.0",
            "sensor_frame": -1,
            "candidate_count": 0,
            "nearest_distance_m": None,
            "nearest_relative_velocity_mps": None,
            "nearest_azimuth_deg": None,
        }

    def close(self) -> None:
        for sensor in self.sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        self.sensors.clear()
