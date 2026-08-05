"""Frame-synchronized in-memory RGB rig for online VLA inference."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

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
    """Own four CARLA RGB sensors and expose only complete frame bundles."""

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
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self._condition = threading.Condition()
        self._frames: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self._latest_complete_frame = -1
        self._latest_returned_frame = -1
        self._retained_frames = int(retained_frames)
        self.sensors: list[Any] = []
        library = world.get_blueprint_library()
        carla = __import__("carla")
        attributes = dict(camera_attributes or {})
        for name in CAMERA_ORDER:
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
                if len(bundle) == len(CAMERA_ORDER):
                    self._latest_complete_frame = max(
                        self._latest_complete_frame, frame
                    )
                    self._condition.notify_all()
                while len(self._frames) > self._retained_frames:
                    self._frames.popitem(last=False)

        return receive

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
            if any(name not in bundle for name in CAMERA_ORDER):
                raise RuntimeError("latest camera frame bundle is incomplete")
            images = torch.stack([bundle[name] for name in CAMERA_ORDER]).unsqueeze(0)
            mask = torch.ones((1, len(CAMERA_ORDER)), dtype=torch.bool)
            self._latest_returned_frame = frame
        return frame, images, mask, (time.monotonic() - started) * 1000.0

    def close(self) -> None:
        for sensor in self.sensors:
            try:
                if sensor.is_alive:
                    sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        self.sensors.clear()
