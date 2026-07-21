"""CARLA 0.9.16 camera, collision and lane-invasion sensor management.

CARLA is imported lazily during :meth:`CarlaSensorManager.setup`, so event
normalization and unit tests remain usable before a simulator is installed.
"""

from __future__ import annotations

import importlib
import math
import threading
from itertools import count
from pathlib import Path
from typing import Any


def _enum_name(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().split(".")[-1].lower()
    return text or "unknown"


class SensorEventBuffer:
    """Thread-safe normalization and frame-based draining of sensor events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._collisions: list[dict[str, Any]] = []
        self._lane_invasions: list[dict[str, Any]] = []
        self._sequence = count(1)

    def record_collision(self, event: Any) -> dict[str, Any]:
        impulse = event.normal_impulse
        impulse_xyz = {
            "x": float(impulse.x),
            "y": float(impulse.y),
            "z": float(impulse.z),
        }
        magnitude = math.sqrt(sum(value * value for value in impulse_xyz.values()))
        other_actor = getattr(event, "other_actor", None)
        with self._lock:
            sequence = next(self._sequence)
            record = {
                "event_id": f"collision_{int(event.frame):08d}_{sequence:04d}",
                "frame": int(event.frame),
                "timestamp_s": float(event.timestamp),
                "other_actor_id": str(getattr(other_actor, "id", "unknown")),
                "normal_impulse_ns": impulse_xyz,
                "impulse_magnitude_ns": magnitude,
            }
            self._collisions.append(record)
        return record

    def record_lane_invasion(self, event: Any) -> dict[str, Any]:
        markings = [
            _enum_name(getattr(marking, "type", marking))
            for marking in event.crossed_lane_markings
        ]
        with self._lock:
            sequence = next(self._sequence)
            record = {
                "event_id": f"lane_invasion_{int(event.frame):08d}_{sequence:04d}",
                "frame": int(event.frame),
                "timestamp_s": float(event.timestamp),
                "crossed_lane_markings": markings,
            }
            self._lane_invasions.append(record)
        return record

    def drain_through(self, frame: int) -> dict[str, list[dict[str, Any]]]:
        """Return and remove events whose CARLA frame is at most ``frame``."""

        with self._lock:
            collisions, future_collisions = self._partition(self._collisions, frame)
            invasions, future_invasions = self._partition(self._lane_invasions, frame)
            self._collisions = future_collisions
            self._lane_invasions = future_invasions
        return {"collisions": collisions, "lane_invasions": invasions}

    @staticmethod
    def _partition(
        events: list[dict[str, Any]], frame: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ready: list[dict[str, Any]] = []
        future: list[dict[str, Any]] = []
        for event in events:
            (ready if event["frame"] <= frame else future).append(event)
        return ready, future

    def clear(self) -> None:
        with self._lock:
            self._collisions.clear()
            self._lane_invasions.clear()


class CarlaSensorManager:
    """Own the ego camera, collision sensor and lane-invasion sensor."""

    def __init__(
        self,
        world: Any,
        ego_vehicle: Any,
        *,
        output_dir: str | Path = "outputs/carla_sensors",
        enable_camera: bool = True,
        image_width: int = 800,
        image_height: int = 600,
        camera_fov_deg: float = 90.0,
        camera_sensor_tick_s: float = 0.0,
        carla_module: Any | None = None,
    ) -> None:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("camera image dimensions must be positive")
        if not 0 < camera_fov_deg < 180:
            raise ValueError("camera_fov_deg must be between 0 and 180")
        if camera_sensor_tick_s < 0:
            raise ValueError("camera_sensor_tick_s must be non-negative")
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.output_dir = Path(output_dir)
        self.enable_camera = enable_camera
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.camera_fov_deg = float(camera_fov_deg)
        self.camera_sensor_tick_s = float(camera_sensor_tick_s)
        self._carla = carla_module
        self.event_buffer = SensorEventBuffer()
        self._sensors: dict[str, Any] = {}
        self._camera_lock = threading.Lock()
        self._latest_camera_frame: dict[str, Any] | None = None

    @property
    def is_setup(self) -> bool:
        return bool(self._sensors)

    def setup(self) -> None:
        """Spawn and start the configured sensors exactly once."""

        if self.is_setup:
            raise RuntimeError("sensors are already set up")
        carla = self._carla or importlib.import_module("carla")
        self._carla = carla
        library = self.world.get_blueprint_library()

        try:
            self._spawn(
                name="collision",
                blueprint=library.find("sensor.other.collision"),
                transform=carla.Transform(),
                callback=self.event_buffer.record_collision,
            )
            self._spawn(
                name="lane_invasion",
                blueprint=library.find("sensor.other.lane_invasion"),
                transform=carla.Transform(),
                callback=self.event_buffer.record_lane_invasion,
            )
            if self.enable_camera:
                camera_bp = library.find("sensor.camera.rgb")
                for key, value in (
                    ("image_size_x", self.image_width),
                    ("image_size_y", self.image_height),
                    ("fov", self.camera_fov_deg),
                    ("sensor_tick", self.camera_sensor_tick_s),
                ):
                    camera_bp.set_attribute(key, str(value))
                camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
                self._spawn(
                    name="front_rgb",
                    blueprint=camera_bp,
                    transform=camera_transform,
                    callback=self._record_camera_frame,
                )
        except Exception:
            self.destroy()
            raise

    def _spawn(self, *, name: str, blueprint: Any, transform: Any, callback: Any) -> None:
        sensor = self.world.spawn_actor(
            blueprint,
            transform,
            attach_to=self.ego_vehicle,
        )
        self._sensors[name] = sensor
        sensor.listen(callback)

    def _record_camera_frame(self, image: Any) -> None:
        camera_dir = self.output_dir / "front_rgb"
        camera_dir.mkdir(parents=True, exist_ok=True)
        path = camera_dir / f"{int(image.frame):08d}.png"
        image.save_to_disk(str(path))
        record = {
            "camera_name": "front_rgb",
            "frame": int(image.frame),
            "timestamp_s": float(image.timestamp),
            "width": int(image.width),
            "height": int(image.height),
            "image_path": str(path.resolve()),
        }
        with self._camera_lock:
            self._latest_camera_frame = record

    def latest_camera_frame(self) -> dict[str, Any] | None:
        with self._camera_lock:
            return dict(self._latest_camera_frame) if self._latest_camera_frame else None

    def drain_events_through(self, frame: int) -> dict[str, list[dict[str, Any]]]:
        return self.event_buffer.drain_through(frame)

    def destroy(self) -> None:
        """Stop and destroy all owned sensors; safe to call more than once."""

        for sensor in reversed(list(self._sensors.values())):
            try:
                sensor.stop()
            except (RuntimeError, AttributeError):
                pass
            try:
                if getattr(sensor, "is_alive", True):
                    sensor.destroy()
            except RuntimeError:
                pass
        self._sensors.clear()
