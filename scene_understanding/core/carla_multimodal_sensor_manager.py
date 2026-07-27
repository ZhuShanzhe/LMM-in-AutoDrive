"""CARLA multi-camera and LiDAR exact-frame synchronization utilities.

The frame buffer is independent of the CARLA Python package so its safety
properties can be tested on a login node. CARLA sensor callbacks can add
records later without allowing data from adjacent simulation frames to mix.
"""

from __future__ import annotations

from copy import deepcopy
import importlib
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


LIDAR_POINT_BYTES = 16


def _positive_integer(value: Any, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _frame_number(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            "frame must be a non-negative integer"
        )
    return value


def _timestamp(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            "timestamp_s must be a finite non-negative number"
        )
    return float(value)


def camera_intrinsic_matrix(
    *,
    width_px: int,
    height_px: int,
    horizontal_fov_degrees: float,
) -> list[float]:
    """Return a row-major pinhole-camera 3x3 intrinsic matrix."""

    width_px = _positive_integer(width_px, "width_px")
    height_px = _positive_integer(height_px, "height_px")

    if (
        isinstance(horizontal_fov_degrees, bool)
        or not isinstance(
            horizontal_fov_degrees,
            (int, float),
        )
        or not math.isfinite(
            float(horizontal_fov_degrees)
        )
    ):
        raise ValueError(
            "horizontal_fov_degrees must be finite"
        )

    fov = float(horizontal_fov_degrees)
    if not 0.0 < fov < 180.0:
        raise ValueError(
            "horizontal_fov_degrees must be between 0 and 180"
        )

    focal_length = width_px / (
        2.0 * math.tan(math.radians(fov) / 2.0)
    )

    return [
        focal_length,
        0.0,
        width_px / 2.0,
        0.0,
        focal_length,
        height_px / 2.0,
        0.0,
        0.0,
        1.0,
    ]


def lidar_point_count(raw_data: Any) -> int:
    """Return point count for CARLA XYZI float32 LiDAR bytes."""

    try:
        byte_count = memoryview(raw_data).nbytes
    except TypeError as exc:
        raise TypeError(
            "raw_data must support the buffer protocol"
        ) from exc

    if byte_count % LIDAR_POINT_BYTES != 0:
        raise ValueError(
            "CARLA LiDAR raw_data length must be divisible "
            "by 16 bytes"
        )

    return byte_count // LIDAR_POINT_BYTES


def _camera_names(
    values: Iterable[str],
) -> tuple[str, ...]:
    names = tuple(values)

    if any(
        not isinstance(name, str) or not name
        for name in names
    ):
        raise ValueError(
            "required camera names must be non-empty strings"
        )

    if len(set(names)) != len(names):
        raise ValueError(
            "required camera names must be unique"
        )

    return names


def _copy_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("sensor record must be a mapping")
    return deepcopy(dict(record))


class MultimodalSensorFrameBuffer:
    """Thread-safe exact-frame camera and LiDAR record buffer."""

    def __init__(
        self,
        *,
        required_camera_names: Iterable[str],
        require_lidar: bool,
        max_frames: int = 128,
    ) -> None:
        if not isinstance(require_lidar, bool):
            raise ValueError("require_lidar must be boolean")

        self.required_camera_names = _camera_names(
            required_camera_names
        )
        self.require_lidar = require_lidar
        self.max_frames = _positive_integer(
            max_frames,
            "max_frames",
        )

        self._frames: dict[int, dict[str, Any]] = {}
        self._condition = threading.Condition(
            threading.RLock()
        )

    def _slot_unlocked(
        self,
        simulation_frame: int,
    ) -> dict[str, Any]:
        slot = self._frames.get(simulation_frame)

        if slot is None:
            slot = {
                "cameras": {},
                "lidar": None,
            }
            self._frames[simulation_frame] = slot

        return slot

    def _evict_unlocked(self) -> None:
        while len(self._frames) > self.max_frames:
            oldest_frame = min(self._frames)
            del self._frames[oldest_frame]

    def add_camera(
        self,
        record: Mapping[str, Any],
    ) -> None:
        """Add one camera record under its exact simulation frame."""

        stored = _copy_record(record)
        sensor_name = stored.get("sensor_name")

        if sensor_name not in self.required_camera_names:
            raise ValueError(
                f"undeclared camera: {sensor_name!r}"
            )

        frame = _frame_number(
            stored.get("frame")
        )
        stored["timestamp_s"] = _timestamp(
            stored.get("timestamp_s")
        )

        with self._condition:
            slot = self._slot_unlocked(frame)
            existing = slot["cameras"].get(sensor_name)

            if existing is not None and existing != stored:
                raise ValueError(
                    "conflicting camera record for "
                    f"{sensor_name!r} at frame {frame}"
                )

            slot["cameras"][sensor_name] = stored
            self._evict_unlocked()
            self._condition.notify_all()

    def add_lidar(
        self,
        record: Mapping[str, Any],
    ) -> None:
        """Add one LiDAR record under its exact simulation frame."""

        stored = _copy_record(record)

        sensor_name = stored.get("sensor_name")
        if not isinstance(sensor_name, str) or not sensor_name:
            raise ValueError(
                "LiDAR sensor_name must be a non-empty string"
            )

        frame = _frame_number(
            stored.get("frame")
        )
        stored["timestamp_s"] = _timestamp(
            stored.get("timestamp_s")
        )

        point_count = stored.get("point_count")
        if (
            isinstance(point_count, bool)
            or not isinstance(point_count, int)
            or point_count < 0
        ):
            raise ValueError(
                "LiDAR point_count must be a "
                "non-negative integer"
            )

        with self._condition:
            slot = self._slot_unlocked(frame)
            existing = slot["lidar"]

            if existing is not None and existing != stored:
                raise ValueError(
                    "conflicting LiDAR record at "
                    f"frame {frame}"
                )

            slot["lidar"] = stored
            self._evict_unlocked()
            self._condition.notify_all()

    def _snapshot_unlocked(
        self,
        simulation_frame: int,
    ) -> dict[str, Any]:
        slot = self._frames.get(simulation_frame)

        if slot is None:
            camera_records: dict[str, Any] = {}
            lidar_record = None
        else:
            camera_records = slot["cameras"]
            lidar_record = slot["lidar"]

        cameras = [
            deepcopy(camera_records[name])
            for name in self.required_camera_names
            if name in camera_records
        ]

        missing_modalities = [
            name
            for name in self.required_camera_names
            if name not in camera_records
        ]

        if self.require_lidar and lidar_record is None:
            missing_modalities.append("lidar")

        return {
            "simulation_frame": simulation_frame,
            "complete": not missing_modalities,
            "missing_modalities": missing_modalities,
            "cameras": cameras,
            "lidar": deepcopy(lidar_record),
        }

    def snapshot(
        self,
        simulation_frame: int,
    ) -> dict[str, Any]:
        """Return only records belonging to the requested frame."""

        frame = _frame_number(simulation_frame)

        with self._condition:
            return self._snapshot_unlocked(frame)

    def wait_for_snapshot(
        self,
        simulation_frame: int,
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Wait until the exact frame is complete or timeout expires."""

        frame = _frame_number(simulation_frame)

        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) < 0.0
        ):
            raise ValueError(
                "timeout_s must be a finite non-negative number"
            )

        deadline = time.monotonic() + float(timeout_s)

        with self._condition:
            while True:
                snapshot = self._snapshot_unlocked(frame)

                if snapshot["complete"]:
                    return snapshot

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return snapshot

                self._condition.wait(remaining)

    def available_frames(self) -> list[int]:
        """Return buffered frames in increasing order."""

        with self._condition:
            return sorted(self._frames)

    def clear_before(
        self,
        simulation_frame: int,
    ) -> None:
        """Remove records older than the specified frame."""

        frame = _frame_number(simulation_frame)

        with self._condition:
            stale_frames = [
                item
                for item in self._frames
                if item < frame
            ]

            for stale_frame in stale_frames:
                del self._frames[stale_frame]


def _pose_component(
    value: Any,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"{field_name} must be a finite number"
        )
    return float(value)


def transform_to_matrix(
    transform: Any,
) -> list[float]:
    """Convert a CARLA relative transform to a row-major 4x4 matrix."""

    location = getattr(transform, "location", None)
    rotation = getattr(transform, "rotation", None)

    if location is None or rotation is None:
        raise ValueError(
            "transform must expose location and rotation"
        )

    x = _pose_component(location.x, "location.x")
    y = _pose_component(location.y, "location.y")
    z = _pose_component(location.z, "location.z")

    pitch = math.radians(
        _pose_component(rotation.pitch, "rotation.pitch")
    )
    yaw = math.radians(
        _pose_component(rotation.yaw, "rotation.yaw")
    )
    roll = math.radians(
        _pose_component(rotation.roll, "rotation.roll")
    )

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cr = math.cos(roll)
    sr = math.sin(roll)

    return [
        cp * cy,
        cy * sp * sr - sy * cr,
        -cy * sp * cr - sy * sr,
        x,
        cp * sy,
        sy * sp * sr + cy * cr,
        -sy * sp * cr + cy * sr,
        y,
        sp,
        -cp * sr,
        cp * cr,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


class CarlaMultimodalSensorManager:
    """Own four ego RGB cameras and one roof-mounted ray-cast LiDAR."""

    CAMERA_RIG = (
        {
            "sensor_name": "front_rgb",
            "location": (1.5, 0.0, 2.4),
            "rotation": (0.0, 0.0, 0.0),
        },
        {
            "sensor_name": "left_rgb",
            "location": (0.0, -0.5, 2.4),
            "rotation": (0.0, -90.0, 0.0),
        },
        {
            "sensor_name": "right_rgb",
            "location": (0.0, 0.5, 2.4),
            "rotation": (0.0, 90.0, 0.0),
        },
        {
            "sensor_name": "rear_rgb",
            "location": (-1.5, 0.0, 2.4),
            "rotation": (0.0, 180.0, 0.0),
        },
    )

    LIDAR_NAME = "roof_lidar"
    LIDAR_LOCATION = (0.0, 0.0, 2.5)
    LIDAR_ROTATION = (0.0, 0.0, 0.0)

    def __init__(
        self,
        world: Any,
        ego_vehicle: Any,
        *,
        output_dir: str | Path = (
            "outputs/carla_multimodal_sensors"
        ),
        image_width: int = 800,
        image_height: int = 600,
        camera_fov_deg: float = 90.0,
        camera_sensor_tick_s: float = 0.0,
        lidar_sensor_tick_s: float = 0.0,
        lidar_range_m: float = 100.0,
        lidar_channels: int = 32,
        lidar_points_per_second: int = 56000,
        lidar_rotation_frequency_hz: float = 20.0,
        lidar_upper_fov_deg: float = 10.0,
        lidar_lower_fov_deg: float = -30.0,
        history_size: int = 128,
        frame_filter: Callable[[int], bool] | None = None,
        carla_module: Any | None = None,
    ) -> None:
        self.intrinsic_matrix = camera_intrinsic_matrix(
            width_px=image_width,
            height_px=image_height,
            horizontal_fov_degrees=camera_fov_deg,
        )

        if camera_sensor_tick_s < 0:
            raise ValueError(
                "camera_sensor_tick_s must be non-negative"
            )
        if lidar_sensor_tick_s < 0:
            raise ValueError(
                "lidar_sensor_tick_s must be non-negative"
            )
        if lidar_range_m <= 0:
            raise ValueError(
                "lidar_range_m must be positive"
            )
        if lidar_rotation_frequency_hz <= 0:
            raise ValueError(
                "lidar_rotation_frequency_hz must be positive"
            )
        if lidar_upper_fov_deg <= lidar_lower_fov_deg:
            raise ValueError(
                "lidar upper FOV must exceed lower FOV"
            )
        if frame_filter is not None and not callable(
            frame_filter
        ):
            raise TypeError("frame_filter must be callable")

        self.world = world
        self.ego_vehicle = ego_vehicle
        self.output_dir = Path(output_dir)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.camera_fov_deg = float(camera_fov_deg)
        self.camera_sensor_tick_s = float(
            camera_sensor_tick_s
        )
        self.lidar_sensor_tick_s = float(
            lidar_sensor_tick_s
        )
        self.lidar_range_m = float(lidar_range_m)
        self.lidar_channels = _positive_integer(
            lidar_channels,
            "lidar_channels",
        )
        self.lidar_points_per_second = _positive_integer(
            lidar_points_per_second,
            "lidar_points_per_second",
        )
        self.lidar_rotation_frequency_hz = float(
            lidar_rotation_frequency_hz
        )
        self.lidar_upper_fov_deg = float(
            lidar_upper_fov_deg
        )
        self.lidar_lower_fov_deg = float(
            lidar_lower_fov_deg
        )
        self.frame_filter = frame_filter
        self._carla = carla_module
        self._sensors: dict[str, Any] = {}
        self._sensor_transforms: dict[str, Any] = {}

        self.buffer = MultimodalSensorFrameBuffer(
            required_camera_names=tuple(
                item["sensor_name"]
                for item in self.CAMERA_RIG
            ),
            require_lidar=True,
            max_frames=history_size,
        )

    @property
    def is_setup(self) -> bool:
        return bool(self._sensors)

    def _make_transform(
        self,
        *,
        location: tuple[float, float, float],
        rotation: tuple[float, float, float],
    ) -> Any:
        carla = self._carla
        pitch, yaw, roll = rotation

        return carla.Transform(
            carla.Location(
                x=location[0],
                y=location[1],
                z=location[2],
            ),
            carla.Rotation(
                pitch=pitch,
                yaw=yaw,
                roll=roll,
            ),
        )

    def _spawn(
        self,
        *,
        name: str,
        blueprint: Any,
        transform: Any,
        callback: Any,
    ) -> None:
        sensor = self.world.spawn_actor(
            blueprint,
            transform,
            attach_to=self.ego_vehicle,
        )
        self._sensors[name] = sensor
        self._sensor_transforms[name] = transform
        sensor.listen(callback)

    def setup(self) -> None:
        """Spawn the four cameras and LiDAR exactly once."""

        if self.is_setup:
            raise RuntimeError(
                "multimodal sensors are already setup"
            )

        self._carla = self._carla or importlib.import_module(
            "carla"
        )
        library = self.world.get_blueprint_library()

        try:
            camera_bp = library.find("sensor.camera.rgb")

            for key, value in (
                ("image_size_x", self.image_width),
                ("image_size_y", self.image_height),
                ("fov", self.camera_fov_deg),
                ("sensor_tick", self.camera_sensor_tick_s),
            ):
                camera_bp.set_attribute(key, str(value))

            for spec in self.CAMERA_RIG:
                name = spec["sensor_name"]
                transform = self._make_transform(
                    location=spec["location"],
                    rotation=spec["rotation"],
                )

                self._spawn(
                    name=name,
                    blueprint=camera_bp,
                    transform=transform,
                    callback=(
                        lambda image,
                        sensor_name=name,
                        sensor_transform=transform:
                        self._record_camera(
                            sensor_name,
                            sensor_transform,
                            image,
                        )
                    ),
                )

            lidar_bp = library.find(
                "sensor.lidar.ray_cast"
            )

            for key, value in (
                ("range", self.lidar_range_m),
                ("channels", self.lidar_channels),
                (
                    "points_per_second",
                    self.lidar_points_per_second,
                ),
                (
                    "rotation_frequency",
                    self.lidar_rotation_frequency_hz,
                ),
                ("upper_fov", self.lidar_upper_fov_deg),
                ("lower_fov", self.lidar_lower_fov_deg),
                ("sensor_tick", self.lidar_sensor_tick_s),
            ):
                lidar_bp.set_attribute(key, str(value))

            lidar_transform = self._make_transform(
                location=self.LIDAR_LOCATION,
                rotation=self.LIDAR_ROTATION,
            )

            self._spawn(
                name=self.LIDAR_NAME,
                blueprint=lidar_bp,
                transform=lidar_transform,
                callback=(
                    lambda measurement,
                    sensor_transform=lidar_transform:
                    self._record_lidar(
                        sensor_transform,
                        measurement,
                    )
                ),
            )
        except Exception:
            self.destroy()
            raise

    def _selected_frame(self, frame: int) -> bool:
        return (
            self.frame_filter is None
            or self.frame_filter(frame)
        )

    def _record_camera(
        self,
        sensor_name: str,
        sensor_transform: Any,
        image: Any,
    ) -> None:
        frame = int(image.frame)

        if not self._selected_frame(frame):
            return

        sensor_dir = (
            self.output_dir / "sensors" / sensor_name
        )
        sensor_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_path = sensor_dir / f"{frame:08d}.png"
        image.save_to_disk(str(image_path))

        width = int(image.width)
        height = int(image.height)

        self.buffer.add_camera(
            {
                "sensor_name": sensor_name,
                "frame": frame,
                "timestamp_s": float(image.timestamp),
                "image_path": str(image_path.resolve()),
                "image_size": {
                    "width": width,
                    "height": height,
                },
                "intrinsic_matrix": (
                    camera_intrinsic_matrix(
                        width_px=width,
                        height_px=height,
                        horizontal_fov_degrees=(
                            self.camera_fov_deg
                        ),
                    )
                ),
                "sensor_to_ego": transform_to_matrix(
                    sensor_transform
                ),
            }
        )

    def _record_lidar(
        self,
        sensor_transform: Any,
        measurement: Any,
    ) -> None:
        frame = int(measurement.frame)

        if not self._selected_frame(frame):
            return

        raw_data = bytes(measurement.raw_data)
        sensor_dir = (
            self.output_dir / "sensors" / self.LIDAR_NAME
        )
        sensor_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cloud_path = sensor_dir / f"{frame:08d}.bin"
        cloud_path.write_bytes(raw_data)

        self.buffer.add_lidar(
            {
                "sensor_name": self.LIDAR_NAME,
                "frame": frame,
                "timestamp_s": float(
                    measurement.timestamp
                ),
                "point_cloud_path": str(
                    cloud_path.resolve()
                ),
                "point_count": lidar_point_count(
                    raw_data
                ),
                "coordinate_frame": "sensor",
                "sensor_to_ego": transform_to_matrix(
                    sensor_transform
                ),
            }
        )

    def sensor_actor(
        self,
        sensor_name: str,
    ) -> Any | None:
        return self._sensors.get(sensor_name)

    def snapshot(
        self,
        simulation_frame: int,
    ) -> dict[str, Any]:
        return self.buffer.snapshot(simulation_frame)

    def wait_for_snapshot(
        self,
        simulation_frame: int,
        *,
        timeout_s: float = 1.0,
    ) -> dict[str, Any]:
        return self.buffer.wait_for_snapshot(
            simulation_frame,
            timeout_s=timeout_s,
        )

    def destroy(self) -> None:
        """Stop and destroy all owned actors; safe repeatedly."""

        for sensor in reversed(
            list(self._sensors.values())
        ):
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
        self._sensor_transforms.clear()
