"""Strict same-frame multimodal evidence for the Town05 Scene 2 run."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from scene2_runtime_interface import (
    build_multimodal_frame_bundle,
    build_scheduled_driving_intent,
)


REQUIRED_SENSOR_NAMES = (
    "front_rgb",
    "left_rgb",
    "right_rgb",
    "rear_rgb",
    "lidar",
)


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def write(self, payload: Mapping[str, Any]) -> None:
        self._handle.write(
            json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
            + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class ExactFrameSensorSuite:
    """Record four RGB views and LiDAR behind an exact-frame barrier.

    A frame is marked available only after its artifact has been atomically
    moved to the final path.  The simulation thread can therefore wait for
    all five artifacts before publishing a MultimodalFrameBundle.
    """

    CAMERA_TRANSFORMS = {
        "front_rgb": (1.5, 0.0, 2.4, 0.0),
        "left_rgb": (0.0, -0.35, 2.2, -90.0),
        "right_rgb": (0.0, 0.35, 2.2, 90.0),
        "rear_rgb": (-1.5, 0.0, 2.2, 180.0),
    }

    def __init__(
        self,
        world: Any,
        ego: Any,
        registry: Any,
        output_dir: Path,
        sensor_tick: float,
        image_width: int = 960,
        image_height: int = 540,
    ) -> None:
        self.world = world
        self.ego = ego
        self.registry = registry
        self.output_dir = output_dir
        self.sensor_tick = float(sensor_tick)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.frame_counts: Counter[str] = Counter()
        self.latest_frames: dict[str, int] = {}
        self._saved_frames = {
            name: set() for name in REQUIRED_SENSOR_NAMES
        }
        self._condition = threading.Condition()
        self.expected_frame_checks = 0
        self.complete_frame_checks = 0
        self.incomplete_frame_checks = 0

    def start(self) -> None:
        import carla

        library = self.world.get_blueprint_library()
        for name, values in self.CAMERA_TRANSFORMS.items():
            blueprint = library.find("sensor.camera.rgb")
            for attribute, value in (
                ("image_size_x", str(self.image_width)),
                ("image_size_y", str(self.image_height)),
                ("fov", "90"),
                ("sensor_tick", str(self.sensor_tick)),
                ("gamma", "2.2"),
            ):
                if blueprint.has_attribute(attribute):
                    blueprint.set_attribute(attribute, value)
            x, y, z, yaw = values
            actor = self.world.spawn_actor(
                blueprint,
                carla.Transform(
                    carla.Location(x=x, y=y, z=z),
                    carla.Rotation(yaw=yaw),
                ),
                attach_to=self.ego,
                attachment_type=carla.AttachmentType.Rigid,
            )
            directory = self.output_dir / "rgb" / name
            directory.mkdir(parents=True, exist_ok=True)
            actor.listen(
                lambda image, sensor_name=name, path=directory: (
                    self._save_image(sensor_name, path, image)
                )
            )
            self.registry.add(actor)

        blueprint = library.find("sensor.lidar.ray_cast")
        for attribute, value in (
            ("sensor_tick", str(self.sensor_tick)),
            ("channels", "32"),
            ("range", "80"),
            ("points_per_second", "56000"),
            ("rotation_frequency", str(1.0 / self.sensor_tick)),
        ):
            if blueprint.has_attribute(attribute):
                blueprint.set_attribute(attribute, value)
        directory = self.output_dir / "lidar"
        directory.mkdir(parents=True, exist_ok=True)
        lidar = self.world.spawn_actor(
            blueprint,
            carla.Transform(carla.Location(z=2.6)),
            attach_to=self.ego,
            attachment_type=carla.AttachmentType.Rigid,
        )
        lidar.listen(
            lambda measurement: self._save_lidar(directory, measurement)
        )
        self.registry.add(lidar)

    def _mark_saved(self, sensor_name: str, frame: int) -> None:
        with self._condition:
            self.frame_counts[sensor_name] += 1
            self.latest_frames[sensor_name] = int(frame)
            frames = self._saved_frames[sensor_name]
            frames.add(int(frame))
            cutoff = int(frame) - 64
            if len(frames) > 96:
                frames.difference_update(
                    {saved for saved in frames if saved < cutoff}
                )
            self._condition.notify_all()

    def _save_image(
        self,
        sensor_name: str,
        directory: Path,
        image: Any,
    ) -> None:
        frame = int(image.frame)
        final_path = directory / "{0:08d}.png".format(frame)
        temporary_path = directory / ".{0:08d}.tmp.png".format(frame)
        image.save_to_disk(str(temporary_path))
        temporary_path.replace(final_path)
        self._mark_saved(sensor_name, frame)

    def _save_lidar(self, directory: Path, measurement: Any) -> None:
        frame = int(measurement.frame)
        final_path = directory / "{0:08d}.ply".format(frame)
        temporary_path = directory / ".{0:08d}.tmp.ply".format(frame)
        measurement.save_to_disk(str(temporary_path))
        temporary_path.replace(final_path)
        self._mark_saved("lidar", frame)

    def _frame_is_complete(self, frame: int) -> bool:
        return all(
            int(frame) in self._saved_frames[name]
            for name in REQUIRED_SENSOR_NAMES
        )

    def wait_for_frame(
        self,
        frame: int,
        timeout_s: float,
    ) -> tuple[bool, dict[str, int]]:
        """Wait until every artifact for ``frame`` is safely on disk."""

        frame = int(frame)
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            self.expected_frame_checks += 1
            while not self._frame_is_complete(frame):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            complete = self._frame_is_complete(frame)
            if complete:
                self.complete_frame_checks += 1
            else:
                self.incomplete_frame_checks += 1
            frames = {
                name: (
                    frame
                    if frame in self._saved_frames[name]
                    else self.latest_frames.get(name)
                )
                for name in REQUIRED_SENSOR_NAMES
            }
        return complete, {
            name: value
            for name, value in frames.items()
            if value is not None
        }

    def wait_for_stable_phase(
        self,
        candidate_frames: list[int],
        expected_stride: int,
        timeout_s: float,
    ) -> tuple[int | None, list[int]]:
        """Determine cadence from three consecutive complete sensor frames.

        CARLA may emit the first post-spawn sample with a one-off interval.
        Using that first frame as the modulo phase shifts every later bundle.
        A phase is accepted only after two consecutive intervals equal the
        configured steady-state stride.
        """

        candidates = [int(frame) for frame in candidate_frames]
        stride = max(1, int(expected_stride))
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while True:
                complete_frames = [
                    frame
                    for frame in candidates
                    if self._frame_is_complete(frame)
                ]
                for index in range(2, len(complete_frames)):
                    if (
                        complete_frames[index]
                        - complete_frames[index - 1]
                        == stride
                        and complete_frames[index - 1]
                        - complete_frames[index - 2]
                        == stride
                    ):
                        return complete_frames[index], complete_frames
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None, complete_frames
                self._condition.wait(remaining)

    def summary(self) -> dict[str, Any]:
        expected = int(self.expected_frame_checks)
        complete = int(self.complete_frame_checks)
        return {
            "required_modalities": list(REQUIRED_SENSOR_NAMES),
            "sensor_tick_s": self.sensor_tick,
            "frames_recorded": dict(self.frame_counts),
            "bundle_frames_expected": expected,
            "bundle_frames_complete": complete,
            "bundle_frames_incomplete": int(
                self.incomplete_frame_checks
            ),
            "exact_completion_ratio": (
                complete / expected if expected else None
            ),
            "adjacent_frame_fill_used": False,
        }


class Scene2RuntimeInterface:
    """Configurable JSON boundary for Town05 ASR/VLA/control evidence."""

    def __init__(
        self,
        output_dir: Path,
        config: Mapping[str, Any],
    ) -> None:
        self.scene_id = str(config["scene_id"])
        self.intent_log = _JsonlWriter(
            output_dir / "driving_intent.jsonl"
        )
        self.world_state_log = _JsonlWriter(
            output_dir / "world_state.jsonl"
        )
        self.bundle_log = _JsonlWriter(
            output_dir / "multimodal_frame_bundle.jsonl"
        )
        manifest = {
            "scene_id": self.scene_id,
            **dict(config["interfaces"]),
            "producer": "run_complex_avoidance_town05.py",
            "ground_truth_schema": "FrameGroundTruth/1.0.0",
            "synchronization_barrier": (
                "all required sensor artifacts saved for exact frame"
            ),
            "policy_boundary": (
                "VLA proposals are untrusted until deterministic safety "
                "gating produces ControlDecision."
            ),
        }
        (output_dir / "interface_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def publish_command(
        self,
        command: Mapping[str, Any],
        frame: int,
        route_s_m: float,
        elapsed_s: float,
    ) -> dict[str, Any]:
        intent = build_scheduled_driving_intent(
            command,
            frame,
            route_s_m,
            elapsed_s,
        )
        self.intent_log.write(intent)
        return intent

    def publish_world_state(
        self,
        world: Any,
        ego: Any,
        frame: int,
        route_s_m: float,
        elapsed_s: float,
        safety: Any,
        speed_kmh: Any,
    ) -> dict[str, Any]:
        transform = ego.get_transform()
        waypoint = world.get_map().get_waypoint(
            transform.location,
            project_to_road=True,
        )
        weather = world.get_weather()
        state = {
            "schema_version": "1.0.0",
            "scene_id": self.scene_id,
            "simulation_frame": int(frame),
            "timestamp_s": round(float(elapsed_s), 3),
            "route_s_m": round(float(route_s_m), 3),
            "ego": {
                "actor_id": int(ego.id),
                "location": {
                    "x": transform.location.x,
                    "y": transform.location.y,
                    "z": transform.location.z,
                },
                "rotation": {
                    "pitch": transform.rotation.pitch,
                    "yaw": transform.rotation.yaw,
                    "roll": transform.rotation.roll,
                },
                "speed_kmh": round(float(speed_kmh(ego)), 3),
            },
            "lane": {
                "road_id": int(waypoint.road_id) if waypoint else None,
                "section_id": (
                    int(waypoint.section_id) if waypoint else None
                ),
                "lane_id": int(waypoint.lane_id) if waypoint else None,
                "is_junction": (
                    bool(waypoint.is_junction) if waypoint else False
                ),
            },
            "weather": {
                "cloudiness": weather.cloudiness,
                "fog_density": weather.fog_density,
                "sun_altitude_angle": weather.sun_altitude_angle,
            },
            "safety": {
                "collision_count": len(safety.collisions),
                "lane_invasion_count": len(safety.lane_invasions),
                "restricted_lane_invasion_count": len(
                    safety.restricted_lane_invasions
                ),
            },
        }
        self.world_state_log.write(state)
        return state

    def publish_bundle(
        self,
        frame: int,
        world_state: Mapping[str, Any],
        sensor_frames: Mapping[str, int],
        intent: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        bundle = build_multimodal_frame_bundle(
            self.scene_id,
            frame,
            int(world_state["simulation_frame"]),
            sensor_frames,
            intent.get("request_id") if intent else None,
        )
        self.bundle_log.write(bundle)
        return bundle

    def close(self) -> None:
        self.intent_log.close()
        self.world_state_log.close()
        self.bundle_log.close()
