"""Bench2Drive adapter for the unified three-scene sensor-policy VLA.

The adapter implements the official ``AutonomousAgent`` lifecycle while
reusing the same checkpoint, instruction FSM, temporal safety supervisor and
route PID as the three long-form competition scenarios.  It consumes only the
declared physical sensors and ego/map state; no scenario or event identifiers
enter the online policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import carla
import numpy as np
import torch

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from carla_multiview_sensor import (
    CAMERA_ORDER,
    SynchronizedMultiviewCameraRig,
    radar_closing_speed_mps,
    radar_relative_height_m,
)
from control.generic_route_pid import GenericRoutePID
from universal_vla_controller import UniversalVLAController


def get_entry_point() -> str:
    return "UniversalThreeSceneVLAAgent"


def _distance_2d(left: Any, right: Any) -> float:
    return math.hypot(float(left.x) - float(right.x), float(left.y) - float(right.y))


@dataclass
class _RouteProgressTracker:
    route: Sequence[tuple[Any, Any]]
    distances_m: Sequence[float]
    index: int = 0

    def update(self, location: Any) -> float:
        lower = max(0, self.index - 10)
        upper = min(len(self.route), self.index + 30)
        candidates = range(lower, upper)
        closest = min(
            candidates,
            key=lambda item: _distance_2d(
                self.route[item][0].transform.location,
                location,
            ),
        )
        self.index = max(self.index, closest)
        return float(self.distances_m[self.index])


def _empty_radar(direction: str, frame: int = -1) -> dict[str, Any]:
    return {
        "schema_version": f"physical_{direction}_radar/1.0",
        "direction": direction,
        "sensor_frame": int(frame),
        "candidate_count": 0,
        "obstacle_candidate_count": 0,
        "closing_candidate_count": 0,
        "azimuth_obstacle_bins": [],
        "nearest_distance_m": None,
        "nearest_relative_velocity_mps": None,
        "nearest_azimuth_deg": None,
        "nearest_closing_distance_m": None,
        "nearest_closing_velocity_mps": None,
    }


class _Bench2DriveSensorRig:
    """Expose Bench2Drive ``input_data`` through the validated rig contract."""

    def __init__(self) -> None:
        self.available_cameras = tuple(CAMERA_ORDER)
        self.enable_lidar = True
        self._frame = -1
        self._images: torch.Tensor | None = None
        self._view_mask = torch.ones((1, 4), dtype=torch.bool)
        self._lidar_bev: torch.Tensor | None = None
        self._radar = {
            "front": _empty_radar("front"),
            "rear": _empty_radar("rear"),
        }

    @staticmethod
    def _rgb_tensor(bgra: np.ndarray) -> torch.Tensor:
        if bgra.ndim != 3 or bgra.shape[2] != 4:
            raise ValueError("Bench2Drive RGB input must have shape [H,W,4]")
        rgb = np.ascontiguousarray(bgra[:, :, 2::-1])
        return torch.from_numpy(rgb).permute(2, 0, 1)

    @staticmethod
    def _lidar_tensor(points: np.ndarray) -> torch.Tensor:
        class _Measurement:
            raw_data = np.asarray(points, dtype=np.float32).tobytes()

        return SynchronizedMultiviewCameraRig._rasterize_lidar(_Measurement())

    @staticmethod
    def _radar_observation(
        direction: str,
        frame: int,
        points: np.ndarray,
    ) -> dict[str, Any]:
        candidates: list[dict[str, float]] = []
        for point in np.asarray(points, dtype=np.float32).reshape(-1, 4):
            depth_m, azimuth_rad, altitude_rad, velocity_mps = (
                float(value) for value in point
            )
            azimuth_deg = float(np.degrees(azimuth_rad))
            altitude_deg = float(np.degrees(altitude_rad))
            if (
                0.5 <= depth_m <= 80.0
                and abs(azimuth_deg) <= 8.0
                and abs(altitude_deg) <= 8.0
            ):
                candidates.append(
                    {
                        "distance_m": depth_m,
                        "relative_velocity_mps": velocity_mps,
                        "closing_speed_mps": radar_closing_speed_mps(
                            velocity_mps
                        ),
                        "azimuth_deg": azimuth_deg,
                        "relative_height_m": radar_relative_height_m(
                            depth_m,
                            altitude_rad,
                        ),
                    }
                )
        obstacles = [
            item for item in candidates if item["relative_height_m"] >= -0.65
        ]
        nearest_by_azimuth: dict[int, dict[str, float]] = {}
        for item in obstacles:
            key = int(round(item["azimuth_deg"]))
            previous = nearest_by_azimuth.get(key)
            if previous is None or item["distance_m"] < previous["distance_m"]:
                nearest_by_azimuth[key] = item
        nearest = min(obstacles, key=lambda item: item["distance_m"], default=None)
        closing = [item for item in obstacles if item["closing_speed_mps"] > 0.5]
        nearest_closing = min(
            closing,
            key=lambda item: item["distance_m"],
            default=None,
        )
        result = _empty_radar(direction, frame)
        result.update(
            {
                "candidate_count": len(candidates),
                "obstacle_candidate_count": len(obstacles),
                "closing_candidate_count": len(closing),
                "azimuth_obstacle_bins": [
                    {
                        name: round(float(item[name]), 4)
                        for name in (
                            "distance_m",
                            "relative_velocity_mps",
                            "closing_speed_mps",
                            "azimuth_deg",
                            "relative_height_m",
                        )
                    }
                    for _key, item in sorted(nearest_by_azimuth.items())
                ],
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
                "nearest_closing_distance_m": (
                    round(float(nearest_closing["distance_m"]), 3)
                    if nearest_closing
                    else None
                ),
                "nearest_closing_velocity_mps": (
                    round(float(nearest_closing["closing_speed_mps"]), 3)
                    if nearest_closing
                    else None
                ),
            }
        )
        return result

    def update(self, input_data: Mapping[str, tuple[int, Any]]) -> None:
        required = [*CAMERA_ORDER, "lidar", "front_radar", "rear_radar"]
        missing = [name for name in required if name not in input_data]
        if missing:
            raise RuntimeError(
                "Bench2Drive sensor bundle is incomplete: " + ", ".join(missing)
            )
        frames = {int(input_data[name][0]) for name in required}
        if len(frames) != 1:
            raise RuntimeError(
                f"Bench2Drive sensor bundle is not frame-synchronized: {frames}"
            )
        frame = frames.pop()
        images = [self._rgb_tensor(input_data[name][1]) for name in CAMERA_ORDER]
        self._frame = frame
        self._images = torch.stack(images).unsqueeze(0)
        self._lidar_bev = self._lidar_tensor(input_data["lidar"][1]).unsqueeze(0)
        self._radar = {
            direction: self._radar_observation(
                direction,
                frame,
                input_data[f"{direction}_radar"][1],
            )
            for direction in ("front", "rear")
        }

    def view_available(self, name: str) -> bool:
        return name in self.available_cameras

    def latest_multisensor(
        self,
        *,
        minimum_frame: int | None = None,
        timeout_s: float = 0.08,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        del timeout_s
        if self._images is None or self._lidar_bev is None:
            raise RuntimeError("no synchronized RGB/LiDAR frame is available")
        if minimum_frame is not None and self._frame < int(minimum_frame):
            raise RuntimeError("Bench2Drive sensor bundle is older than control frame")
        return (
            self._frame,
            self._images,
            self._view_mask,
            self._lidar_bev,
            0.0,
        )

    def latest(
        self,
        *,
        minimum_frame: int | None = None,
        timeout_s: float = 0.08,
    ) -> tuple[int, torch.Tensor, torch.Tensor, float]:
        frame, images, mask, _lidar, wait_ms = self.latest_multisensor(
            minimum_frame=minimum_frame,
            timeout_s=timeout_s,
        )
        return frame, images, mask, wait_ms

    def latest_radar(
        self,
        direction: str = "front",
        *,
        maximum_frame: int | None = None,
    ) -> dict[str, Any]:
        if direction not in self._radar:
            raise ValueError("radar direction must be 'front' or 'rear'")
        observation = self._radar[direction]
        if maximum_frame is not None and int(observation["sensor_frame"]) > int(
            maximum_frame
        ):
            return _empty_radar(direction)
        return dict(observation)

    def close(self) -> None:
        self._images = None
        self._lidar_bev = None


class UniversalThreeSceneVLAAgent(AutonomousAgent):
    """Official Bench2Drive entry point for the final unified VLA."""

    def __init__(self, carla_host: str, carla_port: int, debug: bool = False):
        self._full_world_plan: list[tuple[Any, Any]] = []
        self._controller: UniversalVLAController | None = None
        self._sensor_bridge = _Bench2DriveSensorRig()
        self._output_dir: Path | None = None
        super().__init__(carla_host, carla_port, debug)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord) -> None:
        self._full_world_plan = list(global_plan_world_coord)
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    @staticmethod
    def _load_config(path_with_run_name: str) -> tuple[dict[str, Any], str]:
        parts = str(path_with_run_name).split("+")
        path = Path(parts[0]).expanduser().resolve()
        run_name = parts[-1] if len(parts) > 1 else "manual_run"
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        return config, run_name

    @staticmethod
    def _route_distances(route: Sequence[tuple[Any, Any]]) -> list[float]:
        distances = [0.0]
        for index in range(1, len(route)):
            distances.append(
                distances[-1]
                + _distance_2d(
                    route[index - 1][0].transform.location,
                    route[index][0].transform.location,
                )
            )
        return distances

    @staticmethod
    def _road_option_name(option: Any) -> str:
        return str(getattr(option, "name", option)).split(".")[-1].upper()

    def _text_commands(
        self,
        route: Sequence[tuple[Any, Any]],
        distances: Sequence[float],
        speed_kmh: float,
    ) -> list[dict[str, Any]]:
        commands: list[dict[str, Any]] = [
            {
                "id": "bench2drive_route_000",
                "announce_at_m": 0.0,
                "text": (
                    "Follow the planned route, keep lane, obey hazards, and "
                    f"drive at no more than {speed_kmh:.0f} km/h."
                ),
                "semantic_goal": ["keep_lane", "set_speed"],
            }
        ]
        mapping = {
            "LEFT": ("Turn left at the upcoming junction, then keep lane.", ["turn_left"]),
            "RIGHT": ("Turn right at the upcoming junction, then keep lane.", ["turn_right"]),
            "CHANGELANELEFT": (
                "Change to the left lane when legal and safe, then keep lane.",
                ["lane_change_left"],
            ),
            "CHANGELANERIGHT": (
                "Change to the right lane when legal and safe, then keep lane.",
                ["lane_change_right"],
            ),
            "STRAIGHT": ("Continue straight through the junction and keep lane.", ["keep_lane"]),
        }
        previous = "LANEFOLLOW"
        for index, (_waypoint, option) in enumerate(route):
            name = self._road_option_name(option)
            if name == previous or name not in mapping:
                previous = name
                continue
            text, goals = mapping[name]
            commands.append(
                {
                    "id": f"bench2drive_route_{len(commands):03d}",
                    "announce_at_m": max(0.0, float(distances[index]) - 25.0),
                    "end_progress_m": float(distances[index]) + 45.0,
                    "text": text,
                    "semantic_goal": goals,
                }
            )
            previous = name
        return commands

    def setup(self, path_to_conf_file: str) -> None:
        config, run_name = self._load_config(path_to_conf_file)
        self.track = Track.SENSORS
        world = CarlaDataProvider.get_world()
        ego = self.hero_actor
        if ego is None:
            raise RuntimeError("Bench2Drive hero vehicle is unavailable")
        if not self._full_world_plan:
            raise RuntimeError("Bench2Drive global route is empty")
        world_map = world.get_map()
        route: list[tuple[Any, Any]] = []
        for transform, road_option in self._full_world_plan:
            waypoint = world_map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is not None:
                route.append((waypoint, road_option))
        if len(route) < 2:
            raise RuntimeError("Bench2Drive route could not be projected to driving lanes")
        distances = self._route_distances(route)
        tracker = _RouteProgressTracker(route, distances)
        route_context = SimpleNamespace(
            distances_m=distances,
            tracker=tracker,
            adapter=None,
        )
        speed_kmh = float(config.get("default_speed_kmh", 40.0))
        route_pid = GenericRoutePID(
            world,
            ego,
            target_speed_kmh=speed_kmh,
            fixed_delta_seconds=float(config.get("fixed_delta_seconds", 0.05)),
            route_context=route_context,
            route_plan=route,
        )
        output_root = Path(config.get("output_root", "/workspace/outputs/bench2drive"))
        self._output_dir = output_root / run_name
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._controller = UniversalVLAController(
            world=world,
            ego=ego,
            route_controller=route_pid,
            commands=self._text_commands(route, distances, speed_kmh),
            checkpoint_path=Path(config["checkpoint_path"]),
            config_path=Path(config["model_config_path"]),
            parser_model_path=Path(config["parser_model_path"]),
            output_path=self._output_dir / "vla_control_decisions.jsonl",
            device=str(config.get("device", "cuda")),
            precision=str(config.get("precision", "fp16")),
            decision_interval_frames=int(config.get("decision_interval_frames", 3)),
            fixed_delta_seconds=float(config.get("fixed_delta_seconds", 0.05)),
            available_cameras=CAMERA_ORDER,
            enable_lidar=True,
            default_speed_kmh=speed_kmh,
            sensor_rig=self._sensor_bridge,
        )
        manifest = {
            "schema_version": "bench2drive_universal_vla_agent/1.0",
            "agent": type(self).__name__,
            "track": self.track.value,
            "online_scene_or_event_id_access": False,
            "route_points": len(route),
            "route_length_m": round(float(distances[-1]), 3),
            "command_count": len(self._controller.commands),
            "checkpoint_path": str(config["checkpoint_path"]),
            "model_config_path": str(config["model_config_path"]),
            "parser_model_path": str(config["parser_model_path"]),
        }
        (self._output_dir / "agent_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def sensors(self) -> list[dict[str, Any]]:
        camera_positions = {
            "front": (1.45, 0.0, 1.55, -3.0, 0.0),
            "left": (0.15, -0.65, 1.50, -2.0, -90.0),
            "right": (0.15, 0.65, 1.50, -2.0, 90.0),
            "rear": (-1.35, 0.0, 1.50, -2.0, 180.0),
        }
        sensors: list[dict[str, Any]] = []
        for name in CAMERA_ORDER:
            x, y, z, pitch, yaw = camera_positions[name]
            sensors.append(
                {
                    "type": "sensor.camera.rgb",
                    "x": x,
                    "y": y,
                    "z": z,
                    "roll": 0.0,
                    "pitch": pitch,
                    "yaw": yaw,
                    "width": 224,
                    "height": 224,
                    "fov": 100,
                    "id": name,
                }
            )
        sensors.extend(
            [
                {
                    "type": "sensor.lidar.ray_cast",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 2.6,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "id": "lidar",
                },
                {
                    "type": "sensor.other.radar",
                    "x": 2.0,
                    "y": 0.0,
                    "z": 1.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "horizontal_fov": 30.0,
                    "vertical_fov": 10.0,
                    "id": "front_radar",
                },
                {
                    "type": "sensor.other.radar",
                    "x": -2.0,
                    "y": 0.0,
                    "z": 1.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 180.0,
                    "horizontal_fov": 30.0,
                    "vertical_fov": 10.0,
                    "id": "rear_radar",
                },
            ]
        )
        return sensors

    def run_step(self, input_data, timestamp):
        del timestamp
        if self._controller is None:
            raise RuntimeError("Universal VLA controller is not initialized")
        self._sensor_bridge.update(input_data)
        return self._controller.run_step()

    def destroy(self) -> None:
        if self._controller is not None:
            summary = self._controller.summary()
            if self._output_dir is not None:
                (self._output_dir / "controller_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            self._controller.close()
            self._controller = None
