"""Run the deterministic 8 km CARLA Scene 2 competition environment.

This module replaces the original coordinate-based traffic demo with a
contract-driven runner:

* actors are placed with CARLA OpenDRIVE waypoints;
* simulation and Traffic Manager use one synchronous clock;
* an ego vehicle drives the 8 km route while a third-person chase camera
  records the same presentation view used by Scenes 1 and 3;
* commands and visible events are scheduled from a versioned JSON contract;
* collision, lane-invasion, command, event and multimodal interface records
  are written with CARLA ``simulation_frame`` as their synchronization key.

The script intentionally does not claim that an autopilot action proves VLA
success.  External decision/control code can consume ``driving_intent.jsonl``
and write step feedback through the public ``Scene2RuntimeInterface`` methods.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, field
import glob
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET


SCENE_ID = "scene_2_complex_avoidance_8km"
SCHEMA_VERSION = "scene_2_complex_summary/v1"
ROAD_ID = 1
ROUTE_LENGTH_M = 8000.0
SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name == "maps":
    EXPERIMENT_ROOT = SCRIPT_DIR.parent
    DEFAULT_CONFIG = (
        EXPERIMENT_ROOT
        / "configs"
        / "scene_2_complex_avoidance_8km_runtime.json"
    )
    DEFAULT_XODR = (
        SCRIPT_DIR / "maps" / "output" / "VLA_ComplexRoad_8km.xodr"
    )
    DEFAULT_OUTPUT = (
        EXPERIMENT_ROOT / "outputs" / "scene2_complex_avoidance"
    )
else:
    EXPERIMENT_ROOT = SCRIPT_DIR
    DEFAULT_CONFIG = (
        SCRIPT_DIR / "scene_2_complex_avoidance_8km_runtime.json"
    )
    DEFAULT_XODR = SCRIPT_DIR / "VLA_ComplexRoad_8km.xodr"
    DEFAULT_OUTPUT = SCRIPT_DIR / "outputs"
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
from scene2_runtime_interface import (
    build_multimodal_frame_bundle,
    build_scheduled_driving_intent,
)

CAR_BLUEPRINTS = (
    "vehicle.tesla.model3",
    "vehicle.audi.tt",
    "vehicle.toyota.prius",
    "vehicle.lincoln.mkz_2020",
)
BUS_BLUEPRINTS = (
    "vehicle.mitsubishi.fusorosa",
    "vehicle.volkswagen.t2_2021",
    "vehicle.volkswagen.t2",
)
BICYCLE_BLUEPRINTS = (
    "vehicle.bh.crossbike",
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
)
EGO_BLUEPRINTS = (
    "vehicle.tesla.model3",
    "vehicle.lincoln.mkz_2020",
    "vehicle.audi.etron",
)


def python_abi_tag() -> str:
    return "cp{0}{1}".format(
        sys.version_info.major,
        sys.version_info.minor,
    )


def setup_carla_api() -> None:
    """Locate CARLA without assuming a Windows-only installation."""

    if importlib.util.find_spec("carla") is not None:
        return
    carla_root = os.environ.get("CARLA_ROOT")
    if not carla_root:
        raise RuntimeError(
            "CARLA Python API is unavailable. Set CARLA_ROOT or install "
            "a CARLA 0.9.16 Python package matching this interpreter."
        )
    api_dir = Path(carla_root) / "PythonAPI" / "carla"
    dist_dir = api_dir / "dist"
    candidates = sorted(
        glob.glob(str(dist_dir / "carla-*.egg"))
        + glob.glob(str(dist_dir / "carla-*.whl"))
    )
    expected_tag = python_abi_tag()
    matching = [
        candidate
        for candidate in candidates
        if expected_tag in Path(candidate).name
    ]
    if not matching:
        package_tags = []
        for candidate in candidates:
            match = re.search(r"cp\d+", Path(candidate).name)
            package_tags.append(
                match.group(0) if match else Path(candidate).name
            )
        raise RuntimeError(
            "No CARLA package matches {0}; available packages: {1}".format(
                expected_tag,
                ", ".join(package_tags) or "none",
            )
        )
    sys.path.insert(0, matching[0])
    sys.path.insert(0, str(api_dir))


def import_carla() -> Any:
    setup_carla_api()
    import carla

    return carla


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the competition-aligned CARLA Scene 2 mixed-traffic "
            "environment and emit auditable interface records."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--xodr", type=Path, default=DEFAULT_XODR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Simulation seconds; 0 runs until route completion.",
    )
    parser.add_argument(
        "--fixed-delta-seconds",
        type=float,
        default=0.05,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ego-speed-kmh", type=float, default=45.0)
    parser.add_argument("--stationary-ego", action="store_true")
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--video-overlay", action="store_true")
    parser.add_argument(
        "--video-profile",
        choices=("realtime", "quality"),
        default="quality",
    )
    parser.add_argument(
        "--record-images",
        action="store_true",
        help="Also save chase-camera PNG evidence.",
    )
    parser.add_argument(
        "--record-multimodal",
        action="store_true",
        help="Spawn four RGB cameras and LiDAR and save synchronized evidence.",
    )
    parser.add_argument("--sensor-tick", type=float, default=0.2)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
    )
    parser.add_argument(
        "--require-complete-scene",
        action="store_true",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_runtime_config(
    config: Mapping[str, Any],
) -> None:
    if config.get("scene_id") != SCENE_ID:
        raise ValueError("unexpected scene_id")
    route = config.get("route", {})
    if float(route.get("length_m", 0.0)) != ROUTE_LENGTH_M:
        raise ValueError("Scene 2 route must be exactly 8000 m")

    commands = config.get("voice_commands", [])
    if len(commands) != 15:
        raise ValueError("Scene 2 requires 15 online demo commands")
    announce_positions = [
        float(item["announce_at_m"])
        for item in commands
    ]
    if announce_positions != sorted(announce_positions):
        raise ValueError("voice commands must be monotonic")
    command_ids = [str(item["id"]) for item in commands]
    if len(command_ids) != len(set(command_ids)):
        raise ValueError("voice command ids must be unique")
    for command in commands:
        steps = command.get("steps", [])
        if len(steps) < 3:
            raise ValueError(
                "{0} must preserve at least three ordered steps".format(
                    command.get("id")
                )
            )
        if not str(command.get("spoken_text", "")).strip():
            raise ValueError(
                "{0} has no spoken text".format(command.get("id"))
            )

    events = config.get("events", [])
    if len(events) != 6:
        raise ValueError("Scene 2 requires six visible special events")
    event_ids = [str(item["id"]) for item in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event ids must be unique")
    for event in events:
        if float(event["activate_at_m"]) >= float(
            event["resolve_at_m"]
        ):
            raise ValueError(
                "{0} has an invalid event window".format(event["id"])
            )
        if not event.get("actors"):
            raise ValueError(
                "{0} has no visible actors".format(event["id"])
            )

    traffic = config.get("traffic", {})
    required_counts = {
        "private_cars": 24,
        "city_buses": 3,
        "bicycles": 6,
        "sidewalk_pedestrians": 18,
    }
    for name, expected in required_counts.items():
        if int(traffic.get(name, -1)) != expected:
            raise ValueError(
                "traffic.{0} must equal {1}".format(name, expected)
            )

    required_sensors = set(
        config.get("sensors", {}).get("required", [])
    )
    expected_sensors = {
        "front_rgb",
        "left_rgb",
        "right_rgb",
        "rear_rgb",
        "lidar",
        "vehicle_state",
        "collision",
        "lane_invasion",
    }
    if not expected_sensors.issubset(required_sensors):
        raise ValueError(
            "missing required sensor contracts: {0}".format(
                sorted(expected_sensors - required_sensors)
            )
        )
    interfaces = config.get("interfaces", {})
    if interfaces.get("driving_intent_schema") != (
        "DrivingIntent/1.2.0"
    ):
        raise ValueError("DrivingIntent 1.2.0 is required")
    if not interfaces.get("simulation_frame_is_sync_key"):
        raise ValueError(
            "simulation_frame must be the synchronization key"
        )
    if interfaces.get("allow_adjacent_frame_fill"):
        raise ValueError(
            "adjacent-frame multimodal filling is forbidden"
        )


def validate_xodr(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    if root.tag != "OpenDRIVE":
        raise ValueError("OpenDRIVE root element is missing")
    roads = root.findall("road")
    junctions = root.findall("junction")
    main = next(
        (
            road
            for road in roads
            if int(road.get("id", "-1")) == ROAD_ID
        ),
        None,
    )
    if main is None:
        raise ValueError("main road id=1 is missing")
    if abs(float(main.get("length", "0")) - ROUTE_LENGTH_M) > 0.01:
        raise ValueError("main road must be 8000 m")
    lane_types = Counter(
        lane.get("type")
        for lane in main.findall("./lanes/laneSection/*/lane")
    )
    if lane_types["driving"] < 4:
        raise ValueError(
            "main road requires two driving lanes per direction"
        )
    if lane_types["biking"] < 2:
        raise ValueError("protected bicycle lanes are missing")
    if lane_types["sidewalk"] < 2:
        raise ValueError("sidewalks are missing")
    objects = main.findall("./objects/object")
    object_types = Counter(item.get("type") for item in objects)
    if object_types["busStop"] < 1:
        raise ValueError("bus stop declaration is missing")
    if not junctions:
        raise ValueError("a real OpenDRIVE junction is required")
    return {
        "road_count": len(roads),
        "junction_count": len(junctions),
        "main_road_length_m": float(main.get("length", "0")),
        "lane_types": dict(lane_types),
        "object_types": dict(object_types),
        "topology_note": (
            "The canonical Scene 2 map supplies one real X junction. "
            "Later signal and U-turn tasks remain route-bound events; "
            "physical junction geometry is not yet available "
            "at S2-05 and S2-06."
        ),
    }


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("w", encoding="utf-8")

    def write(self, payload: Mapping[str, Any]) -> None:
        self._handle.write(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def speed_kmh(actor: Any) -> float:
    velocity = actor.get_velocity()
    return 3.6 * math.sqrt(
        velocity.x * velocity.x
        + velocity.y * velocity.y
        + velocity.z * velocity.z
    )


def set_role_name(blueprint: Any, role_name: str) -> None:
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)


def available_blueprints(
    library: Any,
    identifiers: Sequence[str],
) -> list[Any]:
    result = []
    for identifier in identifiers:
        try:
            blueprint = library.find(identifier)
        except (IndexError, RuntimeError):
            continue
        if blueprint is not None:
            result.append(blueprint)
    return result


def first_filtered_blueprint(
    library: Any,
    pattern: str,
) -> Any | None:
    matches = list(library.filter(pattern))
    return matches[0] if matches else None


def waypoint_transform(
    carla_map: Any,
    route_s: float,
    lane_id: int,
    z_offset: float = 0.35,
) -> Any:
    waypoint = carla_map.get_waypoint_xodr(
        ROAD_ID,
        int(lane_id),
        float(route_s),
    )
    if waypoint is None:
        raise RuntimeError(
            "no waypoint: road={0} lane={1} s={2}".format(
                ROAD_ID,
                lane_id,
                route_s,
            )
        )
    transform = waypoint.transform
    transform.location.z += float(z_offset)
    return transform


def try_spawn(
    world: Any,
    blueprint: Any,
    transform: Any,
    description: str,
) -> Any | None:
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is None:
        print("WARNING: failed to spawn {0}".format(description))
    return actor


@dataclass
class ActorRegistry:
    actors: list[Any] = field(default_factory=list)
    roles: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        actor: Any | None,
        role: str | None = None,
    ) -> Any | None:
        if actor is not None:
            self.actors.append(actor)
            if role:
                self.roles[role] = actor
        return actor

    def destroy(self, client: Any | None = None) -> None:
        for actor in reversed(self.actors):
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
            except RuntimeError:
                pass
        if client is not None and self.actors:
            carla = import_carla()
            commands = [
                carla.command.DestroyActor(actor.id)
                for actor in self.actors
                if actor is not None
            ]
            client.apply_batch_sync(commands, False)
        self.actors.clear()
        self.roles.clear()


class SafetyMonitor:
    def __init__(
        self,
        world: Any,
        ego: Any,
        registry: ActorRegistry,
    ) -> None:
        self.world = world
        self.ego = ego
        self.registry = registry
        self.collisions: list[dict[str, Any]] = []
        self.lane_invasions: list[dict[str, Any]] = []
        self._sim_time_s = 0.0

    def start(self) -> None:
        library = self.world.get_blueprint_library()
        carla = import_carla()
        collision = self.world.spawn_actor(
            library.find("sensor.other.collision"),
            carla.Transform(),
            attach_to=self.ego,
        )
        lane = self.world.spawn_actor(
            library.find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=self.ego,
        )
        collision.listen(self._on_collision)
        lane.listen(self._on_lane_invasion)
        self.registry.add(collision, "collision_sensor")
        self.registry.add(lane, "lane_invasion_sensor")

    def set_sim_time(self, sim_time_s: float) -> None:
        self._sim_time_s = float(sim_time_s)

    def _on_collision(self, event: Any) -> None:
        other = getattr(event, "other_actor", None)
        self.collisions.append(
            {
                "frame": int(event.frame),
                "timestamp_s": self._sim_time_s,
                "other_actor_id": (
                    int(other.id) if other is not None else None
                ),
                "other_actor_type": (
                    str(other.type_id)
                    if other is not None
                    else "unknown"
                ),
            }
        )

    def _on_lane_invasion(self, event: Any) -> None:
        markings = [
            str(marking.type)
            for marking in getattr(
                event,
                "crossed_lane_markings",
                [],
            )
        ]
        self.lane_invasions.append(
            {
                "frame": int(event.frame),
                "timestamp_s": self._sim_time_s,
                "crossed_lane_markings": markings,
            }
        )

    @property
    def violation_count(self) -> int:
        """Count solid/curb crossings, not legal broken-line changes."""

        return sum(
            1
            for event in self.lane_invasions
            if any(
                token in marking.lower()
                for marking in event["crossed_lane_markings"]
                for token in ("solid", "curb", "grass")
            )
        )


class LightweightSensorSuite:
    """Optional four-camera/LiDAR evidence with exact frame metadata."""

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
        registry: ActorRegistry,
        output_dir: Path,
        sensor_tick: float,
    ) -> None:
        self.world = world
        self.ego = ego
        self.registry = registry
        self.output_dir = output_dir
        self.sensor_tick = float(sensor_tick)
        self.frame_counts: Counter[str] = Counter()
        self.latest_frames: dict[str, int] = {}

    def start(self) -> None:
        carla = import_carla()
        library = self.world.get_blueprint_library()
        for name, values in self.CAMERA_TRANSFORMS.items():
            blueprint = library.find("sensor.camera.rgb")
            for attribute, value in (
                ("image_size_x", "960"),
                ("image_size_y", "540"),
                ("fov", "90"),
                ("sensor_tick", str(self.sensor_tick)),
                ("gamma", "2.2"),
            ):
                if blueprint.has_attribute(attribute):
                    blueprint.set_attribute(attribute, value)
            x, y, z, yaw = values
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(yaw=yaw),
            )
            actor = self.world.spawn_actor(
                blueprint,
                transform,
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
            self.registry.add(actor, name)

        lidar_blueprint = library.find("sensor.lidar.ray_cast")
        for attribute, value in (
            ("sensor_tick", str(self.sensor_tick)),
            ("channels", "32"),
            ("range", "80"),
            ("points_per_second", "56000"),
            ("rotation_frequency", str(1.0 / self.sensor_tick)),
        ):
            if lidar_blueprint.has_attribute(attribute):
                lidar_blueprint.set_attribute(attribute, value)
        lidar_dir = self.output_dir / "lidar"
        lidar_dir.mkdir(parents=True, exist_ok=True)
        lidar = self.world.spawn_actor(
            lidar_blueprint,
            carla.Transform(carla.Location(z=2.6)),
            attach_to=self.ego,
            attachment_type=carla.AttachmentType.Rigid,
        )
        lidar.listen(
            lambda measurement: self._save_lidar(
                lidar_dir,
                measurement,
            )
        )
        self.registry.add(lidar, "lidar")

    def _save_image(
        self,
        sensor_name: str,
        directory: Path,
        image: Any,
    ) -> None:
        frame = int(image.frame)
        final_path = directory / "{0:08d}.png".format(frame)
        temporary_path = directory / ".{0:08d}.tmp.png".format(
            frame
        )
        image.save_to_disk(str(temporary_path))
        temporary_path.replace(final_path)
        self.frame_counts[sensor_name] += 1
        self.latest_frames[sensor_name] = frame

    def _save_lidar(
        self,
        directory: Path,
        measurement: Any,
    ) -> None:
        frame = int(measurement.frame)
        final_path = directory / "{0:08d}.ply".format(frame)
        temporary_path = directory / ".{0:08d}.tmp.ply".format(
            frame
        )
        measurement.save_to_disk(str(temporary_path))
        temporary_path.replace(final_path)
        self.frame_counts["lidar"] += 1
        self.latest_frames["lidar"] = frame


class Scene2RuntimeInterface:
    """Stable JSON boundary for ASR, VLA, safety and control modules."""

    def __init__(
        self,
        output_dir: Path,
        config: Mapping[str, Any],
    ) -> None:
        self.output_dir = output_dir
        self.config = config
        self.intent_log = JsonlWriter(
            output_dir / "driving_intent.jsonl"
        )
        self.world_state_log = JsonlWriter(
            output_dir / "world_state.jsonl"
        )
        self.bundle_log = JsonlWriter(
            output_dir / "multimodal_frame_bundle.jsonl"
        )
        manifest = {
            "scene_id": SCENE_ID,
            **dict(config["interfaces"]),
            "producer": "decorate_complex_scene.py",
            "policy_boundary": (
                "VLA proposals are untrusted. RiskAssessment and the "
                "deterministic safety gate must produce ControlDecision."
            ),
        }
        (output_dir / "interface_manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
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
        safety: SafetyMonitor,
    ) -> dict[str, Any]:
        carla_map = world.get_map()
        transform = ego.get_transform()
        waypoint = carla_map.get_waypoint(
            transform.location,
            project_to_road=True,
        )
        state = {
            "schema_version": "1.0.0",
            "scene_id": SCENE_ID,
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
                "speed_kmh": round(speed_kmh(ego), 3),
            },
            "lane": {
                "road_id": (
                    int(waypoint.road_id)
                    if waypoint is not None
                    else None
                ),
                "section_id": (
                    int(waypoint.section_id)
                    if waypoint is not None
                    else None
                ),
                "lane_id": (
                    int(waypoint.lane_id)
                    if waypoint is not None
                    else None
                ),
                "is_junction": (
                    bool(waypoint.is_junction)
                    if waypoint is not None
                    else False
                ),
            },
            "weather": {
                "cloudiness": world.get_weather().cloudiness,
                "fog_density": world.get_weather().fog_density,
                "sun_altitude_angle": (
                    world.get_weather().sun_altitude_angle
                ),
            },
            "safety": {
                "collision_count": len(safety.collisions),
                "lane_invasion_count": len(
                    safety.lane_invasions
                ),
                "violation_count": safety.violation_count,
            },
        }
        self.world_state_log.write(state)
        return state

    def publish_bundle(
        self,
        frame: int,
        world_state: Mapping[str, Any],
        latest_sensor_frames: Mapping[str, int],
        intent: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        bundle = build_multimodal_frame_bundle(
            SCENE_ID,
            frame,
            world_state["simulation_frame"],
            latest_sensor_frames,
            intent.get("request_id") if intent else None,
        )
        self.bundle_log.write(bundle)
        return bundle

    def close(self) -> None:
        self.intent_log.close()
        self.world_state_log.close()
        self.bundle_log.close()


class ProgressScheduler:
    def __init__(
        self,
        config: Mapping[str, Any],
        event_log: JsonlWriter,
        command_log: JsonlWriter,
        runtime_interface: Scene2RuntimeInterface,
    ) -> None:
        self.commands = list(config["voice_commands"])
        self.events = list(config["events"])
        self.event_log = event_log
        self.command_log = command_log
        self.runtime_interface = runtime_interface
        self.command_index = 0
        self.event_states = {
            item["id"]: "PENDING"
            for item in self.events
        }
        self.last_intent: dict[str, Any] | None = None

    def update(
        self,
        route_s_m: float,
        frame: int,
        elapsed_s: float,
    ) -> None:
        while self.command_index < len(self.commands):
            command = self.commands[self.command_index]
            if route_s_m < float(command["announce_at_m"]):
                break
            self.last_intent = (
                self.runtime_interface.publish_command(
                    command,
                    frame,
                    route_s_m,
                    elapsed_s,
                )
            )
            record = {
                "scene_id": SCENE_ID,
                "command_id": command["id"],
                "state": "ANNOUNCED",
                "route_s_m": round(route_s_m, 3),
                "simulation_frame": int(frame),
                "elapsed_s": round(elapsed_s, 3),
                "step_count": len(command["steps"]),
                "spoken_text": command["spoken_text"],
            }
            self.command_log.write(record)
            print(
                "COMMAND ANNOUNCED | {0} | route={1:.1f} m".format(
                    command["id"],
                    route_s_m,
                )
            )
            self.command_index += 1

        for event in self.events:
            event_id = event["id"]
            state = self.event_states[event_id]
            if (
                state == "PENDING"
                and route_s_m >= float(event["activate_at_m"])
            ):
                self.event_states[event_id] = "ACTIVE"
                self._record_event(
                    event,
                    "ACTIVE",
                    route_s_m,
                    frame,
                    elapsed_s,
                )
            if (
                self.event_states[event_id] == "ACTIVE"
                and route_s_m >= float(event["resolve_at_m"])
            ):
                self.event_states[event_id] = "RESOLVED"
                self._record_event(
                    event,
                    "RESOLVED",
                    route_s_m,
                    frame,
                    elapsed_s,
                )

    def _record_event(
        self,
        event: Mapping[str, Any],
        state: str,
        route_s_m: float,
        frame: int,
        elapsed_s: float,
    ) -> None:
        record = {
            "scene_id": SCENE_ID,
            "event_id": event["id"],
            "state": state,
            "route_s_m": round(route_s_m, 3),
            "simulation_frame": int(frame),
            "elapsed_s": round(elapsed_s, 3),
            "actors": list(event["actors"]),
            "completion_contract": event["completion"],
        }
        self.event_log.write(record)
        print(
            "EVENT {0:<8} | {1} | route={2:.1f} m".format(
                state,
                event["id"],
                route_s_m,
            )
        )

    @property
    def all_events_resolved(self) -> bool:
        return all(
            state == "RESOLVED"
            for state in self.event_states.values()
        )


def configure_vehicle_speed(
    traffic_manager: Any,
    actor: Any,
    speed_kmh_value: float,
    speed_limit_kmh: float = 60.0,
) -> None:
    difference = (
        100.0
        * (float(speed_limit_kmh) - float(speed_kmh_value))
        / float(speed_limit_kmh)
    )
    traffic_manager.vehicle_percentage_speed_difference(
        actor,
        difference,
    )


def spawn_ego(
    world: Any,
    carla_map: Any,
    library: Any,
    traffic_manager: Any,
    traffic_manager_port: int,
    registry: ActorRegistry,
    config: Mapping[str, Any],
    stationary: bool,
    desired_speed_kmh: float,
) -> Any:
    blueprints = available_blueprints(
        library,
        EGO_BLUEPRINTS,
    )
    if not blueprints:
        raise RuntimeError("no ego vehicle blueprint is available")
    blueprint = blueprints[0]
    set_role_name(blueprint, "hero")
    route = config["route"]
    ego = try_spawn(
        world,
        blueprint,
        waypoint_transform(
            carla_map,
            route["start_s_m"],
            route["start_lane_id"],
            z_offset=0.45,
        ),
        "ego vehicle",
    )
    if ego is None:
        raise RuntimeError("failed to spawn ego vehicle")
    registry.add(ego, "hero")
    if not stationary:
        ego.set_autopilot(True, traffic_manager_port)
        configure_vehicle_speed(
            traffic_manager,
            ego,
            desired_speed_kmh,
        )
    print(
        "Ego spawned: road={0}, lane={1}, s={2:.0f} m".format(
            route["start_road_id"],
            route["start_lane_id"],
            route["start_s_m"],
        )
    )
    print(
        "Ego mode: {0}".format(
            "stationary"
            if stationary
            else "autopilot {0:.1f} km/h".format(
                desired_speed_kmh
            )
        )
    )
    return ego


def spawn_vehicle_group(
    world: Any,
    carla_map: Any,
    library: Any,
    traffic_manager: Any,
    traffic_manager_port: int,
    registry: ActorRegistry,
    random_generator: random.Random,
    blueprint_ids: Sequence[str],
    count: int,
    role_prefix: str,
    positions: Sequence[tuple[float, int]],
    speed_range: tuple[float, float],
    autopilot: bool = True,
) -> list[Any]:
    blueprints = available_blueprints(library, blueprint_ids)
    if not blueprints:
        raise RuntimeError(
            "no blueprints for {0}".format(role_prefix)
        )
    result = []
    for index, (route_s, lane_id) in enumerate(positions):
        if len(result) >= count:
            break
        blueprint = random_generator.choice(blueprints)
        role = "{0}_{1:02d}".format(role_prefix, index + 1)
        set_role_name(blueprint, role)
        actor = try_spawn(
            world,
            blueprint,
            waypoint_transform(
                carla_map,
                route_s,
                lane_id,
                z_offset=0.35,
            ),
            role,
        )
        if actor is None:
            continue
        registry.add(actor, role)
        result.append(actor)
        if autopilot:
            actor.set_autopilot(True, traffic_manager_port)
            configure_vehicle_speed(
                traffic_manager,
                actor,
                random_generator.uniform(*speed_range),
            )
    return result


def spawn_walkers(
    world: Any,
    carla_map: Any,
    library: Any,
    registry: ActorRegistry,
    count: int,
    positions: Sequence[tuple[float, int]],
) -> list[Any]:
    carla = import_carla()
    blueprints = list(library.filter("walker.pedestrian.*"))
    if not blueprints:
        raise RuntimeError("no walker blueprints are available")
    result = []
    for index, (route_s, lane_id) in enumerate(positions):
        if len(result) >= count:
            break
        blueprint = blueprints[index % len(blueprints)]
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        role = "scene2_pedestrian_{0:02d}".format(index + 1)
        set_role_name(blueprint, role)
        try:
            transform = waypoint_transform(
                carla_map,
                route_s,
                lane_id,
                z_offset=1.0,
            )
        except RuntimeError:
            continue
        actor = try_spawn(world, blueprint, transform, role)
        if actor is None:
            continue
        registry.add(actor, role)
        result.append(actor)
        forward = transform.get_forward_vector()
        actor.set_target_velocity(
            carla.Vector3D(
                x=forward.x * 1.1,
                y=forward.y * 1.1,
                z=0.0,
            )
        )
    return result


def spawn_competition_traffic(
    world: Any,
    carla_map: Any,
    library: Any,
    traffic_manager: Any,
    traffic_manager_port: int,
    registry: ActorRegistry,
    config: Mapping[str, Any],
    random_generator: random.Random,
) -> dict[str, int]:
    traffic = config["traffic"]
    reserved = [
        (880.0, -1),
        (1770.0, -2),
        (3000.0, 1),
        (3950.0, -2),
        (5150.0, -1),
        (7400.0, 1),
    ]
    car_positions = []
    for index in range(40):
        route_s = 350.0 + index * 180.0
        lane_id = -1 if index % 2 == 0 else -2
        if all(abs(route_s - item[0]) > 100.0 for item in reserved):
            car_positions.append((route_s, lane_id))
    cars = spawn_vehicle_group(
        world,
        carla_map,
        library,
        traffic_manager,
        traffic_manager_port,
        registry,
        random_generator,
        CAR_BLUEPRINTS,
        int(traffic["private_cars"]),
        "scene2_private_car",
        car_positions,
        (35.0, 52.0),
    )
    bus_positions = [
        (1770.0, -4),
        (5650.0, -4),
        (6300.0, 4),
    ]
    buses = spawn_vehicle_group(
        world,
        carla_map,
        library,
        traffic_manager,
        traffic_manager_port,
        registry,
        random_generator,
        BUS_BLUEPRINTS,
        int(traffic["city_buses"]),
        "scene2_city_bus",
        bus_positions,
        (22.0, 32.0),
        autopilot=False,
    )
    bike_positions = [
        (1300.0, -3),
        (2050.0, -3),
        (3950.0, -3),
        (4400.0, -3),
        (5900.0, -3),
        (7050.0, -3),
    ]
    bicycles = spawn_vehicle_group(
        world,
        carla_map,
        library,
        traffic_manager,
        traffic_manager_port,
        registry,
        random_generator,
        BICYCLE_BLUEPRINTS,
        int(traffic["bicycles"]),
        "scene2_bicycle",
        bike_positions,
        (12.0, 20.0),
    )
    walker_positions = []
    for index in range(30):
        route_s = 300.0 + index * 250.0
        lane_id = -5 if index % 2 == 0 else 5
        walker_positions.append((route_s, lane_id))
    walkers = spawn_walkers(
        world,
        carla_map,
        library,
        registry,
        int(traffic["sidewalk_pedestrians"]),
        walker_positions,
    )
    counts = {
        "private_cars": len(cars),
        "city_buses": len(buses),
        "bicycles": len(bicycles),
        "sidewalk_pedestrians": len(walkers),
    }
    for name, target in (
        ("private_cars", int(traffic["private_cars"])),
        ("city_buses", int(traffic["city_buses"])),
        ("bicycles", int(traffic["bicycles"])),
        (
            "sidewalk_pedestrians",
            int(traffic["sidewalk_pedestrians"]),
        ),
    ):
        print(
            "{0}: spawned={1} configured={2}".format(
                name,
                counts[name],
                target,
            )
        )
    return counts


def apply_weather(
    world: Any,
    weather_config: Mapping[str, Any],
) -> None:
    carla = import_carla()
    weather = carla.WeatherParameters(
        cloudiness=float(weather_config["cloudiness"]),
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=10.0,
        sun_azimuth_angle=float(
            weather_config["sun_azimuth_angle"]
        ),
        sun_altitude_angle=float(
            weather_config["sun_altitude_angle"]
        ),
        fog_density=float(weather_config["fog_density"]),
        fog_distance=float(weather_config["fog_distance"]),
        wetness=float(weather_config["wetness"]),
    )
    world.set_weather(weather)
    print("Weather: {0}".format(weather_config["preset"]))


def route_progress_m(
    carla_map: Any,
    ego: Any,
    previous_s: float,
) -> float:
    waypoint = carla_map.get_waypoint(
        ego.get_location(),
        project_to_road=True,
    )
    if waypoint is None:
        return previous_s
    if int(waypoint.road_id) == ROAD_ID:
        return max(previous_s, float(waypoint.s))
    return previous_s


def start_presentation_camera(
    world: Any,
    ego: Any,
    output_dir: Path,
    args: argparse.Namespace,
) -> Any:
    try:
        from evaluation.camera import ExperimentCamera
    except ImportError as error:
        raise RuntimeError(
            "evaluation.camera.ExperimentCamera is required for "
            "the shared Scene 1/2/3 chase-camera and H.264 interface"
        ) from error

    video_output = args.video_output
    if video_output is None:
        video_output = output_dir / (
            "scene2_complex_avoidance_8km.mp4"
        )
    camera = ExperimentCamera(
        world,
        ego,
        str(output_dir / "rgb" / "chase_rgb"),
        every_n_frames=1,
        width=args.camera_width,
        height=args.camera_height,
        save_images=args.record_images,
        video_output=str(video_output),
        video_fps=args.video_fps,
        ffmpeg_path=args.ffmpeg,
        video_overlay=args.video_overlay,
        camera_pose=(-10.0, 0.0, 3.0, -8.0, 0.0),
    )
    camera.start()
    print("RGB camera: chase_rgb (third-person shared view)")
    print("H.264 video output: {0}".format(video_output))
    return camera


def camera_overlay(
    scheduler: ProgressScheduler,
    route_s_m: float,
    ego: Any,
    elapsed_s: float,
    safety: SafetyMonitor,
    traffic_counts: Mapping[str, int],
) -> dict[str, Any]:
    intent = scheduler.last_intent or {}
    intent_body = intent.get("intent", {})
    steps = intent_body.get("steps", [])
    first_step = steps[0] if steps else {}
    return {
        "status": "RUNNING",
        "route_progress_m": route_s_m,
        "route_length_m": ROUTE_LENGTH_M,
        "speed_kmh": speed_kmh(ego),
        "target_speed_kmh": 45.0,
        "asr_text": intent.get("voice_text", ""),
        "source_step_action": first_step.get(
            "action",
            "KEEP_LANE",
        ),
        "active_step_id": first_step.get("step_id", "PENDING"),
        "parse_status": intent.get(
            "parse_result",
            {},
        ).get("status", "PENDING"),
        "risk_level": (
            "HIGH" if safety.collisions else "LOW"
        ),
        "traffic_count": (
            int(traffic_counts.get("private_cars", 0))
            + int(traffic_counts.get("city_buses", 0))
            + int(traffic_counts.get("bicycles", 0))
        ),
        "pedestrian_count": int(
            traffic_counts.get("sidewalk_pedestrians", 0)
        ),
        "policy_state": "EXTERNAL VLA READY",
        "sim_time_s": elapsed_s,
        "collisions": len(safety.collisions),
        "lane_events": len(safety.lane_invasions),
    }


def write_summary(
    output_dir: Path,
    route_completed: bool,
    scheduler: ProgressScheduler,
    safety: SafetyMonitor,
    traffic_counts: Mapping[str, int],
    multimodal_counts: Mapping[str, int],
    map_contract: Mapping[str, Any],
    direct_video: Mapping[str, Any] | None,
) -> dict[str, Any]:
    event_counts = Counter(scheduler.event_states.values())
    traffic_targets_met = all(
        int(traffic_counts.get(name, 0))
        == expected
        for name, expected in (
            ("private_cars", 24),
            ("city_buses", 3),
            ("bicycles", 6),
            ("sidewalk_pedestrians", 18),
        )
    )
    complete_scene_success = (
        route_completed
        and scheduler.all_events_resolved
        and scheduler.command_index == len(scheduler.commands)
        and not safety.collisions
        and safety.violation_count <= 1
        and traffic_targets_met
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": SCENE_ID,
        "route_completed": route_completed,
        "event_states": {
            "PENDING": event_counts["PENDING"],
            "ACTIVE": event_counts["ACTIVE"],
            "RESOLVED": event_counts["RESOLVED"],
        },
        "commands_announced": scheduler.command_index,
        "commands_required": len(scheduler.commands),
        "command_execution_source": (
            "external DrivingIntent/VLA/ControlDecision feedback "
            "required for competition metric claims"
        ),
        "traffic_counts": dict(traffic_counts),
        "traffic_targets_met": traffic_targets_met,
        "safety": {
            "collision_count": len(safety.collisions),
            "collisions": safety.collisions,
            "lane_invasion_event_count": len(
                safety.lane_invasions
            ),
            "violation_count": safety.violation_count,
            "lane_invasions": safety.lane_invasions,
        },
        "multimodal_frame_counts": dict(multimodal_counts),
        "map_contract": dict(map_contract),
        "direct_video": direct_video,
        "complete_scene_success": complete_scene_success,
        "competition_metrics_pending": [
            "asr_accuracy",
            "ordered_step_recall",
            "semantic_alignment_accuracy",
            "perception_to_action_latency_ms",
        ],
    }
    (output_dir / "scene_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    xodr_path = args.xodr.expanduser().resolve()
    config = load_json(config_path)
    validate_runtime_config(config)
    map_contract = validate_xodr(xodr_path)
    print("Runtime config: {0}".format(config_path))
    print("Configured commands: {0}".format(
        len(config["voice_commands"])
    ))
    print("Configured events: {0}".format(len(config["events"])))
    print(
        "OpenDRIVE: roads={0}, junctions={1}, route={2:.0f} m".format(
            map_contract["road_count"],
            map_contract["junction_count"],
            map_contract["main_road_length_m"],
        )
    )
    if args.validate_config_only:
        print("SCENE 2 RUNTIME CONFIG VALIDATION: PASS")
        return 0

    carla = import_carla()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime_config_snapshot.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    event_log = JsonlWriter(output_dir / "event_timeline.jsonl")
    command_log = JsonlWriter(
        output_dir / "command_timeline.jsonl"
    )
    runtime_interface = Scene2RuntimeInterface(
        output_dir,
        config,
    )
    registry = ActorRegistry()
    camera = None
    sensor_suite = None
    safety = None
    scheduler = ProgressScheduler(
        config,
        event_log,
        command_log,
        runtime_interface,
    )
    world = None
    client = None
    traffic_manager = None
    original_settings = None
    route_completed = False
    traffic_counts: dict[str, int] = {}
    random_generator = random.Random(
        args.seed
        if args.seed is not None
        else int(config["traffic"]["seed"])
    )

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        print(
            "CARLA server: {0}".format(
                client.get_server_version()
            )
        )
        world = client.generate_opendrive_world(
            xodr_path.read_text(encoding="utf-8")
        )
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(
            args.traffic_manager_port
        )
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(
            args.seed
            if args.seed is not None
            else int(config["traffic"]["seed"])
        )
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)

        apply_weather(world, config["weather"])
        carla_map = world.get_map()
        library = world.get_blueprint_library()
        ego = spawn_ego(
            world,
            carla_map,
            library,
            traffic_manager,
            args.traffic_manager_port,
            registry,
            config,
            args.stationary_ego,
            args.ego_speed_kmh,
        )
        safety = SafetyMonitor(world, ego, registry)
        safety.start()
        traffic_counts = spawn_competition_traffic(
            world,
            carla_map,
            library,
            traffic_manager,
            args.traffic_manager_port,
            registry,
            config,
            random_generator,
        )
        camera = start_presentation_camera(
            world,
            ego,
            output_dir,
            args,
        )
        if args.record_multimodal:
            sensor_suite = LightweightSensorSuite(
                world,
                ego,
                registry,
                output_dir,
                args.sensor_tick,
            )
            sensor_suite.start()
            print(
                "Multimodal sensors: front/left/right/rear RGB + LiDAR"
            )

        start_time = time.monotonic()
        route_s_m = float(config["route"]["start_s_m"])
        last_progress_m = route_s_m
        last_progress_time = time.monotonic()
        latest_world_state = None
        last_interface_frame = -1
        while True:
            frame = int(world.tick())
            elapsed_s = (
                time.monotonic() - start_time
            )
            safety.set_sim_time(elapsed_s)
            route_s_m = route_progress_m(
                carla_map,
                ego,
                route_s_m,
            )
            if route_s_m >= last_progress_m + 1.0:
                last_progress_m = route_s_m
                last_progress_time = time.monotonic()
            elif time.monotonic() - last_progress_time >= 120.0:
                raise RuntimeError(
                    "ego stalled for 120 s at route={0:.1f} m".format(
                        route_s_m
                    )
                )
            scheduler.update(
                route_s_m,
                frame,
                elapsed_s,
            )
            overlay = camera_overlay(
                scheduler,
                route_s_m,
                ego,
                elapsed_s,
                safety,
                traffic_counts,
            )
            camera.save_frame(
                frame,
                overlay=overlay,
                timeout_s=2.0,
            )

            interface_stride = max(
                1,
                round(args.sensor_tick / args.fixed_delta_seconds),
            )
            if frame - last_interface_frame >= interface_stride:
                latest_world_state = (
                    runtime_interface.publish_world_state(
                        world,
                        ego,
                        frame,
                        route_s_m,
                        elapsed_s,
                        safety,
                    )
                )
                if sensor_suite is not None:
                    runtime_interface.publish_bundle(
                        frame,
                        latest_world_state,
                        sensor_suite.latest_frames,
                        scheduler.last_intent,
                    )
                last_interface_frame = frame

            if route_s_m >= float(
                config["route"]["finish_s_m"]
            ):
                route_completed = True
                print(
                    "ROUTE FINISH REACHED | s={0:.1f} m".format(
                        route_s_m
                    )
                )
                break
            if args.duration > 0.0 and elapsed_s >= args.duration:
                print("Scene duration completed")
                break

        direct_video = None
        if camera is not None:
            writer = camera.video_writer
            video_fps = camera.video_fps
            hud_overlay = camera.video_overlay
            camera.destroy()
            camera = None
            if writer is not None:
                direct_video = {
                    "codec": "H.264",
                    "fps": video_fps,
                    "frames": writer.frame_count,
                    "dropped_frames": writer.dropped_frames,
                    "hud_overlay": hud_overlay,
                    "camera": "chase_rgb",
                }
        summary = write_summary(
            output_dir,
            route_completed,
            scheduler,
            safety,
            traffic_counts,
            (
                sensor_suite.frame_counts
                if sensor_suite is not None
                else {}
            ),
            map_contract,
            direct_video,
        )
        print(
            "Event scheduler summary: pending={0}, active={1}, "
            "resolved={2}".format(
                summary["event_states"]["PENDING"],
                summary["event_states"]["ACTIVE"],
                summary["event_states"]["RESOLVED"],
            )
        )
        print(
            "Safety audit: collisions={0}, violations={1}".format(
                summary["safety"]["collision_count"],
                summary["safety"]["violation_count"],
            )
        )
        if args.require_complete_scene and not summary[
            "complete_scene_success"
        ]:
            raise RuntimeError(
                "complete scene requirement failed"
            )
        print("SCENE 2 COMPLEX AVOIDANCE CAPTURE: PASS")
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 130
    except Exception as error:
        print("ERROR: {0}".format(error), file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            try:
                camera.destroy()
            except RuntimeError as error:
                print(
                    "WARNING: camera cleanup failed: {0}".format(
                        error
                    )
                )
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except RuntimeError:
                pass
        event_log.close()
        command_log.close()
        registry.destroy(client)
        runtime_interface.close()
        print("Scene actors cleaned up")


if __name__ == "__main__":
    raise SystemExit(main())
