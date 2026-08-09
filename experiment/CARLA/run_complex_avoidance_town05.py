"""Run the Scene 2 environment on packaged CARLA Town05 assets."""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_bootstrap import setup_carla_api
from evaluation.camera import ExperimentCamera
from evaluation.ground_truth import (
    FrameGroundTruthRecorder,
    validate_event_ground_truth_contracts,
)
from evaluation.multimodal import (
    ExactFrameSensorSuite,
    REQUIRED_SENSOR_NAMES,
    Scene2RuntimeInterface,
)
from scenarios.complex.town05_scene2 import (
    ActorRegistry,
    DeterministicSceneEvents,
    RouteProgressTracker,
    TownTrafficFlow,
    build_repeated_route,
    choose_curved_route_destination,
    route_curvature_degrees,
    speed_kmh,
)
from scene2_runtime_interface import build_scheduled_driving_intent


DEFAULT_CONFIG = (
    ROOT / "configs" / "scene_2_town05_runtime.json"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "scene2_town05"
EGO_BLUEPRINTS = (
    "vehicle.audi.a2",
    "vehicle.nissan.micra",
    "vehicle.tesla.model3",
    "vehicle.lincoln.mkz_2020",
)

COMMAND_HUD_FALLBACKS = {
    "s2_t05_cmd_01": "Keep lane, slow to 45 km/h, then continue straight",
    "s2_t05_cmd_02": "Check the road, keep right and resume 50 km/h",
    "s2_t05_cmd_03": "Yield to the pedestrian, then pass the slow car left",
    "s2_t05_cmd_04": (
        "After the pedestrian clears, return right and resume 45"
    ),
    "s2_t05_cmd_05": "Bus stop ahead: slow to 30 km/h and yield to passengers",
    "s2_t05_cmd_06": "Wait for passengers, then continue in the right lane",
    "s2_t05_cmd_07": "Pass the cyclist on the left, then return right",
    "s2_t05_cmd_08": (
        "After passing the cyclist, return right and keep a safe gap"
    ),
    "s2_t05_cmd_09": "Go straight, move left safely and hold 40 km/h",
    "s2_t05_cmd_10": "After the junction and lane change, resume 45 km/h",
    "s2_t05_cmd_11": "Slow to 30 km/h before turning right",
    "s2_t05_cmd_12": "Change left safely, keep a gap, then return",
    "s2_t05_cmd_13": "Keep a gap, turn right, straight, right, then resume 45",
    "s2_t05_cmd_14": "Slow to 30, check the crosswalk, then go straight",
    "s2_t05_cmd_15": "Go straight, turn right twice, then continue to finish",
}

RESTRICTED_LANE_MARKINGS = {
    "Solid",
    "SolidSolid",
    "SolidBroken",
    "BrokenSolid",
    "Curb",
    "Grass",
}


def lane_marking_name(value: Any) -> str:
    """Return the stable enum suffix used by CARLA lane markings."""

    return str(value).rsplit(".", 1)[-1]


def lane_invasion_is_restricted(event: Mapping[str, Any]) -> bool:
    return any(
        lane_marking_name(marking) in RESTRICTED_LANE_MARKINGS
        for marking in event.get("markings", [])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the competition Scene 2 route on Town05 with stable "
            "Traffic Manager flow and deterministic special actors."
        )
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument(
        "--traffic-hybrid-physics",
        action="store_true",
        help=(
            "Use full vehicle physics near the ego and kinematic Traffic "
            "Manager updates for distant background vehicles."
        ),
    )
    parser.add_argument(
        "--traffic-hybrid-radius-m",
        type=float,
        default=100.0,
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--reuse-current-world",
        action="store_true",
        help=(
            "Reuse an actor-clean matching CARLA world. Disabled by default "
            "because some server builds stall while applying new settings."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--duration",
        type=float,
        default=90.0,
        help="Simulation seconds; 0 runs until 8 km route completion.",
    )
    parser.add_argument(
        "--fixed-delta-seconds",
        type=float,
        default=0.05,
    )
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--ffmpeg")
    parser.add_argument(
        "--hud-font",
        type=Path,
        help=(
            "Optional CJK-capable TTF/TTC/OTF font. The CARLA_HUD_FONT "
            "environment variable is also supported."
        ),
    )
    parser.add_argument("--record-images", action="store_true")
    parser.add_argument("--video-overlay", action="store_true")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "Disable the demonstration camera and video encoder for fast "
            "scene/route regression runs. In a VLA competition run the "
            "online multimodal cameras remain rendered and synchronized; "
            "this flag only disables the separate presentation camera and "
            "video encoder."
        ),
    )
    parser.add_argument(
        "--competition-logs-only",
        action="store_true",
        help=(
            "Keep the formal route, traffic, online VLA sensors, control, "
            "event and safety logs, but skip the duplicate on-disk RGB/LiDAR "
            "evidence rig, per-frame actor truth and demonstration video. "
            "This mode requires --competition-run and --vla-checkpoint."
        ),
    )
    parser.add_argument("--record-ground-truth", action="store_true")
    parser.add_argument("--ground-truth-every-n", type=int, default=1)
    parser.add_argument(
        "--record-multimodal",
        action="store_true",
        help="Record exact-frame four-view RGB and LiDAR evidence.",
    )
    parser.add_argument(
        "--sensor-tick",
        type=float,
        help="Multimodal sampling period; defaults to config sensors value.",
    )
    parser.add_argument(
        "--sensor-sync-timeout",
        type=float,
        default=5.0,
        help="Wall-clock seconds to wait for all exact-frame artifacts.",
    )
    parser.add_argument(
        "--competition-run",
        action="store_true",
        help=(
            "Require the formal 8 km external-control run and enable all "
            "competition evidence recorders."
        ),
    )
    parser.add_argument(
        "--variant-index",
        type=int,
        default=0,
        help=(
            "Deterministically select event micro-scenario variants; "
            "use 0, 1, 2 for three reproducible evaluation episodes."
        ),
    )
    parser.add_argument(
        "--start-progress-m",
        type=float,
        default=0.0,
        help="Diagnostic start along the route; normal runs use 0.",
    )
    parser.add_argument(
        "--external-ego-control",
        action="store_true",
        help="Spawn and instrument ego but do not apply preview-agent control.",
    )
    parser.add_argument(
        "--vla-checkpoint",
        type=Path,
        help=(
            "Universal VLA checkpoint. When set, the model consumes raw "
            "four-view RGB, LiDAR, vehicle/environment state, and the active "
            "compound-command substep, then controls ego through the PID "
            "executor."
        ),
    )
    parser.add_argument("--vla-config", type=Path)
    parser.add_argument("--command-parser-model", type=Path)
    parser.add_argument("--vla-device", default="cuda")
    parser.add_argument(
        "--vla-precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument(
        "--vla-decision-every-n",
        type=int,
        default=3,
        help=(
            "Run the high-level VLA every N 20-Hz simulation frames and "
            "hold the latest approved decision in the PID between updates; "
            "the default is a 150-ms policy period."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the runtime contract without connecting to CARLA.",
    )
    parser.add_argument(
        "--route-preflight",
        action="store_true",
        help=(
            "Build the CARLA route, audit structured turn/straight steps, "
            "then exit before spawning actors."
        ),
    )
    return parser.parse_args()


def build_vla_command_schedule(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compile reviewed Scene-2 commands into compound DrivingIntents."""

    result = []
    for command in config["commands"]:
        intent = build_scheduled_driving_intent(
            command,
            simulation_frame=0,
            route_s_m=float(command["announce_at_m"]),
            timestamp_s=0.0,
        )
        intent["request_id"] = "{0}-configured".format(command["id"])
        result.append(
            {
                "id": str(command["id"]),
                "announce_at_m": float(command["announce_at_m"]),
                "activate_at_m": float(command["announce_at_m"]),
                "voice_text": str(command["spoken_text"]),
                "target_speed_kmh": float(
                    config["route"]["target_speed_kmh"]
                ),
                "action": "keep_lane",
                "driving_intent": intent,
            }
        )
    return result


def vla_route_context(
    route: list[tuple[Any, Any]],
    route_distances: list[float],
    tracker: Any,
    progress_m: float,
    simulation_time_s: float,
    default_speed_kmh: float,
) -> dict[str, Any]:
    """Return the same short-horizon geometry contract as the unified runner."""

    target_index = bisect.bisect_left(route_distances, progress_m + 10.0)
    target_index = min(max(int(tracker.index), target_index), len(route) - 1)
    target_transform = route[target_index][0].transform
    reference_transform = route[int(tracker.index)][0].transform
    return {
        "progress_m": float(progress_m),
        "simulation_time_s": float(simulation_time_s),
        "default_speed_kmh": float(default_speed_kmh),
        "route_target": {
            "x": float(target_transform.location.x),
            "y": float(target_transform.location.y),
            "z": float(target_transform.location.z),
            "yaw": float(target_transform.rotation.yaw),
        },
        "turn_route_target": {
            "x": float(target_transform.location.x),
            "y": float(target_transform.location.y),
            "z": float(target_transform.location.z),
            "yaw": float(target_transform.rotation.yaw),
        },
        "route_reference": {
            "x": float(reference_transform.location.x),
            "y": float(reference_transform.location.y),
            "z": float(reference_transform.location.z),
            "yaw": float(reference_transform.rotation.yaw),
        },
    }


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "scene_2_town05_runtime/v1":
        raise ValueError("unsupported Town05 Scene 2 config schema")
    if payload.get("map") != "Town05_Opt":
        raise ValueError("Town05 Scene 2 must use Town05_Opt")
    route = payload["route"]
    if float(route["target_length_m"]) < 8000.0:
        raise ValueError("Scene 2 route must be at least 8 km")
    if float(route.get("minimum_curvature_degrees", 0.0)) < 180.0:
        raise ValueError("Scene 2 route must require visible curved driving")
    if len(payload.get("commands", [])) != 15:
        raise ValueError("Scene 2 requires exactly 15 demonstration commands")
    for command in payload["commands"]:
        required_command_fields = {
            "id",
            "announce_at_m",
            "text",
            "spoken_text",
            "category",
            "urgency",
            "steps",
        }
        missing = required_command_fields.difference(command)
        if missing:
            raise ValueError(
                "structured command fields are missing for {0}: {1}".format(
                    command.get("id", "unknown"),
                    sorted(missing),
                )
            )
        if len(command["steps"]) < 2:
            raise ValueError(
                "competition commands must contain ordered compound steps"
            )
        if str(command["text"]) != str(command["spoken_text"]):
            raise ValueError(
                "command display text and spoken text must match exactly: "
                + str(command["id"])
            )
    kinds = {
        str(event["kind"])
        for event in payload.get("special_events", [])
    }
    required = {
        "slow_vehicle",
        "crossing_pedestrian",
        "bus_stop",
        "cyclist",
    }
    if not required.issubset(kinds):
        raise ValueError(
            "special events must include slow vehicle, pedestrian, "
            "bus stop, and cyclist"
        )
    for event in payload.get("special_events", []):
        if len(event.get("variants", [])) < 2:
            raise ValueError(
                "every Town05 special event requires at least two variants: "
                + str(event.get("id"))
            )
    validate_event_ground_truth_contracts(payload["special_events"])
    traffic = payload["traffic"]
    if bool(traffic["hybrid_physics"]):
        raise ValueError(
            "hybrid physics is disabled to prevent visible NPC teleporting"
        )
    if bool(traffic["respawn_dormant_vehicles"]):
        raise ValueError(
            "runtime respawn is disabled to prevent visible NPC pop-in"
        )
    weather = payload["weather"]
    if (
        weather.get("preset") != "cloudy-evening"
        or float(weather["cloudiness"]) < 70.0
        or float(weather["sun_altitude_angle"]) > 10.0
    ):
        raise ValueError(
            "Scene 2 competition weather must be cloudy evening/low light"
        )
    sensors = payload["sensors"]
    if not set(REQUIRED_SENSOR_NAMES).issubset(sensors["required"]):
        raise ValueError("four RGB views and LiDAR are required")
    interfaces = payload["interfaces"]
    if not interfaces.get("simulation_frame_is_sync_key"):
        raise ValueError("simulation_frame must be the synchronization key")
    if interfaces.get("allow_adjacent_frame_fill"):
        raise ValueError("adjacent-frame modality filling is forbidden")
    acceptance = payload["acceptance"]
    if float(acceptance["asr_accuracy_min"]) < 0.96:
        raise ValueError("ASR acceptance must be at least 96%")
    if float(acceptance["semantic_alignment_accuracy_min"]) < 0.985:
        raise ValueError("semantic alignment acceptance must be at least 98.5%")
    if int(acceptance["violation_count_at_most"]) > 1:
        raise ValueError("competition allows at most one violation")
    return payload


def road_option_name(value: Any) -> str:
    # CARLA releases expose RoadOption as either Enum or IntEnum.  On modern
    # Python, str(IntEnumMember) is the numeric value (for example ``2``), so
    # parsing only its string representation silently loses every maneuver.
    enum_name = getattr(value, "name", None)
    if enum_name:
        return str(enum_name).upper()
    return str(value).rsplit(".", 1)[-1].upper()


def route_aware_preview_speed_kmh(
    progress_m: float,
    maneuvers: list[Mapping[str, Any]],
    cruise_speed_kmh: float,
    turn_speed_kmh: float,
    approach_m: float,
    exit_m: float,
) -> float:
    """Cap preview speed around planned turns to avoid corner cutting."""

    cruise = max(0.0, float(cruise_speed_kmh))
    turn_speed = min(cruise, max(0.0, float(turn_speed_kmh)))
    approach = max(0.0, float(approach_m))
    exit_distance = max(0.0, float(exit_m))
    progress = float(progress_m)
    for maneuver in maneuvers:
        if str(maneuver.get("route_option", "")).upper() not in {
            "LEFT",
            "RIGHT",
        }:
            continue
        delta_m = float(maneuver["progress_m"]) - progress
        if -exit_distance <= delta_m <= approach:
            return turn_speed
    return cruise


def planned_turn_window_active(
    progress_m: float,
    maneuvers: list[Mapping[str, Any]],
    approach_m: float,
    exit_m: float,
) -> bool:
    """Return whether progress is inside a planned turn control window."""

    progress = float(progress_m)
    approach = max(0.0, float(approach_m))
    exit_distance = max(0.0, float(exit_m))
    return any(
        str(maneuver.get("route_option", "")).upper()
        in {"LEFT", "RIGHT"}
        and -exit_distance
        <= float(maneuver["progress_m"]) - progress
        <= approach
        for maneuver in maneuvers
    )


def route_centering_steer_correction(
    ego_location: Any,
    route_waypoint: Any,
    lateral_gain: float,
    maximum_correction: float,
) -> tuple[float, float]:
    """Return a bounded steering correction toward the route centerline.

    CARLA steering is positive to the right.  Signed lateral error is also
    positive to the waypoint's right, so the stabilizing correction has the
    opposite sign.  This guard is applied only around audited route turns.
    """

    transform = route_waypoint.transform
    center = transform.location
    yaw = math.radians(float(transform.rotation.yaw))
    right_x = -math.sin(yaw)
    right_y = math.cos(yaw)
    dx = float(ego_location.x) - float(center.x)
    dy = float(ego_location.y) - float(center.y)
    lateral_error_m = dx * right_x + dy * right_y
    limit = max(0.0, float(maximum_correction))
    correction = -float(lateral_gain) * lateral_error_m
    correction = max(-limit, min(limit, correction))
    return correction, lateral_error_m


def audit_command_route_alignment(
    commands: list[Mapping[str, Any]],
    route: list[tuple[Any, Any]],
    route_distances: list[float],
    default_horizon_m: float = 350.0,
) -> dict[str, Any]:
    """Check explicit upcoming turn/straight steps against the GRP route."""

    required_options = {
        "TURN:LEFT": "LEFT",
        "TURN:RIGHT": "RIGHT",
        "PROCEED:STRAIGHT_THROUGH_JUNCTION": "STRAIGHT",
    }
    maneuver_names = {"LEFT", "RIGHT", "STRAIGHT"}
    global_maneuvers = []
    previous_name = None
    for index, distance in enumerate(route_distances):
        name = road_option_name(route[index][1])
        if name in maneuver_names and name != previous_name:
            global_maneuvers.append(
                {
                    "route_option": name,
                    "progress_m": round(float(distance), 3),
                }
            )
        previous_name = name

    records = []
    for command in commands:
        required = [
            required_options[step]
            for step in command.get("steps", [])
            if step in required_options
        ]
        start_m = float(command["announce_at_m"])
        end_m = start_m + float(
            command.get("route_alignment_horizon_m", default_horizon_m)
        )
        maneuvers = [
            maneuver
            for maneuver in global_maneuvers
            if start_m <= maneuver["progress_m"] <= end_m
        ]
        encountered = [
            maneuver["route_option"] for maneuver in maneuvers
        ]
        cursor = iter(encountered)
        matched = all(
            any(actual == expected for actual in cursor)
            for expected in required
        )
        records.append(
            {
                "command_id": str(command["id"]),
                "announce_at_m": start_m,
                "horizon_end_m": end_m,
                "required_route_options": required,
                "encountered_route_options": encountered,
                "maneuvers": maneuvers,
                "matched": matched,
            }
        )
    mismatches = [
        record for record in records if not record["matched"]
    ]
    return {
        "schema_version": "scene_2_route_command_audit/v1",
        "commands_audited": len(records),
        "mismatch_count": len(mismatches),
        "competition_ready": not mismatches,
        "global_maneuvers": global_maneuvers,
        "records": records,
    }


def setup_navigation_agents(carla_root: str | None) -> None:
    try:
        __import__("agents.navigation.behavior_agent")
        return
    except ModuleNotFoundError as error:
        if error.name and error.name != "agents":
            raise RuntimeError(
                "CARLA navigation agent dependency is missing: "
                + error.name
            ) from error
    configured = carla_root or os.environ.get("CARLA_ROOT")
    if configured:
        api_path = Path(configured) / "PythonAPI" / "carla"
        if api_path.exists() and str(api_path) not in sys.path:
            sys.path.insert(0, str(api_path))
    try:
        __import__("agents.navigation.behavior_agent")
    except ModuleNotFoundError as error:
        if error.name and not error.name.startswith("agents"):
            raise RuntimeError(
                "CARLA navigation agent dependency is missing: "
                + error.name
            ) from error
        raise RuntimeError(
            "CARLA navigation agents are required for preview control; "
            "set CARLA_ROOT to the extracted CARLA package"
        ) from error


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
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


@dataclass
class SafetyMonitor:
    world: Any
    ego: Any
    registry: ActorRegistry

    def __post_init__(self) -> None:
        # ``sensor.other.collision`` emits one callback for every contact
        # frame.  Keep those raw samples for diagnostics, but group adjacent
        # samples from the same actor into one physical collision episode.
        self.collision_samples: list[dict[str, Any]] = []
        self.collisions: list[dict[str, Any]] = []
        self._open_collision_by_actor: dict[str, dict[str, Any]] = {}
        self.lane_invasions: list[dict[str, Any]] = []
        self.simulation_time_s = 0.0

    def start(self) -> None:
        import carla

        library = self.world.get_blueprint_library()
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
        self.registry.add(collision)
        self.registry.add(lane)

    def _on_collision(self, event: Any) -> None:
        other = getattr(event, "other_actor", None)
        frame = int(event.frame)
        actor_id = getattr(other, "id", None)
        actor_type = getattr(other, "type_id", "unknown")
        actor_attributes = getattr(other, "attributes", {}) or {}
        actor_role = actor_attributes.get("role_name")
        sample = {
            "frame": frame,
            "simulation_time_s": self.simulation_time_s,
            "other_actor_id": actor_id,
            "other_actor_type": actor_type,
            "other_actor_role": actor_role,
        }
        self.collision_samples.append(sample)

        actor_key = (
            "id:{0}".format(actor_id)
            if actor_id is not None
            else "type:{0}".format(actor_type)
        )
        episode = self._open_collision_by_actor.get(actor_key)
        if (
            episode is not None
            and frame - int(episode["last_frame"]) <= 10
        ):
            episode["last_frame"] = frame
            episode["last_time_s"] = self.simulation_time_s
            episode["contact_samples"] += 1
            return

        episode = {
            "first_frame": frame,
            "last_frame": frame,
            "first_time_s": self.simulation_time_s,
            "last_time_s": self.simulation_time_s,
            "other_actor_id": actor_id,
            "other_actor_type": actor_type,
            "other_actor_role": actor_role,
            "contact_samples": 1,
        }
        self.collisions.append(episode)
        self._open_collision_by_actor[actor_key] = episode

    def _on_lane_invasion(self, event: Any) -> None:
        payload = {
            "frame": int(event.frame),
            "simulation_time_s": self.simulation_time_s,
            "markings": [
                lane_marking_name(marking.type)
                for marking in event.crossed_lane_markings
            ],
        }
        payload["restricted"] = lane_invasion_is_restricted(payload)
        self.lane_invasions.append(payload)

    @property
    def restricted_lane_invasions(self) -> list[dict[str, Any]]:
        return [
            event
            for event in self.lane_invasions
            if lane_invasion_is_restricted(event)
        ]


def apply_weather(world: Any, config: Mapping[str, Any]) -> None:
    import carla

    world.set_weather(
        carla.WeatherParameters(
            cloudiness=float(config["cloudiness"]),
            precipitation=float(config["precipitation"]),
            precipitation_deposits=float(
                config["precipitation_deposits"]
            ),
            wind_intensity=float(config["wind_intensity"]),
            sun_azimuth_angle=float(config["sun_azimuth_angle"]),
            sun_altitude_angle=float(config["sun_altitude_angle"]),
            fog_density=float(config["fog_density"]),
            fog_distance=float(config["fog_distance"]),
            fog_falloff=float(config["fog_falloff"]),
            wetness=float(config["wetness"]),
        )
    )


def load_packaged_map_layers(world: Any) -> bool:
    """Ensure optional Town assets are visible on ``*_Opt`` maps."""

    import carla

    loader = getattr(world, "load_map_layer", None)
    map_layer = getattr(carla, "MapLayer", None)
    if not callable(loader) or map_layer is None:
        return False
    try:
        loader(map_layer.All)
    except RuntimeError as error:
        print("WARNING: optional Town map layers were not reloaded: {0}".format(error))
        return False
    print("Packaged Town optional map layers: ALL loaded")
    return True


def spawn_ego(world: Any, transform: Any) -> Any:
    library = world.get_blueprint_library()
    for identifier in EGO_BLUEPRINTS:
        try:
            blueprint = library.find(identifier)
        except (IndexError, RuntimeError):
            continue
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")
        candidate = transform
        candidate.location.z += 0.35
        actor = world.try_spawn_actor(blueprint, candidate)
        if actor is not None:
            return actor
    raise RuntimeError("failed to spawn Town05 Scene 2 ego")


def overlay_payload(
    progress_m: float,
    route_length_m: float,
    ego: Any,
    current_command: Mapping[str, Any] | None,
    event_states: Mapping[str, str],
    nearby: Mapping[str, int],
) -> dict[str, Any]:
    command_text = (
        str(current_command["text"])
        if current_command is not None
        else "WAITING"
    )
    return {
        "status": "RUNNING",
        "route_progress_m": progress_m,
        "route_length_m": route_length_m,
        "speed_kmh": speed_kmh(ego),
        "target_speed_kmh": 45.0,
        "asr_text": command_text,
        "asr_text_ascii": (
            COMMAND_HUD_FALLBACKS.get(
                str(current_command["id"]),
                "Composite Scene 2 voice command",
            )
            if current_command is not None
            else "Waiting for the next voice command"
        ),
        "source_step_action": (
            "COMPOSITE_COMMAND"
            if current_command is not None
            else "WAITING"
        ),
        "active_step_id": (
            str(current_command["id"])
            if current_command is not None
            else "WAITING"
        ),
        "parse_status": "PRESET_TEXT",
        "risk_level": (
            "MEDIUM"
            if any(state == "ACTIVE" for state in event_states.values())
            else "LOW"
        ),
        "reason": (
            "Town05 packaged map / deterministic event actors"
        ),
        "traffic_count": int(nearby["vehicles"]),
        "pedestrian_count": int(nearby["walkers"]),
    }


def command_event_requirements_met(
    command: Mapping[str, Any],
    event_states: Mapping[str, str],
) -> bool:
    """Return whether all event-state gates for a command are satisfied."""

    requirements = command.get("requires_event_states", {})
    if requirements is None:
        return True
    if not isinstance(requirements, Mapping):
        raise ValueError(
            "command requires_event_states must be a mapping"
        )
    return all(
        event_states.get(str(event_id)) == str(required_state)
        for event_id, required_state in requirements.items()
    )


def ready_commands_in_order(
    commands: list[Mapping[str, Any]],
    announced: set[str],
    progress_m: float,
    event_states: Mapping[str, str],
) -> list[Mapping[str, Any]]:
    """Return the contiguous, ordered prefix ready for announcement.

    A command whose progress or event gate is not ready blocks every later
    command.  This preserves the competition requirement that compound
    instructions are delivered and evaluated in their configured order.
    """

    ready = []
    for command in commands:
        command_id = str(command["id"])
        if command_id in announced:
            continue
        if progress_m < float(command["announce_at_m"]):
            break
        if not command_event_requirements_met(command, event_states):
            break
        ready.append(command)
    return ready


def configure_traffic_manager_physics(
    traffic_manager: Any,
    *,
    hybrid_enabled: bool,
    hybrid_radius_m: float,
) -> None:
    """Configure transparent near-field physics for dense traffic runs."""

    radius_m = float(hybrid_radius_m)
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("traffic hybrid radius must be positive")
    traffic_manager.set_hybrid_physics_mode(bool(hybrid_enabled))
    if hybrid_enabled:
        traffic_manager.set_hybrid_physics_radius(radius_m)


def configure_competition_artifacts(args: Any, vla_enabled: bool) -> None:
    """Configure evidence writers without changing the online VLA sensor rig."""

    if args.competition_logs_only and not args.competition_run:
        raise ValueError("--competition-logs-only requires --competition-run")
    if args.competition_logs_only and not vla_enabled:
        raise ValueError("--competition-logs-only requires --vla-checkpoint")
    if not args.competition_run:
        return
    if args.competition_logs_only:
        args.no_video = True
        args.record_multimodal = False
        args.record_ground_truth = False
        args.video_overlay = False
        return
    args.record_multimodal = True
    args.record_ground_truth = True
    args.video_overlay = True


def carla_world_can_be_reused(
    world: Any,
    requested_map: str,
) -> bool:
    """Return whether an already loaded, actor-clean CARLA world is reusable."""

    try:
        current_name = str(world.get_map().name)
        current_asset = current_name.replace("\\", "/").rstrip("/").split("/")[-1]
        requested_asset = (
            str(requested_map).replace("\\", "/").rstrip("/").split("/")[-1]
        )
        if current_asset != requested_asset:
            return False
        blocking_prefixes = (
            "vehicle.",
            "walker.pedestrian.",
            "controller.ai.walker",
            "sensor.",
        )
        return not any(
            str(getattr(actor, "type_id", "")).startswith(blocking_prefixes)
            for actor in world.get_actors()
        )
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    vla_enabled = args.vla_checkpoint is not None
    if vla_enabled and (
        args.vla_config is None or args.command_parser_model is None
    ):
        raise ValueError(
            "--vla-checkpoint requires --vla-config and "
            "--command-parser-model"
        )
    if args.vla_decision_every_n <= 0:
        raise ValueError("--vla-decision-every-n must be positive")
    configure_competition_artifacts(args, vla_enabled)
    if args.competition_run:
        if args.no_video and not vla_enabled:
            raise ValueError(
                "--competition-run --no-video requires --vla-checkpoint"
            )
        if args.duration != 0.0:
            raise ValueError("--competition-run requires --duration 0")
        if args.start_progress_m != 0.0:
            raise ValueError(
                "--competition-run requires --start-progress-m 0"
            )
        if not args.external_ego_control:
            raise ValueError(
                "--competition-run requires --external-ego-control"
            )
    print("Validated runtime config: {0}".format(config_path))
    print("Map: {0}".format(config["map"]))
    print("Commands: {0}".format(len(config["commands"])))
    print(
        "Traffic: {0} vehicles, {1} ambient walkers".format(
            config["traffic"]["vehicles"],
            config["traffic"]["ambient_walkers"],
        )
    )
    if args.validate_only:
        return 0

    carla_root = setup_carla_api()
    import carla

    # GlobalRoutePlanner is required by every real CARLA run, including
    # --route-preflight and external-control competition runs.  Only the
    # BehaviorAgent controller itself is optional.
    setup_navigation_agents(carla_root)
    BehaviorAgent = None
    if (
        not args.external_ego_control
        and not args.route_preflight
        and not vla_enabled
    ):
        from agents.navigation.behavior_agent import BehaviorAgent

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_output = None
    if not args.no_video:
        video_output = (
            args.video_output.resolve()
            if args.video_output is not None
            else output_dir / "scene2_town05_preview.mp4"
        )
    runtime_log = JsonlWriter(output_dir / "runtime.jsonl")
    event_log = JsonlWriter(output_dir / "events.jsonl")
    command_log = JsonlWriter(output_dir / "commands.jsonl")
    registry = ActorRegistry()
    client = None
    world = None
    original_settings = None
    traffic_manager = None
    camera = None
    ground_truth = None
    sensor_suite = None
    runtime_interface = None
    safety = None
    unified_vla = None
    unified_route_pid = None
    summary: dict[str, Any] = {}

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        current_world = client.get_world()
        world_reused = bool(
            args.reuse_current_world
            and carla_world_can_be_reused(
                current_world, str(config["map"])
            )
        )
        if world_reused:
            world = current_world
            print("CARLA world: reused clean {0}".format(config["map"]))
        else:
            world = client.load_world(str(config["map"]))
            print("CARLA world: loaded {0}".format(config["map"]))
        load_packaged_map_layers(world)
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(
            args.fixed_delta_seconds
        )
        # Camera-free regression runs do not need Unreal rendering.  Physics,
        # traffic lights, Traffic Manager, collision sensors, and event actors
        # continue to advance in synchronous mode.
        settings.no_rendering_mode = bool(
            args.no_video
            and not vla_enabled
            and not args.record_multimodal
        )
        world.apply_settings(settings)
        apply_weather(world, config["weather"])

        traffic_manager = client.get_trafficmanager(
            args.traffic_manager_port
        )
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(
            int(config["traffic"]["seed"])
        )
        traffic_manager.set_global_distance_to_leading_vehicle(
            float(
                config["traffic"][
                    "global_distance_to_leading_vehicle_m"
                ]
            )
        )
        traffic_manager.global_percentage_speed_difference(
            float(
                config["traffic"]["global_speed_difference_pct"]
            )
        )
        configure_traffic_manager_physics(
            traffic_manager,
            hybrid_enabled=bool(args.traffic_hybrid_physics),
            hybrid_radius_m=float(args.traffic_hybrid_radius_m),
        )
        traffic_manager.set_respawn_dormant_vehicles(False)

        sensor_tick = float(
            args.sensor_tick
            if args.sensor_tick is not None
            else config["sensors"]["sensor_tick_s"]
        )
        sensor_stride = max(
            1,
            round(sensor_tick / float(args.fixed_delta_seconds)),
        )
        if not math.isclose(
            sensor_tick,
            sensor_stride * float(args.fixed_delta_seconds),
            abs_tol=1e-9,
        ):
            raise ValueError(
                "sensor tick must be an integer multiple of fixed delta"
            )

        destination = config["route"]["turnaround_spawn_index"]
        if destination == "auto":
            destination, leg_length, leg_curvature = (
                choose_curved_route_destination(
                    world.get_map(),
                    int(config["route"]["start_spawn_index"]),
                    float(config["route"]["route_sampling_m"]),
                )
            )
            print(
                "Auto-selected Town05 destination: index={0}, leg={1:.1f} "
                "m, curvature={2:.1f} deg".format(
                    destination,
                    leg_length,
                    leg_curvature,
                )
            )
        route, route_distances = build_repeated_route(
            world.get_map(),
            int(config["route"]["start_spawn_index"]),
            int(destination),
            float(config["route"]["target_length_m"]),
            float(config["route"]["route_sampling_m"]),
        )
        route_length_m = float(route_distances[-1])
        curvature_degrees = route_curvature_degrees(route)
        if curvature_degrees < float(
            config["route"]["minimum_curvature_degrees"]
        ):
            raise RuntimeError(
                "selected route is too straight for Scene 2: "
                "{0:.1f} deg".format(curvature_degrees)
            )
        print(
            "Town05 route geometry: {0:.1f} m, accumulated curvature "
            "{1:.1f} deg".format(route_length_m, curvature_degrees)
        )
        route_command_audit = audit_command_route_alignment(
            config["commands"],
            route,
            route_distances,
        )
        (output_dir / "route_command_audit.json").write_text(
            json.dumps(
                route_command_audit,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "Route-command audit: {0} commands, {1} mismatches".format(
                route_command_audit["commands_audited"],
                route_command_audit["mismatch_count"],
            )
        )
        if args.route_preflight:
            print(json.dumps(route_command_audit, ensure_ascii=False, indent=2))
            return 0 if route_command_audit["competition_ready"] else 2
        if args.competition_run and not route_command_audit[
            "competition_ready"
        ]:
            raise RuntimeError(
                "structured voice commands do not match the planned route; "
                "inspect route_command_audit.json"
            )
        start_progress_m = max(0.0, float(args.start_progress_m))
        start_route_index = min(
            range(len(route_distances)),
            key=lambda index: abs(
                route_distances[index] - start_progress_m
            ),
        )
        ego = spawn_ego(
            world,
            route[start_route_index][0].transform,
        )
        registry.add(ego)
        print("Ego vehicle blueprint: " + str(ego.type_id))

        events = DeterministicSceneEvents(
            world,
            traffic_manager,
            registry,
            route,
            route_distances,
            config["special_events"],
            int(config["traffic"]["seed"]),
            episode_index=args.variant_index,
        )
        events.spawn()
        print(
            "Scene 2 variants: "
            + json.dumps(events.selected_variants, ensure_ascii=False)
        )
        (output_dir / "selected_variants.json").write_text(
            json.dumps(
                {
                    "variant_index": int(args.variant_index),
                    "events": events.selected_variants,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        traffic = TownTrafficFlow(
            client,
            world,
            traffic_manager,
            registry,
            route,
            config["traffic"],
        )
        traffic.spawn(events.reserved_locations, ego.get_location())
        safety = SafetyMonitor(world, ego, registry)
        safety.start()
        if args.record_ground_truth:
            ground_truth = FrameGroundTruthRecorder(
                output_dir / "frame_ground_truth.jsonl",
                scene_id=str(config["scene_id"]),
                events=events.events,
                every_n_frames=args.ground_truth_every_n,
                model_output_used=bool(vla_enabled),
            )

        agent = None
        behavior_profile = None
        agent_target_speed_kmh = None
        if not args.external_ego_control and not vla_enabled:
            assert BehaviorAgent is not None
            agent = BehaviorAgent(ego, behavior="normal")
            cruise_speed_kmh = float(
                config["route"]["target_speed_kmh"]
            )
            agent.set_target_speed(cruise_speed_kmh)
            agent_target_speed_kmh = cruise_speed_kmh
            agent.set_global_plan(
                list(route[start_route_index:]),
                stop_waypoint_creation=True,
                clean_queue=True,
            )
            behavior_profile = getattr(agent, "_behavior", None)
            if behavior_profile is None or not hasattr(
                behavior_profile,
                "tailgate_counter",
            ):
                raise RuntimeError(
                    "BehaviorAgent tailgating guard is unavailable"
                )
            # CARLA's stock BehaviorAgent may change lane merely because a
            # faster vehicle approaches from behind.  Such an uncommanded
            # maneuver is unsafe for the competition route and previously
            # caused restricted-line invasions.  A route-length-scale
            # cooldown disables that opportunistic behavior while preserving
            # GlobalRoutePlanner turns and ordinary obstacle following.
            behavior_profile.tailgate_counter = 1_000_000_000
            print(
                "BehaviorAgent autonomous tailgating lane changes: disabled"
            )

        for _ in range(30):
            ego.apply_control(carla.VehicleControl(brake=1.0))
            world.tick()

        if not args.no_video:
            assert video_output is not None
            camera = ExperimentCamera(
                world,
                ego,
                str(output_dir / "camera_frames"),
                every_n_frames=1,
                width=args.camera_width,
                height=args.camera_height,
                save_images=args.record_images,
                video_output=str(video_output),
                video_fps=args.video_fps,
                ffmpeg_path=args.ffmpeg,
                video_overlay=args.video_overlay,
                camera_attributes={
                    "gamma": "2.2",
                    "exposure_mode": "histogram",
                    "exposure_compensation": "-0.3",
                    "exposure_speed_up": "3.0",
                    "exposure_speed_down": "1.0",
                    "bloom_intensity": "0.15",
                    "lens_flare_intensity": "0.10",
                    "motion_blur_intensity": "0.10",
                },
                camera_pose=(-9.0, 0.0, 3.2, -9.0, 0.0),
                hud_font_path=args.hud_font,
            )
            camera.start()

        sensor_phase_frame = None
        if args.record_multimodal:
            sensor_suite = ExactFrameSensorSuite(
                world,
                ego,
                registry,
                output_dir,
                sensor_tick,
                image_width=int(config["sensors"]["image_width"]),
                image_height=int(config["sensors"]["image_height"]),
            )
            sensor_suite.start()
            runtime_interface = Scene2RuntimeInterface(
                output_dir,
                config,
            )
            warmup_frames = [
                int(world.tick())
                for _ in range(sensor_stride * 3 + 2)
            ]
            (
                sensor_phase_frame,
                observed_sensor_frames,
            ) = sensor_suite.wait_for_stable_phase(
                warmup_frames,
                sensor_stride,
                args.sensor_sync_timeout,
            )
            if sensor_phase_frame is None:
                raise RuntimeError(
                    "multimodal sensor warm-up did not establish a stable "
                    "four-RGB/LiDAR cadence; complete frames={0}".format(
                        observed_sensor_frames
                    )
                )
            print(
                "Multimodal exact-frame barrier ready: phase={0}, "
                "stride={1}, observed={2}".format(
                    sensor_phase_frame,
                    sensor_stride,
                    observed_sensor_frames,
                )
            )

        tracker = RouteProgressTracker(
            route,
            route_distances,
            index=start_route_index,
        )
        if vla_enabled:
            from types import SimpleNamespace

            from control.generic_route_pid import GenericRoutePID
            from universal_vla_controller import UniversalVLAController

            vla_commands = build_vla_command_schedule(config)
            route_context_view = SimpleNamespace(
                distances_m=route_distances,
                tracker=tracker,
                adapter=None,
            )
            unified_route_pid = GenericRoutePID(
                world,
                ego,
                target_speed_kmh=float(
                    config["route"]["target_speed_kmh"]
                ),
                fixed_delta_seconds=float(args.fixed_delta_seconds),
                route_context=route_context_view,
                route_plan=route,
            )
            # Give the unified VLA's sensor rig one complete cadence before
            # the timed loop starts.
            for _ in range(int(args.vla_decision_every_n) + 2):
                world.tick()
            unified_vla = UniversalVLAController(
                world=world,
                ego=ego,
                route_controller=unified_route_pid,
                commands=vla_commands,
                checkpoint_path=args.vla_checkpoint.resolve(),
                config_path=args.vla_config.resolve(),
                parser_model_path=args.command_parser_model.resolve(),
                output_path=output_dir / "vla_control_decisions.jsonl",
                device=args.vla_device,
                precision=args.vla_precision,
                decision_interval_frames=int(args.vla_decision_every_n),
                fixed_delta_seconds=float(args.fixed_delta_seconds),
                available_cameras=("front", "left", "right", "rear"),
                enable_lidar=True,
                default_speed_kmh=float(
                    config["route"]["target_speed_kmh"]
                ),
            )
            # Give the unified controller's sensor rig one complete cadence.
            for _ in range(int(args.vla_decision_every_n) + 2):
                world.tick()
            print(
                "Scene 2 universal VLA control enabled: unified batch -> "
                "VLA pipeline -> temporal supervisor -> instruction FSM -> "
                "route PID"
            )
        announced: set[str] = set()
        current_command = None
        latest_intent = None
        start_snapshot = world.get_snapshot()
        start_time = float(start_snapshot.timestamp.elapsed_seconds)
        frame_counter = 0
        progress_m = 0.0
        turn_centering_active = False
        while True:
            snapshot = world.get_snapshot()
            simulation_time_s = (
                float(snapshot.timestamp.elapsed_seconds) - start_time
            )
            if safety is not None:
                safety.simulation_time_s = simulation_time_s

            progress_m = tracker.update(ego.get_location())
            for command in ready_commands_in_order(
                config["commands"],
                announced,
                progress_m,
                events.states,
            ):
                command_id = str(command["id"])
                announced.add(command_id)
                current_command = command
                command_log.write(
                    {
                        "frame": int(snapshot.frame),
                        "simulation_time_s": simulation_time_s,
                        "route_progress_m": progress_m,
                        **command,
                    }
                )
                if runtime_interface is not None:
                    latest_intent = runtime_interface.publish_command(
                        command,
                        int(snapshot.frame),
                        progress_m,
                        simulation_time_s,
                    )
                print(
                    "COMMAND | {0} | {1:.1f} m | {2}".format(
                        command_id,
                        progress_m,
                        command["text"],
                    )
                )

            for change in events.update(progress_m):
                event_log.write(
                    {
                        "frame": int(snapshot.frame),
                        "simulation_time_s": simulation_time_s,
                        **change,
                    }
                )
                print(
                    "EVENT | {0} | {1}".format(
                        change["event_id"],
                        change["state"],
                    )
                )

            if unified_vla is not None:
                control = unified_vla.run_step()
                ego.apply_control(control)
            elif agent is not None:
                desired_speed_kmh = route_aware_preview_speed_kmh(
                    progress_m,
                    route_command_audit["global_maneuvers"],
                    float(config["route"]["target_speed_kmh"]),
                    float(config["route"].get("turn_speed_kmh", 18.0)),
                    float(
                        config["route"].get(
                            "turn_slowdown_approach_m",
                            60.0,
                        )
                    ),
                    float(
                        config["route"].get(
                            "turn_slowdown_exit_m",
                            30.0,
                        )
                    ),
                )
                if (
                    agent_target_speed_kmh is None
                    or not math.isclose(
                        desired_speed_kmh,
                        agent_target_speed_kmh,
                        abs_tol=1e-6,
                    )
                ):
                    assert behavior_profile is not None
                    behavior_profile.max_speed = desired_speed_kmh
                    agent.set_target_speed(desired_speed_kmh)
                    agent_target_speed_kmh = desired_speed_kmh
                control = agent.run_step()
                centering_now = planned_turn_window_active(
                    progress_m,
                    route_command_audit["global_maneuvers"],
                    float(
                        config["route"].get(
                            "turn_centering_approach_m",
                            25.0,
                        )
                    ),
                    float(
                        config["route"].get(
                            "turn_centering_exit_m",
                            45.0,
                        )
                    ),
                )
                lateral_error_m = 0.0
                if centering_now:
                    correction, lateral_error_m = (
                        route_centering_steer_correction(
                            ego.get_location(),
                            route[tracker.index][0],
                            float(
                                config["route"].get(
                                    "turn_centering_lateral_gain",
                                    0.22,
                                )
                            ),
                            float(
                                config["route"].get(
                                    "turn_centering_max_steer_correction",
                                    0.35,
                                )
                            ),
                        )
                    )
                    control.steer = max(
                        -1.0,
                        min(1.0, float(control.steer) + correction),
                    )
                if centering_now != turn_centering_active:
                    print(
                        "Route turn centering: {0} at {1:.1f} m "
                        "(lateral_error={2:.3f} m)".format(
                            "ACTIVE" if centering_now else "INACTIVE",
                            progress_m,
                            lateral_error_m,
                        )
                    )
                    turn_centering_active = centering_now
                control.manual_gear_shift = False
                ego.apply_control(control)

            frame = world.tick()
            post_snapshot = world.get_snapshot()
            simulation_time_s = (
                float(post_snapshot.timestamp.elapsed_seconds) - start_time
            )
            if safety is not None:
                safety.simulation_time_s = simulation_time_s
            nearby = traffic.nearby_counts(ego.get_location())
            if camera is not None:
                camera.save_frame(
                    frame,
                    overlay=overlay_payload(
                        progress_m,
                        route_length_m,
                        ego,
                        current_command,
                        events.states,
                        nearby,
                    ),
                )
            if ground_truth is not None:
                ground_truth.record(
                    world=world,
                    ego=ego,
                    simulation_frame=int(frame),
                    timestamp_s=simulation_time_s,
                    route_s_m=progress_m,
                    event_states=events.states,
                    actor_bindings=(
                        events.ground_truth_actor_bindings()
                    ),
                    runtime_state=(
                        events.ground_truth_runtime_state()
                    ),
                )
            if (
                sensor_suite is not None
                and runtime_interface is not None
                and sensor_phase_frame is not None
                and (int(frame) - sensor_phase_frame) % sensor_stride == 0
            ):
                complete, sensor_frames = sensor_suite.wait_for_frame(
                    int(frame),
                    args.sensor_sync_timeout,
                )
                world_state = runtime_interface.publish_world_state(
                    world,
                    ego,
                    int(frame),
                    progress_m,
                    simulation_time_s,
                    safety,
                    speed_kmh,
                )
                bundle = runtime_interface.publish_bundle(
                    int(frame),
                    world_state,
                    sensor_frames,
                    latest_intent,
                )
                if not complete or bundle["status"] != "COMPLETE":
                    message = (
                        "exact multimodal synchronization timed out at "
                        "frame {0}: {1}".format(frame, sensor_frames)
                    )
                    if args.competition_run:
                        raise RuntimeError(message)
                    print("WARNING | " + message)
            if frame_counter % 20 == 0:
                runtime_log.write(
                    {
                        "frame": int(frame),
                        "simulation_time_s": simulation_time_s,
                        "route_progress_m": progress_m,
                        "route_length_m": route_length_m,
                        "ego_speed_kmh": speed_kmh(ego),
                        "nearby_vehicles_85m": nearby["vehicles"],
                        "nearby_walkers_85m": nearby["walkers"],
                        "event_states": dict(events.states),
                        "selected_variants": dict(
                            events.selected_variants
                        ),
                        "collisions": len(safety.collisions),
                        "collision_contact_samples": len(
                            safety.collision_samples
                        ),
                        "lane_invasions": len(
                            safety.lane_invasions
                        ),
                        "restricted_lane_invasions": len(
                            safety.restricted_lane_invasions
                        ),
                    }
                )
            frame_counter += 1

            if progress_m >= float(config["route"]["target_length_m"]):
                print("Town05 Scene 2 route completed")
                break
            if (
                args.duration > 0.0
                and simulation_time_s >= args.duration
            ):
                print("Town05 Scene 2 preview duration completed")
                break

        summary = {
            "schema_version": "scene_2_town05_summary/v1",
            "map": world.get_map().name,
            "world_reused": bool(world_reused),
            "route_start_spawn_index": int(
                config["route"]["start_spawn_index"]
            ),
            "route_turnaround_spawn_index": int(destination),
            "route_length_m": route_length_m,
            "route_curvature_degrees": curvature_degrees,
            "route_progress_m": progress_m,
            "route_completed": progress_m
            >= float(config["route"]["target_length_m"]),
            "route_command_audit": route_command_audit,
            "commands_announced": len(announced),
            "traffic_vehicles_spawned": len(traffic.vehicles),
            "ambient_walkers_spawned": len(traffic.walkers),
            "traffic_hybrid_physics": {
                "enabled": bool(args.traffic_hybrid_physics),
                "radius_m": float(args.traffic_hybrid_radius_m),
            },
            "event_states": dict(events.states),
            "event_summary": events.summary(),
            "variant_index": int(args.variant_index),
            "selected_variants": dict(events.selected_variants),
            "spawn_diagnostics": dict(events.spawn_diagnostics),
            "collision_count": len(safety.collisions),
            "collision_contact_samples": len(
                safety.collision_samples
            ),
            "collision_events": list(safety.collisions),
            "lane_invasion_count": len(safety.lane_invasions),
            "lane_invasion_events": list(safety.lane_invasions),
            "restricted_lane_invasion_count": len(
                safety.restricted_lane_invasions
            ),
            "restricted_lane_invasion_events": list(
                safety.restricted_lane_invasions
            ),
            "video_output": (
                str(video_output) if video_output is not None else None
            ),
            "artifact_recording": {
                "mode": (
                    "structured_logs_only"
                    if args.competition_logs_only
                    else "full_artifacts"
                ),
                "online_vla_sensors_enabled": bool(vla_enabled),
            },
            "preview_controller": {
                "source": (
                    "universal multisensor VLA + deterministic safety gate + PID"
                    if vla_enabled
                    else
                    "external"
                    if args.external_ego_control
                    else "CARLA BehaviorAgent demonstration controller"
                ),
                "competition_metric_eligible": bool(vla_enabled),
            },
        }
        if vla_enabled:
            summary["vla_control"] = unified_vla.summary()
            summary["vla_control"]["checkpoint"] = str(args.vla_checkpoint)
            summary["vla_control"]["config"] = str(args.vla_config)
        if sensor_suite is not None:
            summary["multimodal_evidence"] = sensor_suite.summary()
            summary["multimodal_evidence"][
                "competition_required_exact_ratio"
            ] = 1.0
        if args.competition_logs_only:
            vla_summary = summary.get("vla_control") or {}
            sensor_evidence_available = bool(
                int(vla_summary.get("decision_count", 0)) > 0
                and "4view_rgb" in str(vla_summary.get("input_mode", ""))
                and vla_summary.get("sensor_batch_schema_version")
                == "unified_sensor_batch/1.0"
            )
        else:
            sensor_evidence_available = bool(
                sensor_suite is not None
                and sensor_suite.summary()["exact_completion_ratio"] == 1.0
            )
        measurable_checks = {
            "route_completed": bool(summary["route_completed"]),
            "collision_count_equals_0": (
                summary["collision_count"] == 0
            ),
            "restricted_lane_invasions_at_most_1": (
                summary["restricted_lane_invasion_count"] <= 1
            ),
            "all_commands_announced": (
                summary["commands_announced"] == len(config["commands"])
            ),
            "all_events_resolved": all(
                state == "RESOLVED"
                for state in summary["event_states"].values()
            ),
            "route_commands_aligned": bool(
                route_command_audit["competition_ready"]
            ),
            "sensor_evidence_available": sensor_evidence_available,
            "vla_fallback_count_equals_0": (
                not vla_enabled
                or summary["vla_control"]["fallback_count"] == 0
            ),
        }
        summary["competition_acceptance"] = {
            "measurable_checks": measurable_checks,
            "measurable_checks_passed": all(measurable_checks.values()),
            "pending_external_metrics": [
                "asr_accuracy",
                "ordered_step_completion_and_omissions",
                "semantic_alignment_accuracy",
                "instruction_parse_latency_ms",
                "end_to_end_decision_latency_ms",
                "task_completion_rate",
                "traffic_rule_violations_beyond_lane_invasion",
            ],
            "final_competition_pass_claimed": False,
        }
        if ground_truth is not None:
            summary["ground_truth"] = ground_truth.summary()
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.competition_run and not all(measurable_checks.values()):
            return 2
        return 0
    finally:
        if unified_vla is not None:
            try:
                unified_vla.close()
            except RuntimeError:
                pass
        if runtime_interface is not None:
            runtime_interface.close()
        if ground_truth is not None:
            ground_truth.close()
        runtime_log.close()
        event_log.close()
        command_log.close()
        if camera is not None:
            try:
                camera.destroy()
            except RuntimeError:
                pass
        if client is not None:
            registry.destroy(client)
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


if __name__ == "__main__":
    raise SystemExit(main())
