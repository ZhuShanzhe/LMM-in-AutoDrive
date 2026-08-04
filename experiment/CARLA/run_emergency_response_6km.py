"""Run the 6 km emergency-response scene on official CARLA Town05_Opt.

The runner loads the official town asset (buildings, vegetation, streetlights
and road furniture), builds a route-progress coordinate system over a 6 km
urban expressway corridor, applies physical rain/fog/wet-friction settings,
and reuses the seven recoverable emergency event controllers.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from carla_bootstrap import setup_carla_api
from emergency_scene_3_events import (
    EmergencySceneActorRuntime,
)
from evaluation.camera import ExperimentCamera
from evaluation.ground_truth import (
    FrameGroundTruthRecorder,
    validate_event_ground_truth_contracts,
)
from scene3_town05_route import (
    LOGICAL_CENTRE_LANE,
    LOGICAL_ROUTE_ID,
    Town05RouteContext,
    build_town05_route_context,
    validate_scene3_event_anchors,
)


MAP_NAME = "Town05_Opt"
MAP_ASSET_PATH = "/Game/Carla/Maps/Town05_Opt"
ROAD_ID = LOGICAL_ROUTE_ID
EGO_LANE_ID = LOGICAL_CENTRE_LANE
EGO_START_S_M = 0.0
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "emergency_scene_3"
)
DEFAULT_RUNTIME_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "scene_3_emergency_6km_runtime.json"
)

EGO_BLUEPRINT_IDS = (
    "vehicle.lincoln.mkz_2020",
    "vehicle.tesla.model3",
    "vehicle.audi.tt",
)

CAMERA_TRANSFORMS = {
    "front_rgb": (
        1.6,
        0.0,
        1.7,
        0.0,
    ),
    "left_rgb": (
        0.2,
        -0.5,
        1.7,
        -90.0,
    ),
    "right_rgb": (
        0.2,
        0.5,
        1.7,
        90.0,
    ),
    "rear_rgb": (
        -1.6,
        0.0,
        1.7,
        180.0,
    ),
}

CHASE_CAMERA_TRANSFORMS = {
    "chase_rgb": (
        -5.5,
        0.0,
        2.8,
        -15.0,
        0.0,
    ),
}

PRESENTATION_WEATHER = {
    "cloudiness": 95.0,
    "precipitation": 55.0,
    "precipitation_deposits": 80.0,
    "wind_intensity": 25.0,
    "sun_altitude_angle": 18.0,
    "fog_density": 8.0,
    "fog_distance": 250.0,
    "wetness": 100.0,
}

CLEAR_PRESENTATION_WEATHER = {
    "cloudiness": 15.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "wind_intensity": 5.0,
    "sun_altitude_angle": 65.0,
    "fog_density": 0.0,
    "fog_distance": 1000.0,
    "wetness": 0.0,
}

EVENT_VIDEO_INTENTS = {
    "scene3_cut_in": {
        "asr_text": "突发车辆加塞，紧急避让",
        "action": "decelerate",
        "target_speed_kmh": 30.0,
        "emergency": True,
        "risk_level": "HIGH",
        "policy_state": "CUT_IN_RESPONSE",
    },
    "scene3_advance_warning": {
        "asr_text": "前方进入施工区域，注意观察",
        "action": "decelerate",
        "target_speed_kmh": 35.0,
        "emergency": False,
        "risk_level": "MEDIUM",
        "policy_state": "ADVANCE_WARNING",
    },
    "scene3_cone_taper": {
        "asr_text": "施工路段，减速并道至左侧车道",
        "action": "keep_lane",
        "target_speed_kmh": 30.0,
        "emergency": False,
        "risk_level": "MEDIUM",
        "policy_state": "CONE_TAPER",
    },
    "scene3_work_zone": {
        "asr_text": "前方路况危险，保持安全车速",
        "action": "keep_lane",
        "target_speed_kmh": 25.0,
        "emergency": False,
        "risk_level": "MEDIUM",
        "policy_state": "WORK_ZONE",
    },
    "scene3_temporary_pedestrian": {
        "asr_text": "有人突然横穿，立即减速让行",
        "action": "decelerate",
        "target_speed_kmh": 15.0,
        "emergency": True,
        "risk_level": "HIGH",
        "policy_state": "WORKER_CROSSING",
    },
    "scene3_blocked_lane": {
        "asr_text": "前方车道受阻，确认安全后向左避让",
        "action": "change_lane_left",
        "target_speed_kmh": 20.0,
        "emergency": True,
        "risk_level": "HIGH",
        "policy_state": "GAP_CHECK",
    },
    "scene3_work_zone_exit": {
        "asr_text": "危险路段结束，保持安全并逐步恢复车速",
        "action": "accelerate",
        "target_speed_kmh": 60.0,
        "emergency": False,
        "risk_level": "LOW",
        "policy_state": "ZONE_EXIT",
    },
}

DIRECT_PRESENTATION_CAMERA_ATTRIBUTES = {
    "gamma": "2.2",
    "exposure_mode": "manual",
    "exposure_compensation": "0.0",
    "shutter_speed": "200.0",
    "iso": "100.0",
    "fstop": "5.6",
}

PRESENTATION_DASH_LENGTH_M = 4.0
PRESENTATION_DASH_PERIOD_M = 10.0
PRESENTATION_MARK_Z_M = 0.08


def camera_sensor_attributes(
    *,
    camera_name: str,
    image_width: int,
    image_height: int,
    fov: float,
    camera_tick: float,
    low_signal_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    attributes = {
        "image_size_x": str(image_width),
        "image_size_y": str(image_height),
        "fov": str(fov),
        "sensor_tick": str(camera_tick),
        "gamma": "3.0",
        "exposure_mode": "histogram",
        "exposure_compensation": "3.0",
        "exposure_speed_up": "3.0",
        "exposure_speed_down": "1.0",
    }
    if camera_name == "chase_rgb":
        attributes.update(
            {
                "gamma": "2.2",
                # Lock exposure for presentation video. Histogram
                # auto-exposure brightens over several seconds and
                # makes debug lane paint bloom even in clear weather.
                "exposure_mode": "manual",
                "exposure_compensation": "0.0",
                "shutter_speed": "200.0",
                "iso": "100.0",
                "fstop": "8.0",
            }
        )
    if low_signal_config and low_signal_config.get("enabled") is True:
        for key in (
            "exposure_mode",
            "exposure_compensation",
            "shutter_speed",
            "iso",
            "fstop",
            "gamma",
            "bloom_intensity",
            "motion_blur_intensity",
            "lens_flare_intensity",
        ):
            if key in low_signal_config:
                attributes[key] = str(low_signal_config[key])
    return attributes


def draw_presentation_lane_markings(
    world: Any,
    *,
    road_length_m: float,
    life_time_s: float,
) -> int:
    """Draw visible lane paint over the minimal OpenDRIVE mesh."""

    # Match the restrained decoration values used by the team's
    # main-road preview.  Full-intensity, thick debug lines bloom
    # heavily in rainy Epic-quality rendering.
    white = carla.Color(205, 205, 195)
    yellow = carla.Color(220, 180, 55)
    line_count = 0

    # The presentation ego drives on the right carriageway.  Its
    # three 3.5 m lanes are separated at y=3.5 and y=7.0.
    for lateral_y in (3.5, 7.0):
        start_x = 0.0
        while start_x < road_length_m:
            end_x = min(
                start_x
                + PRESENTATION_DASH_LENGTH_M,
                road_length_m,
            )
            world.debug.draw_line(
                carla.Location(
                    x=start_x,
                    y=lateral_y,
                    z=PRESENTATION_MARK_Z_M,
                ),
                carla.Location(
                    x=end_x,
                    y=lateral_y,
                    z=PRESENTATION_MARK_Z_M,
                ),
                thickness=0.045,
                color=white,
                life_time=0.0,
            )
            line_count += 1
            start_x += PRESENTATION_DASH_PERIOD_M

    # Highlight the carriageway center and outside edge as continuous
    # paint so the minimal standalone mesh reads like a real highway.
    for lateral_y, color, thickness in (
        (0.0, yellow, 0.055),
        (10.5, white, 0.045),
    ):
        world.debug.draw_line(
            carla.Location(
                x=0.0,
                y=lateral_y,
                z=PRESENTATION_MARK_Z_M,
            ),
            carla.Location(
                x=road_length_m,
                y=lateral_y,
                z=PRESENTATION_MARK_Z_M,
            ),
            thickness=thickness,
            color=color,
            life_time=0.0,
        )
        line_count += 1

    print(
        "PRESENTATION LANE MARKINGS DRAWN | "
        f"segments={line_count} | "
        "white_dashed=2 | yellow_center=1 | "
        "white_edge=1"
    )
    return line_count


def camera_transforms_for_mode(
    camera_mode: str,
) -> dict[str, tuple[float, ...]]:
    transforms: dict[str, tuple[float, ...]] = {}
    if camera_mode in {
        "four-view",
        "four-view-plus-chase",
    }:
        transforms.update(
            {
                name: (
                    x,
                    y,
                    z,
                    0.0,
                    yaw,
                )
                for name, (
                    x,
                    y,
                    z,
                    yaw,
                ) in CAMERA_TRANSFORMS.items()
            }
        )
    if camera_mode in {
        "chase-only",
        "four-view-plus-chase",
    }:
        transforms.update(
            CHASE_CAMERA_TRANSFORMS
        )
    if not transforms:
        raise ValueError(
            f"unsupported camera mode: {camera_mode}"
        )
    return transforms


def python_abi_tag() -> str:
    return "cp{0}{1}".format(
        sys.version_info.major,
        sys.version_info.minor,
    )


def load_carla_api() -> Any:
    setup_carla_api()
    return importlib.import_module("carla")


carla: Any = None


class CaptureState:
    """Thread-safe image counters and first-frame notifications."""

    def __init__(
        self,
        camera_names: Sequence[str],
    ) -> None:
        self._lock = threading.Lock()
        self._counts = {
            name: 0
            for name in camera_names
        }
        self._first_frame_events = {
            name: threading.Event()
            for name in camera_names
        }

    def record(
        self,
        camera_name: str,
    ) -> None:
        with self._lock:
            self._counts[camera_name] += 1
        self._first_frame_events[
            camera_name
        ].set()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def wait_for_first_frames(
        self,
        timeout_s: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        for event in self._first_frame_events.values():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            if not event.wait(remaining):
                return False
        return True


class SafetyAuditState:
    """Thread-safe collision, lane-invasion, and lane-use audit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._collision_frames: list[int] = []
        self._lane_invasion_frames: list[int] = []
        self._invalid_lane_samples = 0

    def record_collision(
        self,
        event: Any,
    ) -> None:
        with self._lock:
            self._collision_frames.append(
                int(event.frame)
            )
            collision_index = len(self._collision_frames)
        if collision_index <= 5:
            other_actor = getattr(event, "other_actor", None)
            print(
                "COLLISION SAMPLE | "
                f"frame={int(event.frame)} | "
                f"other={getattr(other_actor, 'type_id', 'unknown')}"
            )

    def record_lane_invasion(
        self,
        event: Any,
    ) -> None:
        with self._lock:
            self._lane_invasion_frames.append(
                int(event.frame)
            )

    def record_lane_id(
        self,
        lane_id: int,
        legal_lane_ids: set[int] | None = None,
        *,
        road_id: int | None = None,
        route_progress_m: float | None = None,
    ) -> None:
        legal = (
            {-1, -2, -3}
            if legal_lane_ids is None
            else legal_lane_ids
        )
        if lane_id in legal:
            return
        with self._lock:
            self._invalid_lane_samples += 1
            sample_index = self._invalid_lane_samples
        if sample_index <= 20:
            print(
                "INVALID LANE SAMPLE | "
                f"route={route_progress_m} | road={road_id} | "
                f"lane={lane_id} | legal={sorted(legal)}"
            )

    def record_route_deviation(
        self,
        distance_m: float,
        *,
        maximum_m: float = 8.0,
        road_id: int | None = None,
        lane_id: int | None = None,
        route_progress_m: float | None = None,
    ) -> None:
        """Audit physical distance to the declared plan, including junctions."""

        if float(distance_m) <= float(maximum_m):
            return
        with self._lock:
            self._invalid_lane_samples += 1
            sample_index = self._invalid_lane_samples
        if sample_index <= 20:
            print(
                "ROUTE DEVIATION SAMPLE | "
                f"route={route_progress_m} | road={road_id} | "
                f"lane={lane_id} | deviation={float(distance_m):.2f} m | "
                f"maximum={float(maximum_m):.2f} m"
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            collision_frames = list(
                self._collision_frames
            )
            lane_invasion_frames = list(
                self._lane_invasion_frames
            )
            invalid_lane_samples = (
                self._invalid_lane_samples
            )
        return {
            "collision_count": len(
                collision_frames
            ),
            "collision_frames": collision_frames,
            "lane_invasion_event_count": len(
                lane_invasion_frames
            ),
            "lane_invasion_frames": (
                lane_invasion_frames
            ),
            "invalid_lane_samples": (
                invalid_lane_samples
            ),
        }


class EmergencyEventScheduler:
    """Activate and resolve configured events by route distance."""

    def __init__(
        self,
        events: Sequence[dict[str, Any]],
        *,
        output_path: Path,
        event_handler: Any | None = None,
    ) -> None:
        self._events = [
            dict(event)
            for event in events
        ]
        self._states = {
            event["id"]: "PENDING"
            for event in self._events
        }
        self._output_path = output_path
        self._event_handler = event_handler
        self._output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._output_path.write_text(
            "",
            encoding="utf-8",
        )

    def _record(
        self,
        *,
        event: dict[str, Any],
        state: str,
        route_s_m: float,
        simulation_frame: int,
        elapsed_s: float,
    ) -> None:
        record = {
            "scene_id": "scene_3_emergency_6km",
            "event_id": event["id"],
            "scenario": event["scenario"],
            "voice_command_id": event.get("voice_command_id"),
            "state": state,
            "route_s_m": round(route_s_m, 3),
            "simulation_frame": simulation_frame,
            "elapsed_s": round(elapsed_s, 3),
        }
        with self._output_path.open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(
            "EVENT "
            f"{state:<8} | {event['id']} | "
            f"route={route_s_m:.1f} m"
        )

    def update(
        self,
        *,
        route_s_m: float,
        simulation_frame: int,
        elapsed_s: float,
    ) -> None:
        for event in self._events:
            event_id = event["id"]
            state = self._states[event_id]
            if (
                state == "PENDING"
                and route_s_m
                >= float(event["activate_at_m"])
            ):
                if self._event_handler is not None:
                    self._event_handler.on_activate(
                        event,
                        route_s_m=route_s_m,
                        simulation_frame=(
                            simulation_frame
                        ),
                        elapsed_s=elapsed_s,
                    )
                self._states[event_id] = "ACTIVE"
                self._record(
                    event=event,
                    state="ACTIVE",
                    route_s_m=route_s_m,
                    simulation_frame=simulation_frame,
                    elapsed_s=elapsed_s,
                )

        if self._event_handler is not None:
            self._event_handler.update(
                route_s_m=route_s_m,
                simulation_frame=simulation_frame,
                elapsed_s=elapsed_s,
            )

        for event in self._events:
            event_id = event["id"]
            state = self._states[event_id]
            if (
                state == "ACTIVE"
                and route_s_m
                >= float(event["resolve_after_m"])
            ):
                if self._event_handler is not None:
                    self._event_handler.on_resolve(
                        event,
                        route_s_m=route_s_m,
                        simulation_frame=(
                            simulation_frame
                        ),
                        elapsed_s=elapsed_s,
                    )
                self._states[event_id] = "RESOLVED"
                self._record(
                    event=event,
                    state="RESOLVED",
                    route_s_m=route_s_m,
                    simulation_frame=simulation_frame,
                    elapsed_s=elapsed_s,
                )

    def summary(self) -> dict[str, int]:
        return {
            state: sum(
                current == state
                for current in self._states.values()
            )
            for state in (
                "PENDING",
                "ACTIVE",
                "RESOLVED",
            )
        }

    def active_event(
        self,
    ) -> dict[str, Any] | None:
        """Return the currently active event for the video HUD."""

        for event in self._events:
            if self._states[event["id"]] == "ACTIVE":
                return dict(event)
        return None

    def state_snapshot(
        self,
    ) -> dict[str, str]:
        return dict(self._states)

    def policy_observation(
        self,
        *,
        route_s_m: float,
        simulation_frame: int,
    ) -> dict[str, Any]:
        """Expose lifecycle metadata without leaking CARLA actor truth.

        This payload is suitable for logging and synchronization.  The policy
        must still infer hazards from sensors/WorldState; event distance,
        configured actor poses and the scheduler's future timeline are omitted.
        """

        active = [
            event
            for event in self._events
            if self._states[event["id"]] == "ACTIVE"
        ]
        return {
            "schema_version": "scene_event_observation/1.0",
            "scene_id": "scene_3_emergency_6km",
            "simulation_frame": int(simulation_frame),
            "route_progress_m": round(float(route_s_m), 3),
            "active_event_count": len(active),
            "active_event_types": [str(event["scenario"]) for event in active],
            "policy_use": "synchronization_only",
        }


def vehicle_speed_kmh(
    vehicle: Any,
) -> float:
    velocity = vehicle.get_velocity()
    return (
        math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )
        * 3.6
    )


class VehicleStateRecorder:
    """Persist the dynamic ego-state modality using the simulation frame key."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")

    @staticmethod
    def _vector(value: Any) -> dict[str, float]:
        return {
            "x": round(float(value.x), 4),
            "y": round(float(value.y), 4),
            "z": round(float(value.z), 4),
        }

    def record(
        self,
        *,
        ego: Any,
        waypoint: Any,
        simulation_frame: int,
        timestamp_s: float,
        route_progress_m: float,
    ) -> None:
        control = ego.get_control()
        payload = {
            "schema_version": "vehicle_state/1.0",
            "simulation_frame": int(simulation_frame),
            "timestamp_s": round(float(timestamp_s), 4),
            "route_progress_m": round(float(route_progress_m), 3),
            "speed_kmh": round(vehicle_speed_kmh(ego), 3),
            "velocity_mps": self._vector(ego.get_velocity()),
            "acceleration_mps2": self._vector(ego.get_acceleration()),
            "angular_velocity_deg_s": self._vector(
                ego.get_angular_velocity()
            ),
            "lane": {
                "road_id": int(waypoint.road_id),
                "section_id": int(waypoint.section_id),
                "lane_id": int(waypoint.lane_id),
            },
            "control": {
                "throttle": round(float(control.throttle), 4),
                "steer": round(float(control.steer), 4),
                "brake": round(float(control.brake), 4),
                "hand_brake": bool(control.hand_brake),
                "reverse": bool(control.reverse),
                "gear": int(control.gear),
            },
        }
        self._stream.write(
            json.dumps(payload, ensure_ascii=False) + "\n"
        )

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()


def make_video_overlay(
    *,
    ego: Any,
    scheduler: EmergencyEventScheduler,
    safety_audit: SafetyAuditState,
    route_s_m: float,
    finish_s_m: float,
    simulation_frame: int,
    elapsed_s: float,
    cruise_speed_kmh: float,
) -> dict[str, Any]:
    """Build the per-tick HUD payload used by ExperimentCamera."""

    active_event = scheduler.active_event()
    intent = {
        "asr_text": "Continue on the 6 km emergency route",
        "action": "keep_lane",
        "target_speed_kmh": cruise_speed_kmh,
        "emergency": False,
        "risk_level": "LOW",
        "policy_state": "CRUISE",
    }
    if active_event is not None:
        intent.update(
            EVENT_VIDEO_INTENTS.get(
                active_event["id"],
                {},
            )
        )

    control = ego.get_control()
    event_summary = scheduler.summary()
    safety_summary = safety_audit.snapshot()
    status = "RUNNING"
    if (
        route_s_m >= finish_s_m
        and event_summary["RESOLVED"] == 7
        and safety_summary["collision_count"] == 0
        and safety_summary["invalid_lane_samples"]
        == 0
    ):
        status = "SUCCESS"
    elif safety_summary["collision_count"]:
        status = "FAILURE"

    return {
        "scenario": "scene_3_emergency_6km",
        "frame": int(simulation_frame),
        "sim_time_s": float(elapsed_s),
        "route_s_m": float(route_s_m),
        "status": status,
        "speed_kmh": vehicle_speed_kmh(ego),
        "throttle": float(control.throttle),
        "brake": float(control.brake),
        "steer": float(control.steer),
        "collisions": safety_summary[
            "collision_count"
        ],
        "lane_events": safety_summary[
            "lane_invasion_event_count"
        ],
        **intent,
    }


def load_runtime_config(
    path: Path,
) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except OSError as error:
        raise ValueError(
            f"cannot read runtime config: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            "runtime config is not valid JSON: "
            f"{error}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "runtime config must be a JSON object"
        )
    if (
        data.get("schema_version")
        != "emergency_response_town05_runtime/v2"
    ):
        raise ValueError(
            "unsupported runtime config schema"
        )

    map_config = data.get("map")
    if not isinstance(map_config, dict):
        raise ValueError(
            "runtime config map must be an object"
        )
    expected_map_values = {
        "name": MAP_NAME,
        "asset_path": MAP_ASSET_PATH,
        "official_carla_asset": True,
        "target_length_m": 6000.0,
        "finish_progress_m": 6000.0,
    }
    for key, expected in expected_map_values.items():
        if map_config.get(key) != expected:
            raise ValueError(
                f"runtime config map.{key} must be "
                f"{expected!r}"
            )

    route_config = map_config.get("route")
    if not isinstance(route_config, dict):
        raise ValueError("runtime config map.route must be an object")
    if float(route_config.get("target_length_m", 0.0)) != 6000.0:
        raise ValueError("Town05 Scene 3 route must be exactly 6 km")
    for key in ("start_spawn_index", "turnaround_spawn_index"):
        value = route_config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"runtime config map.route.{key} is invalid")

    events = data.get("events")
    if not isinstance(events, list):
        raise ValueError(
            "runtime config events must be an array"
        )
    if len(events) != 7:
        raise ValueError(
            "runtime config must declare 7 events"
        )

    event_ids: set[str] = set()
    previous_distance = -math.inf
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            raise ValueError(
                f"event {index} must be an object"
            )
        event_id = event.get("id")
        if (
            not isinstance(event_id, str)
            or not event_id
            or event_id in event_ids
        ):
            raise ValueError(
                f"event {index} has an invalid or "
                "duplicate id"
            )
        event_ids.add(event_id)
        if event.get("order") != index:
            raise ValueError(
                f"event {event_id} order must be "
                f"{index}"
            )
        for key in (
            "distance_m",
            "activate_at_m",
            "resolve_after_m",
        ):
            value = event.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise ValueError(
                    f"event {event_id}.{key} must "
                    "be numeric"
                )
        distance = float(event["distance_m"])
        if distance <= previous_distance:
            raise ValueError(
                "event distances must be strictly "
                "increasing"
            )
        if not (
            0.0
            <= float(event["activate_at_m"])
            < distance
            < float(event["resolve_after_m"])
            <= float(map_config["target_length_m"])
        ):
            raise ValueError(
                f"event {event_id} has an invalid "
                "activation or resolution window"
            )
        safety = event.get("safety")
        if (
            not isinstance(safety, dict)
            or safety.get("recoverable") is not True
        ):
            raise ValueError(
                f"event {event_id} must be "
                "recoverable"
            )
        previous_distance = distance

    validate_event_ground_truth_contracts(events)

    traffic = data.get("traffic")
    expected_counts = {
        "private_vehicle_count": 16,
        "work_vehicle_count": 2,
        "maintenance_vehicle_count": 1,
        "worker_count": 2,
    }
    if not isinstance(traffic, dict):
        raise ValueError(
            "runtime config traffic must be an object"
        )
    for key, expected in expected_counts.items():
        if traffic.get(key) != expected:
            raise ValueError(
                f"runtime config traffic.{key} "
                f"must be {expected}"
            )

    camera_names = (
        data.get("sensors", {})
        .get("required_camera_names")
    )
    if camera_names != list(CAMERA_TRANSFORMS):
        raise ValueError(
            "runtime config must require the "
            "front/left/right/rear RGB cameras"
        )

    interfaces = data.get("decision_interfaces")
    if not isinstance(interfaces, dict):
        raise ValueError("runtime config decision_interfaces must be an object")
    if interfaces.get("simulation_frame_is_sync_key") is not True:
        raise ValueError("Scene 3 decision inputs must share the simulation frame")
    if interfaces.get("carla_truth_allowed_for_policy") is not False:
        raise ValueError("CARLA truth must not be exposed to the Scene 3 policy")
    if interfaces.get("carla_truth_allowed_for_evaluation") is not True:
        raise ValueError("CARLA truth must remain available to offline evaluation")
    if float(interfaces.get("maximum_source_skew_ms", 0.0)) <= 0.0:
        raise ValueError("maximum_source_skew_ms must be positive")
    if float(interfaces.get("maximum_decision_latency_ms", 0.0)) <= 0.0:
        raise ValueError("maximum_decision_latency_ms must be positive")
    if float(interfaces["maximum_decision_latency_ms"]) > 120.0:
        raise ValueError("Scene 3 emergency response latency budget exceeds 120 ms")

    voice_input = data.get("voice_input")
    if not isinstance(voice_input, dict):
        raise ValueError("runtime config voice_input must be an object")
    commands = voice_input.get("commands")
    if not isinstance(commands, list) or len(commands) < 6:
        raise ValueError("Scene 3 requires ambiguous/emergency voice commands")
    command_ids = {str(item.get("id")) for item in commands if isinstance(item, dict)}
    referenced_ids = {str(event.get("voice_command_id")) for event in events}
    if not referenced_ids.issubset(command_ids):
        missing = sorted(referenced_ids - command_ids)
        raise ValueError("voice command definitions missing: " + ", ".join(missing))
    noise = voice_input.get("environment_noise", {})
    if noise.get("enabled") is not True or float(noise.get("snr_db", 99.0)) > 25.0:
        raise ValueError("Scene 3 voice input must include light environment noise")

    success = data.get("success_conditions", {})
    if float(success.get("maximum_emergency_response_latency_ms", 999.0)) > 120.0:
        raise ValueError("success latency must be no greater than 120 ms")
    if float(success.get("minimum_multimodal_semantic_alignment_accuracy", 0.0)) < 0.97:
        raise ValueError("semantic alignment target must be at least 0.97")

    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="CARLA server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA RPC port",
    )
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        default=8000,
        help="CARLA Traffic Manager port",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG_PATH,
        help=(
            "Scene 3 event and safety runtime "
            "configuration"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Scene artifacts and RGB image directory",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help=(
            "Simulation duration in seconds; "
            "use 0 to run until Ctrl+C"
        ),
    )
    parser.add_argument(
        "--fixed-delta-seconds",
        type=float,
        default=0.05,
        help="Synchronous simulation time step",
    )
    parser.add_argument(
        "--camera-tick",
        type=float,
        default=1.0,
        help="Seconds between saved RGB frames",
    )
    parser.add_argument(
        "--camera-mode",
        choices=(
            "four-view",
            "chase-only",
            "four-view-plus-chase",
        ),
        default="four-view",
        help=(
            "Camera recording layout; chase-only "
            "is intended for presentation video"
        ),
    )
    parser.add_argument(
        "--presentation-lighting",
        choices=(
            "official-rainy-night",
            "rainy-daylight",
            "clear-daylight",
        ),
        default="official-rainy-night",
        help=(
            "Lighting profile. clear-daylight is the "
            "recommended presentation profile; the "
            "default preserves the official rainy-night "
            "contract"
        ),
    )
    parser.add_argument(
        "--draw-presentation-lane-markings",
        action="store_true",
        help=(
            "Draw optional debug lane markings. "
            "Disabled by default because debug paint "
            "can bloom in recorded video"
        ),
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=1280,
        help="RGB image width",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=720,
        help="RGB image height",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Horizontal camera field of view",
    )
    parser.add_argument(
        "--record-images",
        action="store_true",
        help=(
            "Save synchronized front RGB frames in "
            "addition to any direct video"
        ),
    )
    parser.add_argument(
        "--record-ground-truth",
        action="store_true",
        help=(
            "Write independent exact-frame CARLA truth to "
            "frame_ground_truth.jsonl."
        ),
    )
    parser.add_argument(
        "--ground-truth-every-n",
        type=int,
        default=1,
        help="Record one ground-truth row every N simulation frames.",
    )
    parser.add_argument(
        "--record-every-n",
        type=int,
        default=1,
        help=(
            "Record one synchronized front frame "
            "every N simulation frames"
        ),
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1920,
        help="Direct front-camera/video width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=1080,
        help="Direct front-camera/video height",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        help=(
            "Write chase RGB directly to an H.264 MP4 "
            "through ffmpeg"
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=30.0,
        help="Direct H.264 video playback frame rate",
    )
    parser.add_argument(
        "--ffmpeg",
        help="Optional path to the ffmpeg executable",
    )
    parser.add_argument(
        "--video-overlay",
        action="store_true",
        help=(
            "Draw the experiment HUD on each frame "
            "before sending it to ffmpeg"
        ),
    )
    parser.add_argument(
        "--terminal-hold-s",
        type=float,
        default=2.0,
        help=(
            "Seconds to hold the final encoded video "
            "frame"
        ),
    )
    parser.add_argument(
        "--ego-speed-kmh",
        type=float,
        default=40.0,
        help="Preview ego target speed",
    )
    parser.add_argument(
        "--ego-controller",
        choices=(
            "route-pid",
            "behavior-agent",
            "external",
        ),
        default="route-pid",
        help=(
            "Ego control backend. route-pid follows the checked-in "
            "Scene 3 plan deterministically; behavior-agent is the CARLA "
            "baseline; external leaves role_name=hero available to a model "
            "or ROS/CARLA client."
        ),
    )
    parser.add_argument(
        "--stationary-ego",
        action="store_true",
        help="Keep the ego stationary for camera checks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
        help="Traffic Manager deterministic seed",
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help=(
            "Validate the runtime config without "
            "connecting to CARLA"
        ),
    )
    parser.add_argument(
        "--require-complete-scene",
        action="store_true",
        help=(
            "Fail unless the route and all 7 events "
            "finish without a collision or invalid "
            "lane occupancy"
        ),
    )
    return parser


def write_voice_command_schedule(
    output_path: Path,
    voice_config: dict[str, Any],
) -> None:
    """Materialize the ASR/audio-injection contract used by the preview run."""

    noise = dict(voice_config["environment_noise"])
    rows = []
    for command in voice_config["commands"]:
        rows.append(
            {
                "schema_version": "scene3_voice_command/1.0",
                "command_id": command["id"],
                "trigger_progress_m": float(command["trigger_progress_m"]),
                "text": command["text"],
                "semantic_goal": list(command["semantic_goal"]),
                "audio_injection": {
                    "enabled": bool(noise["enabled"]),
                    "noise_type": noise["type"],
                    "snr_db": float(noise["snr_db"]),
                    "maximum_peak_dbfs": float(noise["maximum_peak_dbfs"]),
                },
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.duration < 0.0:
        raise ValueError(
            "--duration must be non-negative"
        )
    if args.ground_truth_every_n < 1:
        raise ValueError(
            "--ground-truth-every-n must be at least 1"
        )
    if args.fixed_delta_seconds <= 0.0:
        raise ValueError(
            "--fixed-delta-seconds must be positive"
        )
    if args.camera_tick <= 0.0:
        raise ValueError(
            "--camera-tick must be positive"
        )
    if args.image_width <= 0:
        raise ValueError(
            "--image-width must be positive"
        )
    if args.image_height <= 0:
        raise ValueError(
            "--image-height must be positive"
        )
    if args.record_every_n <= 0:
        raise ValueError(
            "--record-every-n must be positive"
        )
    if args.camera_width <= 0:
        raise ValueError(
            "--camera-width must be positive"
        )
    if args.camera_height <= 0:
        raise ValueError(
            "--camera-height must be positive"
        )
    if args.video_fps <= 0.0:
        raise ValueError(
            "--video-fps must be positive"
        )
    if args.terminal_hold_s < 0.0:
        raise ValueError(
            "--terminal-hold-s must be non-negative"
        )
    if not 1.0 <= args.fov <= 179.0:
        raise ValueError(
            "--fov must be in [1, 179]"
        )
    if args.ego_speed_kmh <= 0.0:
        raise ValueError(
            "--ego-speed-kmh must be positive"
        )
    if (
        args.presentation_lighting
        != "official-rainy-night"
        and args.camera_mode == "four-view"
        and args.video_output is None
        and not args.record_images
    ):
        raise ValueError(
            "Presentation lighting "
            "requires chase-only or "
            "four-view-plus-chase"
        )


def first_available_blueprint(
    library: Any,
    blueprint_ids: Sequence[str],
) -> Any:
    for blueprint_id in blueprint_ids:
        try:
            return library.find(blueprint_id)
        except RuntimeError:
            continue
    raise RuntimeError(
        "No configured ego vehicle blueprint "
        "is available"
    )


def apply_emergency_weather(
    world: Any,
    weather_config: dict[str, Any],
    *,
    profile: str = "official-rainy-night",
) -> None:
    effective_config = dict(weather_config)
    if profile == "rainy-daylight":
        effective_config.update(
            PRESENTATION_WEATHER
        )
    elif profile == "clear-daylight":
        effective_config.update(
            CLEAR_PRESENTATION_WEATHER
        )
    world.set_weather(
        carla.WeatherParameters(
            cloudiness=float(
                effective_config["cloudiness"]
            ),
            precipitation=float(
                effective_config["precipitation"]
            ),
            precipitation_deposits=float(
                effective_config[
                    "precipitation_deposits"
                ]
            ),
            wind_intensity=float(
                effective_config["wind_intensity"]
            ),
            sun_azimuth_angle=float(
                effective_config.get("sun_azimuth_angle", 280.0)
            ),
            sun_altitude_angle=float(
                effective_config[
                    "sun_altitude_angle"
                ]
            ),
            fog_density=float(
                effective_config["fog_density"]
            ),
            fog_distance=float(
                effective_config["fog_distance"]
            ),
            fog_falloff=0.2,
            wetness=float(
                effective_config["wetness"]
            ),
            scattering_intensity=1.0,
            mie_scattering_scale=0.8,
            rayleigh_scattering_scale=0.02,
        )
    )
    if profile == "rainy-daylight":
        print(
            "Weather: presentation rainy daylight, "
            "wet road, fog density 8"
        )
    elif profile == "clear-daylight":
        print(
            "Weather: presentation clear daylight, "
            "dry road, fog density 0"
        )
    else:
        print(
            "Weather: rainy night, wet road, "
            "fog density 35"
        )


def activate_town05_night_lighting(world: Any) -> int:
    """Turn on official-map street and building lights for the night scene."""

    manager = world.get_lightmanager()
    groups = []
    light_group = getattr(carla, "LightGroup", None)
    if light_group is not None:
        for name in ("Street", "Building", "Other"):
            value = getattr(light_group, name, None)
            if value is not None:
                groups.append(value)
    lights: list[Any] = []
    for group in groups:
        lights.extend(manager.get_all_lights(group))
    unique = {int(light.id): light for light in lights}
    if unique:
        manager.turn_on(list(unique.values()))
        manager.set_intensity(list(unique.values()), 65.0)
    print(f"Town05 night lights enabled: {len(unique)}")
    return len(unique)


def spawn_wet_surface_friction_triggers(
    world: Any,
    route_context: Town05RouteContext,
    surface_config: dict[str, Any],
) -> list[Any]:
    """Apply configured wet-road friction to the actual Town05 road surface."""

    trigger_config = surface_config.get("friction_triggers", {})
    if not isinstance(trigger_config, dict) or trigger_config.get("enabled") is not True:
        return []
    spacing_m = float(trigger_config["spacing_m"])
    if spacing_m <= 0.0:
        raise ValueError("friction trigger spacing_m must be positive")
    library = world.get_blueprint_library()
    blueprint = library.find("static.trigger.friction")
    attributes = {
        "friction": trigger_config["friction"],
        "extent_x": trigger_config["extent_x_m"],
        "extent_y": trigger_config["extent_y_m"],
        "extent_z": trigger_config["extent_z_m"],
    }
    for name, value in attributes.items():
        if not blueprint.has_attribute(name):
            raise RuntimeError(
                "CARLA static.trigger.friction does not expose required "
                f"attribute {name!r}; wet-road physics cannot be guaranteed"
            )
        blueprint.set_attribute(name, str(value))

    actors: list[Any] = []
    seen_physical_tiles: set[tuple[int, int]] = set()
    progress_m = 0.0
    while progress_m <= route_context.length_m:
        waypoint = route_context.adapter.route_waypoint(progress_m)
        location = waypoint.transform.location
        tile = (round(float(location.x) / 8.0), round(float(location.y) / 8.0))
        if tile not in seen_physical_tiles:
            seen_physical_tiles.add(tile)
            base = waypoint.transform
            transform = carla.Transform(
                carla.Location(
                    x=base.location.x,
                    y=base.location.y,
                    z=base.location.z + 0.2,
                ),
                carla.Rotation(
                    pitch=base.rotation.pitch,
                    yaw=base.rotation.yaw,
                    roll=base.rotation.roll,
                ),
            )
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is None:
                raise RuntimeError(
                    f"failed to apply wet friction near route {progress_m:.1f} m"
                )
            actors.append(actor)
        progress_m += spacing_m
    if not actors:
        raise RuntimeError("wet friction is enabled but no trigger was spawned")
    print(
        "Wet-road friction triggers: "
        f"count={len(actors)}, friction={float(trigger_config['friction']):.2f}"
    )
    return actors


def spawn_ego(
    world: Any,
    carla_map: Any,
    *,
    traffic_manager: Any,
    traffic_manager_port: int,
    target_speed_kmh: float,
    stationary: bool,
    autopilot: bool = True,
    spawn_waypoint: Any | None = None,
    color: str = "30,30,30",
    lights_enabled: bool = True,
    controller_label: str = "route-pid",
) -> Any:
    waypoint = spawn_waypoint or carla_map.get_waypoint_xodr(
        ROAD_ID, EGO_LANE_ID, EGO_START_S_M
    )
    if waypoint is None:
        raise RuntimeError(
            "Emergency-road ego waypoint is missing"
        )

    transform = waypoint.transform
    transform.location.z += 0.5
    library = world.get_blueprint_library()
    blueprint = first_available_blueprint(
        library,
        EGO_BLUEPRINT_IDS,
    )
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute(
            "role_name",
            "hero",
        )
    if blueprint.has_attribute("color"):
        blueprint.set_attribute(
            "color",
            color,
        )

    ego = world.try_spawn_actor(
        blueprint,
        transform,
    )
    if ego is None:
        raise RuntimeError(
            "Failed to spawn the emergency-scene ego"
        )

    light_state = 0
    if lights_enabled:
        light_state = (
            carla.VehicleLightState.Position
            | carla.VehicleLightState.LowBeam
            | carla.VehicleLightState.Fog
        )
    ego.set_light_state(
        carla.VehicleLightState(light_state)
    )

    if stationary:
        ego.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=True,
            )
        )
    elif autopilot:
        ego.set_autopilot(
            True,
            traffic_manager_port,
        )
        traffic_manager.set_desired_speed(
            ego,
            target_speed_kmh,
        )
        # Keep the ego in the declared middle lane through the
        # right-lane work zone.  The blocked-lane event later issues
        # one explicit, gap-checked change to the left lane.
        traffic_manager.auto_lane_change(
            ego,
            False,
        )
        traffic_manager.update_vehicle_lights(
            ego,
            lights_enabled,
        )

    print(
        "Ego spawned: "
        f"road={ROAD_ID}, lane={EGO_LANE_ID}, "
        f"s={EGO_START_S_M:.0f} m"
    )
    print(
        "Ego mode:",
        (
            "stationary"
            if stationary
            else (
                f"autopilot {target_speed_kmh:.1f} km/h"
                if autopilot
                else f"{controller_label} {target_speed_kmh:.1f} km/h"
            )
        ),
    )
    return ego


def build_ego_route_plan(
    route_context: Any,
    events: Sequence[dict[str, Any]],
) -> list[tuple[Any, Any]]:
    """Build the drivable lane plan for the seven configured events."""

    work_zone = next(
        event
        for event in events
        if event["scenario"] == "right_lane_work_zone"
    )
    taper = next(
        event
        for event in events
        if event["scenario"] == "progressive_lane_closure"
    )
    blockage = next(
        event
        for event in events
        if event["scenario"] == "maintenance_vehicle_blockage"
    )
    recovery = next(
        event
        for event in events
        if event["scenario"] == "work_zone_exit"
    )
    cut_in = next(
        event
        for event in events
        if event["scenario"] == "cut_in_vehicle"
    )
    warning = next(
        event
        for event in events
        if event["scenario"] == "work_zone_advance_warning"
    )
    centre_windows = (
        (
            float(cut_in["activate_at_m"]) - 100.0,
            float(cut_in["resolve_after_m"]) + 100.0,
        ),
        (
            float(warning["activate_at_m"]) - 50.0,
            float(work_zone["resolve_after_m"]) + 100.0,
        ),
        (
            float(blockage["activate_at_m"]) - 50.0,
            float(recovery["resolve_after_m"]) + 100.0,
        ),
    )
    lane_windows = (
        (
            float(taper["activate_at_m"]),
            float(work_zone["resolve_after_m"]),
            int(taper["closure"]["merge_target_lane_id"]),
        ),
        (
            float(blockage["distance_m"]),
            float(recovery["resolve_after_m"]),
            int(blockage["blockage"]["target_lane_id"]),
        ),
    )
    plan: list[tuple[Any, Any]] = []
    for (waypoint, _road_option), progress_m in zip(
        route_context.route,
        route_context.distances_m,
    ):
        desired_lane_id: int | None = None
        for start_m, end_m in centre_windows:
            if start_m <= float(progress_m) <= end_m:
                desired_lane_id = EGO_LANE_ID
                break
        for start_m, end_m, lane_id in lane_windows:
            if start_m <= float(progress_m) <= end_m:
                desired_lane_id = lane_id
                break
        selected_waypoint = waypoint
        if (
            desired_lane_id is not None
            and not bool(getattr(waypoint, "is_junction", False))
        ):
            selected_waypoint = (
                route_context.adapter.logical_waypoint(
                    desired_lane_id,
                    float(progress_m),
                )
                or waypoint
            )
        plan.append((selected_waypoint, _road_option))
    if len(plan) < 2:
        raise RuntimeError("Town05 ego route plan is too short")
    return plan


def _speed_kmh(actor: Any) -> float:
    velocity = actor.get_velocity()
    return 3.6 * math.sqrt(
        float(velocity.x) ** 2
        + float(velocity.y) ** 2
        + float(velocity.z) ** 2
    )


class Scene3RouteController:
    """Small deterministic controller over the shared Scene 3 route plan.

    The controller deliberately has the same ``run_step`` surface as CARLA's
    BehaviorAgent.  The runner can therefore swap this baseline for an
    external multimodal policy without changing scenario actors, sensors,
    ground truth, or evaluation artifacts.
    """

    def __init__(
        self,
        world: Any,
        ego: Any,
        route_context: Town05RouteContext,
        route_plan: Sequence[tuple[Any, Any]],
        target_speed_kmh: float,
        fixed_delta_seconds: float,
    ) -> None:
        if len(route_plan) != len(route_context.distances_m):
            raise ValueError("route plan and route distances must be aligned")
        self.world = world
        self.ego = ego
        self.route_context = route_context
        self.route_plan = list(route_plan)
        self.target_speed_kmh = float(target_speed_kmh)
        from agents.navigation.controller import VehiclePIDController

        dt = float(fixed_delta_seconds)
        self._vehicle_pid = VehiclePIDController(
            ego,
            args_lateral={
                "K_P": 1.45,
                "K_D": 0.12,
                "K_I": 0.02,
                "dt": dt,
            },
            args_longitudinal={
                "K_P": 0.9,
                "K_D": 0.05,
                "K_I": 0.04,
                "dt": dt,
            },
            max_throttle=0.60,
            max_brake=1.0,
            max_steering=0.85,
        )

    def _waypoint_ahead(self, lookahead_m: float) -> Any:
        progress_m = float(
            self.route_context.distances_m[
                self.route_context.tracker.index
            ]
        )
        target_index = bisect.bisect_left(
            self.route_context.distances_m,
            progress_m + lookahead_m,
        )
        target_index = min(target_index, len(self.route_plan) - 1)
        return self.route_plan[target_index][0]

    def _target_waypoint(self, speed_kmh: float) -> Any:
        return self._waypoint_ahead(
            min(13.0, max(5.0, 4.5 + speed_kmh * 0.17))
        )

    def _obstacle_distance_m(self) -> float | None:
        transform = self.ego.get_transform()
        origin = transform.location
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        closest: float | None = None
        for pattern in ("vehicle.*", "walker.pedestrian.*"):
            for actor in self.world.get_actors().filter(pattern):
                if int(actor.id) == int(self.ego.id):
                    continue
                try:
                    location = actor.get_location()
                except RuntimeError:
                    continue
                dx = float(location.x - origin.x)
                dy = float(location.y - origin.y)
                dz = float(location.z - origin.z)
                longitudinal = dx * float(forward.x) + dy * float(forward.y)
                lateral = abs(dx * float(right.x) + dy * float(right.y))
                if longitudinal <= 0.0 or longitudinal > 55.0:
                    continue
                if abs(dz) > 2.5 or lateral > 3.0:
                    continue
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                if closest is None or distance < closest:
                    closest = distance
        return closest

    def _must_stop_for_light(self) -> bool:
        try:
            if not self.ego.is_at_traffic_light():
                return False
            state = self.ego.get_traffic_light_state()
            return state in (
                carla.TrafficLightState.Red,
                carla.TrafficLightState.Yellow,
            )
        except (AttributeError, RuntimeError):
            return False

    def run_step(self) -> Any:
        speed_kmh = _speed_kmh(self.ego)
        target_waypoint = self._target_waypoint(speed_kmh)
        target = target_waypoint.transform.location
        transform = self.ego.get_transform()
        origin = transform.location
        yaw_rad = math.radians(float(transform.rotation.yaw))
        target_heading = math.atan2(
            float(target.y - origin.y),
            float(target.x - origin.x),
        )
        heading_error = math.atan2(
            math.sin(target_heading - yaw_rad),
            math.cos(target_heading - yaw_rad),
        )
        far_target = self._waypoint_ahead(30.0).transform.location
        far_heading = math.atan2(
            float(far_target.y - origin.y),
            float(far_target.x - origin.x),
        )
        far_heading_error = math.atan2(
            math.sin(far_heading - yaw_rad),
            math.cos(far_heading - yaw_rad),
        )

        target_speed_kmh = self.target_speed_kmh
        curve_error = max(abs(heading_error), abs(far_heading_error))
        if curve_error > 0.08:
            curve_fraction = max(0.38, 1.0 - 1.35 * curve_error)
            target_speed_kmh = min(
                target_speed_kmh,
                max(13.0, self.target_speed_kmh * curve_fraction),
            )
        obstacle_distance_m = self._obstacle_distance_m()
        if obstacle_distance_m is not None:
            speed_mps = speed_kmh / 3.6
            # Actor locations are at bounding-box centres.  Eight metres
            # preserves roughly one vehicle length of clearance even while
            # creeping at a red light or behind a stopped work vehicle.
            emergency_distance_m = 8.0 + speed_mps * 0.80
            caution_distance_m = max(
                14.0,
                emergency_distance_m + speed_mps * 1.4,
            )
            if obstacle_distance_m <= emergency_distance_m:
                target_speed_kmh = 0.0
            elif obstacle_distance_m < caution_distance_m:
                fraction = (
                    (obstacle_distance_m - emergency_distance_m)
                    / (caution_distance_m - emergency_distance_m)
                )
                target_speed_kmh = min(
                    target_speed_kmh,
                    max(4.0, self.target_speed_kmh * fraction),
                )
        if self._must_stop_for_light():
            target_speed_kmh = 0.0

        control = self._vehicle_pid.run_step(
            target_speed_kmh,
            target_waypoint,
        )
        if target_speed_kmh <= 0.1 and speed_kmh < 0.5:
            control.throttle = 0.0
            control.brake = 1.0
        control.hand_brake = False
        control.manual_gear_shift = False
        return control


def set_traffic_manager_plan(
    traffic_manager: Any,
    ego: Any,
    plan: Sequence[tuple[Any, Any]],
    distances_m: Sequence[float],
    *,
    spacing_m: float = 5.0,
) -> int:
    if spacing_m <= 0.0:
        raise ValueError("route path spacing must be positive")
    locations: list[Any] = []
    next_progress_m = spacing_m
    for (waypoint, _road_option), progress_m in zip(plan, distances_m):
        if float(progress_m) + 1e-6 < next_progress_m:
            continue
        locations.append(waypoint.transform.location)
        next_progress_m = float(progress_m) + spacing_m
    final_location = plan[-1][0].transform.location
    if not locations or locations[-1] is not final_location:
        locations.append(final_location)
    traffic_manager.set_path(ego, locations)
    print(
        "Traffic Manager route assigned: "
        f"waypoints={len(locations)}, spacing={spacing_m:.1f} m"
    )
    return len(locations)


def make_camera_callback(
    *,
    camera_name: str,
    output_dir: Path,
    capture_state: CaptureState,
) -> Any:
    camera_dir = output_dir / "rgb" / camera_name
    camera_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    pending_dir = camera_dir / ".pending"
    pending_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    def save_image(image: Any) -> None:
        path = (
            camera_dir
            / f"{image.frame:08d}.png"
        )
        pending_path = (
            pending_dir
            / f"{image.frame:08d}.png"
        )
        image.save_to_disk(str(pending_path))
        if (
            not pending_path.is_file()
            or pending_path.stat().st_size <= 0
        ):
            print(
                "WARNING: incomplete RGB frame "
                f"discarded: {image.frame}",
                file=sys.stderr,
            )
            return
        pending_path.replace(path)
        capture_state.record(camera_name)

    return save_image


def spawn_rgb_cameras(
    world: Any,
    ego: Any,
    *,
    output_dir: Path,
    image_width: int,
    image_height: int,
    fov: float,
    camera_tick: float,
    camera_mode: str,
    low_signal_config: dict[str, Any] | None = None,
) -> tuple[list[Any], CaptureState]:
    library = world.get_blueprint_library()
    camera_transforms = (
        camera_transforms_for_mode(
            camera_mode
        )
    )
    capture_state = CaptureState(
        tuple(camera_transforms),
    )
    cameras: list[Any] = []

    for camera_name, values in (
        camera_transforms.items()
    ):
        x, y, z, pitch, yaw = values
        blueprint = library.find(
            "sensor.camera.rgb"
        )
        attributes = camera_sensor_attributes(
            camera_name=camera_name,
            image_width=image_width,
            image_height=image_height,
            fov=fov,
            camera_tick=camera_tick,
            low_signal_config=low_signal_config,
        )
        for name, value in attributes.items():
            if blueprint.has_attribute(name):
                blueprint.set_attribute(
                    name,
                    value,
                )

        camera = world.spawn_actor(
            blueprint,
            carla.Transform(
                carla.Location(
                    x=x,
                    y=y,
                    z=z,
                ),
                carla.Rotation(
                    pitch=pitch,
                    yaw=yaw,
                    roll=0.0,
                ),
            ),
            attach_to=ego,
            # SpringArmGhost collapses into the road mesh on
            # generated OpenDRIVE maps.  A rigid mount preserves the
            # requested chase-camera pose.
            attachment_type=(
                carla.AttachmentType.Rigid
            ),
        )
        camera.listen(
            make_camera_callback(
                camera_name=camera_name,
                output_dir=output_dir,
                capture_state=capture_state,
            )
        )
        cameras.append(camera)

    print(
        "RGB cameras:",
        ", ".join(camera_transforms),
    )
    print(
        "RGB output:",
        output_dir / "rgb",
    )
    return cameras, capture_state


def spawn_safety_sensors(
    world: Any,
    ego: Any,
) -> tuple[list[Any], SafetyAuditState]:
    """Attach collision and lane-invasion sensors to the ego."""

    library = world.get_blueprint_library()
    audit = SafetyAuditState()
    sensors: list[Any] = []

    collision = world.spawn_actor(
        library.find("sensor.other.collision"),
        carla.Transform(),
        attach_to=ego,
    )
    collision.listen(audit.record_collision)
    sensors.append(collision)

    lane_invasion = world.spawn_actor(
        library.find("sensor.other.lane_invasion"),
        carla.Transform(),
        attach_to=ego,
    )
    lane_invasion.listen(
        audit.record_lane_invasion
    )
    sensors.append(lane_invasion)

    print(
        "Safety sensors: collision, lane_invasion"
    )
    return sensors, audit


def set_spectator_view(
    world: Any,
    ego: Any,
) -> None:
    ego_transform = ego.get_transform()
    location = ego_transform.location
    rotation = ego_transform.rotation
    world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(
                x=location.x - 12.0,
                y=location.y,
                z=7.0,
            ),
            carla.Rotation(
                pitch=-18.0,
                yaw=rotation.yaw,
                roll=0.0,
            ),
        )
    )


def destroy_actors(
    actors: Sequence[Any],
) -> None:
    for actor in reversed(actors):
        if actor is None:
            continue
        try:
            if hasattr(actor, "stop"):
                actor.stop()
        except RuntimeError:
            pass
        try:
            if actor.is_alive:
                actor.destroy()
        except RuntimeError:
            pass


def run_simulation(
    *,
    world: Any,
    carla_map: Any,
    route_context: Town05RouteContext | None = None,
    ego: Any,
    scheduler: EmergencyEventScheduler,
    safety_audit: SafetyAuditState,
    finish_s_m: float,
    duration_s: float,
    fixed_delta_seconds: float,
    video_camera: ExperimentCamera | None = None,
    cruise_speed_kmh: float = 40.0,
    ground_truth_recorder: (
        FrameGroundTruthRecorder | None
    ) = None,
    event_actor_runtime: (
        EmergencySceneActorRuntime | None
    ) = None,
    vehicle_state_recorder: VehicleStateRecorder | None = None,
    ego_controller: Any | None = None,
    ego_route_plan: Sequence[tuple[Any, Any]] | None = None,
) -> bool:
    tick_count: int | None = None
    if duration_s > 0.0:
        tick_count = int(
            math.ceil(
                duration_s
                / fixed_delta_seconds
            )
        )

    index = 0
    last_route_s_m: float | None = None
    spectator_warning_printed = False
    while (
        tick_count is None
        or index < tick_count
    ):
        if ego_controller is not None:
            control = ego_controller.run_step()
            control.manual_gear_shift = False
            ego.apply_control(control)
        frame = world.tick()
        try:
            ego_location = ego.get_location()
        except RuntimeError as error:
            ego_id = getattr(ego, "id", None)
            refreshed_ego = (
                world.get_actor(int(ego_id))
                if ego_id is not None
                else None
            )
            if refreshed_ego is None:
                route_context = (
                    "unknown"
                    if last_route_s_m is None
                    else f"{last_route_s_m:.1f} m"
                )
                raise RuntimeError(
                    "ego actor was destroyed after "
                    f"route position {route_context}"
                ) from error
            ego = refreshed_ego
            ego_location = ego.get_location()
            print(
                "EGO ACTOR HANDLE REFRESHED | "
                f"id={int(ego.id)}"
            )
        waypoint = carla_map.get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            raise RuntimeError(
                "ego is no longer on a driving lane"
            )
        route_progress_m = (
            route_context.progress(ego_location)
            if route_context is not None
            else float(waypoint.s)
        )
        if route_context is None and waypoint.road_id != ROAD_ID:
            raise RuntimeError("ego left the emergency route")
        if route_context is not None and ego_route_plan is not None:
            route_index = int(route_context.tracker.index)
            lower = max(0, route_index - 15)
            upper = min(len(ego_route_plan), route_index + 31)
            route_distance_m = min(
                math.hypot(
                    float(candidate[0].transform.location.x - ego_location.x),
                    float(candidate[0].transform.location.y - ego_location.y),
                )
                for candidate in ego_route_plan[lower:upper]
            )
            safety_audit.record_route_deviation(
                route_distance_m,
                road_id=int(waypoint.road_id),
                lane_id=int(waypoint.lane_id),
                route_progress_m=route_progress_m,
            )
        elif not bool(getattr(waypoint, "is_junction", False)):
            safety_audit.record_lane_id(
                int(waypoint.lane_id),
                (
                    route_context.adapter.legal_driving_lane_ids(
                        route_progress_m
                    )
                    if route_context is not None
                    else None
                ),
                road_id=int(waypoint.road_id),
                route_progress_m=route_progress_m,
            )
        last_route_s_m = route_progress_m
        scheduler.update(
            route_s_m=last_route_s_m,
            simulation_frame=int(frame),
            elapsed_s=(
                (index + 1)
                * fixed_delta_seconds
            ),
        )
        elapsed_s = (
            (index + 1)
            * fixed_delta_seconds
        )
        if vehicle_state_recorder is not None:
            vehicle_state_recorder.record(
                ego=ego,
                waypoint=waypoint,
                simulation_frame=int(frame),
                timestamp_s=elapsed_s,
                route_progress_m=last_route_s_m,
            )
        if ground_truth_recorder is not None:
            if event_actor_runtime is None:
                raise RuntimeError(
                    "event_actor_runtime is required "
                    "for Scene 3 ground truth"
                )
            ground_truth_recorder.record(
                world=world,
                ego=ego,
                simulation_frame=int(frame),
                timestamp_s=elapsed_s,
                route_s_m=last_route_s_m,
                event_states=(
                    scheduler.state_snapshot()
                ),
                actor_bindings=(
                    event_actor_runtime
                    .ground_truth_actor_bindings()
                ),
                runtime_state=(
                    event_actor_runtime
                    .ground_truth_runtime_state()
                ),
            )
        if video_camera is not None:
            video_camera.save_frame(
                frame,
                overlay=make_video_overlay(
                    ego=ego,
                    scheduler=scheduler,
                    safety_audit=safety_audit,
                    route_s_m=last_route_s_m,
                    finish_s_m=finish_s_m,
                    simulation_frame=int(frame),
                    elapsed_s=elapsed_s,
                    cruise_speed_kmh=(
                        cruise_speed_kmh
                    ),
                ),
            )
        if index % 20 == 0:
            try:
                set_spectator_view(world, ego)
            except RuntimeError as error:
                if not spectator_warning_printed:
                    print(
                        "WARNING: spectator view "
                        "update skipped: "
                        f"{error}"
                    )
                    spectator_warning_printed = True
        if last_route_s_m >= finish_s_m:
            print(
                "ROUTE FINISH REACHED | "
                f"s={last_route_s_m:.1f} m"
            )
            return True
        index += 1

    return False


def main(
    argv: list[str] | None = None,
) -> int:
    global carla

    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    runtime_config_path = (
        args.runtime_config.expanduser().resolve()
    )
    try:
        runtime_config = load_runtime_config(
            runtime_config_path
        )
    except ValueError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    print(
        "Runtime config:",
        runtime_config_path,
    )
    print(
        "Configured events:",
        len(runtime_config["events"]),
    )
    if args.validate_config_only:
        print(
            "SCENE 3 RUNTIME CONFIG VALIDATION: PASS"
        )
        return 0

    output_dir = args.output_dir.expanduser().resolve()

    try:
        carla = load_carla_api()
    except RuntimeError as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1

    client: Any | None = None
    world: Any | None = None
    original_settings: Any | None = None
    traffic_manager: Any | None = None
    actors: list[Any] = []
    capture_state: CaptureState | None = None
    direct_camera: ExperimentCamera | None = None
    direct_camera_closed = False
    safety_audit: SafetyAuditState | None = None
    ground_truth: FrameGroundTruthRecorder | None = None
    vehicle_state_recorder: VehicleStateRecorder | None = None
    event_actor_runtime: (
        EmergencySceneActorRuntime | None
    ) = None
    ego_controller: Any | None = None
    ego_plan: list[tuple[Any, Any]] | None = None
    route_context: Town05RouteContext | None = None
    result = 1

    try:
        client = carla.Client(
            args.host,
            args.port,
        )
        client.set_timeout(60.0)
        print(
            "CARLA server:",
            client.get_server_version(),
        )

        world = client.load_world(MAP_NAME)
        map_layer = getattr(carla, "MapLayer", None)
        if map_layer is not None and hasattr(world, "load_map_layer"):
            world.load_map_layer(map_layer.All)
            print("Town05_Opt layers: All (buildings, foliage, streetlights, props)")
        official_map = world.get_map()
        loaded_map_name = str(getattr(official_map, "name", ""))
        if not loaded_map_name.endswith(MAP_NAME):
            raise RuntimeError(
                f"CARLA loaded {loaded_map_name!r}, expected official {MAP_NAME}"
            )
        route_context = build_town05_route_context(
            official_map,
            runtime_config["map"]["route"],
        )
        validate_scene3_event_anchors(
            route_context,
            runtime_config["events"],
        )
        print(
            "Official map and route: "
            f"{loaded_map_name}, length={route_context.length_m:.1f} m"
        )

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            args.fixed_delta_seconds
        )
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager(
            args.traffic_manager_port
        )
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(
            args.seed
        )
        traffic_manager.set_global_distance_to_leading_vehicle(
            5.0
        )
        # Town05_Opt ships native CARLA lane topology; OSM mode is only for
        # imported OpenDRIVE/OSM networks and would weaken route following.
        traffic_manager.set_osm_mode(False)

        apply_emergency_weather(
            world,
            runtime_config["weather"],
            profile=args.presentation_lighting,
        )
        if args.presentation_lighting == "official-rainy-night":
            activate_town05_night_lighting(world)
        actors.extend(
            spawn_wet_surface_friction_triggers(
                world,
                route_context,
                runtime_config["surface_and_visibility"],
            )
        )
        effective_ego_speed_kmh = min(
            float(args.ego_speed_kmh),
            float(
                runtime_config["surface_and_visibility"].get(
                    "wet_speed_limit_kmh", args.ego_speed_kmh
                )
            ),
        )
        print(
            "Wet-weather ego speed cap: "
            f"requested={args.ego_speed_kmh:.1f}, "
            f"effective={effective_ego_speed_kmh:.1f} km/h"
        )
        if (
            args.draw_presentation_lane_markings
        ):
            print(
                "WARNING: --draw-presentation-lane-markings is ignored on "
                "Town05_Opt because the official map already contains native "
                "lane markings"
            )
        ego = spawn_ego(
            world,
            route_context.adapter,
            traffic_manager=traffic_manager,
            traffic_manager_port=(
                args.traffic_manager_port
            ),
            target_speed_kmh=effective_ego_speed_kmh,
            stationary=args.stationary_ego,
            autopilot=False,
            spawn_waypoint=route_context.route[0][0],
            color=(
                "40,100,180"
                if args.presentation_lighting
                != "official-rainy-night"
                else "30,30,30"
            ),
            lights_enabled=(
                args.presentation_lighting
                == "official-rainy-night"
            ),
            controller_label=args.ego_controller,
        )
        actors.append(ego)
        if not args.stationary_ego:
            ego_plan = build_ego_route_plan(
                route_context,
                runtime_config["events"],
            )
            if args.ego_controller == "route-pid":
                ego_controller = Scene3RouteController(
                    world,
                    ego,
                    route_context,
                    ego_plan,
                    effective_ego_speed_kmh,
                    args.fixed_delta_seconds,
                )
                print(
                    "Scene3 route-PID controller assigned: "
                    f"waypoints={len(ego_plan)}"
                )
            elif args.ego_controller == "behavior-agent":
                from agents.navigation.behavior_agent import BehaviorAgent

                ego_controller = BehaviorAgent(ego, behavior="normal")
                ego_controller.set_target_speed(effective_ego_speed_kmh)
                ego_controller.set_global_plan(
                    ego_plan,
                    stop_waypoint_creation=True,
                    clean_queue=True,
                )
                print(
                    "BehaviorAgent route assigned: "
                    f"waypoints={len(ego_plan)}"
                )
            else:
                print(
                    "External ego control enabled: role_name=hero; "
                    "the runner will not apply vehicle control."
                )

        direct_recording = (
            args.video_output is not None
            or args.record_images
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        write_voice_command_schedule(
            output_dir / "voice_command_schedule.jsonl",
            runtime_config["voice_input"],
        )
        vehicle_state_recorder = VehicleStateRecorder(
            output_dir / "vehicle_state.jsonl"
        )
        print("Dynamic vehicle state:", vehicle_state_recorder.path)
        if direct_recording:
            video_output = (
                str(
                    args.video_output
                    .expanduser()
                    .resolve()
                )
                if args.video_output is not None
                else None
            )
            direct_camera = ExperimentCamera(
                world,
                ego,
                str(
                    output_dir
                    / "camera_frames"
                ),
                args.record_every_n,
                args.camera_width,
                args.camera_height,
                args.record_images,
                video_output,
                args.video_fps,
                args.ffmpeg,
                args.video_overlay,
                (
                    DIRECT_PRESENTATION_CAMERA_ATTRIBUTES
                    if args.presentation_lighting
                    != "official-rainy-night"
                    else {
                        key: str(value)
                        for key, value in runtime_config["sensors"]
                        .get("low_signal_rgb", {})
                        .items()
                        if key != "enabled"
                    }
                ),
                camera_pose=(
                    CHASE_CAMERA_TRANSFORMS[
                        "chase_rgb"
                    ]
                ),
            )
            direct_camera.start()
            print(
                "RGB camera: chase_rgb "
                "(synchronized direct recorder)"
            )
            if video_output is not None:
                print(
                    "H.264 video output:",
                    video_output,
                )
        if (
            not direct_recording
            or args.camera_mode in {"four-view", "four-view-plus-chase"}
        ):
            auxiliary_camera_mode = (
                "four-view"
                if direct_recording
                else args.camera_mode
            )
            cameras, capture_state = (
                spawn_rgb_cameras(
                    world,
                    ego,
                    output_dir=output_dir,
                    image_width=args.image_width,
                    image_height=args.image_height,
                    fov=args.fov,
                    camera_tick=args.camera_tick,
                    camera_mode=auxiliary_camera_mode,
                    low_signal_config=runtime_config["sensors"].get(
                        "low_signal_rgb"
                    ),
                )
            )
            actors.extend(cameras)
        safety_sensors, safety_audit = (
            spawn_safety_sensors(
                world,
                ego,
            )
        )
        actors.extend(safety_sensors)
        set_spectator_view(world, ego)

        (
            output_dir
            / "runtime_config_snapshot.json"
        ).write_text(
            json.dumps(
                runtime_config,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        event_actor_runtime = (
            EmergencySceneActorRuntime(
                carla_module=carla,
                world=world,
                carla_map=route_context.adapter,
                traffic_manager=traffic_manager,
                traffic_manager_port=(
                    args.traffic_manager_port
                ),
                ego_actor=ego,
                actor_sink=actors,
                lights_enabled=(
                    args.presentation_lighting
                    == "official-rainy-night"
                ),
            )
        )
        event_actor_runtime.spawn_background_traffic(
            runtime_config["traffic"]
        )
        scheduler = EmergencyEventScheduler(
            runtime_config["events"],
            output_path=(
                output_dir
                / "event_timeline.jsonl"
            ),
            event_handler=event_actor_runtime,
        )
        if args.record_ground_truth:
            ground_truth = FrameGroundTruthRecorder(
                output_dir
                / "frame_ground_truth.jsonl",
                scene_id=runtime_config["scene_id"],
                events=runtime_config["events"],
                every_n_frames=(
                    args.ground_truth_every_n
                ),
            )
            print(
                "Frame ground truth:",
                ground_truth.path,
            )

        world.tick()
        route_completed = run_simulation(
            world=world,
            carla_map=route_context.adapter,
            route_context=route_context,
            ego=ego,
            scheduler=scheduler,
            safety_audit=safety_audit,
            finish_s_m=float(
                runtime_config["map"][
                    "finish_progress_m"
                ]
            ),
            duration_s=args.duration,
            fixed_delta_seconds=(
                args.fixed_delta_seconds
            ),
            video_camera=direct_camera,
            cruise_speed_kmh=effective_ego_speed_kmh,
            ground_truth_recorder=ground_truth,
            event_actor_runtime=event_actor_runtime,
            vehicle_state_recorder=vehicle_state_recorder,
            ego_controller=ego_controller,
            ego_route_plan=ego_plan,
        )

        vehicle_state_recorder.close()
        vehicle_state_recorder = None

        if direct_camera is not None:
            direct_camera.hold_last_video_frame(
                args.terminal_hold_s
            )
            direct_camera.destroy()
            direct_camera_closed = True

        event_summary = scheduler.summary()
        print("Event scheduler summary:")
        for state in (
            "PENDING",
            "ACTIVE",
            "RESOLVED",
        ):
            print(
                f"  {state.lower()}: "
                f"{event_summary[state]}"
            )
        print(
            "Event timeline:",
            output_dir / "event_timeline.jsonl",
        )

        safety_summary = safety_audit.snapshot()
        print("Safety audit:")
        print(
            "  collisions:",
            safety_summary["collision_count"],
        )
        print(
            "  lane-invasion events:",
            safety_summary[
                "lane_invasion_event_count"
            ],
        )
        print(
            "  invalid-lane samples:",
            safety_summary[
                "invalid_lane_samples"
            ],
        )

        counts: dict[str, int] = {}
        if capture_state is not None:
            capture_state.wait_for_first_frames(timeout_s=5.0)
            counts.update(capture_state.snapshot())
        if direct_camera is not None:
            counts.update({
                "chase_rgb": max(
                    direct_camera.saved_frames,
                    direct_camera.written_video_frames,
                )
            })
        print("RGB frame counts:")
        for camera_name in counts:
            print(
                f"  {camera_name}: "
                f"{counts[camera_name]}"
            )

        missing = [
            name
            for name, count in counts.items()
            if count == 0
        ]
        if missing:
            raise RuntimeError(
                "No RGB frames received from: "
                + ", ".join(missing)
            )

        complete_scene_success = (
            route_completed
            and event_summary["RESOLVED"]
            == len(runtime_config["events"])
            and safety_summary[
                "collision_count"
            ]
            == 0
            and safety_summary[
                "invalid_lane_samples"
            ]
            == 0
        )
        scene_summary_path = (
            output_dir / "scene_summary.json"
        )
        scene_summary_path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "scene_3_emergency_summary/v1"
                    ),
                    "scene_id": runtime_config[
                        "scene_id"
                    ],
                    "map": {
                        "name": MAP_NAME,
                        "asset_path": MAP_ASSET_PATH,
                        "official_carla_asset": True,
                    },
                    "route": {
                        "target_length_m": float(
                            runtime_config["map"]["target_length_m"]
                        ),
                        "generated_length_m": round(
                            float(route_context.length_m), 3
                        ),
                        "finish_progress_m": float(
                            runtime_config["map"]["finish_progress_m"]
                        ),
                        "last_progress_m": (
                            round(
                                float(
                                    route_context.distances_m[
                                        route_context.tracker.index
                                    ]
                                ),
                                3,
                            )
                        ),
                    },
                    "route_completed": (
                        route_completed
                    ),
                    "event_states": event_summary,
                    "safety": safety_summary,
                    "rgb_frame_counts": counts,
                    "direct_video": (
                        {
                            "path": str(
                                args.video_output
                                .expanduser()
                                .resolve()
                            ),
                            "codec": "H.264",
                            "fps": args.video_fps,
                            "frames": (
                                direct_camera
                                .written_video_frames
                            ),
                            "dropped_frames": (
                                direct_camera
                                .dropped_video_frames
                            ),
                            "hud_overlay": (
                                args.video_overlay
                            ),
                        }
                        if (
                            direct_camera is not None
                            and args.video_output
                            is not None
                        )
                        else None
                    ),
                    "ground_truth": (
                        ground_truth.summary()
                        if ground_truth is not None
                        else None
                    ),
                    "complete_scene_success": (
                        complete_scene_success
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "Scene summary:",
            scene_summary_path,
        )

        if (
            args.require_complete_scene
            and not complete_scene_success
        ):
            incomplete_reasons: list[str] = []
            if not route_completed:
                incomplete_reasons.append(
                    "route not completed"
                )
            if (
                event_summary["RESOLVED"]
                != len(runtime_config["events"])
            ):
                incomplete_reasons.append(
                    "not all events resolved"
                )
            if safety_summary["collision_count"]:
                incomplete_reasons.append(
                    "collision detected"
                )
            if safety_summary[
                "invalid_lane_samples"
            ]:
                incomplete_reasons.append(
                    "invalid lane occupancy"
                )
            raise RuntimeError(
                "complete scene requirement failed: "
                + ", ".join(incomplete_reasons)
            )

        if direct_camera is not None:
            print(
                "CHASE RGB DIRECT H.264 "
                "CAPTURE: PASS"
            )
        else:
            print(
                "RAINY-NIGHT "
                + (
                    "FOUR-CAMERA"
                    if args.camera_mode
                    == "four-view"
                    else args.camera_mode.upper()
                )
                + " CAPTURE: PASS"
            )
        result = 0
    except KeyboardInterrupt:
        print("Interrupted by user")
        result = 130
    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        result = 1
    finally:
        # Release navigation helpers while their actor handles are alive.
        ego_controller = None
        event_actor_runtime = None
        import gc

        gc.collect()
        if vehicle_state_recorder is not None:
            vehicle_state_recorder.close()
        if ground_truth is not None:
            ground_truth.close()
        if (
            direct_camera is not None
            and not direct_camera_closed
        ):
            try:
                direct_camera.destroy()
            except (
                OSError,
                RuntimeError,
            ) as error:
                print(
                    "WARNING: direct camera cleanup "
                    f"failed: {error}",
                    file=sys.stderr,
                )
        destroy_actors(actors)
        if traffic_manager is not None:
            try:
                traffic_manager.set_synchronous_mode(
                    False
                )
            except RuntimeError:
                pass
        if (
            world is not None
            and original_settings is not None
        ):
            try:
                world.apply_settings(
                    original_settings
                )
            except RuntimeError:
                pass
        if actors:
            print("Scene actors cleaned up")
        if __name__ == "__main__":
            # CARLA 0.9.16 may throw from Python wrapper finalizers while the
            # main frame unwinds after actors were explicitly destroyed.
            # Exit here, after flushing, to avoid touching dead actor handles.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(int(result))

    return result


if __name__ == "__main__":
    exit_status = int(main() or 0)

    # Actors and sensors have already been explicitly cleaned up. Avoid the
    # CARLA 0.9.16 libcarla finalization abort observed under Python 3.11.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_status)
