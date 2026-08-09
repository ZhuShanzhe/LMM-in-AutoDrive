"""Universal online text + raw multiview + state VLA controller.

This is the single formal online controller entry point for every CARLA
scene.  The chain is fixed:

  UnifiedSensorBatch
    -> Universal VLA Pipeline
    -> Generic Temporal Risk Supervisor
    -> Generic Instruction FSM
    -> Route PID
    -> carla.VehicleControl

The controller never branches on scene ids, event ids or command ids.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from carla_multiview_sensor import CAMERA_ORDER, SynchronizedMultiviewCameraRig
from control.generic_instruction_fsm import GenericInstructionFSM
from control.generic_temporal_risk_supervisor import (
    GenericTemporalRiskSupervisor,
    TemporalRiskSupervisorConfig,
)
from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
from lightweight_vla_adapter.src.unified_sensor_batch import (
    CAMERA_VIEW_NAMES,
    UNIFIED_SENSOR_BATCH_SCHEMA_VERSION,
    UnifiedSensorBatch,
    default_modality_mask,
)
from structured_command_parser.src.modernbert_service import ModernBertCommandService


def _speed_mps(actor: Any) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(
        float(velocity.x) ** 2
        + float(velocity.y) ** 2
        + float(velocity.z) ** 2
    )


def filter_forward_radar_to_route_corridor(
    radar_observation: Mapping[str, Any],
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw_deg: float,
    route_polyline: Sequence[Sequence[float]],
    corridor_half_width_m: float,
    sensor_offset_m: float = 2.0,
) -> dict[str, Any]:
    """Keep only physical radar returns intersecting the planned corridor.

    A narrow forward cone can hit the barrier straight ahead while the route
    bends safely away at a junction.  Each binned physical return is projected
    into world coordinates and compared with the planner-owned route polyline.
    Straight-lane vehicles stay inside the swept corridor; outside barriers
    are retained only as audit fields and cannot trigger longitudinal braking.
    Missing binned detections fail closed by returning the original aggregate.
    """

    result = dict(radar_observation)
    bins = radar_observation.get("azimuth_obstacle_bins")
    try:
        points = [
            (float(point[0]), float(point[1]))
            for point in route_polyline
        ]
    except (TypeError, ValueError, IndexError):
        return result
    if not isinstance(bins, list) or len(points) < 2:
        return result

    yaw_rad = math.radians(float(ego_yaw_deg))
    sensor_x = float(ego_x) + float(sensor_offset_m) * math.cos(yaw_rad)
    sensor_y = float(ego_y) + float(sensor_offset_m) * math.sin(yaw_rad)
    half_width_m = max(0.5, float(corridor_half_width_m))

    def distance_to_segment(
        point_x: float,
        point_y: float,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        norm = dx * dx + dy * dy
        if norm <= 1e-9:
            return math.hypot(point_x - start[0], point_y - start[1])
        fraction = (
            (point_x - start[0]) * dx + (point_y - start[1]) * dy
        ) / norm
        fraction = max(0.0, min(1.0, fraction))
        return math.hypot(
            point_x - (start[0] + fraction * dx),
            point_y - (start[1] + fraction * dy),
        )

    relevant: list[dict[str, Any]] = []
    for raw in bins:
        if not isinstance(raw, Mapping):
            continue
        try:
            distance_m = float(raw["distance_m"])
            azimuth_rad = math.radians(float(raw["azimuth_deg"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(distance_m) or distance_m <= 0.0:
            continue
        point_x = sensor_x + distance_m * math.cos(yaw_rad + azimuth_rad)
        point_y = sensor_y + distance_m * math.sin(yaw_rad + azimuth_rad)
        lateral_distance_m = min(
            distance_to_segment(point_x, point_y, start, end)
            for start, end in zip(points, points[1:])
        )
        if lateral_distance_m <= half_width_m:
            item = dict(raw)
            item["route_lateral_distance_m"] = round(
                lateral_distance_m, 3
            )
            relevant.append(item)

    nearest = min(
        relevant,
        key=lambda item: float(item["distance_m"]),
        default=None,
    )
    closing = [
        item
        for item in relevant
        if float(item.get("closing_speed_mps", 0.0) or 0.0) > 0.5
    ]
    nearest_closing = min(
        closing,
        key=lambda item: float(item["distance_m"]),
        default=None,
    )
    result.update(
        route_corridor_filter_applied=True,
        route_corridor_half_width_m=round(half_width_m, 3),
        route_corridor_obstacle_bins=relevant,
        unfiltered_obstacle_candidate_count=radar_observation.get(
            "obstacle_candidate_count"
        ),
        unfiltered_nearest_distance_m=radar_observation.get(
            "nearest_distance_m"
        ),
        unfiltered_nearest_azimuth_deg=radar_observation.get(
            "nearest_azimuth_deg"
        ),
        obstacle_candidate_count=len(relevant),
        closing_candidate_count=len(closing),
        nearest_distance_m=(
            round(float(nearest["distance_m"]), 3) if nearest else None
        ),
        nearest_relative_velocity_mps=(
            round(float(nearest.get("relative_velocity_mps", 0.0)), 3)
            if nearest else None
        ),
        nearest_azimuth_deg=(
            round(float(nearest["azimuth_deg"]), 3) if nearest else None
        ),
        nearest_closing_distance_m=(
            round(float(nearest_closing["distance_m"]), 3)
            if nearest_closing else None
        ),
        nearest_closing_velocity_mps=(
            round(float(nearest_closing["closing_speed_mps"]), 3)
            if nearest_closing else None
        ),
    )
    return result


def fuse_forward_radar_risk(
    learned_risk: Mapping[str, Any],
    radar_observation: Mapping[str, Any],
    *,
    ego_speed_kmh: float,
) -> dict[str, Any]:
    """Fuse a physical forward-radar safety envelope with learned risk.

    This is a scene-agnostic safety cage.  It uses only the ego speed and a
    narrow forward physical-radar return; no actor truth, scenario ids, event
    ids or scheduled distances are accepted by the function.
    """

    result = dict(learned_risk)
    speed_mps = max(0.0, float(ego_speed_kmh) / 3.6)
    # Emergency envelope: standstill buffer + controller/sensor reaction
    # travel + braking distance at 6 m/s^2.  The six-metre floor keeps a
    # stopped vehicle latched instead of allowing the liveness gate to crawl
    # back into a close obstacle.
    emergency_distance_m = max(
        6.0,
        1.8 + speed_mps * 0.65 + speed_mps * speed_mps / (2.0 * 6.0),
    )
    # Earlier comfortable-deceleration envelope at 3.5 m/s^2.
    caution_distance_m = max(
        12.0,
        2.5 + speed_mps + speed_mps * speed_mps / (2.0 * 3.5),
    )
    result["physical_forward_radar"] = {
        **dict(radar_observation),
        "emergency_distance_m": round(emergency_distance_m, 3),
        "caution_distance_m": round(caution_distance_m, 3),
    }
    learned_probabilities = dict(result.get("probabilities") or {})
    if learned_probabilities:
        result["learned_probabilities"] = learned_probabilities
    reasons = list(result.get("reason_codes") or [])
    # Preserve an explicit no-obstacle snapshot for the temporal supervisor.
    # Without the thresholds and sensor frame, a learned visual false positive
    # cannot be cleared even after multiple physical empty-road frames.
    distance_value = radar_observation.get("nearest_distance_m")
    if not isinstance(distance_value, (int, float)):
        return result
    distance_m = float(distance_value)
    if not math.isfinite(distance_m) or distance_m <= 0.0:
        return result

    if distance_m <= emergency_distance_m:
        reasons.append("physical_forward_radar_emergency_distance")
        result.update(
            risk_level="high",
            raw_argmax_level=str(result.get("raw_argmax_level", "low")),
            recommended_action="emergency_brake",
            reason_codes=list(dict.fromkeys(reasons)),
            source="learned_visual_risk+physical_forward_radar",
            probabilities={"low": 0.0, "medium": 0.0, "high": 1.0},
            risk_score=1.0,
            horizon_probabilities=[1.0, 1.0, 0.0],
        )
        return result

    learned_level = str(result.get("risk_level", "low")).lower()
    if distance_m <= caution_distance_m and learned_level == "low":
        reasons.append("physical_forward_radar_caution_distance")
        result.update(
            risk_level="medium",
            raw_argmax_level=str(result.get("raw_argmax_level", "low")),
            recommended_action="decelerate",
            reason_codes=list(dict.fromkeys(reasons)),
            source="learned_visual_risk+physical_forward_radar",
            probabilities={"low": 0.0, "medium": 1.0, "high": 0.0},
            risk_score=max(float(result.get("risk_score", 0.0) or 0.0), 0.5),
        )
    return result


def assess_rear_radar_collision(
    radar_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify a rapidly closing rear radar return using physical TTC."""

    distance_value = radar_observation.get("nearest_closing_distance_m")
    closing_value = radar_observation.get("nearest_closing_velocity_mps")
    if not isinstance(distance_value, (int, float)):
        nearest_velocity = radar_observation.get(
            "nearest_relative_velocity_mps"
        )
        if isinstance(nearest_velocity, (int, float)) and float(
            nearest_velocity
        ) < -0.5:
            distance_value = radar_observation.get("nearest_distance_m")
            closing_value = -float(nearest_velocity)
    assessment = {
        "schema_version": "rear_collision_assessment/1.0",
        "collision_risk": False,
        "distance_m": None,
        "closing_speed_mps": None,
        "ttc_s": None,
        "reason": "no_closing_rear_return",
    }
    if not all(
        isinstance(value, (int, float))
        for value in (distance_value, closing_value)
    ):
        return assessment
    distance_m = float(distance_value)
    closing_speed_mps = float(closing_value)
    if (
        not math.isfinite(distance_m)
        or not math.isfinite(closing_speed_mps)
        or distance_m <= 0.0
        or closing_speed_mps <= 0.0
    ):
        return assessment
    ttc_s = distance_m / closing_speed_mps
    collision_risk = (
        closing_speed_mps >= 2.0
        and distance_m <= 30.0
        and ttc_s <= 3.0
    ) or (
        closing_speed_mps >= 1.0
        and distance_m <= 6.0
        and ttc_s <= 4.0
    )
    assessment.update(
        collision_risk=collision_risk,
        distance_m=round(distance_m, 3),
        closing_speed_mps=round(closing_speed_mps, 3),
        ttc_s=round(ttc_s, 3),
        reason=(
            "rear_radar_ttc_collision_risk"
            if collision_risk
            else "rear_radar_outside_collision_envelope"
        ),
    )
    return assessment


def apply_directional_collision_response(
    final_decision: Mapping[str, Any],
    *,
    front_risk: Mapping[str, Any] | None,
    forward_radar: Mapping[str, Any],
    rear_radar: Mapping[str, Any],
    ego_speed_kmh: float,
    road_speed_limit_kmh: float,
    route_speed_cap_kmh: float,
    lane_options: Mapping[str, Mapping[str, Any]] | None = None,
    allow_evasive_motion: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Apply front-first, speed-limited rear-collision avoidance.

    Front risk always wins.  A rear TTC threat may raise speed only below
    both the map speed limit and the route controller cap.  At that cap it
    may select an adjacent lane only when static-map legality and a learned
    side-view risk probe both pass.
    """

    decision = dict(final_decision)
    rear = assess_rear_radar_collision(rear_radar)
    assessment: dict[str, Any] = {
        "schema_version": "directional_collision_response/1.0",
        "front_priority": True,
        "rear": rear,
        "front_risk_level": str(
            (front_risk or {}).get("risk_level", "unknown")
        ).lower(),
        "selected_response": "none",
    }
    if not rear["collision_risk"]:
        return decision, assessment, None

    front_level = assessment["front_risk_level"]
    front_distance = forward_radar.get("nearest_distance_m")
    front_caution = forward_radar.get("caution_distance_m")
    physical_front_conflict = all(
        isinstance(value, (int, float))
        for value in (front_distance, front_caution)
    ) and float(front_distance) <= float(front_caution)
    if front_level in {"medium", "high"} or physical_front_conflict:
        assessment["selected_response"] = "front_brake_priority"
        return decision, assessment, None
    if not allow_evasive_motion:
        assessment["selected_response"] = "text_hazard_stop_priority"
        return decision, assessment, None

    caps = [
        float(value)
        for value in (road_speed_limit_kmh, route_speed_cap_kmh)
        if isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    ]
    if not caps:
        assessment["selected_response"] = "missing_speed_limit"
        return decision, assessment, None
    speed_cap_kmh = min(caps)
    assessment["speed_cap_kmh"] = round(speed_cap_kmh, 3)
    if float(ego_speed_kmh) < speed_cap_kmh - 1.5:
        target_speed_kmh = min(
            speed_cap_kmh,
            max(
                float(ego_speed_kmh) + 8.0,
                float(decision.get("target_speed_kmh", 0.0) or 0.0),
            ),
        )
        decision.update(
            action="accelerate",
            target_speed_kmh=target_speed_kmh,
            target_lane=None,
            emergency=False,
            reason="physical_rear_radar_acceleration_escape",
            blocked_reason_codes=[],
        )
        assessment["selected_response"] = "accelerate_within_speed_limit"
        return decision, assessment, "physical_rear_radar_acceleration_escape"

    safe_lanes: list[tuple[float, str]] = []
    for direction in ("left", "right"):
        option = dict((lane_options or {}).get(direction) or {})
        side_risk = dict(option.get("risk") or {})
        if not bool(option.get("legal", False)):
            continue
        if str(side_risk.get("risk_level", "high")).lower() != "low":
            continue
        high_probability = float(
            (side_risk.get("probabilities") or {}).get("high", 0.0) or 0.0
        )
        safe_lanes.append((high_probability, direction))
    if safe_lanes:
        _probability, direction = min(safe_lanes)
        override = f"physical_rear_radar_lane_change_{direction}_escape"
        decision.update(
            action=f"lane_change_{direction}",
            target_speed_kmh=speed_cap_kmh,
            target_lane=direction,
            emergency=False,
            reason=override,
            blocked_reason_codes=[],
        )
        assessment["selected_response"] = f"safe_{direction}_lane_change"
        return decision, assessment, override

    assessment["selected_response"] = "hold_no_legal_clear_escape"
    return decision, assessment, None


def build_sensor_policy_state(
    world: Any,
    ego: Any,
    *,
    frame_id: str,
) -> dict[str, Any]:
    """Build the policy state without enumerating CARLA actors or objects."""

    ego_acceleration = ego.get_acceleration()
    ego_angular = ego.get_angular_velocity()
    control = ego.get_control()
    snapshot = world.get_snapshot()
    map_waypoint = world.get_map().get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=__import__("carla").LaneType.Driving,
    )
    return {
        "schema_version": "unified_sensor_policy_state/1.0",
        "frame_id": frame_id,
        "timestamp_s": float(snapshot.timestamp.elapsed_seconds),
        "ego": {
            "speed_mps": _speed_mps(ego),
            "acceleration_mps2": math.sqrt(
                float(ego_acceleration.x) ** 2
                + float(ego_acceleration.y) ** 2
                + float(ego_acceleration.z) ** 2
            ),
            "yaw_rate_rps": math.radians(float(ego_angular.z)),
            "speed_limit_mps": float(ego.get_speed_limit()) / 3.6,
            "control": {
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
            },
            "is_junction": bool(
                map_waypoint is not None
                and bool(getattr(map_waypoint, "is_junction", False))
            ),
        },
        "environment": {"at_junction": False},
        "objects": [],
    }


def vehicle_state_tensor(
    world: Any,
    ego: Any,
    *,
    include_junction: bool = True,
) -> torch.Tensor:
    """Return the fixed 8-dim vehicle-state modality."""

    policy_state = build_sensor_policy_state(
        world,
        ego,
        frame_id="vehicle_state",
    )
    ego_state = policy_state["ego"]
    controls = ego_state["control"]
    values = [
        float(ego_state["speed_mps"]),
        float(ego_state["acceleration_mps2"]),
        float(ego_state["yaw_rate_rps"]),
        float(controls.get("steer", 0.0)),
        float(controls.get("throttle", 0.0)),
        float(controls.get("brake", 0.0)),
        float(ego_state["speed_limit_mps"]),
        float(bool(ego_state.get("is_junction", False)))
        if include_junction
        else 0.0,
    ]
    return torch.tensor([values], dtype=torch.float32)


def environment_feature_tensor(
    world: Any,
    ego: Any | None = None,
    control_speed_cap_kmh: float | None = None,
) -> torch.Tensor:
    """Return the fixed 14-dim environment-state modality."""

    weather = world.get_weather()
    road_speed_limit_kmh = (
        float(ego.get_speed_limit()) if ego is not None else 100.0
    )
    control_cap = (
        float(control_speed_cap_kmh)
        if control_speed_cap_kmh is not None
        else road_speed_limit_kmh
    )
    values = (
        float(weather.cloudiness) / 100.0,
        float(weather.precipitation) / 100.0,
        float(weather.precipitation_deposits) / 100.0,
        float(weather.wind_intensity) / 100.0,
        float(weather.sun_azimuth_angle) / 360.0,
        max(-1.0, min(1.0, float(weather.sun_altitude_angle) / 90.0)),
        float(weather.fog_density) / 100.0,
        float(weather.fog_distance) / 1000.0,
        0.02,
        float(weather.wetness) / 100.0,
        0.10,
        0.80,
        min(1.0, max(0.0, road_speed_limit_kmh / 100.0)),
        min(1.0, max(0.0, control_cap / 100.0)),
    )
    return torch.tensor([values], dtype=torch.float32)


class UniversalVLAController:
    """Run VLA decisions online and execute them through one route PID."""

    def __init__(
        self,
        *,
        world: Any,
        ego: Any,
        route_controller: Any,
        commands: Sequence[Mapping[str, Any]],
        checkpoint_path: Path,
        config_path: Path,
        parser_model_path: Path,
        output_path: Path,
        device: str = "cuda",
        precision: str = "fp16",
        decision_interval_frames: int = 3,
        camera_attributes: Mapping[str, Any] | None = None,
        fixed_delta_seconds: float = 0.05,
        available_cameras: Sequence[str] | None = None,
        enable_lidar: bool = False,
        default_speed_kmh: float = 40.0,
        hold_seconds: float = 20.0,
        modality_schema_version: str = UNIFIED_SENSOR_BATCH_SCHEMA_VERSION,
    ) -> None:
        if decision_interval_frames < 1:
            raise ValueError("decision_interval_frames must be at least 1")
        if available_cameras is None:
            available_cameras = CAMERA_ORDER
        self.world = world
        self.ego = ego
        self.route_controller = route_controller
        self.commands = [dict(item) for item in commands]
        self.decision_interval_frames = int(decision_interval_frames)
        self._last_frame = -10**12
        self._last_intent: str | None = None
        self._fixed_delta_seconds = float(fixed_delta_seconds)
        self._last_overlay: dict[str, Any] = {}
        self._adapter_warmed = False
        self._fallback_count = 0
        self._accepted_count = 0
        self._liveness_overrides = Counter()
        self._decision_count = 0
        self._proposal_actions = Counter()
        self._final_actions = Counter()
        self._latencies_ms: list[float] = []
        self._camera_wait_ms: list[float] = []
        self._sensor_frame_lag: list[int] = []
        self._response_latency_ms: list[float] = []
        self._sensor_to_decision_response_ms: list[float] = []
        self._stream = output_path.open("w", encoding="utf-8")

        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        self.teacher_force_control = bool(
            config.get("teacher_force_control", False)
        )
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision]
        self.pipeline = LightweightVLAPipeline.from_checkpoint(
            build_model(config),
            str(checkpoint_path),
            model_name=config["model_name"],
            device=device,
            dtype=dtype,
            strict_checkpoint=not bool(
                config.get("allow_legacy_checkpoint", False)
            ),
            high_confidence_threshold=float(
                config.get("high_confidence_threshold", 0.55)
            ),
        )
        self.parser = ModernBertCommandService(
            str(parser_model_path),
            device=device,
        )
        self.parser.warmup()
        self.fsm = GenericInstructionFSM(
            default_speed_kmh=float(default_speed_kmh),
            parser=self.parser,
        )
        self.supervisor = GenericTemporalRiskSupervisor(
            TemporalRiskSupervisorConfig(hold_seconds=float(hold_seconds))
        )
        self.modality_schema_version = modality_schema_version
        self.available_cameras = tuple(available_cameras)
        self.enable_lidar = bool(enable_lidar)
        self.camera_rig = SynchronizedMultiviewCameraRig(
            world,
            ego,
            width=int(config.get("camera_input_width", 224)),
            height=int(config.get("camera_input_height", 224)),
            fov=float(config.get("camera_input_fov", 100.0)),
            sensor_tick=float(fixed_delta_seconds) * self.decision_interval_frames,
            camera_attributes=dict(camera_attributes or {}),
            enable_lidar=self.enable_lidar,
            available_cameras=self.available_cameras,
        )
        self._camera_wait_deque: deque[float] = deque(maxlen=64)

    def predict_target_lane_risk(
        self,
        batch: Any,
        direction: str,
    ) -> dict[str, Any]:
        """Probe the learned risk head on one camera view only.

        ``direction`` must be ``"left"`` or ``"right"``.  The probe reuses the
        same checkpoint and risk head, selects the view purely through
        ``camera_view_mask``, and never overwrites the primary fused-view
        temporal risk state.
        """

        if direction not in {"front", "left", "right"}:
            raise ValueError("direction must be 'front', 'left' or 'right'")
        view_name = direction
        if view_name not in self.available_cameras:
            return {
                "risk_level": "low",
                "recommended_action": "keep_lane",
                "reason_codes": [],
                "matched_entity_id": None,
                "source": "unavailable_view",
            }
        view_index = CAMERA_ORDER.index(view_name)
        masked_view_mask = torch.zeros_like(batch.camera_view_mask)
        masked_view_mask[:, view_index] = batch.camera_view_mask[
            :, view_index
        ]
        masked_batch = replace(
            batch,
            camera_view_mask=masked_view_mask,
        )
        risk = self.pipeline.predict_visual_risk(masked_batch)
        risk["source"] = f"learned_{view_name}_camera_visual_risk_head"
        return risk

    def _progress_m(self) -> float:
        return float(self.route_controller.progress_m())

    def active_command(self) -> dict[str, Any]:
        """Return the route-triggered command currently selected by the FSM."""

        return self.fsm.active_command(self.commands, self._progress_m())

    def _decide(self, frame: int) -> None:
        progress_m = self._progress_m()
        command = self.fsm.active_command(self.commands, progress_m)
        parsed = self.fsm.parse(command)
        intent_key = parsed.parsed_intent
        if parsed.requested_lane_direction is not None:
            intent_key = f"{parsed.parsed_intent}_{parsed.requested_lane_direction}"
        if intent_key != self._last_intent:
            self.pipeline.reset_temporal_state()
            self.supervisor.reset()
            self._last_intent = intent_key

        tokens, text_mask = self.fsm.encode_tokens(
            parsed,
            cache_key=f"{intent_key}:{parsed.target_speed_kmh}",
        )
        frame_id = f"carla_{frame}"
        policy_state = build_sensor_policy_state(
            self.world, self.ego, frame_id=frame_id
        )
        timestamp_s = float(policy_state.get("timestamp_s", 0.0))

        if self.enable_lidar:
            sensor_frame, images, view_mask, lidar_bev, camera_wait_ms = (
                self.camera_rig.latest_multisensor(
                    minimum_frame=frame - self.decision_interval_frames,
                    timeout_s=0.05,
                )
            )
        else:
            sensor_frame, images, view_mask, camera_wait_ms = self.camera_rig.latest(
                minimum_frame=frame - self.decision_interval_frames,
                timeout_s=0.05,
            )
            lidar_bev = torch.zeros((1, 4, 64, 64), dtype=torch.float32)

        modality_mask = default_modality_mask(
            text=True,
            front_rgb="front" in self.available_cameras,
            left_rgb="left" in self.available_cameras,
            right_rgb="right" in self.available_cameras,
            rear_rgb="rear" in self.available_cameras,
            lidar_bev=self.enable_lidar,
            vehicle_state=True,
            environment_state=True,
        )
        forward_radar = self.camera_rig.latest_radar(
            "front",
            maximum_frame=sensor_frame,
        )
        # Convert the conic radar aggregate into a route-swept observation.
        # Fail closed on missing legacy bins or unexpected map errors by
        # retaining the original physical distance envelope.
        try:
            route_polyline = self.route_controller.radar_corridor_polyline(
                85.0
            )
            ego_transform = self.ego.get_transform()
            ego_location = ego_transform.location
            vehicle_half_width_m = float(
                getattr(self.ego.bounding_box.extent, "y", 1.0) or 1.0
            )
            forward_radar = filter_forward_radar_to_route_corridor(
                forward_radar,
                ego_x=float(ego_location.x),
                ego_y=float(ego_location.y),
                ego_yaw_deg=float(ego_transform.rotation.yaw),
                route_polyline=route_polyline,
                corridor_half_width_m=vehicle_half_width_m + 0.35,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        rear_radar = self.camera_rig.latest_radar(
            "rear",
            maximum_frame=sensor_frame
        )
        policy_state["sensor_observations"] = {
            "forward_radar": forward_radar,
            "rear_radar": rear_radar,
        }
        view_tensors = {
            name: images[:, index]
            for index, name in enumerate(CAMERA_ORDER)
        }
        vehicle_state = vehicle_state_tensor(self.world, self.ego)
        environment_state = environment_feature_tensor(
            self.world,
            self.ego,
            float(self.route_controller.target_speed_kmh),
        )
        unified_batch = UnifiedSensorBatch(
            schema_version=self.modality_schema_version,
            text_tokens=tokens,
            text_mask=text_mask,
            front_rgb=view_tensors["front"],
            left_rgb=view_tensors["left"],
            right_rgb=view_tensors["right"],
            rear_rgb=view_tensors["rear"],
            lidar_bev=lidar_bev,
            vehicle_state=vehicle_state,
            environment_state=environment_state,
            camera_view_mask=view_mask,
            modality_mask=modality_mask,
            frame_id=frame_id,
            timestamp_s=timestamp_s,
        )
        batch = unified_batch.to_sensor_batch()
        self._camera_wait_ms.append(camera_wait_ms)
        self._sensor_frame_lag.append(frame - sensor_frame)
        if not self._adapter_warmed:
            self.pipeline.warmup(batch, iterations=5)
            self._adapter_warmed = True

        ego_speed_kmh = 3.6 * float(policy_state["ego"]["speed_mps"])
        canonical = self.fsm.canonical_decision(
            parsed,
            frame_id=frame_id,
            request_id=f"unified-{frame}",
            risk={
                "risk_level": "low",
                "recommended_action": "keep_lane",
                "reason_codes": [],
                "matched_entity_id": None,
                "lane_change": {
                    "left": {"is_safe": True, "reason_codes": []},
                    "right": {"is_safe": True, "reason_codes": []},
                },
            },
            ego_speed_kmh=ego_speed_kmh,
        )
        started = time.perf_counter()
        proposal = self.pipeline.predict_proposal(
            batch,
            request_id=canonical["request_id"],
            frame_id=frame_id,
            candidate_entity_ids=[[]],
            world_state=policy_state,
            stream_id=intent_key,
            use_model_risk_assessment=True,
        )
        learned_risk = self.pipeline.last_visual_risk_assessment
        risk = fuse_forward_radar_risk(
            learned_risk,
            forward_radar,
            ego_speed_kmh=ego_speed_kmh,
        )

        target_lane_risk = None
        front_view_risk = None
        rear_assessment = assess_rear_radar_collision(rear_radar)
        lane_options: dict[str, dict[str, Any]] = {}
        if rear_assessment["collision_risk"]:
            front_view_risk = self.predict_target_lane_risk(batch, "front")
            road_speed_limit_kmh = 3.6 * float(
                policy_state["ego"]["speed_limit_mps"]
            )
            effective_cap_kmh = min(
                road_speed_limit_kmh,
                float(self.route_controller.target_speed_kmh),
            )
            front_level = str(
                front_view_risk.get("risk_level", "high")
            ).lower()
            if (
                parsed.parsed_intent
                not in {"YIELD", "STOP", "EMERGENCY_BRAKE"}
                and front_level == "low"
                and ego_speed_kmh >= effective_cap_kmh - 1.5
            ):
                legality_probe = getattr(
                    self.route_controller,
                    "autonomous_lane_change_legal",
                    None,
                )
                for direction in ("left", "right"):
                    legal = bool(
                        callable(legality_probe)
                        and legality_probe(direction)
                    )
                    lane_options[direction] = {"legal": legal}
                    if legal:
                        lane_options[direction]["risk"] = (
                            self.predict_target_lane_risk(batch, direction)
                        )

        if parsed.requested_lane_direction in {"left", "right"}:
            requested_direction = parsed.requested_lane_direction
            target_lane_risk = dict(
                lane_options.get(requested_direction, {}).get("risk")
                or self.predict_target_lane_risk(batch, requested_direction)
            )

        canonical = self.fsm.canonical_decision(
            parsed,
            frame_id=frame_id,
            request_id=canonical["request_id"],
            risk=risk,
            ego_speed_kmh=ego_speed_kmh,
        )
        gated_decision = gate_vla_proposal(proposal, canonical, risk)

        self.supervisor.observe(
            frame=frame,
            timestamp_s=timestamp_s,
            parsed_intent=parsed.parsed_intent,
            risk_level=str(risk.get("risk_level", "high")),
            target_lane_risk_level=(
                str(target_lane_risk.get("risk_level"))
                if target_lane_risk is not None
                else None
            ),
            ego_speed_kmh=ego_speed_kmh,
            requested_lane_direction=parsed.requested_lane_direction,
        )
        stationary_elapsed_s = self.supervisor.stationary_elapsed_s(
            frame,
            self._fixed_delta_seconds,
        )
        resume_active = (
            self.supervisor.resume_intent == parsed.parsed_intent
            and self.supervisor.resume_intent is not None
        )
        final_decision, liveness_override = self.supervisor.apply(
            gated_decision,
            canonical,
            risk,
            parsed_intent=parsed.parsed_intent,
            requested_lane_direction=parsed.requested_lane_direction,
            target_lane_risk=target_lane_risk,
            stationary_elapsed_s=stationary_elapsed_s,
            resume_active=resume_active,
            resume_speed_kmh=min(
                40.0, float(self.route_controller.target_speed_kmh)
            ),
            frame=frame,
            fixed_delta_seconds=self._fixed_delta_seconds,
        )
        directional_override = None
        directional_assessment = {
            "schema_version": "directional_collision_response/1.0",
            "front_priority": True,
            "rear": rear_assessment,
            "front_risk_level": "not_probed",
            "selected_response": "teacher_force_disabled"
            if self.teacher_force_control
            else "none",
        }
        if not self.teacher_force_control:
            final_decision, directional_assessment, directional_override = (
                apply_directional_collision_response(
                    final_decision,
                    front_risk=front_view_risk,
                    forward_radar=risk.get("physical_forward_radar") or {},
                    rear_radar=rear_radar,
                    ego_speed_kmh=ego_speed_kmh,
                    road_speed_limit_kmh=3.6
                    * float(policy_state["ego"]["speed_limit_mps"]),
                    route_speed_cap_kmh=float(
                        self.route_controller.target_speed_kmh
                    ),
                    lane_options=lane_options,
                    allow_evasive_motion=parsed.parsed_intent
                    not in {"YIELD", "STOP", "EMERGENCY_BRAKE"},
                )
            )
        if self.teacher_force_control:
            final_decision = dict(canonical)
            final_decision["reason"] = "training_teacher_force_control"
            final_decision["blocked_reason_codes"] = []
        self.supervisor.record_decision(
            frame=frame,
            risk_level=str(risk.get("risk_level", "high")),
            action=str(final_decision["action"]),
            target_speed_kmh=float(
                final_decision.get("target_speed_kmh", 0.0)
            ),
            override=directional_override or liveness_override,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response_latency_ms = camera_wait_ms + elapsed_ms
        self._latencies_ms.append(elapsed_ms)
        self._response_latency_ms.append(response_latency_ms)
        self._sensor_to_decision_response_ms.append(
            camera_wait_ms + elapsed_ms
        )
        self.route_controller.set_high_level_decision(final_decision)
        accepted = (
            str(gated_decision.get("reason", "")).startswith("vla_accepted_")
            and liveness_override is None
            and directional_override is None
            and not self.teacher_force_control
        )
        self._decision_count += 1
        self._accepted_count += int(accepted)
        self._proposal_actions[proposal["action"]] += 1
        self._final_actions[final_decision["action"]] += 1
        control_override = directional_override or liveness_override
        if control_override is not None:
            self._liveness_overrides[control_override] += 1
        record = {
            "schema_version": "unified_vla_decision/1.0",
            "simulation_frame": frame,
            "route_s_m": round(progress_m, 3),
            "source_text": str(command.get("text", "")),
            "parsed_intent": parsed.parsed_intent,
            "requested_lane_direction": parsed.requested_lane_direction,
            "target_speed_envelope_kmh": parsed.target_speed_kmh,
            "semantic_text": self.fsm.semantic_text(parsed),
            "input_mode": (
                "text_raw_4view_rgb_lidar_bidirectional_radar_vehicle_environment"
                if self.enable_lidar
                else "text_raw_rgb_bidirectional_radar_vehicle_environment"
            ),
            "sensor_frame": sensor_frame,
            "sensor_frame_lag": frame - sensor_frame,
            "camera_wait_ms": round(camera_wait_ms, 3),
            "candidate_count": 0,
            "safety_observation_candidate_count": int(
                forward_radar.get("candidate_count", 0) or 0
            ) + int(
                rear_radar.get("candidate_count", 0) or 0
            ),
            "physical_safety_observation": {
                "forward_radar": forward_radar,
                "rear_radar": rear_radar,
            },
            "policy_truth_access": False,
            "modality_mask": {
                key: bool(value) for key, value in modality_mask.items()
            },
            "camera_view_mask": view_mask[0].tolist(),
            "sensor_batch_schema_version": self.modality_schema_version,
            "risk_assessment": risk,
            "target_lane_risk_assessment": target_lane_risk,
            "front_view_risk_assessment": front_view_risk,
            "directional_collision_assessment": directional_assessment,
            "vla_proposal": proposal,
            "control_decision": final_decision,
            "model_output_applied": accepted,
            "training_teacher_force_control": self.teacher_force_control,
            "liveness_override": control_override,
            "full_decision_latency_ms": round(elapsed_ms, 3),
            "sensor_to_decision_response_ms": round(response_latency_ms, 3),
        }
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()
        self._last_overlay = {
            "asr_text": str(command.get("text", "")),
            "parsed_intent": parsed.parsed_intent,
            "action": final_decision["action"],
            "target_speed_kmh": final_decision["target_speed_kmh"],
            "emergency": final_decision["emergency"],
            "risk_level": str(risk["risk_level"]).upper(),
            "policy_state": "VLA_ACCEPTED" if accepted else "SAFETY_GATE",
        }

    def run_step(self) -> Any:
        try:
            frame = int(self.world.get_snapshot().frame)
            if frame - self._last_frame >= self.decision_interval_frames:
                self._decide(frame)
                self._last_frame = frame
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            self._fallback_count += 1
            self.route_controller.set_high_level_decision(
                {
                    "action": "stop",
                    "target_speed_kmh": 0.0,
                    "emergency": False,
                    "reason": "vla_runtime_safe_stop",
                }
            )
            self._last_overlay = {
                "asr_text": "VLA runtime fallback",
                "action": "stop",
                "target_speed_kmh": 0.0,
                "emergency": False,
                "risk_level": "HIGH",
                "policy_state": type(error).__name__,
            }
            self._stream.write(
                json.dumps(
                    {
                        "status": "fallback_safe_stop",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._stream.flush()
        return self.route_controller.run_step()

    def overlay(self) -> dict[str, Any]:
        return dict(self._last_overlay)

    def summary(self) -> dict[str, Any]:
        latencies = self._latencies_ms
        response = self._response_latency_ms
        return {
            "controller": "universal-vla-controller",
            "input_mode": (
                "text_raw_4view_rgb_lidar_bidirectional_radar_vehicle_environment"
                if self.enable_lidar
                else "text_raw_4view_rgb_bidirectional_radar_vehicle_environment"
            ),
            "model_output_used": self._accepted_count > 0,
            "training_teacher_force_control": self.teacher_force_control,
            "decision_count": self._decision_count,
            "model_accepted_count": self._accepted_count,
            "safety_gate_or_canonical_count": (
                self._decision_count - self._accepted_count
            ),
            "fallback_count": self._fallback_count,
            "liveness_override_counts": dict(self._liveness_overrides),
            "proposal_action_counts": dict(self._proposal_actions),
            "final_action_counts": dict(self._final_actions),
            "full_decision_latency_ms": {
                "mean": round(statistics.fmean(latencies), 3)
                if latencies
                else None,
                "median": round(statistics.median(latencies), 3)
                if latencies
                else None,
                "p95": round(
                    statistics.quantiles(
                        latencies, n=100, method="inclusive"
                    )[94],
                    3,
                )
                if len(latencies) >= 2
                else (round(latencies[0], 3) if latencies else None),
                "max": round(max(latencies), 3) if latencies else None,
            },
            "sensor_to_decision_response_ms": {
                "mean": round(statistics.fmean(response), 3)
                if response
                else None,
                "median": round(statistics.median(response), 3)
                if response
                else None,
                "p95": round(
                    statistics.quantiles(
                        response, n=100, method="inclusive"
                    )[94],
                    3,
                )
                if len(response) >= 2
                else (round(response[0], 3) if response else None),
                "max": round(max(response), 3) if response else None,
                "within_120_ms_rate": round(
                    sum(value <= 120.0 for value in response)
                    / len(response),
                    6,
                )
                if response
                else None,
            },
            "sensor_batch_schema_version": self.modality_schema_version,
            "supervisor": self.supervisor.diagnostics(),
        }

    def close(self) -> None:
        self.camera_rig.close()
        if not self._stream.closed:
            self._stream.close()
