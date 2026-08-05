"""Online text + raw multiview + state VLA controller for CARLA Scene 3."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer
from structured_command_parser.src.modernbert_service import ModernBertCommandService
from carla_multiview_sensor import CAMERA_ORDER, SynchronizedMultiviewCameraRig


class Scene3TrainingRecorder:
    """Persist exact-frame multimodal samples and expert supervision."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.image_dir = output_dir / "images"
        self.tensor_dir = output_dir / "tensors"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self._stream = (output_dir / "manifest.jsonl").open(
            "w", encoding="utf-8"
        )
        self.count = 0

    def record(
        self,
        *,
        frame: int,
        route_s_m: float,
        command_id: str,
        source_text: str,
        images: torch.Tensor,
        batch: Any,
        label: Mapping[str, Any],
        risk: Mapping[str, Any],
    ) -> None:
        from torchvision.io import write_jpeg

        sample_id = f"frame_{frame:08d}"
        image_paths = []
        for view_index, view_name in enumerate(CAMERA_ORDER):
            relative = Path("images") / f"{sample_id}_{view_name}.jpg"
            write_jpeg(
                images[0, view_index].contiguous(),
                str(self.output_dir / relative),
                quality=88,
            )
            image_paths.append(relative.as_posix())
        tensor_relative = Path("tensors") / f"{sample_id}.pt"
        torch.save(
            {
                "camera_bev": batch.camera_bev[0].cpu(),
                "lidar_bev": batch.lidar_bev[0].cpu(),
                "ego_features": batch.ego_features[0].cpu(),
                "candidate_features": batch.candidate_features[0].cpu(),
                "candidate_mask": batch.candidate_mask[0].cpu(),
                "intent_tokens": batch.intent_tokens[0].cpu(),
                "intent_mask": batch.intent_mask[0].cpu(),
                "environment_features": batch.environment_features[0].cpu(),
            },
            self.output_dir / tensor_relative,
        )
        row = {
            "schema_version": "scene3_multimodal_training_sample/1.0",
            "sample_id": sample_id,
            "frame": int(frame),
            "route_s_m": round(float(route_s_m), 3),
            "command_id": command_id,
            "source_text": source_text,
            "camera_order": list(CAMERA_ORDER),
            "image_paths": image_paths,
            "tensor_path": tensor_relative.as_posix(),
            "label": {
                "action": label["action"],
                "target_speed_kmh": float(label["target_speed_kmh"]),
                "target_lane": label.get("target_lane"),
            },
            "risk_level": risk.get("risk_level", "low"),
            "risk_reason_codes": list(risk.get("reason_codes", [])),
        }
        self._stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._stream.flush()
        self.count += 1

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


COMMAND_PROFILES = {
    "scene3_cruise": {
        "text_en": "Keep the current lane.",
        "action": "keep_lane",
        "target_speed_kmh": 40.0,
    },
    "scene3_general_hazard": {
        "text_en": "Slow down and keep the current lane.",
        "action": "decelerate",
        "target_speed_kmh": 30.0,
    },
    "scene3_cut_in_decelerate": {
        "text_en": "Brake immediately to avoid the vehicle ahead.",
        "action": "decelerate",
        "target_speed_kmh": 18.0,
    },
    "scene3_work_zone_warning": {
        "text_en": "Slow down and keep the current lane.",
        "action": "decelerate",
        "target_speed_kmh": 30.0,
    },
    "scene3_right_lane_closure": {
        "text_en": "Move to the left lane when safe.",
        "action": "lane_change_left",
        "target_speed_kmh": 25.0,
    },
    "scene3_pass_work_zone": {
        "text_en": "Keep the current lane and slow down.",
        "action": "keep_lane",
        "target_speed_kmh": 25.0,
    },
    "scene3_worker_crossing": {
        "text_en": "Brake immediately to avoid the pedestrian ahead.",
        "action": "decelerate",
        "target_speed_kmh": 10.0,
    },
    "scene3_blocked_lane_change_left": {
        "text_en": "Move to the left lane when safe.",
        "action": "lane_change_left",
        "target_speed_kmh": 20.0,
    },
    "scene3_resume_normal_driving": {
        "text_en": "Accelerate to 40 kilometers per hour and keep the current lane.",
        "action": "accelerate",
        "target_speed_kmh": 40.0,
    },
}


def active_text_command(
    commands: Sequence[Mapping[str, Any]],
    progress_m: float,
) -> dict[str, Any]:
    """Return the newest route-triggered command whose window is active."""
    active = {
        "id": "scene3_cruise",
        "trigger_progress_m": 0.0,
        "text": "继续沿当前车道安全行驶",
        "semantic_goal": ["keep_lane"],
    }
    for command in sorted(
        commands,
        key=lambda item: float(item["trigger_progress_m"]),
    ):
        if progress_m + 1e-6 < float(command["trigger_progress_m"]):
            break
        end_progress_m = command.get("end_progress_m")
        if (
            end_progress_m is not None
            and progress_m >= float(end_progress_m) - 1e-6
        ):
            continue
        active = dict(command)
    return active


def _speed_mps(actor: Any) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(
        float(velocity.x) ** 2
        + float(velocity.y) ** 2
        + float(velocity.z) ** 2
    )


def _actor_category(type_id: str) -> str:
    lowered = type_id.lower()
    if lowered.startswith("vehicle."):
        return "vehicle"
    if "walker.pedestrian" in lowered:
        return "pedestrian"
    if "trafficcone" in lowered or "cone" in lowered:
        return "traffic_cone"
    if "barrier" in lowered:
        return "road_barrier"
    return "other"


def waypoint_lane_relation(
    ego_waypoint: Any | None,
    actor_waypoint: Any | None,
    lateral_m: float,
) -> str:
    """Prefer CARLA lane identity; use lateral geometry only as a fallback."""
    if ego_waypoint is not None and actor_waypoint is not None:
        same_road = (
            int(ego_waypoint.road_id) == int(actor_waypoint.road_id)
            and int(ego_waypoint.section_id) == int(actor_waypoint.section_id)
        )
        if same_road:
            if int(ego_waypoint.lane_id) == int(actor_waypoint.lane_id):
                return "ego_lane"
            return (
                "left_adjacent_lane"
                if lateral_m < 0.0
                else "right_adjacent_lane"
            )
    if lateral_m < -2.2:
        return "left_adjacent_lane"
    if lateral_m > 2.2:
        return "right_adjacent_lane"
    return "ego_lane"


def build_carla_world_state(
    world: Any,
    ego: Any,
    *,
    frame_id: str,
    maximum_distance_m: float = 70.0,
) -> dict[str, Any]:
    """Build the metric WorldState consumed by StructuredBEVRasterizer."""
    transform = ego.get_transform()
    origin = transform.location
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    ego_velocity = ego.get_velocity()
    ego_acceleration = ego.get_acceleration()
    ego_angular = ego.get_angular_velocity()
    control = ego.get_control()
    carla_map = world.get_map()
    ego_waypoint = carla_map.get_waypoint(
        origin,
        project_to_road=True,
    )
    objects = []
    for actor in world.get_actors():
        if int(actor.id) == int(ego.id):
            continue
        type_id = str(getattr(actor, "type_id", ""))
        if not (
            type_id.startswith("vehicle.")
            or "walker.pedestrian" in type_id
            or "trafficcone" in type_id.lower()
            or "barrier" in type_id.lower()
        ):
            continue
        try:
            location = actor.get_location()
            velocity = actor.get_velocity()
        except RuntimeError:
            continue
        dx = float(location.x - origin.x)
        dy = float(location.y - origin.y)
        dz = float(location.z - origin.z)
        longitudinal = dx * float(forward.x) + dy * float(forward.y)
        lateral = dx * float(right.x) + dy * float(right.y)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance > maximum_distance_m:
            continue
        rvx = float(velocity.x - ego_velocity.x)
        rvy = float(velocity.y - ego_velocity.y)
        relative_longitudinal = rvx * float(forward.x) + rvy * float(forward.y)
        relative_lateral = rvx * float(right.x) + rvy * float(right.y)
        actor_waypoint = carla_map.get_waypoint(
            location,
            project_to_road=True,
        )
        lane_relation = waypoint_lane_relation(
            ego_waypoint,
            actor_waypoint,
            lateral,
        )
        objects.append(
            {
                "entity_id": f"carla_actor_{int(actor.id)}",
                "category": _actor_category(type_id),
                "type": type_id,
                "relative_position_m": {
                    "x": longitudinal,
                    "y": lateral,
                    "z": dz,
                },
                "relative_velocity_mps": {
                    "x": relative_longitudinal,
                    "y": relative_lateral,
                },
                "distance_m": distance,
                "lane_relation": lane_relation,
                "confidence": 1.0,
            }
        )
    objects.sort(key=lambda item: float(item["distance_m"]))
    return {
        "schema_version": "scene3_carla_world_state/1.0",
        "frame_id": frame_id,
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
        },
        "environment": {"at_junction": False},
        "objects": objects[:32],
    }


def environment_feature_tensor(world: Any) -> torch.Tensor:
    """Return the explicit environment modality used by the learned policy."""
    weather = world.get_weather()
    values = (
        float(weather.cloudiness) / 100.0,
        float(weather.precipitation) / 100.0,
        float(weather.precipitation_deposits) / 100.0,
        float(weather.wind_intensity) / 100.0,
        float(weather.sun_azimuth_angle) / 360.0,
        max(-1.0, min(1.0, float(weather.sun_altitude_angle) / 90.0)),
        float(weather.fog_density) / 100.0,
        float(weather.fog_distance) / 1000.0,
        float(weather.fog_falloff) / 10.0,
        float(weather.wetness) / 100.0,
        float(getattr(weather, "scattering_intensity", 0.0)) / 10.0,
        float(getattr(weather, "mie_scattering_scale", 0.0)) / 0.1,
    )
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def assess_scene3_risk(world_state: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a small deterministic collision and lane-change safety gate."""
    ego_speed = float(world_state["ego"]["speed_mps"])
    closest = None
    left_safe = True
    reason_codes = []
    for entity in world_state.get("objects", []):
        position = entity["relative_position_m"]
        longitudinal = float(position["x"])
        lateral = float(position["y"])
        if (
            entity.get("lane_relation") == "left_adjacent_lane"
            and -18.0 < longitudinal < 35.0
            and abs(lateral) < 7.0
        ):
            left_safe = False
        if (
            entity.get("lane_relation") == "ego_lane"
            and longitudinal > 0.0
            and abs(lateral) < 3.0
            and (closest is None or longitudinal < closest[0])
        ):
            closest = (longitudinal, entity)
    recommended = "keep_lane"
    level = "low"
    matched = None
    if closest is not None:
        distance, entity = closest
        matched = entity["entity_id"]
        relative_speed = -float(entity["relative_velocity_mps"]["x"])
        ttc = distance / relative_speed if relative_speed > 0.1 else math.inf
        emergency_distance = 7.5 + 0.8 * ego_speed
        caution_distance = max(16.0, emergency_distance + 1.5 * ego_speed)
        if distance <= emergency_distance or ttc < 1.5:
            recommended = "emergency_brake"
            level = "high"
            reason_codes.append("front_collision_imminent")
        elif distance < caution_distance or ttc < 3.0:
            recommended = "decelerate"
            level = "medium"
            reason_codes.append("front_gap_requires_deceleration")
    if not left_safe:
        reason_codes.append("left_lane_occupied")
    return {
        "risk_level": level,
        "recommended_action": recommended,
        "reason_codes": reason_codes,
        "matched_entity_id": matched,
        "lane_change": {
            "left": {
                "is_safe": left_safe,
                "reason_codes": [] if left_safe else ["left_lane_occupied"],
            },
            "right": {"is_safe": True, "reason_codes": []},
        },
    }


def build_canonical_decision(
    *,
    command_id: str,
    frame_id: str,
    profile: Mapping[str, Any],
    parse_result: Mapping[str, Any],
    risk: Mapping[str, Any],
) -> dict[str, Any]:
    action = str(profile["action"])
    target_speed = float(profile["target_speed_kmh"])
    if risk.get("recommended_action") == "emergency_brake":
        action = "emergency_brake"
        target_speed = 0.0
    elif risk.get("recommended_action") == "decelerate" and action in {
        "accelerate",
        "keep_lane",
        "lane_change_left",
        "lane_change_right",
    }:
        action = "decelerate"
        target_speed = min(target_speed, 15.0)
    if action in {"lane_change_left", "lane_change_right"}:
        direction = action.removeprefix("lane_change_")
        if risk.get("lane_change", {}).get(direction, {}).get("is_safe") is not True:
            action = "decelerate"
            target_speed = min(target_speed, 15.0)
    status = str(parse_result.get("status") or "VALID")
    if status not in {"VALID", "NEEDS_CLARIFICATION", "UNSUPPORTED", "INVALID"}:
        status = "VALID"
    confidence = parse_result.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = None
    source_action = {
        "keep_lane": "KEEP_LANE",
        "accelerate": "ADJUST_SPEED",
        "decelerate": "ADJUST_SPEED",
        "stop": "STOP",
        "emergency_brake": "EMERGENCY_BRAKE",
        "lane_change_left": "CHANGE_LANE",
        "lane_change_right": "CHANGE_LANE",
    }[action]
    return {
        "schema_version": "1.0.0",
        "request_id": f"scene3-{command_id}",
        "frame_id": frame_id,
        "decision_status": "READY",
        "action": action,
        "target_speed_kmh": target_speed,
        "target_lane": (
            action.removeprefix("lane_change_")
            if action.startswith("lane_change_")
            else None
        ),
        "target_location": None,
        "emergency": action == "emergency_brake",
        "reason": f"reviewed_text_envelope_{command_id}",
        "parse_status": status,
        "parse_confidence": confidence,
        "source_step_id": "step_1",
        "source_step_action": source_action,
        "source_step_count": 1,
        "matched_entity_id": risk.get("matched_entity_id"),
        "risk_level": str(risk.get("risk_level", "low")),
        "risk_reason_codes": list(risk.get("reason_codes", [])),
        "blocked_reason_codes": [],
    }


def apply_scene3_liveness_gate(
    final_decision: Mapping[str, Any],
    canonical: Mapping[str, Any],
    risk: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Legacy API retained as an identity function for log compatibility."""
    del canonical, risk
    return dict(final_decision), None


class Scene3VlaController:
    """Run VLA decisions online and execute them through the route PID."""

    def __init__(
        self,
        *,
        world: Any,
        ego: Any,
        route_context: Any,
        route_controller: Any,
        commands: Sequence[Mapping[str, Any]],
        checkpoint_path: Path,
        config_path: Path,
        parser_model_path: Path,
        output_path: Path,
        device: str = "cuda",
        precision: str = "fp16",
        decision_interval_frames: int = 4,
        camera_attributes: Mapping[str, Any] | None = None,
        fixed_delta_seconds: float = 0.05,
        training_data_output: Path | None = None,
    ) -> None:
        if decision_interval_frames < 1:
            raise ValueError("decision_interval_frames must be at least 1")
        self.world = world
        self.ego = ego
        self.route_context = route_context
        self.route_controller = route_controller
        self.commands = [dict(item) for item in commands]
        self.decision_interval_frames = int(decision_interval_frames)
        self._last_frame = -10**12
        self._last_command_id = None
        self._last_overlay = {}
        self._token_cache = {}
        self._parse_cache = {}
        self._adapter_warmed = False
        self._fallback_count = 0
        self._accepted_count = 0
        self._liveness_overrides = Counter()
        self._decision_count = 0
        self._proposal_actions = Counter()
        self._final_actions = Counter()
        self._latencies_ms = []
        self._camera_wait_ms = []
        self._sensor_frame_lag = []
        self._stream = output_path.open("w", encoding="utf-8")
        self.training_recorder = (
            Scene3TrainingRecorder(training_data_output)
            if training_data_output is not None
            else None
        )

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
        )
        self.rasterizer = StructuredBEVRasterizer(
            max_candidates=int(config.get("max_candidates", 32))
        )
        self.parser = ModernBertCommandService(
            str(parser_model_path),
            device=device,
        )
        self.parser.warmup()
        self.camera_rig = SynchronizedMultiviewCameraRig(
            world,
            ego,
            width=int(config.get("camera_input_width", 224)),
            height=int(config.get("camera_input_height", 224)),
            fov=float(config.get("camera_input_fov", 100.0)),
            sensor_tick=float(fixed_delta_seconds) * self.decision_interval_frames,
            camera_attributes=dict(camera_attributes or {}),
        )

    def _intent_features(self, command: Mapping[str, Any]):
        command_id = str(command["id"])
        cached = self._token_cache.get(command_id)
        if cached is not None:
            return cached
        profile = COMMAND_PROFILES[command_id]
        english_text = str(profile["text_en"])
        parsed = self.parser.parse_text(
            english_text,
            request_id=f"scene3-{command_id}",
            modality="TEXT",
            source_text=str(command.get("text", "")),
            source_language="zh-CN",
        )
        self._parse_cache[command_id] = parsed.get("parse_result", {})
        parser = self.parser.parser
        parser.load()
        encoded = parser.tokenizer(
            english_text,
            return_tensors="pt",
            truncation=True,
            max_length=parser.max_length,
        )
        encoded = {
            name: tensor.to(parser.device)
            for name, tensor in encoded.items()
        }
        with torch.inference_mode():
            tokens = parser.model.backbone(**encoded).last_hidden_state
        result = (
            tokens.detach().float().cpu(),
            encoded["attention_mask"].detach().bool().cpu(),
        )
        self._token_cache[command_id] = result
        return result

    def _progress_m(self) -> float:
        return float(
            self.route_context.distances_m[self.route_context.tracker.index]
        )

    def _decide(self, frame: int) -> None:
        progress_m = self._progress_m()
        command = active_text_command(self.commands, progress_m)
        command_id = str(command["id"])
        profile = COMMAND_PROFILES[command_id]
        if command_id != self._last_command_id:
            self.pipeline.reset_temporal_state()
            self._last_command_id = command_id
        tokens, mask = self._intent_features(command)
        frame_id = f"carla_{frame}"
        world_state = build_carla_world_state(
            self.world,
            self.ego,
            frame_id=frame_id,
        )
        risk = assess_scene3_risk(world_state)
        batch, entity_ids = self.rasterizer.build(
            world_state,
            intent_tokens=tokens,
            intent_mask=mask,
        )
        sensor_frame, images, view_mask, camera_wait_ms = self.camera_rig.latest(
            minimum_frame=frame - self.decision_interval_frames,
        )
        batch.camera_images = images
        batch.camera_view_mask = view_mask
        batch.environment_features = environment_feature_tensor(self.world)
        self._camera_wait_ms.append(camera_wait_ms)
        self._sensor_frame_lag.append(frame - sensor_frame)
        if not self._adapter_warmed:
            self.pipeline.warmup(batch, iterations=10)
            self._adapter_warmed = True
        canonical = build_canonical_decision(
            command_id=command_id,
            frame_id=frame_id,
            profile=profile,
            parse_result=self._parse_cache.get(command_id, {}),
            risk=risk,
        )
        if self.training_recorder is not None:
            self.training_recorder.record(
                frame=frame,
                route_s_m=progress_m,
                command_id=command_id,
                source_text=str(command.get("text", "")),
                images=images,
                batch=batch,
                label=canonical,
                risk=risk,
            )
        started = time.perf_counter()
        proposal = self.pipeline.predict_proposal(
            batch,
            request_id=canonical["request_id"],
            frame_id=frame_id,
            candidate_entity_ids=entity_ids,
            world_state=world_state,
            risk_assessment=risk,
            stream_id=canonical["request_id"],
        )
        gated_decision = gate_vla_proposal(proposal, canonical, risk)
        final_decision, liveness_override = apply_scene3_liveness_gate(
            gated_decision,
            canonical,
            risk,
        )
        if self.teacher_force_control:
            final_decision = dict(canonical)
            final_decision["reason"] = "training_teacher_force_control"
            final_decision["blocked_reason_codes"] = []
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.route_controller.set_high_level_decision(
            final_decision,
            command_id=command_id,
        )
        accepted = (
            str(gated_decision.get("reason", "")).startswith("vla_accepted_")
            and liveness_override != "unprompted_stop"
            and not self.teacher_force_control
        )
        self._decision_count += 1
        self._accepted_count += int(accepted)
        self._proposal_actions[proposal["action"]] += 1
        self._final_actions[final_decision["action"]] += 1
        self._latencies_ms.append(elapsed_ms)
        if liveness_override is not None:
            self._liveness_overrides[liveness_override] += 1
        record = {
            "simulation_frame": frame,
            "route_s_m": round(progress_m, 3),
            "command_id": command_id,
            "source_text": command.get("text", ""),
            "normalized_text": profile["text_en"],
            "input_mode": "text_raw_4view_rgb_vehicle_environment",
            "sensor_frame": sensor_frame,
            "sensor_frame_lag": frame - sensor_frame,
            "camera_wait_ms": round(camera_wait_ms, 3),
            "candidate_count": len(entity_ids[0]),
            "risk_assessment": risk,
            "vla_proposal": proposal,
            "control_decision": final_decision,
            "model_output_applied": accepted,
            "training_teacher_force_control": self.teacher_force_control,
            "liveness_override": liveness_override,
            "full_decision_latency_ms": round(elapsed_ms, 3),
        }
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()
        self._last_overlay = {
            "asr_text": str(command.get("text", "")),
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
                },
                command_id="vla_runtime_fallback",
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
        return {
            "controller": "vla-route-pid",
            "input_mode": "text_raw_4view_rgb_vehicle_environment",
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
                "mean": round(statistics.fmean(latencies), 3) if latencies else None,
                "median": round(statistics.median(latencies), 3) if latencies else None,
                "max": round(max(latencies), 3) if latencies else None,
            },
            "camera_wait_ms": {
                "mean": round(statistics.fmean(self._camera_wait_ms), 3)
                if self._camera_wait_ms
                else None,
                "max": round(max(self._camera_wait_ms), 3)
                if self._camera_wait_ms
                else None,
            },
            "sensor_frame_lag": {
                "mean": round(statistics.fmean(self._sensor_frame_lag), 3)
                if self._sensor_frame_lag
                else None,
                "max": max(self._sensor_frame_lag)
                if self._sensor_frame_lag
                else None,
            },
            "training_samples_recorded": (
                self.training_recorder.count
                if self.training_recorder is not None
                else 0
            ),
        }

    def close(self) -> None:
        self.camera_rig.close()
        if self.training_recorder is not None:
            self.training_recorder.close()
        if not self._stream.closed:
            self._stream.close()
