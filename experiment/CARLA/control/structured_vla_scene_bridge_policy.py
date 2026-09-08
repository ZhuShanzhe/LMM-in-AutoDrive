"""Universal raw-camera/LiDAR VLA bridge with a narrow rule safety boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer
from carla_multiview_sensor import SynchronizedMultiviewCameraRig


class StructuredVlaSceneBridgePolicy:
    """Use one multisensor VLA checkpoint as the high-level proposal source.

    ``ScheduledSceneBridgePolicy`` remains the canonical rule/FSM producer.
    This class consumes its persisted ControlDecision, gates one learned
    proposal with the model's raw-camera visual-risk head, then atomically
    persists the accepted or overridden result back to the exact controller
    JSON boundary.  The rule/FSM decision remains an independently auditable
    fail-safe envelope and is never rasterized into model inputs.
    """

    def __init__(
        self,
        rule_policy,
        *,
        checkpoint_path,
        config_path,
        device="cuda",
        precision="fp16",
        world=None,
        ego=None,
        sensor_tick=0.05,
        front_only=False,
    ):
        self.rule_policy = rule_policy
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.config_path = os.path.abspath(config_path)
        self.device = device
        self.precision = precision
        if world is None or ego is None:
            raise ValueError("world and ego are required for raw multisensor VLA")
        with open(self.config_path, encoding="utf-8") as handle:
            self._config = json.load(handle)
        self._front_only = bool(front_only)
        self._sensor_max_age_frames = max(
            1, int(round(float(sensor_tick) / 0.05))
        )
        self._pipeline = None
        self._rasterizer = StructuredBEVRasterizer()
        self._tokens_by_text = {}
        self._adapter_warmed = False
        self._sensor_rig = SynchronizedMultiviewCameraRig(
            world,
            ego,
            width=int(self._config.get("camera_input_width", 224)),
            height=int(self._config.get("camera_input_height", 224)),
            fov=float(self._config.get("camera_input_fov", 100.0)),
            sensor_tick=float(sensor_tick),
            enable_lidar=not self._front_only,
        )
        self._last_vla = {"status": "not_ready", "input_mode": "raw_rgb_lidar_state"}

    def warmup(self):
        self.rule_policy.warmup()
        self._ensure_pipeline()

    def set_context(self, context):
        self.rule_policy.set_context(context)

    def set_scene_world_state(self, world_state):
        self.rule_policy.set_scene_world_state(world_state)

    def decide(self, controller_frame):
        canonical = self.rule_policy.decide(controller_frame)
        trace = self.rule_policy.trace()
        world_state = self.rule_policy.scene_world_state
        control_decision = self._read_canonical_decision()
        risk = trace.get("risk_assessment")
        semantic_text = self._canonical_step_text(control_decision)
        if not (
            isinstance(world_state, dict)
            and isinstance(control_decision, dict)
            and isinstance(risk, dict)
            and isinstance(semantic_text, str)
            and semantic_text.strip()
        ):
            self._last_vla = {
                "status": "skipped",
                "reason": "canonical_scene_artifacts_unavailable",
                "input_mode": "raw_rgb_lidar_state",
            }
            return canonical

        try:
            tokens, mask = self._intent_tokens(semantic_text)
            batch, entity_ids = self._rasterizer.build(
                world_state,
                intent_tokens=tokens,
                intent_mask=mask,
            )
            batch, entity_ids = self._strip_privileged_inputs(
                batch, entity_ids
            )
            minimum_frame = None
            if isinstance(controller_frame, dict):
                minimum_frame = self._minimum_sensor_frame(
                    controller_frame.get("simulation_frame"),
                    self._sensor_max_age_frames,
                )
            if self._front_only:
                sensor_frame, images, view_mask, sensor_wait_ms = (
                    self._sensor_rig.latest(
                        minimum_frame=minimum_frame,
                        timeout_s=0.08,
                    )
                )
                lidar_bev = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
            else:
                sensor_frame, images, view_mask, lidar_bev, sensor_wait_ms = (
                    self._sensor_rig.latest_multisensor(
                        minimum_frame=minimum_frame,
                        timeout_s=0.08,
                    )
                )
            if self._front_only:
                view_mask[:, 1:] = False
            batch.camera_images = images
            batch.camera_view_mask = view_mask
            batch.lidar_bev = lidar_bev
            batch.environment_features = self._environment_features(
                control_decision
            )
            pipeline = self._ensure_pipeline()
            if not self._adapter_warmed:
                # Adapter warmup is a startup cost, not a per-frame decision
                # cost. Token features are cached per active utterance below.
                pipeline.warmup(batch, iterations=10)
                self._adapter_warmed = True
            # Temporal stabilization belongs to one atomic FSM step.  Reusing
            # only the compound request id would latch a completed stop/yield
            # into the following lane-change, turn, or resume step.
            temporal_stream_id = self._temporal_stream_id(control_decision)
            policy_world_state = self._sensor_policy_state(world_state)
            proposal = pipeline.predict_proposal(
                batch,
                request_id=control_decision["request_id"],
                frame_id=control_decision["frame_id"],
                candidate_entity_ids=entity_ids,
                world_state=policy_world_state,
                stream_id=temporal_stream_id,
                use_model_risk_assessment=True,
            )
            learned_risk = pipeline.last_visual_risk_assessment
            raw_proposal = dict(proposal)
            proposal = self._normalize_longitudinal_proposal(
                proposal, policy_world_state
            )
            vla_canonical = self._cruise_envelope(
                control_decision, policy_world_state, learned_risk
            )
            final_decision = gate_vla_proposal(
                proposal, vla_canonical, learned_risk
            )
            result = self.rule_policy.persist_external_final_decision(
                final_decision, controller_frame
            )
            self._write_proposal(raw_proposal)
            self._last_vla = {
                "status": "accepted",
                "input_mode": "raw_rgb_lidar_state",
                "sensor_frame": sensor_frame,
                "sensor_wait_ms": round(float(sensor_wait_ms), 6),
                "camera_view_mask": view_mask[0].tolist(),
                "lidar_nonzero_cells": int((lidar_bev[:, 0] > 0).sum()),
                "policy_truth_access": False,
                "learned_risk_assessment": learned_risk,
                "rule_safety_risk_assessment": risk,
                "semantic_step_text": semantic_text,
                "proposal": raw_proposal,
                "normalized_proposal": proposal,
                "final_action": result.get("action"),
                "final_reason": result.get("reason"),
            }
            return result
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            # Preserve the validated deterministic decision if VLA inference
            # is unavailable. A learned proposal never causes an unsafe stop.
            self._last_vla = {
                "status": "fallback_to_rules",
                "reason": type(exc).__name__,
                "detail": str(exc),
                "input_mode": "raw_rgb_lidar_state",
            }
            return canonical

    @staticmethod
    def _strip_privileged_inputs(batch, entity_ids):
        """Remove actor-rasterized tensors while preserving physical LiDAR."""
        batch.camera_bev.zero_()
        batch.candidate_features.zero_()
        batch.candidate_mask.zero_()
        return batch, [[] for _ in entity_ids]

    @staticmethod
    def _sensor_policy_state(world_state):
        """Keep only ego/timing fields used by the temporal supervisor."""
        result = {
            "frame_id": world_state.get("frame_id", "sensor_frame"),
            "ego": dict(world_state.get("ego", {})),
            "objects": [],
        }
        for key in ("timestamp_s", "sim_time_s", "elapsed_seconds"):
            if key in world_state:
                result[key] = world_state[key]
        return result

    @staticmethod
    def _normalize_longitudinal_proposal(proposal, world_state):
        """Make the action label agree with the learned speed setpoint.

        The action classifier and speed regressor are independent heads.  A
        transient ``keep_lane`` proposal below the current speed is therefore
        physically a deceleration request; marking it as such keeps the
        controller, telemetry and violation audit consistent without changing
        the learned target or weakening any risk constraint.
        """
        result = dict(proposal)
        try:
            current_speed_kmh = float(world_state["ego"]["speed_mps"]) * 3.6
            target_speed_kmh = float(result["target_speed_kmh"])
        except (KeyError, TypeError, ValueError):
            return result
        if (
            result.get("action") in {"accelerate", "keep_lane"}
            and target_speed_kmh < current_speed_kmh - 2.0
        ):
            result["action"] = "decelerate"
        return result

    @staticmethod
    def _temporal_stream_id(control_decision):
        """Scope temporal proposal state to one compound-command substep."""
        request_id = str(control_decision.get("request_id") or "request_unknown")
        step_id = str(control_decision.get("source_step_id") or "step_unknown")
        return f"{request_id}:{step_id}"

    @staticmethod
    def _minimum_sensor_frame(simulation_frame, max_age_frames):
        """Accept one complete sensor bundle within the configured cadence."""
        if simulation_frame is None:
            return None
        return max(0, int(simulation_frame) - max(1, int(max_age_frames)))

    def _cruise_envelope(self, canonical, world_state, risk):
        """Give the safety gate a persistent, bounded longitudinal intent.

        A completed ``SET_SPEED`` plan used to emit ``keep_lane`` capped at
        the instantaneous vehicle speed.  That made an otherwise valid VLA
        acceleration proposal incompatible or capped to a few km/h.  Preserve
        the rule command's speed ceiling and express the current speed error
        as a deterministic longitudinal action.  Risk actions deliberately
        bypass this helper and still dominate in ``gate_vla_proposal``.
        """
        if not isinstance(canonical, dict) or not isinstance(world_state, dict):
            return canonical
        if canonical.get("decision_status") != "READY":
            return canonical
        if canonical.get("action") != "keep_lane" or canonical.get("emergency"):
            return canonical
        if risk.get("recommended_action") in {"decelerate", "emergency_brake"}:
            return canonical

        target_speed = canonical.get("target_speed_kmh")
        if str(canonical.get("reason", "")).startswith("plan_completed"):
            scheduled = self.rule_policy.active_scheduled_intent
            target_speed = scheduled.get("target_speed_kmh", target_speed)
        try:
            target_speed = float(target_speed)
            current_speed = float(world_state["ego"]["speed_mps"]) * 3.6
        except (KeyError, TypeError, ValueError):
            return canonical
        if not 0.0 < target_speed <= 100.0 or current_speed < 0.0:
            return canonical

        # A small deadband prevents per-frame action flips at the cruise point.
        deadband_kmh = 2.0
        result = dict(canonical)
        result["target_speed_kmh"] = round(target_speed, 6)
        if current_speed < target_speed - deadband_kmh:
            result["action"] = "accelerate"
            result["reason"] = "vla_cruise_below_speed_setpoint"
        elif current_speed > target_speed + deadband_kmh:
            result["action"] = "decelerate"
            result["reason"] = "vla_cruise_above_speed_setpoint"
        else:
            result["reason"] = "vla_cruise_speed_setpoint_held"
        return result

    def report_execution(self, world_state, intent, controller=None):
        return self.rule_policy.report_execution(world_state, intent, controller)

    def telemetry(self):
        telemetry = self.rule_policy.telemetry()
        telemetry["vla"] = dict(self._last_vla)
        return telemetry

    def trace(self):
        return self.rule_policy.trace()

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        config = self._config
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        self._pipeline = LightweightVLAPipeline.from_checkpoint(
            build_model(config),
            self.checkpoint_path,
            model_name=config["model_name"],
            device=self.device,
            dtype=dtype,
        )
        return self._pipeline

    def _environment_features(self, control_decision):
        world = self._sensor_rig.sensors[0].get_world()
        weather = world.get_weather()
        scheduled = self.rule_policy.active_scheduled_intent
        cap = scheduled.get(
            "target_speed_kmh", control_decision.get("target_speed_kmh", 50.0)
        )
        try:
            cap = max(0.0, min(float(cap), 100.0))
        except (TypeError, ValueError):
            cap = 50.0
        return torch.tensor(
            [[
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
                cap / 100.0,
                cap / 100.0,
            ]],
            dtype=torch.float32,
        )

    def close(self):
        self._sensor_rig.close()

    def _read_canonical_decision(self):
        decision_path = self.rule_policy.decision_path
        if not decision_path:
            return None
        with open(decision_path, encoding="utf-8-sig") as handle:
            return json.load(handle)

    @staticmethod
    def _canonical_step_text(control_decision):
        """Represent the parser/FSM active step in the model's label space."""
        if not isinstance(control_decision, dict):
            return ""
        action = str(control_decision.get("action", "keep_lane"))
        speed = float(control_decision.get("target_speed_kmh", 0.0))
        templates = {
            "keep_lane": "Keep the current lane at {speed:.1f} kilometers per hour.",
            "accelerate": "Accelerate smoothly to {speed:.1f} kilometers per hour when safe.",
            "decelerate": "Slow down smoothly to {speed:.1f} kilometers per hour.",
            "stop": "Stop the vehicle at a safe position.",
            "emergency_brake": "Brake immediately.",
            "lane_change_left": "Change to the left lane when it is safe.",
            "lane_change_right": "Change to the right lane when it is safe.",
            "turn_left": "Turn left safely at the next junction.",
            "turn_right": "Turn right safely at the next junction.",
        }
        return templates.get(action, templates["keep_lane"]).format(speed=speed)

    def _intent_tokens(self, voice_text):
        cached = self._tokens_by_text.get(voice_text)
        if cached is not None:
            return cached
        tokens, mask = self.rule_policy.schedule_policy.encode_intent_tokens(voice_text)
        self._tokens_by_text[voice_text] = (tokens, mask)
        return tokens, mask

    def _write_proposal(self, proposal):
        decision_path = self.rule_policy.decision_path
        if not decision_path:
            return
        output = Path(decision_path).with_name("vla_decision_proposal.json")
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
