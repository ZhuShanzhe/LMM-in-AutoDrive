"""Scene-2 structured-BEV VLA proposal bridge with a rule safety boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer


class StructuredVlaSceneBridgePolicy:
    """Use VLA only as a Scene-2 high-level proposal source.

    ``ScheduledSceneBridgePolicy`` remains the canonical rule/FSM producer.
    This class consumes its persisted ControlDecision, gates one learned
    proposal against that decision and the same-frame risk assessment, then
    atomically persists the accepted or overridden result back to the exact
    controller JSON boundary.
    """

    def __init__(
        self,
        rule_policy,
        *,
        checkpoint_path,
        config_path,
        device="cuda",
        precision="fp16",
    ):
        self.rule_policy = rule_policy
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.config_path = os.path.abspath(config_path)
        self.device = device
        self.precision = precision
        self._pipeline = None
        self._rasterizer = StructuredBEVRasterizer()
        self._tokens_by_text = {}
        self._adapter_warmed = False
        self._last_vla = {"status": "not_ready", "input_mode": "structured_semantic_bev"}

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
        voice_text = canonical.get("voice_text", "")
        if not (
            isinstance(world_state, dict)
            and isinstance(control_decision, dict)
            and isinstance(risk, dict)
            and isinstance(voice_text, str)
            and voice_text.strip()
        ):
            self._last_vla = {
                "status": "skipped",
                "reason": "canonical_scene_artifacts_unavailable",
                "input_mode": "structured_semantic_bev",
            }
            return canonical

        try:
            tokens, mask = self._intent_tokens(voice_text)
            batch, entity_ids = self._rasterizer.build(
                world_state,
                intent_tokens=tokens,
                intent_mask=mask,
            )
            pipeline = self._ensure_pipeline()
            if not self._adapter_warmed:
                # Adapter warmup is a startup cost, not a per-frame decision
                # cost. Token features are cached per active utterance below.
                pipeline.warmup(batch, iterations=10)
                self._adapter_warmed = True
            proposal = pipeline.predict_proposal(
                batch,
                request_id=control_decision["request_id"],
                frame_id=control_decision["frame_id"],
                candidate_entity_ids=entity_ids,
            )
            vla_canonical = self._cruise_envelope(control_decision, world_state, risk)
            final_decision = gate_vla_proposal(proposal, vla_canonical, risk)
            result = self.rule_policy.persist_external_final_decision(
                final_decision, controller_frame
            )
            self._write_proposal(proposal)
            self._last_vla = {
                "status": "accepted",
                "input_mode": "structured_semantic_bev",
                "proposal": proposal,
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
                "input_mode": "structured_semantic_bev",
            }
            return canonical

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
        with open(self.config_path, encoding="utf-8") as handle:
            config = json.load(handle)
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

    def _read_canonical_decision(self):
        decision_path = self.rule_policy.decision_path
        if not decision_path:
            return None
        with open(decision_path, encoding="utf-8-sig") as handle:
            return json.load(handle)

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
