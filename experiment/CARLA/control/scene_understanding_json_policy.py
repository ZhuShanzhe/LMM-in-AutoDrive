"""Consume scene_understanding's persisted ControlDecision at the CARLA boundary."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Mapping

from control.decision_provider import JsonFileDecisionPolicy
from control.scene_bridge_policy import SceneBridgeDecisionPolicy
from scene_understanding.src.control_decision import validate_control_decision


class SceneUnderstandingJsonPolicy:
    """Separate the scene-understanding producer from the CARLA JSON consumer.

    The producer owns ``DrivingIntent + WorldState -> control_decision.json``.
    This policy deliberately gives the controller only the result parsed back
    from that file, subject to the existing same-frame guard.
    """

    def __init__(
        self,
        *,
        driving_intent_path: str | None = None,
        driving_intent: Mapping[str, Any] | None = None,
        output_dir: str,
        default_speed_kmh: float = 25.0,
        max_age_frames: int = 0,
    ) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self._producer = SceneBridgeDecisionPolicy(
            driving_intent_path=driving_intent_path,
            driving_intent=copy.deepcopy(dict(driving_intent)) if driving_intent else None,
            output_dir=self.output_dir,
        )
        self.decision_path = os.path.join(self.output_dir, "control_decision.json")
        self._consumer = JsonFileDecisionPolicy(
            self.decision_path,
            default_speed_kmh=default_speed_kmh,
            max_age_frames=max_age_frames,
        )
        self._last_telemetry: dict[str, Any] = {
            "source": "scene_understanding_json",
            "status": "not_ready",
            "decision_path": self.decision_path,
        }

    def set_context(self, context: Mapping[str, Any] | None) -> None:
        self._producer.set_context(context)

    def set_scene_world_state(self, world_state: Mapping[str, Any] | None) -> None:
        self._producer.set_scene_world_state(world_state)

    def decide(self, controller_frame: Mapping[str, Any]) -> dict[str, Any]:
        producer_decision = self._producer.decide({})
        decision = self._consumer.decide(dict(controller_frame))
        self._last_telemetry = {
            "source": "scene_understanding_json",
            "status": self._consumer.telemetry().get("status"),
            "decision_path": self.decision_path,
            "producer": self._producer.telemetry(),
            "consumer": self._consumer.telemetry(),
            "scene_bridge": self._producer.telemetry(),
        }
        # Keep display/audit metadata only; vehicle action fields originate
        # solely from the JSON document parsed by the consumer above.
        decision["voice_text"] = producer_decision.get("voice_text", "")
        decision["decision_status"] = producer_decision.get("decision_status")
        return decision

    def persist_final_decision(
        self,
        decision: Mapping[str, Any],
        controller_frame: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically persist a final rule adjustment and consume it again.

        Companion rules may stabilize a safe, already-produced decision (for
        example, retain a lateral target while reducing speed).  They must not
        hand an in-memory action directly to CARLA: the same validated JSON
        document remains the sole vehicle-control boundary.
        """

        with open(self.decision_path, "r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("existing ControlDecision must be a JSON object")
        # JsonFileDecisionPolicy intentionally narrows its in-memory return to
        # controller fields. Preserve producer-only provenance and risk fields
        # from the file, while applying every rule field that is available.
        for key in (
            "decision_status", "action", "target_speed_kmh", "target_lane",
            "target_location", "emergency", "reason", "blocked_reason_codes",
        ):
            if key in decision:
                payload[key] = decision[key]
        errors = validate_control_decision(payload)
        if errors:
            raise ValueError("invalid final ControlDecision: " + "; ".join(errors))
        temporary_path = self.decision_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, self.decision_path)
        return self._consumer.decide(dict(controller_frame))

    def report_execution(self, world_state, intent, controller=None):
        return self._producer.report_execution(world_state, intent, controller)

    def telemetry(self) -> dict[str, Any]:
        return copy.deepcopy(self._last_telemetry)

    def trace(self) -> dict[str, Any]:
        return self._producer.trace()
