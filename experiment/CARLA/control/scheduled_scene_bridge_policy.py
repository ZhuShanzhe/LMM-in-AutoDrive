"""Route-scheduled DrivingIntent execution through the scene decision bridge."""

from control.scene_bridge_policy import SceneBridgeDecisionPolicy


class ScheduledSceneBridgePolicy:
    """Combine route-progress command activation with per-frame rule decisions.

    The schedule decides *which* parsed utterance is currently active.  The
    scene bridge decides whether and how that utterance can be executed against
    the current CARLA WorldState.
    """

    def __init__(self, schedule_policy, output_dir=None):
        self.schedule_policy = schedule_policy
        self.output_dir = output_dir
        self._bridge = None
        self._active_command_id = None
        self._scene_world_state = None
        self._context = {}
        self._last_schedule_intent = None

    def warmup(self):
        warmup = getattr(self.schedule_policy, "warmup", None)
        if callable(warmup):
            warmup()

    def set_context(self, context):
        self._context = dict(context or {})
        self.schedule_policy.set_context(self._context)

    def set_scene_world_state(self, world_state):
        self._scene_world_state = dict(world_state or {})
        if self._bridge is not None:
            self._bridge.set_scene_world_state(self._scene_world_state)

    def decide(self, world_state):
        scheduled = self.schedule_policy.decide(world_state)
        self._last_schedule_intent = scheduled
        driving_intent = scheduled.get("driving_intent")
        command_id = scheduled.get("command_id")
        if not isinstance(driving_intent, dict) or not command_id:
            return scheduled

        if command_id != self._active_command_id:
            self._active_command_id = command_id
            self._bridge = SceneBridgeDecisionPolicy(
                driving_intent=driving_intent,
                output_dir=self.output_dir,
            )

        bridge_context = dict(self._context)
        bridge_context["default_speed_kmh"] = float(
            scheduled.get("target_speed_kmh", bridge_context.get("default_speed_kmh", 40.0))
        )
        self._bridge.set_context(bridge_context)
        self._bridge.set_scene_world_state(self._scene_world_state)
        decision = self._bridge.decide(world_state)
        decision = self._preserve_lane_change_for_speed_limit(scheduled, decision)
        decision.update({
            "command_id": command_id,
            "voice_text": scheduled.get("voice_text", ""),
            "structured_command": scheduled.get("structured_command", {}),
        })
        if decision.get("decision_status") == "READY" and not decision.get("emergency"):
            decision["target_speed_kmh"] = float(scheduled.get("target_speed_kmh", 0.0))
        return decision

    @staticmethod
    def _preserve_lane_change_for_speed_limit(scheduled, decision):
        """Keep lateral continuity when risk handling only requests slowing down.

        A medium-risk deceleration is a longitudinal constraint, not evidence
        that the selected adjacent lane is unavailable. Replacing an active
        lane-change action for one tick makes the PID abandon its target lane
        and then reacquire it on the next tick, creating a visible steering
        kick. Emergency braking and explicit lane-change blocks retain their
        normal fail-safe behavior.
        """
        lane_action = scheduled.get("action")
        if (
            lane_action not in {"lane_change_left", "lane_change_right"}
            or decision.get("action") != "decelerate"
            or decision.get("reason") != "risk_requires_deceleration"
            or decision.get("emergency")
        ):
            return decision
        result = dict(decision)
        scheduled_speed = float(scheduled.get("target_speed_kmh", 0.0))
        limited_speed = max(0.0, float(decision.get("target_speed_kmh", scheduled_speed)))
        result.update({
            "decision_status": "READY",
            "action": lane_action,
            "target_speed_kmh": min(scheduled_speed, limited_speed),
            "target_lane": scheduled.get("target_lane"),
            "target_location": None,
            "reason": "risk_speed_limited_lane_change",
        })
        result["blocked_reason_codes"] = list(dict.fromkeys(
            list(result.get("blocked_reason_codes", []))
            + ["speed_limited_during_lane_change"]
        ))
        return result

    def report_execution(self, world_state, intent, controller=None):
        if self._bridge is None:
            return None
        return self._bridge.report_execution(world_state, intent, controller)

    def telemetry(self):
        telemetry = self.schedule_policy.telemetry()
        bridge = self._bridge.telemetry() if self._bridge is not None else {}
        telemetry["scene_bridge"] = bridge
        return telemetry
