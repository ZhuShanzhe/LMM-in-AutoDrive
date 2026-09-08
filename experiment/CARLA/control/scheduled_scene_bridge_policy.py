"""Route-scheduled DrivingIntent execution through the JSON decision boundary."""

import copy

from control.scene_understanding_json_policy import SceneUnderstandingJsonPolicy


class ScheduledSceneBridgePolicy:
    """Combine route-progress command activation with per-frame rule decisions.

    The schedule decides *which* parsed utterance is currently active.  The
    scene-understanding producer decides whether and how that utterance can be
    executed against the current CARLA WorldState, persists a ControlDecision,
    and the CARLA side consumes that JSON on the same frame.
    """

    def __init__(self, schedule_policy, output_dir=None):
        self.schedule_policy = schedule_policy
        self.output_dir = output_dir
        self._bridge = None
        self._active_command_id = None
        self._scene_world_state = None
        self._context = {}
        self._last_schedule_intent = None
        self._risk_recovery_command_id = None
        self._risk_recovery_target_speed_kmh = None
        self._risk_recovery_clear_frames = 0
        self._risk_recovery_hold_frames = 4
        self._lane_change_settling_command_id = None

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
            self._lane_change_settling_command_id = None
            driving_intent = self._bind_planned_turn_target(
                driving_intent, scheduled.get("target_location")
            )
            self._bridge = SceneUnderstandingJsonPolicy(
                driving_intent=driving_intent,
                output_dir=self.output_dir,
                default_speed_kmh=float(scheduled.get("target_speed_kmh", 40.0)),
                max_age_frames=0,
            )

        bridge_context = dict(self._context)
        bridge_context["default_speed_kmh"] = float(
            scheduled.get("target_speed_kmh", bridge_context.get("default_speed_kmh", 40.0))
        )
        self._bridge.set_context(bridge_context)
        self._bridge.set_scene_world_state(self._scene_world_state)
        # ``SceneUnderstandingJsonPolicy`` deliberately ignores the raw
        # controller state.  It produces and re-parses control_decision.json;
        # this caller passes only the frame guard required by that consumer.
        decision = self._bridge.decide(world_state)
        decision = self._preserve_lane_change_for_speed_limit(scheduled, decision)
        if (
            scheduled.get("action") in {"lane_change_left", "lane_change_right"}
            and decision.get("decision_status") == "READY"
            and decision.get("action") == scheduled.get("action")
        ):
            self._lane_change_settling_command_id = command_id
        decision = self._preserve_lane_change_during_settle(scheduled, decision)
        decision = self._stabilize_risk_recovery(scheduled, decision)
        if (
            str(decision.get("reason", "")).startswith("plan_completed")
            and not decision.get("emergency")
            and decision.get("risk_level", "none") in {"none", "low"}
        ):
            # Completion means that the requested transition succeeded; it
            # does not replace the persistent cruise setpoint with whatever
            # speed happened to be measured on the completion frame. Without
            # this restoration, one transient safety brake permanently latches
            # the vehicle at the post-braking speed.
            decision["action"] = "keep_lane"
            decision["target_speed_kmh"] = float(scheduled.get("target_speed_kmh", 30.0))
            decision["target_lane"] = None
            decision["target_location"] = (
                None
                if scheduled.get("action") in {
                    "lane_change_left", "lane_change_right",
                }
                else scheduled.get("target_location")
            )
            decision["reason"] = "plan_completed_keep_lane"
        # These are decision-layer continuity rules, not controller-side
        # overrides. Persist and re-consume the final result so the JSON file,
        # audit trace, and CARLA input are exactly the same decision.
        decision = self._bridge.persist_final_decision(decision, world_state)
        decision.update({
            "command_id": (
                None if scheduled.get("continuous_safety_monitor") else command_id
            ),
            "voice_text": scheduled.get("voice_text", ""),
            "structured_command": scheduled.get("structured_command", {}),
            "command_phase": scheduled.get("command_phase", "EXECUTING"),
            "audio_file": scheduled.get("audio_file"),
            # Preserve planner provenance across the persisted decision
            # boundary. Without this flag the controller can reject the
            # correct route point after a temporary cross-track error and
            # latch onto a nearby map corridor.
            "route_target_trusted": bool(
                scheduled.get("route_target_trusted", False)
                or (
                    scheduled.get("action") in {"turn_left", "turn_right"}
                    and decision.get("target_location") is not None
                    and not self._context.get("turn_uses_local_branch", False)
                )
            ),
        })
        return decision

    def _stabilize_risk_recovery(self, scheduled, decision):
        """Avoid throttle/brake chatter while a non-emergency risk clears.

        The scene layer runs every simulation frame. A target near a TTC or
        distance threshold can therefore alternate between normal driving and
        ordinary deceleration on adjacent frames. Keep the deceleration target
        for a short clear interval; a new risk or emergency always wins.
        """
        command_id = scheduled.get("command_id") or self._active_command_id
        if decision.get("emergency"):
            self._clear_risk_recovery()
            return decision
        if (
            decision.get("action") == "decelerate"
            and decision.get("reason") == "risk_requires_deceleration"
        ):
            self._risk_recovery_command_id = command_id
            self._risk_recovery_target_speed_kmh = max(
                0.0, float(decision.get("target_speed_kmh", 0.0))
            )
            self._risk_recovery_clear_frames = 0
            return decision
        if self._risk_recovery_command_id != command_id:
            self._clear_risk_recovery()
            return decision
        if decision.get("risk_level") in {"medium", "high"}:
            self._risk_recovery_clear_frames = 0
            return decision
        self._risk_recovery_clear_frames += 1
        if self._risk_recovery_clear_frames >= self._risk_recovery_hold_frames:
            self._clear_risk_recovery()
            return decision

        result = dict(decision)
        scheduled_speed = float(scheduled.get("target_speed_kmh", 0.0))
        result.update({
            "decision_status": "READY",
            "action": "decelerate",
            "target_speed_kmh": min(
                scheduled_speed, float(self._risk_recovery_target_speed_kmh)
            ),
            "reason": "risk_recovery_hold",
        })
        result["blocked_reason_codes"] = list(dict.fromkeys(
            list(result.get("blocked_reason_codes", [])) + ["risk_recovery_hold"]
        ))
        return result

    def _clear_risk_recovery(self):
        self._risk_recovery_command_id = None
        self._risk_recovery_target_speed_kmh = None
        self._risk_recovery_clear_frames = 0

    @staticmethod
    def _bind_planned_turn_target(driving_intent, target_location):
        """Resolve a scheduled turn against the route planner's geometry.

        CARLA junction connector geometry starts before ``ego.is_junction``
        becomes true. A route-scheduled turn is already grounded by the route
        planner, so pass that resolved point to the scene decision module
        instead of requiring the ego to enter the junction before alignment.
        """
        if not isinstance(target_location, dict):
            return driving_intent
        result = copy.deepcopy(driving_intent)
        for step in result.get("intent", {}).get("steps", []):
            if str(step.get("action", "")).upper() not in {"TURN", "U_TURN"}:
                continue
            step.pop("target", None)
            step.pop("target_ref", None)
        return result

    def _preserve_lane_change_during_settle(self, scheduled, decision):
        """Keep an active, risk-clear merge through a map-marking transition.

        CARLA's map permission flips after the ego crosses the lane boundary;
        evaluating the same active instruction again then describes a second
        adjacent-lane request and returns ``lane_change_not_permitted``. Keep
        the original action for that narrow static-map condition. Genuine
        scene-risk interventions still take priority.
        """
        lane_action = scheduled.get("action")
        command_id = scheduled.get("command_id") or self._active_command_id
        if (
            lane_action not in {"lane_change_left", "lane_change_right"}
            or self._lane_change_settling_command_id != command_id
            or decision.get("emergency")
            # The JSON consumer intentionally strips audit-only risk fields;
            # a missing value here is the producer's already-accepted normal
            # path, not a new risk signal.
            or decision.get("risk_level", "none") not in {"none", "low"}
            or decision.get("reason")
            not in {
                "lane_change_left_blocked_wait_for_safe",
                "lane_change_right_blocked_wait_for_safe",
            }
        ):
            return decision
        result = dict(decision)
        result.update({
            "decision_status": "READY",
            "action": lane_action,
            # The static map gate's deceleration target shrinks on every
            # frame. Keep the utterance's fixed speed setpoint while the PID
            # completes the active lateral merge.
            "target_speed_kmh": float(scheduled.get("target_speed_kmh", 0.0)),
            "target_lane": lane_action.removeprefix("lane_change_"),
            "target_location": None,
            "reason": "lane_change_settling_after_target_lane_entry",
        })
        result["blocked_reason_codes"] = list(dict.fromkeys(
            list(result.get("blocked_reason_codes", []))
            + ["lane_change_settling"]
        ))
        return result

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
        feedback = self._bridge.report_execution(world_state, intent, controller)
        if (
            isinstance(feedback, dict)
            and feedback.get("outcome") == "COMPLETED"
            and self._active_command_id
            and not bool(
                (self._last_schedule_intent or {}).get(
                    "continuous_safety_monitor", False
                )
            )
        ):
            self.schedule_policy.mark_completed(self._active_command_id)
        return feedback

    def telemetry(self):
        telemetry = self.schedule_policy.telemetry()
        bridge = self._bridge.telemetry() if self._bridge is not None else {}
        telemetry["scene_bridge"] = bridge
        return telemetry

    def trace(self):
        return self._bridge.trace() if self._bridge is not None else {}

    @property
    def decision_path(self):
        return self._bridge.decision_path if self._bridge is not None else None

    @property
    def scene_world_state(self):
        return dict(self._scene_world_state or {})

    @property
    def active_scheduled_intent(self):
        """Return the route command currently represented by the rule plan.

        Companion policies may use this only to preserve the command's
        longitudinal envelope after the executor reports ``plan_completed``.
        It is not a replacement for the persisted scene decision.
        """
        return dict(self._last_schedule_intent or {})

    def persist_external_final_decision(self, decision, controller_frame):
        """Persist a safety-gated companion decision through the JSON boundary."""
        if self._bridge is None:
            raise RuntimeError("scene bridge is not active")
        final = self._bridge.persist_final_decision(decision, controller_frame)
        scheduled = self._last_schedule_intent or {}
        final.update({
            "command_id": scheduled.get("command_id"),
            "voice_text": scheduled.get("voice_text", ""),
            "structured_command": scheduled.get("structured_command", {}),
        })
        return final
