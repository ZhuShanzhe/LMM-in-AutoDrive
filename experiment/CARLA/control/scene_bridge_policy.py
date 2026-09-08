"""In-process adapter from DrivingIntent and CARLA WorldState to ego actions."""

import copy
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scene_understanding.scripts.run_carla_decision_bridge import build_decision
from scene_understanding.src.control_decision import validate_control_decision
from scene_understanding.src.execution_feedback import evaluate_execution_feedback
from control.motion_contract import apply_motion_constraints


class SceneBridgeDecisionPolicy:
    """Run alignment, risk gating, planning, and execution feedback per tick."""

    def __init__(self, driving_intent_path=None, output_dir=None, driving_intent=None):
        if driving_intent_path is None and driving_intent is None:
            raise ValueError("driving_intent_path or driving_intent is required")
        self.driving_intent_path = (
            os.path.abspath(driving_intent_path) if driving_intent_path else None
        )
        self._inline_driving_intent = copy.deepcopy(driving_intent) if driving_intent else None
        self.output_dir = os.path.abspath(output_dir) if output_dir else None
        self._driving_intent = None
        self._intent_signature = None
        self._scene_world_state = None
        self._context = {}
        self._plan_state = None
        self._pending_feedback = None
        self._execution_tracker = None
        self._decision = None
        self._alignment = None
        self._risk = None
        self._junction_entered_steps = set()
        self._last_telemetry = {"source": "scene_bridge", "status": "not_ready"}

    def set_context(self, context):
        self._context = dict(context or {})

    def set_scene_world_state(self, world_state):
        self._scene_world_state = dict(world_state or {})

    def decide(self, _legacy_world_state):
        if not self._scene_world_state:
            return self._safe_stop("scene_world_state_unavailable")
        try:
            intent = self._load_driving_intent()
            prepared_intent = self._with_route_target(intent)
            decision_started = time.perf_counter()
            plan_state, decision, alignment, risk = build_decision(
                prepared_intent,
                self._scene_world_state,
                prior_plan_state=self._plan_state,
                feedback=self._pending_feedback,
            )
            scene_decision_latency_ms = (time.perf_counter() - decision_started) * 1000.0
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return self._safe_stop("scene_bridge_{0}".format(type(exc).__name__.lower()))

        self._pending_feedback = None
        self._plan_state = plan_state
        self._decision = self._add_route_guidance(
            self._await_object_trigger(decision, prepared_intent)
        )
        self._alignment = alignment
        self._risk = risk
        self._write_outputs()
        self._last_telemetry = {
            "source": "scene_bridge",
            "status": "accepted",
            "frame_id": decision["frame_id"],
            "plan_status": plan_state["plan_status"],
            "active_step_id": plan_state["active_step_id"],
            "decision_status": decision["decision_status"],
            "risk_level": risk["risk_level"],
            "scene_decision_latency_ms": round(scene_decision_latency_ms, 3),
        }
        result = dict(self._decision)
        result["voice_text"] = prepared_intent.get("input", {}).get("raw_text", "")
        return result

    def report_execution(self, world_state, intent, controller=None):
        """Queue terminal feedback for the active plan step after a CARLA tick."""
        if not self._plan_state or self._plan_state.get("plan_status") != "ACTIVE":
            return None
        active_step_id = self._plan_state.get("active_step_id")
        step = self._find_step(active_step_id)
        if step is None:
            return None
        completion_type = (step.get("completion") or {}).get("type")
        static_lane_marking_block = (
            self._decision
            and self._decision.get("reason")
            in {
                "lane_change_left_blocked_wait_for_safe",
                "lane_change_right_blocked_wait_for_safe",
            }
            and "lane_change_not_permitted"
            in self._decision.get("blocked_reason_codes", [])
        )
        if (
            not self._decision
            or (
                self._decision.get("decision_status") != "READY"
                and not (
                    completion_type == "LANE_CHANGE_COMPLETED"
                    and static_lane_marking_block
                )
            )
        ):
            return None
        feedback_decision = dict(self._decision)
        # The control was selected before world.tick(); feedback observes the
        # resulting frame. Rebind only this feedback view to that frame while
        # retaining the exact action and active-step provenance.
        feedback_decision["frame_id"] = world_state["frame_id"]
        self._execution_tracker, feedback = evaluate_execution_feedback(
            self._driving_intent,
            self._plan_state,
            feedback_decision,
            world_state,
            tracker=self._execution_tracker,
        )
        if completion_type == "LANE_CHANGE_COMPLETED" and controller is not None:
            # The scene layer can prove that CARLA has entered the target lane,
            # but it cannot prove the ego is centered and heading-aligned. Use
            # the controller's geometric completion check as the authority so
            # a boundary crossing cannot terminate the merge prematurely.
            feedback = self._completion_feedback(world_state, intent, step, controller)
        elif feedback is None:
            feedback = self._completion_feedback(world_state, intent, step, controller)
        if feedback is not None:
            self._pending_feedback = feedback
            self._last_telemetry["pending_feedback"] = feedback
        return feedback

    def telemetry(self):
        return dict(self._last_telemetry)

    def trace(self):
        """Return the exact artefacts used for the current control decision."""
        return {
            "plan_state": copy.deepcopy(self._plan_state),
            "control_decision": copy.deepcopy(self._decision),
            "semantic_alignment": copy.deepcopy(self._alignment),
            "risk_assessment": copy.deepcopy(self._risk),
        }

    def _load_driving_intent(self):
        if self._inline_driving_intent is not None:
            payload = json.dumps(self._inline_driving_intent, sort_keys=True)
            signature = hash(payload)
            if signature != self._intent_signature:
                self._driving_intent = copy.deepcopy(self._inline_driving_intent)
                self._intent_signature = signature
                self._plan_state = None
                self._pending_feedback = None
                self._execution_tracker = None
                self._junction_entered_steps.clear()
            return self._driving_intent
        with open(self.driving_intent_path, "r", encoding="utf-8-sig") as handle:
            payload = handle.read()
        signature = hash(payload)
        if signature != self._intent_signature:
            self._driving_intent = json.loads(payload)
            self._intent_signature = signature
            self._plan_state = None
            self._pending_feedback = None
            self._execution_tracker = None
            self._junction_entered_steps.clear()
        if not isinstance(self._driving_intent, dict):
            raise ValueError("DrivingIntent must be a JSON object")
        return self._driving_intent

    def _with_route_target(self, driving_intent):
        """Supply CARLA route geometry only for a currently active turn step."""
        if self._context.get("turn_uses_local_branch"):
            result = copy.deepcopy(driving_intent)
            for step in result.get("intent", {}).get("steps", []):
                if step.get("action") in {"TURN", "U_TURN"}:
                    step.setdefault("parameters", {}).setdefault("route_managed", True)
            return result
        target = self._context.get("route_target")
        if not isinstance(target, dict) or "x" not in target or "y" not in target:
            return driving_intent
        result = copy.deepcopy(driving_intent)
        steps = result.get("intent", {}).get("steps", [])
        active_id = (
            self._plan_state.get("active_step_id")
            if self._plan_state is not None
            else (steps[0].get("step_id") if steps else None)
        )
        for step in steps:
            if step.get("step_id") != active_id or step.get("action") not in {"TURN", "U_TURN"}:
                continue
            parameters = step.setdefault("parameters", {})
            parameters.setdefault(
                "target_location",
                {
                    key: value
                    for key, value in {
                        "x": target["x"],
                        "y": target["y"],
                        "z": target.get("z", 0.0),
                        "yaw": target.get("yaw"),
                    }.items()
                    if value is not None
                },
            )
            reference = self._context.get("route_reference")
            if isinstance(reference, dict):
                parameters["target_location"].setdefault(
                    "reference",
                    {
                        key: reference[key]
                        for key in ("x", "y", "z", "yaw")
                        if key in reference
                    },
                )
        return result

    def _add_route_guidance(self, decision):
        result = apply_motion_constraints(decision, self._context)
        target = self._context.get("route_target")
        # The route is scenario geometry, not a second decision source. Persist
        # its short-horizon point in the producer's ControlDecision so the
        # JSON consumer, rather than a CARLA-side bypass, supplies stable
        # intersection guidance for longitudinal/keep-lane commands.
        if (
            result.get("target_location") is None
            and result.get("action") in {"keep_lane", "accelerate", "decelerate"}
            and isinstance(target, dict)
            and "x" in target
            and "y" in target
        ):
            result["target_location"] = {
                "x": float(target["x"]),
                "y": float(target["y"]),
                "z": float(target.get("z", 0.0)),
            }
            if "yaw" in target:
                result["target_location"]["yaw"] = float(target["yaw"])
            self._add_route_reference(result["target_location"])
        if (
            not self._context.get("turn_uses_local_branch")
            and result.get("target_location") is None
            and result.get("action") in {"turn_left", "turn_right"}
            and isinstance(target, dict)
            and "x" in target
            and "y" in target
        ):
            result["target_location"] = {
                "x": float(target["x"]),
                "y": float(target["y"]),
                "z": float(target.get("z", 0.0)),
            }
            if "yaw" in target:
                result["target_location"]["yaw"] = float(target["yaw"])
            self._add_route_reference(result["target_location"])
        errors = validate_control_decision(result)
        if errors:
            raise ValueError("invalid motion-constrained ControlDecision: " + "; ".join(errors))
        return result

    def _add_route_reference(self, target_location):
        reference = self._context.get("route_reference")
        if not isinstance(reference, dict):
            return
        if not all(key in reference for key in ("x", "y", "yaw")):
            return
        target_location["reference"] = {
            "x": float(reference["x"]),
            "y": float(reference["y"]),
            "z": float(reference.get("z", 0.0)),
            "yaw": float(reference["yaw"]),
        }

    def _await_object_trigger(self, decision, driving_intent):
        """Keep moving safely until an object-triggered instruction becomes observable."""
        active_step = self._find_step(
            self._plan_state.get("active_step_id") if self._plan_state else None
        )
        trigger = (active_step or {}).get("trigger") or {}
        trigger_type = str(trigger.get("type", "")).upper()
        if (
            trigger_type != "OBJECT_PRESENT"
            or decision.get("reason") != "no_matching_entity_safe_stop"
        ):
            return decision
        waiting = dict(decision)
        waiting.update({
            "action": "keep_lane",
            "target_speed_kmh": float(
                self._context.get("default_speed_kmh", self._context.get("target_speed_kmh", 35.0))
            ),
            "target_lane": None,
            "target_location": None,
            "emergency": False,
            "reason": "awaiting_object_trigger",
        })
        waiting["blocked_reason_codes"] = list(dict.fromkeys(
            list(waiting.get("blocked_reason_codes", []))
            + ["awaiting_object_trigger"]
        ))
        errors = validate_control_decision(waiting)
        if errors:
            raise ValueError("invalid trigger-wait ControlDecision: " + "; ".join(errors))
        return waiting

    def _find_step(self, step_id):
        if not self._driving_intent:
            return None
        for step in self._driving_intent.get("intent", {}).get("steps", []):
            if step.get("step_id") == step_id:
                return step
        return None

    def _completion_feedback(self, world_state, intent, step, controller):
        completion = step.get("completion") or {}
        completion_type = completion.get("type")
        if not completion_type:
            return None
        ego = world_state.get("ego", {})
        speed_kmh = float(ego.get("speed_mps", 0.0)) * 3.6
        step_id = step["step_id"]
        completed = False
        reason = None
        if completion_type == "VEHICLE_STOPPED":
            completed, reason = speed_kmh <= 0.5, "vehicle_stopped"
        elif completion_type == "TARGET_SPEED_REACHED":
            completed = abs(speed_kmh - float(intent["target_speed_kmh"])) <= 2.0
            reason = "target_speed_reached"
        elif completion_type == "LANE_CHANGE_COMPLETED" and controller is not None:
            state_getter = getattr(controller, "get_execution_state", None)
            execution_state = state_getter() if callable(state_getter) else {}
            execution_state = execution_state if isinstance(execution_state, dict) else {}
            completed, reason = bool(execution_state.get("lane_change_completed")), "lane_change_completed"
        elif completion_type == "JUNCTION_EXITED":
            in_junction = bool(ego.get("is_junction"))
            if in_junction:
                self._junction_entered_steps.add(step_id)
            completed = step_id in self._junction_entered_steps and not in_junction
            reason = "junction_exited"
            route_completion = self._context.get("turn_completion_progress_m")
            if (
                not completed
                and self._context.get("turn_uses_local_branch")
                and isinstance(route_completion, (int, float))
                and float(self._context.get("progress_m", 0.0)) >= float(route_completion)
            ):
                completed, reason = True, "route_turn_connector_exited"
        elif completion_type == "TARGET_REACHED":
            target = intent.get("target_location")
            position = ego.get("position_world_m", {})
            if isinstance(target, dict) and isinstance(position, dict):
                dx = float(position.get("x", 0.0)) - float(target["x"])
                dy = float(position.get("y", 0.0)) - float(target["y"])
                completed, reason = dx * dx + dy * dy <= 36.0, "target_reached"
        elif completion_type == "TARGET_CLEARED":
            completed, reason = self._scenario_target_cleared(step)
        if not completed:
            return None
        return {
            "schema_version": "1.0.0",
            "request_id": self._plan_state["request_id"],
            "frame_id": world_state["frame_id"],
            "step_id": step_id,
            "outcome": "COMPLETED",
            "reason_codes": [reason],
        }

    def _scenario_target_cleared(self, step):
        """Use completed scenario events as explicit target-clear evidence."""
        target_type = str((step.get("target") or {}).get("type", "")).upper()
        required_scenarios = {
            "PEDESTRIAN": {"pedestrian_crossing"},
            "SLOW_VEHICLE": {"adjacent_lead_brake", "cut_in_vehicle"},
            "VEHICLE": {"adjacent_lead_brake", "cut_in_vehicle"},
        }.get(target_type, set())
        completed = set(self._context.get("completed_event_scenarios", []))
        if required_scenarios and completed.intersection(required_scenarios):
            return True, "scenario_target_cleared"
        return False, None

    def _safe_stop(self, reason):
        self._last_telemetry = {
            "source": "scene_bridge",
            "status": "safe_stop",
            "reason": reason,
        }
        return {
            "action": "stop",
            "target_speed_kmh": 0.0,
            "emergency": False,
            "reason": reason,
        }

    def _write_outputs(self):
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        documents = {
            "control_plan_state.json": self._plan_state,
            "control_decision.json": self._decision,
            "semantic_alignment.json": self._alignment,
            "risk_assessment.json": self._risk,
        }
        for name, document in documents.items():
            path = os.path.join(self.output_dir, name)
            temporary = path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, path)
