"""VLA-first behaviour coordination with deterministic safety invariants.

The learned policy owns the nominal behaviour decision.  This module keeps
only the state that cannot be represented by a single-frame proposal:
compound-command progress, an in-flight manoeuvre, input freshness and
replanning history.  Rule/FSM decisions may be supplied as a degraded-mode
fallback, but they are not used to constrain every nominal VLA proposal.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping

from scene_understanding.src.control_decision import validate_control_decision

from .contracts import validate_vla_proposal


COORDINATOR_STATE_SCHEMA_VERSION = "1.0.0"
REQUIRED_SOURCES = ("vision", "voice", "vehicle_state", "environment")
SAFE_ACTIONS = {"decelerate", "stop", "emergency_brake"}
LANE_CHANGE_ACTIONS = {"lane_change_left", "lane_change_right"}

STEP_ACTION_COMPATIBILITY = {
    "KEEP_LANE": {"keep_lane", "decelerate", "stop", "emergency_brake"},
    "ACCELERATE": {"accelerate", "keep_lane", "decelerate", "stop"},
    "DECELERATE": {"decelerate", "keep_lane", "stop", "emergency_brake"},
    "SET_SPEED": {"accelerate", "decelerate", "keep_lane", "stop"},
    "STOP": {"stop", "emergency_brake"},
    "EMERGENCY_BRAKE": {"emergency_brake"},
    "CHANGE_LANE": LANE_CHANGE_ACTIONS | SAFE_ACTIONS | {"keep_lane"},
    "OVERTAKE": LANE_CHANGE_ACTIONS | SAFE_ACTIONS | {"keep_lane", "accelerate"},
    "AVOID": LANE_CHANGE_ACTIONS | SAFE_ACTIONS | {"keep_lane"},
    "YIELD": LANE_CHANGE_ACTIONS | SAFE_ACTIONS | {"keep_lane"},
    "PULL_OVER": LANE_CHANGE_ACTIONS | SAFE_ACTIONS | {"keep_lane"},
    "TURN_LEFT": {"turn_left", "decelerate", "stop", "emergency_brake"},
    "TURN_RIGHT": {"turn_right", "decelerate", "stop", "emergency_brake"},
}


@dataclass(frozen=True)
class CoordinatorConfig:
    minimum_vla_confidence: float = 0.55
    maximum_source_age_s: float = 0.35
    maximum_source_skew_s: float = 0.15
    maximum_inference_latency_ms: float = 150.0
    blocked_frames_before_replan: int = 3
    maximum_streams: int = 128

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_vla_confidence <= 1.0:
            raise ValueError("minimum_vla_confidence must be in [0, 1]")
        if self.maximum_source_age_s <= 0.0:
            raise ValueError("maximum_source_age_s must be positive")
        if self.maximum_source_skew_s < 0.0:
            raise ValueError("maximum_source_skew_s must be non-negative")
        if self.maximum_inference_latency_ms <= 0.0:
            raise ValueError("maximum_inference_latency_ms must be positive")
        if self.blocked_frames_before_replan <= 0:
            raise ValueError("blocked_frames_before_replan must be positive")
        if self.maximum_streams <= 0:
            raise ValueError("maximum_streams must be positive")


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _current_time(world_state: Mapping[str, Any]) -> float | None:
    for key in ("timestamp_s", "sim_time_s", "elapsed_seconds"):
        value = world_state.get(key)
        if _finite(value):
            return float(value)
    return None


def infer_input_health(
    driving_intent: Mapping[str, Any],
    world_state: Mapping[str, Any],
    risk_assessment: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build a backward-compatible health view when adapters lack telemetry.

    Integrators should pass explicit source health.  The inferred view keeps
    old callers operational and marks every entry as inferred for audit logs.
    """

    now = _current_time(world_state)
    timestamp = 0.0 if now is None else now
    ego = world_state.get("ego")
    objects = world_state.get("objects")
    parse_result = driving_intent.get("parse_result")
    return {
        "vision": {
            "available": isinstance(objects, list),
            "timestamp_s": timestamp,
            "inferred": True,
        },
        "voice": {
            "available": isinstance(parse_result, Mapping),
            "timestamp_s": timestamp,
            "inferred": True,
        },
        "vehicle_state": {
            "available": isinstance(ego, Mapping),
            "timestamp_s": timestamp,
            "inferred": True,
        },
        "environment": {
            "available": isinstance(risk_assessment, Mapping),
            "timestamp_s": timestamp,
            "inferred": True,
        },
    }


def validate_input_health(
    health: Mapping[str, Any],
    *,
    reference_time_s: float | None,
    maximum_age_s: float,
    maximum_skew_s: float,
) -> list[str]:
    reasons: list[str] = []
    timestamps: list[float] = []
    for source in REQUIRED_SOURCES:
        item = health.get(source)
        if not isinstance(item, Mapping):
            reasons.append(f"source_{source}_missing")
            continue
        if item.get("available") is not True:
            reasons.append(f"source_{source}_unavailable")
        timestamp = item.get("timestamp_s")
        if not _finite(timestamp):
            reasons.append(f"source_{source}_timestamp_invalid")
            continue
        timestamp_s = float(timestamp)
        timestamps.append(timestamp_s)
        if (
            reference_time_s is not None
            and reference_time_s - timestamp_s > maximum_age_s
        ):
            reasons.append(f"source_{source}_stale")
        if reference_time_s is not None and timestamp_s > reference_time_s + 0.02:
            reasons.append(f"source_{source}_from_future")
    if timestamps and max(timestamps) - min(timestamps) > maximum_skew_s:
        reasons.append("multisource_timestamp_skew")
    return reasons


def _steps(driving_intent: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    intent = driving_intent.get("intent")
    if not isinstance(intent, Mapping):
        return []
    steps = intent.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def _matched_entity_ids(alignment: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            entity_id = value.get("entity_id")
            if isinstance(entity_id, str) and entity_id:
                result.add(entity_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(alignment)
    return result


def _action_matches_step(action: str, step: Mapping[str, Any] | None) -> bool:
    if step is None:
        return False
    semantic_action = str(step.get("action", "")).strip().upper()
    allowed = STEP_ACTION_COMPATIBILITY.get(semantic_action)
    if allowed is None or action not in allowed:
        return False
    if semantic_action == "CHANGE_LANE" and action in LANE_CHANGE_ACTIONS:
        parameters = step.get("parameters", {})
        parameters = parameters if isinstance(parameters, Mapping) else {}
        direction = str(parameters.get("direction", "")).strip().lower()
        if direction in {"left", "right"}:
            return action == f"lane_change_{direction}"
    return True


def _initial_state(driving_intent: Mapping[str, Any], frame_id: str) -> dict[str, Any]:
    steps = _steps(driving_intent)
    first_id = steps[0].get("step_id") if steps else None
    return {
        "schema_version": COORDINATOR_STATE_SCHEMA_VERSION,
        "request_id": str(driving_intent.get("request_id", "")),
        "revision": 0,
        "active_step_index": 0 if steps else None,
        "active_step_id": first_id,
        "completed_step_ids": [],
        "maneuver": None,
        "blocked_frames": 0,
        "replan_requested": False,
        "replan_reason_codes": [],
        "last_frame_id": frame_id,
        "last_source_health": {},
        "decision_source": "vla",
    }


def _apply_feedback(
    state: dict[str, Any],
    driving_intent: Mapping[str, Any],
    feedback: Mapping[str, Any] | None,
) -> None:
    if not isinstance(feedback, Mapping):
        return
    if feedback.get("request_id") != state["request_id"]:
        raise ValueError("feedback request_id does not match coordinator state")
    active_step_id = state.get("active_step_id")
    if feedback.get("step_id") != active_step_id:
        raise ValueError("feedback step_id does not match the active step")
    outcome = feedback.get("outcome")
    if outcome == "COMPLETED":
        completed = list(state["completed_step_ids"])
        if active_step_id is not None and active_step_id not in completed:
            completed.append(active_step_id)
        state["completed_step_ids"] = completed
        steps = _steps(driving_intent)
        next_index = int(state["active_step_index"] or 0) + 1
        state["active_step_index"] = next_index if next_index < len(steps) else None
        state["active_step_id"] = (
            steps[next_index].get("step_id") if next_index < len(steps) else None
        )
        state["maneuver"] = None
        state["revision"] += 1
    elif outcome in {"FAILED", "CANCELLED"}:
        state["replan_requested"] = True
        state["replan_reason_codes"] = [f"active_step_{str(outcome).lower()}"]
        state["maneuver"] = None
        state["revision"] += 1


def _fallback_decision(
    proposal: Mapping[str, Any],
    driving_intent: Mapping[str, Any],
    world_state: Mapping[str, Any],
    risk_assessment: Mapping[str, Any],
    step: Mapping[str, Any] | None,
    reasons: list[str],
    supplied_fallback: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if supplied_fallback is not None:
        errors = validate_control_decision(dict(supplied_fallback))
        if errors:
            raise ValueError("invalid supplied fallback: " + "; ".join(errors))
        if supplied_fallback.get("request_id") != proposal["request_id"]:
            raise ValueError("supplied fallback request_id does not match proposal")
        if supplied_fallback.get("frame_id") != world_state["frame_id"]:
            raise ValueError("supplied fallback frame_id does not match WorldState")
        result = copy.deepcopy(dict(supplied_fallback))
        result["decision_status"] = "SAFE_FALLBACK"
        result["reason"] = "vla_degraded_mode_fallback"
        result["blocked_reason_codes"] = list(dict.fromkeys(reasons))
        return result

    parse_result = driving_intent.get("parse_result")
    parse_result = parse_result if isinstance(parse_result, Mapping) else {}
    confidence = parse_result.get("confidence")
    if not _finite(confidence):
        confidence = None
    risk_level = str(risk_assessment.get("risk_level", "high")).lower()
    if risk_level not in {"none", "low", "medium", "high"}:
        risk_level = "high"
    risk_reasons = risk_assessment.get("reason_codes", [])
    if not isinstance(risk_reasons, list):
        risk_reasons = []
    action = "emergency_brake" if (
        risk_assessment.get("recommended_action") == "emergency_brake"
    ) else "stop"
    result = {
        "schema_version": "1.0.0",
        "request_id": str(proposal["request_id"]),
        "frame_id": str(world_state["frame_id"]),
        "decision_status": "SAFE_FALLBACK",
        "action": action,
        "target_speed_kmh": 0.0,
        "target_lane": None,
        "target_location": None,
        "emergency": action == "emergency_brake",
        "reason": "vla_degraded_mode_safe_stop",
        "parse_status": str(parse_result.get("status", "INVALID")),
        "parse_confidence": confidence,
        "source_step_id": None if step is None else step.get("step_id"),
        "source_step_action": None if step is None else step.get("action"),
        "source_step_count": len(_steps(driving_intent)),
        "matched_entity_id": proposal.get("target_entity_id"),
        "risk_level": risk_level,
        "risk_reason_codes": [str(item) for item in risk_reasons if str(item)],
        "blocked_reason_codes": list(dict.fromkeys(reasons)),
    }
    errors = validate_control_decision(result)
    if errors:
        raise ValueError("invalid safe fallback ControlDecision: " + "; ".join(errors))
    return result


class VLAFirstDecisionCoordinator:
    """Coordinate compound driving tasks while keeping VLA policy authority."""

    def __init__(self, config: CoordinatorConfig | None = None) -> None:
        self.config = config or CoordinatorConfig()
        self._states: dict[str, dict[str, Any]] = {}

    def reset(self, request_id: str | None = None) -> None:
        if request_id is None:
            self._states.clear()
        else:
            self._states.pop(request_id, None)

    def state(self, request_id: str) -> dict[str, Any] | None:
        value = self._states.get(request_id)
        return None if value is None else copy.deepcopy(value)

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore persisted coordinator state after a process restart."""

        required = {
            "schema_version",
            "request_id",
            "revision",
            "active_step_index",
            "active_step_id",
            "completed_step_ids",
            "maneuver",
            "blocked_frames",
            "replan_requested",
            "replan_reason_codes",
            "last_frame_id",
            "last_source_health",
            "decision_source",
        }
        missing = required - state.keys()
        if missing:
            raise ValueError(
                "coordinator state is missing fields: " + ", ".join(sorted(missing))
            )
        if state.get("schema_version") != COORDINATOR_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported coordinator state schema_version")
        request_id = state.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("coordinator state request_id must be non-empty")
        if request_id not in self._states and len(self._states) >= self.config.maximum_streams:
            self._states.pop(next(iter(self._states)))
        self._states[request_id] = copy.deepcopy(dict(state))

    def coordinate(
        self,
        proposal: dict[str, Any],
        driving_intent: dict[str, Any],
        world_state: dict[str, Any],
        semantic_alignment: dict[str, Any],
        risk_assessment: dict[str, Any],
        *,
        input_health: Mapping[str, Any] | None = None,
        feedback: Mapping[str, Any] | None = None,
        fallback_decision: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        errors = validate_vla_proposal(proposal)
        if errors:
            raise ValueError("invalid VLADecisionProposal: " + "; ".join(errors))
        request_id = str(driving_intent.get("request_id", ""))
        frame_id = str(world_state.get("frame_id", ""))
        if proposal["request_id"] != request_id:
            raise ValueError("proposal request_id does not match DrivingIntent")
        if proposal["frame_id"] != frame_id:
            raise ValueError("proposal frame_id does not match WorldState")

        if request_id not in self._states and len(self._states) >= self.config.maximum_streams:
            self._states.pop(next(iter(self._states)))

        state = copy.deepcopy(
            self._states.get(request_id) or _initial_state(driving_intent, frame_id)
        )
        _apply_feedback(state, driving_intent, feedback)
        steps = _steps(driving_intent)
        step_index = state.get("active_step_index")
        step = (
            steps[int(step_index)]
            if isinstance(step_index, int) and step_index < len(steps)
            else None
        )
        health = dict(
            input_health
            if input_health is not None
            else infer_input_health(driving_intent, world_state, risk_assessment)
        )
        reasons = validate_input_health(
            health,
            reference_time_s=_current_time(world_state),
            maximum_age_s=self.config.maximum_source_age_s,
            maximum_skew_s=self.config.maximum_source_skew_s,
        )
        parse_result = driving_intent.get("parse_result", {})
        if not isinstance(parse_result, Mapping) or parse_result.get("status") != "VALID":
            reasons.append("driving_intent_not_valid")
        if float(proposal["confidence"]) < self.config.minimum_vla_confidence:
            reasons.append("vla_confidence_below_threshold")
        if float(proposal["latency_ms"]) > self.config.maximum_inference_latency_ms:
            reasons.append("vla_inference_deadline_exceeded")
        target_id = proposal.get("target_entity_id")
        if target_id is not None and target_id not in _matched_entity_ids(semantic_alignment):
            reasons.append("vla_target_not_semantically_grounded")

        action = str(proposal["action"])
        if step is None:
            reasons.append("compound_plan_has_no_active_step")
        elif not _action_matches_step(action, step):
            reasons.append("vla_action_not_aligned_with_active_step")
        recommended = str(risk_assessment.get("recommended_action", ""))
        if recommended == "emergency_brake" and action != "emergency_brake":
            reasons.append("risk_requires_emergency_brake")
        elif recommended == "decelerate" and action not in SAFE_ACTIONS:
            reasons.append("risk_requires_deceleration")
        if action in LANE_CHANGE_ACTIONS:
            direction = action.removeprefix("lane_change_")
            lane_change = risk_assessment.get("lane_change", {})
            judgment = lane_change.get(direction, {}) if isinstance(lane_change, Mapping) else {}
            if not isinstance(judgment, Mapping) or judgment.get("is_safe") is not True:
                reasons.append(f"target_lane_{direction}_unsafe")
        if action in {"turn_left", "turn_right"} and proposal.get("target_location") is None:
            reasons.append("turn_target_location_missing")

        if reasons:
            state["blocked_frames"] = int(state.get("blocked_frames", 0)) + 1
            if state["blocked_frames"] >= self.config.blocked_frames_before_replan:
                state["replan_requested"] = True
                state["replan_reason_codes"] = list(dict.fromkeys(reasons))
                state["revision"] += 1
            state["decision_source"] = (
                "rule_fallback" if fallback_decision is not None else "safety_fallback"
            )
            decision = _fallback_decision(
                proposal,
                driving_intent,
                world_state,
                risk_assessment,
                step,
                reasons,
                fallback_decision,
            )
        else:
            previous_maneuver = state.get("maneuver")
            if not isinstance(previous_maneuver, Mapping) or previous_maneuver.get("action") != action:
                state["maneuver"] = {
                    "maneuver_id": f"{request_id}:{state['revision']}:{frame_id}",
                    "action": action,
                    "phase": "EXECUTING",
                    "started_frame_id": frame_id,
                    "target_lane": proposal.get("target_lane"),
                    "target_entity_id": target_id,
                }
                state["revision"] += 1
            state["blocked_frames"] = 0
            state["replan_requested"] = False
            state["replan_reason_codes"] = []
            state["decision_source"] = "vla"
            risk_reasons = risk_assessment.get("reason_codes", [])
            if not isinstance(risk_reasons, list):
                risk_reasons = []
            confidence = parse_result.get("confidence")
            if not _finite(confidence):
                confidence = None
            risk_level = str(risk_assessment.get("risk_level", "none")).lower()
            if risk_level not in {"none", "low", "medium", "high"}:
                risk_level = "high"
            decision = {
                "schema_version": "1.0.0",
                "request_id": request_id,
                "frame_id": frame_id,
                "decision_status": "READY",
                "action": action,
                "target_speed_kmh": 0.0 if action in {"stop", "emergency_brake"} else round(float(proposal["target_speed_kmh"]), 6),
                "target_lane": proposal.get("target_lane") if action in LANE_CHANGE_ACTIONS else None,
                "target_location": proposal.get("target_location") if action in {"turn_left", "turn_right"} else None,
                "emergency": action == "emergency_brake",
                "reason": f"vla_first_accepted_{proposal['model']}",
                "parse_status": str(parse_result.get("status", "VALID")),
                "parse_confidence": confidence,
                "source_step_id": None if step is None else step.get("step_id"),
                "source_step_action": None if step is None else step.get("action"),
                "source_step_count": len(steps),
                "matched_entity_id": target_id,
                "risk_level": risk_level,
                "risk_reason_codes": [str(item) for item in risk_reasons if str(item)],
                "blocked_reason_codes": [],
            }
            decision_errors = validate_control_decision(decision)
            if decision_errors:
                raise ValueError("invalid VLA-first ControlDecision: " + "; ".join(decision_errors))

        state["last_frame_id"] = frame_id
        state["last_source_health"] = copy.deepcopy(health)
        self._states[request_id] = copy.deepcopy(state)
        return state, decision
