"""Stateful execution of multi-step DrivingIntent plans across JSON frames.

The executor advances only on explicit step feedback.  Scene disappearance is
never treated as proof of completion.  Each call evaluates the active step
against the current semantic alignment and risk assessment, then emits a
persisted state plus one controller-compatible ControlDecision.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from scene_understanding.src.control_decision import (
    build_control_decision,
    validate_control_decision,
)


CONTROL_PLAN_STATE_SCHEMA_VERSION = "1.0.0"
STEP_FEEDBACK_SCHEMA_VERSION = "1.0.0"
PLAN_STATUSES = {
    "ACTIVE",
    "COMPLETED",
    "BLOCKED",
    "FAILED",
    "CANCELLED",
    "SAFE_FALLBACK",
}
STEP_STATUSES = {
    "PENDING",
    "ACTIVE",
    "WAITING",
    "COMPLETED",
    "SKIPPED",
    "BLOCKED",
    "FAILED",
    "CANCELLED",
}
FEEDBACK_OUTCOMES = {"CONTINUE", "COMPLETED", "FAILED", "CANCELLED"}


def _string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{path} must be an array of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{path} values must be unique")
    return list(value)


def _intent_steps(driving_intent: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    intent = driving_intent.get("intent")
    if not isinstance(intent, dict) or not isinstance(intent.get("steps"), list):
        raise ValueError("DrivingIntent intent.steps must be an array")
    steps = intent["steps"]
    step_ids: list[str] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"DrivingIntent intent.steps[{index}] must be an object")
        step_id = step.get("step_id")
        action = step.get("action")
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"DrivingIntent intent.steps[{index}].step_id is invalid")
        if step_id in step_ids:
            raise ValueError(f"DrivingIntent contains duplicate step_id {step_id!r}")
        if not isinstance(action, str) or not action:
            raise ValueError(f"DrivingIntent step {step_id!r} action is invalid")
        dependencies = step.get("depends_on", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) or not item for item in dependencies
        ):
            raise ValueError(f"DrivingIntent step {step_id!r} depends_on is invalid")
        unknown = [item for item in dependencies if item not in step_ids]
        if unknown:
            raise ValueError(
                f"DrivingIntent step {step_id!r} depends on a missing or later step: "
                + ", ".join(unknown)
            )
        step_ids.append(step_id)
    return steps


def validate_step_feedback(data: Any) -> list[str]:
    """Return structural errors for one external step-result event."""

    if not isinstance(data, dict):
        return ["root: expected an object"]
    expected = {
        "schema_version",
        "request_id",
        "frame_id",
        "step_id",
        "outcome",
        "reason_codes",
    }
    errors: list[str] = []
    missing = sorted(expected - data.keys())
    extra = sorted(data.keys() - expected)
    if missing:
        errors.append("root: missing fields: " + ", ".join(missing))
    if extra:
        errors.append("root: unexpected fields: " + ", ".join(extra))
    if data.get("schema_version") != STEP_FEEDBACK_SCHEMA_VERSION:
        errors.append("schema_version: expected '1.0.0'")
    for key in ("request_id", "frame_id", "step_id"):
        if not isinstance(data.get(key), str) or not data[key]:
            errors.append(f"{key}: expected a non-empty string")
    if data.get("outcome") not in FEEDBACK_OUTCOMES:
        errors.append("outcome: invalid value")
    reasons = data.get("reason_codes")
    if not isinstance(reasons, list) or any(
        not isinstance(item, str) or not item for item in reasons
    ):
        errors.append("reason_codes: expected an array of non-empty strings")
    elif len(set(reasons)) != len(reasons):
        errors.append("reason_codes: values must be unique")
    return errors


def validate_control_plan_state(data: Any) -> list[str]:
    """Return structural and active-step consistency errors."""

    if not isinstance(data, dict):
        return ["root: expected an object"]
    expected = {
        "schema_version",
        "request_id",
        "revision",
        "plan_status",
        "active_step_index",
        "active_step_id",
        "step_states",
        "last_frame_id",
        "last_feedback_outcome",
        "reason_codes",
    }
    errors: list[str] = []
    missing = sorted(expected - data.keys())
    extra = sorted(data.keys() - expected)
    if missing:
        errors.append("root: missing fields: " + ", ".join(missing))
    if extra:
        errors.append("root: unexpected fields: " + ", ".join(extra))
    if data.get("schema_version") != CONTROL_PLAN_STATE_SCHEMA_VERSION:
        errors.append("schema_version: expected '1.0.0'")
    if not isinstance(data.get("request_id"), str) or not data["request_id"]:
        errors.append("request_id: expected a non-empty string")
    revision = data.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        errors.append("revision: expected a non-negative integer")
    if data.get("plan_status") not in PLAN_STATUSES:
        errors.append("plan_status: invalid value")
    if not isinstance(data.get("last_frame_id"), str) or not data["last_frame_id"]:
        errors.append("last_frame_id: expected a non-empty string")
    if data.get("last_feedback_outcome") not in FEEDBACK_OUTCOMES | {"INITIALIZED"}:
        errors.append("last_feedback_outcome: invalid value")
    reasons = data.get("reason_codes")
    if not isinstance(reasons, list) or any(
        not isinstance(item, str) or not item for item in reasons
    ):
        errors.append("reason_codes: expected an array of non-empty strings")
    elif len(set(reasons)) != len(reasons):
        errors.append("reason_codes: values must be unique")

    step_states = data.get("step_states")
    if not isinstance(step_states, list):
        errors.append("step_states: expected an array")
        return errors
    ids: set[str] = set()
    active_indices: list[int] = []
    for index, item in enumerate(step_states):
        path = f"step_states[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{path}: expected an object")
            continue
        item_expected = {
            "step_id", "action", "status", "activation_frame_id",
            "terminal_frame_id", "reason_codes",
        }
        if set(item) != item_expected:
            errors.append(f"{path}: fields do not match the state contract")
        step_id = item.get("step_id")
        if not isinstance(step_id, str) or not step_id:
            errors.append(f"{path}.step_id: expected a non-empty string")
        elif step_id in ids:
            errors.append(f"{path}.step_id: duplicate value {step_id!r}")
        else:
            ids.add(step_id)
        if not isinstance(item.get("action"), str) or not item["action"]:
            errors.append(f"{path}.action: expected a non-empty string")
        status = item.get("status")
        if status not in STEP_STATUSES:
            errors.append(f"{path}.status: invalid value")
        if status in {"ACTIVE", "WAITING"}:
            active_indices.append(index)
        for key in ("activation_frame_id", "terminal_frame_id"):
            value = item.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                errors.append(f"{path}.{key}: expected null or a non-empty string")
        item_reasons = item.get("reason_codes")
        if not isinstance(item_reasons, list) or any(
            not isinstance(reason, str) or not reason for reason in item_reasons
        ):
            errors.append(f"{path}.reason_codes: invalid value")

    active_index = data.get("active_step_index")
    active_id = data.get("active_step_id")
    if data.get("plan_status") == "ACTIVE":
        if len(active_indices) != 1:
            errors.append("step_states: ACTIVE plan requires exactly one active/waiting step")
        if isinstance(active_index, bool) or not isinstance(active_index, int):
            errors.append("active_step_index: ACTIVE plan requires an integer")
        elif not 0 <= active_index < len(step_states):
            errors.append("active_step_index: out of range")
        elif step_states[active_index].get("step_id") != active_id:
            errors.append("active_step_id: does not match active_step_index")
        if active_indices and active_index != active_indices[0]:
            errors.append("active_step_index: does not identify active/waiting step")
    elif active_index is not None or active_id is not None:
        errors.append("active step fields must be null for a terminal plan")
    return errors


def _initial_state(
    driving_intent: Mapping[str, Any], frame_id: str
) -> dict[str, Any]:
    steps = _intent_steps(driving_intent)
    request_id = driving_intent.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("DrivingIntent request_id must be a non-empty string")
    parse_status = driving_intent.get("parse_result", {}).get("status")
    active = bool(steps) and parse_status == "VALID"
    step_states = [
        {
            "step_id": step["step_id"],
            "action": step["action"],
            "status": "PENDING",
            "activation_frame_id": None,
            "terminal_frame_id": None,
            "reason_codes": [],
        }
        for index, step in enumerate(steps)
    ]
    active_index = 0 if active else None
    if active and _subsumed_keep_lane_prefix(steps):
        step_states[0].update(
            {
                "status": "SKIPPED",
                "terminal_frame_id": frame_id,
                "reason_codes": ["subsumed_by_set_speed_lateral_control"],
            }
        )
        active_index = 1
    if active_index is not None:
        step_states[active_index]["status"] = "ACTIVE"
        step_states[active_index]["activation_frame_id"] = frame_id
    return {
        "schema_version": CONTROL_PLAN_STATE_SCHEMA_VERSION,
        "request_id": request_id,
        "revision": 0,
        "plan_status": "ACTIVE" if active else "SAFE_FALLBACK",
        "active_step_index": active_index,
        "active_step_id": (
            steps[active_index]["step_id"] if active_index is not None else None
        ),
        "step_states": step_states,
        "last_frame_id": frame_id,
        "last_feedback_outcome": "INITIALIZED",
        "reason_codes": [] if active else [f"parse_status_{str(parse_status).lower()}"],
    }


def _subsumed_keep_lane_prefix(steps: list[Mapping[str, Any]]) -> bool:
    """Recognise one parser form that represents a single lateral/longitudinal action.

    Rule parsing may emit ``KEEP_LANE`` and ``SET_SPEED`` as independent
    immediate clauses for a sentence such as "keep the current lane and set
    speed to 60 km/h".  A SET_SPEED ControlDecision already keeps the lane,
    so the unconstrained KEEP_LANE prefix is compiled away rather than
    inventing a completion event.  Dependent or otherwise constrained lane
    steps keep their normal explicit-feedback semantics.
    """

    if len(steps) < 2:
        return False
    lane_step, speed_step = steps[0], steps[1]
    if str(lane_step.get("action", "")).upper() != "KEEP_LANE":
        return False
    if lane_step.get("completion") is not None or lane_step.get("target") is not None:
        return False
    if lane_step.get("depends_on") or lane_step.get("preconditions"):
        return False
    if lane_step.get("trigger", {}).get("type", "IMMEDIATE") != "IMMEDIATE":
        return False
    if str(speed_step.get("action", "")).upper() != "SET_SPEED":
        return False
    if speed_step.get("depends_on"):
        return False
    return speed_step.get("trigger", {}).get("type", "IMMEDIATE") == "IMMEDIATE"


def _check_state_matches_intent(
    state: Mapping[str, Any], driving_intent: Mapping[str, Any]
) -> None:
    errors = validate_control_plan_state(state)
    if errors:
        raise ValueError("invalid ControlPlanState: " + "; ".join(errors))
    if state["request_id"] != driving_intent.get("request_id"):
        raise ValueError("request_id mismatch between ControlPlanState and DrivingIntent")
    expected = [
        (step["step_id"], step["action"]) for step in _intent_steps(driving_intent)
    ]
    actual = [(item["step_id"], item["action"]) for item in state["step_states"]]
    if actual != expected:
        raise ValueError("ControlPlanState steps do not match DrivingIntent steps")


def _set_terminal(
    state: dict[str, Any], index: int, status: str, frame_id: str,
    reason_codes: list[str],
) -> None:
    item = state["step_states"][index]
    item["status"] = status
    item["terminal_frame_id"] = frame_id
    item["reason_codes"] = list(dict.fromkeys(reason_codes))


def _activate(
    state: dict[str, Any], index: int, frame_id: str,
    driving_intent: Mapping[str, Any],
) -> bool:
    steps = _intent_steps(driving_intent)
    if index >= len(steps):
        state["plan_status"] = "COMPLETED"
        state["active_step_index"] = None
        state["active_step_id"] = None
        state["reason_codes"] = ["all_steps_completed"]
        return False
    settled = {
        item["step_id"]
        for item in state["step_states"]
        if item["status"] in {"COMPLETED", "SKIPPED"}
    }
    missing = [item for item in steps[index].get("depends_on", []) if item not in settled]
    if missing:
        reasons = [f"dependency_not_completed:{item}" for item in missing]
        _set_terminal(state, index, "BLOCKED", frame_id, reasons)
        state["plan_status"] = "BLOCKED"
        state["active_step_index"] = None
        state["active_step_id"] = None
        state["reason_codes"] = reasons
        return False
    item = state["step_states"][index]
    item["status"] = "ACTIVE"
    item["activation_frame_id"] = frame_id
    item["reason_codes"] = []
    state["plan_status"] = "ACTIVE"
    state["active_step_index"] = index
    state["active_step_id"] = item["step_id"]
    state["reason_codes"] = []
    return True


def _apply_feedback(
    state: dict[str, Any], feedback: Mapping[str, Any], frame_id: str,
    driving_intent: Mapping[str, Any],
) -> None:
    errors = validate_step_feedback(feedback)
    if errors:
        raise ValueError("invalid StepFeedback: " + "; ".join(errors))
    if state["plan_status"] != "ACTIVE":
        raise ValueError("StepFeedback cannot be applied to a terminal plan")
    if feedback["request_id"] != state["request_id"]:
        raise ValueError("request_id mismatch between StepFeedback and ControlPlanState")
    if feedback["step_id"] != state["active_step_id"]:
        raise ValueError("StepFeedback step_id does not match the active step")

    index = state["active_step_index"]
    outcome = feedback["outcome"]
    reasons = list(feedback["reason_codes"])
    state["last_feedback_outcome"] = outcome
    if outcome == "CONTINUE":
        state["step_states"][index]["status"] = "ACTIVE"
        state["step_states"][index]["reason_codes"] = reasons
        return
    if outcome == "COMPLETED":
        _set_terminal(state, index, "COMPLETED", feedback["frame_id"], reasons)
        _activate(state, index + 1, frame_id, driving_intent)
        return

    terminal_status = "FAILED" if outcome == "FAILED" else "CANCELLED"
    _set_terminal(state, index, terminal_status, feedback["frame_id"], reasons)
    state["plan_status"] = terminal_status
    state["active_step_index"] = None
    state["active_step_id"] = None
    state["reason_codes"] = reasons or [f"step_{outcome.lower()}"]


def _blocked_policy(step: Mapping[str, Any]) -> str:
    value = str(step.get("on_blocked") or "SAFE_STOP").strip().upper()
    if value in {"WAIT_FOR_SAFE", "WAIT", "SLOW_DOWN"}:
        return "WAIT"
    if value in {"SKIP", "SKIP_STEP", "CONTINUE"}:
        return "SKIP"
    return "STOP"


def _await_target_clearance_feedback(
    step: Mapping[str, Any], reason_codes: list[str]
) -> bool:
    """Keep a target-clearance step active after its target leaves AHEAD.

    A correctly yielded pedestrian or overtaken vehicle no longer satisfies an
    AHEAD/AHEAD_CROSSING alignment reference on its first clearance frame.
    The execution-feedback service must inspect that frame before the plan is
    terminal.  This narrow exception retains the blocked stop decision and
    applies only to the explicit matcher miss used by that transition.
    """

    completion = step.get("completion")
    return (
        isinstance(completion, Mapping)
        and completion.get("type") == "TARGET_CLEARED"
        and "no_matching_entity" in reason_codes
    )


def _replace_decision(
    decision: Mapping[str, Any], *, status: str, action: str, reason: str,
    target_speed_kmh: float, blocked_reason_codes: list[str],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(decision))
    result.update(
        {
            "decision_status": status,
            "action": action,
            "target_speed_kmh": target_speed_kmh,
            "target_lane": None,
            "target_location": None,
            "emergency": action == "emergency_brake",
            "reason": reason,
            "blocked_reason_codes": list(dict.fromkeys(blocked_reason_codes)),
        }
    )
    errors = validate_control_decision(result)
    if errors:
        raise ValueError("invalid terminal ControlDecision: " + "; ".join(errors))
    return result


def _active_step(
    state: Mapping[str, Any], driving_intent: Mapping[str, Any]
) -> Mapping[str, Any]:
    index = state["active_step_index"]
    return _intent_steps(driving_intent)[index]


def advance_control_plan(
    driving_intent: dict[str, Any],
    world_state: dict[str, Any],
    semantic_alignment: dict[str, Any],
    risk_assessment: dict[str, Any],
    *,
    prior_state: dict[str, Any] | None = None,
    feedback: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Advance a plan by at most one feedback transition and emit an action."""

    frame_id = world_state.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("WorldState frame_id must be a non-empty string")
    if prior_state is None:
        if feedback is not None:
            raise ValueError("StepFeedback requires a prior ControlPlanState")
        state = _initial_state(driving_intent, frame_id)
    else:
        _check_state_matches_intent(prior_state, driving_intent)
        state = copy.deepcopy(prior_state)
        state["revision"] += 1
        state["last_frame_id"] = frame_id
        state["last_feedback_outcome"] = "CONTINUE"
        if feedback is not None:
            _apply_feedback(state, feedback, frame_id, driving_intent)

    # Build a template with complete parser/risk provenance.  For terminal
    # states the last plan step supplies source metadata only.
    steps = _intent_steps(driving_intent)
    template_step_id = state.get("active_step_id")
    if template_step_id is None and steps:
        terminal = [
            item for item in state["step_states"] if item["status"] != "PENDING"
        ]
        template_step_id = (terminal[-1] if terminal else state["step_states"][0])["step_id"]
    decision = build_control_decision(
        driving_intent,
        world_state,
        semantic_alignment,
        risk_assessment,
        source_step_id=template_step_id,
    )

    if state["plan_status"] == "SAFE_FALLBACK":
        state["reason_codes"] = list(decision["blocked_reason_codes"])
    elif state["plan_status"] == "COMPLETED":
        # SET_SPEED remains a persistent longitudinal setpoint after its
        # completion event.  Replacing it with instantaneous speed makes the
        # target decay frame by frame and can unintentionally stop the ego.
        if decision.get("source_step_action") == "SET_SPEED":
            decision = _replace_decision(
                decision,
                status="READY",
                action="keep_lane",
                reason="plan_completed",
                target_speed_kmh=float(decision["target_speed_kmh"]),
                blocked_reason_codes=[],
            )
            return state, decision
        current_speed_kmh = round(float(world_state["ego"]["speed_mps"]) * 3.6, 6)
        decision = _replace_decision(
            decision,
            status="READY",
            action="keep_lane",
            reason="plan_completed",
            target_speed_kmh=current_speed_kmh,
            blocked_reason_codes=[],
        )
    elif state["plan_status"] in {"FAILED", "CANCELLED", "BLOCKED"}:
        decision = _replace_decision(
            decision,
            status="SAFE_FALLBACK" if state["plan_status"] != "BLOCKED" else "BLOCKED",
            action="stop",
            reason=f"plan_{state['plan_status'].lower()}",
            target_speed_kmh=0.0,
            blocked_reason_codes=state["reason_codes"] or [
                f"plan_{state['plan_status'].lower()}"
            ],
        )
    else:
        # A SKIP policy may immediately expose another blocked step, so the
        # loop is bounded by the finite number of intent steps.
        for _ in range(len(steps) + 1):
            if decision["decision_status"] != "BLOCKED":
                active = state["step_states"][state["active_step_index"]]
                active["status"] = "ACTIVE"
                active["reason_codes"] = []
                state["reason_codes"] = []
                break
            index = state["active_step_index"]
            step = _active_step(state, driving_intent)
            reasons = list(decision["blocked_reason_codes"])
            policy = _blocked_policy(step)
            if policy == "WAIT":
                state["step_states"][index]["status"] = "WAITING"
                state["step_states"][index]["reason_codes"] = reasons
                state["reason_codes"] = reasons
                break
            if policy == "STOP":
                if _await_target_clearance_feedback(step, reasons):
                    state["step_states"][index]["status"] = "WAITING"
                    state["step_states"][index]["reason_codes"] = reasons
                    state["reason_codes"] = reasons
                    break
                _set_terminal(state, index, "BLOCKED", frame_id, reasons)
                state["plan_status"] = "BLOCKED"
                state["active_step_index"] = None
                state["active_step_id"] = None
                state["reason_codes"] = reasons
                decision = _replace_decision(
                    decision,
                    status="BLOCKED",
                    action="stop",
                    reason="plan_blocked_safe_stop",
                    target_speed_kmh=0.0,
                    blocked_reason_codes=reasons or ["plan_blocked"],
                )
                break

            _set_terminal(state, index, "SKIPPED", frame_id, reasons)
            if not _activate(state, index + 1, frame_id, driving_intent):
                if state["plan_status"] == "COMPLETED":
                    current_speed_kmh = round(
                        float(world_state["ego"]["speed_mps"]) * 3.6, 6
                    )
                    decision = _replace_decision(
                        decision,
                        status="READY",
                        action="keep_lane",
                        reason="plan_completed_after_skips",
                        target_speed_kmh=current_speed_kmh,
                        blocked_reason_codes=[],
                    )
                else:
                    decision = _replace_decision(
                        decision,
                        status="BLOCKED",
                        action="stop",
                        reason="plan_dependency_blocked",
                        target_speed_kmh=0.0,
                        blocked_reason_codes=state["reason_codes"],
                    )
                break
            decision = build_control_decision(
                driving_intent,
                world_state,
                semantic_alignment,
                risk_assessment,
                source_step_id=state["active_step_id"],
            )

    state_errors = validate_control_plan_state(state)
    if state_errors:
        raise ValueError("invalid ControlPlanState: " + "; ".join(state_errors))
    decision_errors = validate_control_decision(decision)
    if decision_errors:
        raise ValueError("invalid ControlDecision: " + "; ".join(decision_errors))
    return state, decision
