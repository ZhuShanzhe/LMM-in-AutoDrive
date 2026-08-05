"""Pure-Python Scene 2 contracts shared with perception and control code."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


DRIVING_INTENT_SCHEMA = "1.2.0"
WORLD_STATE_SCHEMA = "1.0.0"
MULTIMODAL_BUNDLE_SCHEMA = "1.0.0"
VLA_PROPOSAL_SCHEMA = "1.0.0"
CONTROL_DECISION_SCHEMA = "1.0.0"

CONTROL_ACTIONS = {
    "keep_lane",
    "accelerate",
    "decelerate",
    "stop",
    "emergency_brake",
    "lane_change_left",
    "lane_change_right",
    "turn_left",
    "turn_right",
}


def _scene2_step_contract(encoded: Any) -> tuple[str, dict[str, Any], str]:
    """Normalize one compact Scene 2 step into the shared intent contract."""

    action, separator, raw_parameter = str(encoded).partition(":")
    action = action.strip().upper()
    parameter = raw_parameter.strip().upper() if separator else ""
    parameters: dict[str, Any] = {}
    wait_actions = {"CHECK", "WAIT", "CONFIRM", "YIELD"}

    if action == "SET_SPEED":
        value = parameter.removesuffix("MPS")
        try:
            parameters["target_speed_mps"] = float(value)
        except ValueError as error:
            raise ValueError(
                "SET_SPEED requires a numeric mps value: {0}".format(encoded)
            ) from error
    elif action == "ADJUST_SPEED":
        parameters["change"] = parameter or "HOLD"
    elif action in {"CHANGE_LANE", "TURN", "PULL_OVER"}:
        direction = parameter
        if parameter == "RETURN_WHEN_SAFE":
            direction = "RIGHT"
        elif parameter.endswith("_WHEN_SAFE"):
            direction = parameter.removesuffix("_WHEN_SAFE")
        parameters["direction"] = direction
    elif action == "KEEP_LANE" and parameter:
        parameters["direction"] = parameter
    elif parameter:
        parameters["condition"] = parameter

    on_blocked = (
        "WAIT_FOR_SAFE"
        if action in wait_actions
        or parameter.endswith("_WHEN_SAFE")
        else "SAFE_STOP"
    )
    return action, parameters, on_blocked


def build_scheduled_driving_intent(
    command: Mapping[str, Any],
    simulation_frame: int,
    route_s_m: float,
    timestamp_s: float,
) -> dict[str, Any]:
    """Translate one competition schedule entry to DrivingIntent 1.2."""

    steps = []
    encoded_steps: Sequence[Any] = command.get("steps", [])
    previous_step_id = None
    for index, encoded in enumerate(encoded_steps, start=1):
        action, parameters, on_blocked = _scene2_step_contract(encoded)
        step_id = "{0}_step_{1:02d}".format(command["id"], index)
        steps.append(
            {
                "step_id": step_id,
                "action": action,
                "parameters": parameters,
                "depends_on": [previous_step_id] if previous_step_id else [],
                "on_blocked": on_blocked,
                "status": "PENDING",
            }
        )
        previous_step_id = step_id
    return {
        "schema_version": DRIVING_INTENT_SCHEMA,
        "request_id": "{0}-frame-{1}".format(
            command["id"],
            int(simulation_frame),
        ),
        "simulation_frame": int(simulation_frame),
        "route_s_m": round(float(route_s_m), 3),
        "timestamp_s": round(float(timestamp_s), 3),
        "parse_result": {
            "status": "VALID",
            "confidence": 1.0,
            "source": "competition_schedule",
        },
        "intent": {
            "category": command["category"],
            "urgency": command["urgency"],
            "steps": steps,
        },
        "voice_text": command["spoken_text"],
    }


def build_multimodal_frame_bundle(
    scene_id: str,
    simulation_frame: int,
    world_state_frame: int,
    latest_sensor_frames: Mapping[str, int],
    driving_intent_request_id: str | None,
) -> dict[str, Any]:
    """Build a strict multimodal bundle without adjacent-frame filling."""

    required = (
        "front_rgb",
        "left_rgb",
        "right_rgb",
        "rear_rgb",
        "lidar",
    )
    frame = int(simulation_frame)
    sensor_exact = all(
        latest_sensor_frames.get(name) == frame
        for name in required
    )
    world_state_exact = int(world_state_frame) == frame
    exact = sensor_exact and world_state_exact
    return {
        "schema_version": MULTIMODAL_BUNDLE_SCHEMA,
        "scene_id": scene_id,
        "simulation_frame": frame,
        "status": "COMPLETE" if exact else "INCOMPLETE",
        "synchronization": {
            "key": "simulation_frame",
            "exact": exact,
            "sensor_exact": sensor_exact,
            "world_state_exact": world_state_exact,
            "adjacent_frame_fill_used": False,
        },
        "modalities": {
            name: {
                "frame": latest_sensor_frames.get(name),
                "available": name in latest_sensor_frames,
                "exact": latest_sensor_frames.get(name) == frame,
            }
            for name in required
        },
        "world_state_frame": int(world_state_frame),
        "driving_intent_request_id": driving_intent_request_id,
    }


def validate_control_decision(
    decision: Mapping[str, Any],
    simulation_frame: int,
) -> dict[str, Any]:
    """Validate the ControlDecision boundary before CARLA actuation.

    The function does not perform safety gating. The caller must only pass a
    decision already approved by deterministic RiskAssessment/VLA safety
    gating.
    """

    if not isinstance(decision, Mapping):
        raise TypeError("ControlDecision must be a mapping")
    action = str(decision.get("action", "")).strip().lower()
    if action not in CONTROL_ACTIONS:
        raise ValueError(
            "unsupported ControlDecision action: {0}".format(action)
        )
    decision_frame = decision.get("simulation_frame")
    if decision_frame is None:
        raise ValueError("ControlDecision simulation_frame is required")
    if int(decision_frame) != int(simulation_frame):
        raise ValueError(
            "stale ControlDecision: expected frame {0}, got {1}".format(
                int(simulation_frame),
                int(decision_frame),
            )
        )
    target_speed = float(decision.get("target_speed_kmh", 0.0))
    if not 0.0 <= target_speed <= 100.0:
        raise ValueError("target_speed_kmh must be in [0, 100]")
    if decision.get("safety_gate_status") not in {
        "APPROVED",
        "OVERRIDDEN",
    }:
        raise ValueError(
            "ControlDecision must be approved by the safety gate"
        )
    return {
        "schema_version": CONTROL_DECISION_SCHEMA,
        "simulation_frame": int(simulation_frame),
        "action": action,
        "target_speed_kmh": target_speed,
        "target_lane": decision.get("target_lane"),
        "emergency": bool(decision.get("emergency", False)),
        "reason": str(decision.get("reason", "")),
        "request_id": decision.get("request_id"),
        "safety_gate_status": decision["safety_gate_status"],
    }
