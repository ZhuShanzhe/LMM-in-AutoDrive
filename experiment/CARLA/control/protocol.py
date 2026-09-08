"""Stable action protocol shared by decision and control modules.

The decision module may send either a Python dictionary or a JSON string.  This
module normalizes it before it reaches the CARLA-specific controller.
"""

import json


SUPPORTED_ACTIONS = {
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

ACTION_ALIASES = {
    "go_straight": "keep_lane",
    "straight": "keep_lane",
    "brake": "decelerate",
    "slow_down": "decelerate",
    "emergency_stop": "emergency_brake",
    "change_lane_left": "lane_change_left",
    "change_lane_right": "lane_change_right",
}

HIGH_LEVEL_FALLBACK_ACTIONS = {
    "follow": "keep_lane",
    "approach": "keep_lane",
    "navigate_to": "keep_lane",
    "proceed": "keep_lane",
    "wait": "stop",
    "yield": "decelerate",
    "avoid": "decelerate",
    "overtake": "keep_lane",
    "pass_by": "keep_lane",
    "u_turn": "turn_left",
    "park": "stop",
    "reverse": "stop",
}

STEP_ACTION_MAP = {
    "KEEP_LANE": "keep_lane",
    "STOP": "stop",
    "EMERGENCY_BRAKE": "emergency_brake",
    "RESUME": "keep_lane",
    "CANCEL": "keep_lane",
}


def _speed_mps_to_kmh(value, default_speed_kmh):
    try:
        return float(value) * 3.6
    except (TypeError, ValueError):
        return float(default_speed_kmh)


def _select_first_actionable_step(steps):
    valid_steps = [step for step in steps if isinstance(step, dict)]
    if not valid_steps:
        return {}
    first = valid_steps[0]
    # "Keep lane and set speed" is a concurrent longitudinal/lateral command.
    # A flat consumer has no completion feedback for an unconstrained KEEP_LANE
    # prefix, so select the following explicit speed setpoint when present.
    if (
        str(first.get("action", "")).upper() == "KEEP_LANE"
        and not first.get("depends_on")
        and not first.get("preconditions")
        and first.get("completion") is None
        and len(valid_steps) > 1
    ):
        second = valid_steps[1]
        if str(second.get("action", "")).upper() == "SET_SPEED":
            return second
    return first


def _flatten_driving_intent(driving_intent, default_speed_kmh):
    """Map structured_command_parser DrivingIntent to the flat control action."""
    parse_result = driving_intent.get("parse_result", {})
    status = parse_result.get("status")
    request_id = driving_intent.get("request_id")
    if status != "VALID":
        return {
            "action": "stop",
            "target_speed_kmh": 0.0,
            "emergency": False,
            "reason": "parse_status_{0}".format(status or "unknown"),
            "request_id": request_id,
            "parse_status": status,
            "parse_confidence": parse_result.get("confidence"),
        }

    intent = driving_intent.get("intent", {})
    steps = intent.get("steps", [])
    step = _select_first_actionable_step(steps)
    parser_action = str(step.get("action", "KEEP_LANE")).strip().upper()
    parameters = step.get("parameters", {}) or {}
    action = STEP_ACTION_MAP.get(parser_action)

    target_speed_kmh = _speed_mps_to_kmh(
        parameters.get("target_speed_mps"),
        default_speed_kmh,
    )
    if parser_action == "SET_SPEED":
        action = "keep_lane"
    elif parser_action == "ADJUST_SPEED":
        change = str(parameters.get("change", "HOLD")).strip().upper()
        if change == "INCREASE":
            action = "accelerate"
        elif change == "DECREASE":
            action = "decelerate"
        else:
            action = "keep_lane"
    elif parser_action == "CHANGE_LANE":
        direction = str(parameters.get("direction", "")).strip().upper()
        if direction == "LEFT":
            action = "lane_change_left"
        elif direction == "RIGHT":
            action = "lane_change_right"
    elif parser_action == "TURN":
        direction = str(parameters.get("direction", "")).strip().upper()
        if direction == "LEFT":
            action = "turn_left"
        elif direction == "RIGHT":
            action = "turn_right"
        elif direction == "STRAIGHT":
            action = "keep_lane"
    elif parser_action in ("YIELD", "PULL_OVER", "AVOID", "OVERTAKE"):
        direction = str(parameters.get("direction", "")).strip().upper()
        if direction == "LEFT":
            action = "lane_change_left"
        elif direction == "RIGHT":
            action = "lane_change_right"
        else:
            action = "decelerate"

    if action is None:
        action = "stop"
        reason = "unsupported_parser_action_{0}".format(parser_action.lower())
    else:
        reason = "driving_intent_{0}".format(parser_action.lower())

    emergency = (
        action == "emergency_brake"
        or str(intent.get("urgency", "")).strip().upper() == "EMERGENCY"
    )
    if action == "emergency_brake":
        target_speed_kmh = 0.0

    return {
        "action": action,
        "target_speed_kmh": target_speed_kmh,
        "target_lane": None,
        "target_location": None,
        "emergency": emergency,
        "reason": reason,
        "request_id": request_id,
        "parse_status": status,
        "parse_confidence": parse_result.get("confidence"),
        "source_step_id": step.get("step_id"),
        "source_step_action": parser_action,
        "source_step_count": len(steps),
    }


def _compile_high_level_action(action, intent):
    """Safely reduce a VLA semantic action to the controller contract.

    Multi-step ordering remains the scene-planning module's responsibility.
    This adapter only prevents a direct high-level VLA output from bypassing
    the stable ControlDecision action vocabulary at the CARLA boundary.
    """
    target_lane = str(intent.get("target_lane") or "").strip().lower()
    if action in {"merge", "pull_over"}:
        direction = target_lane or "right"
        return (
            "lane_change_{0}".format(direction)
            if direction in {"left", "right"}
            else "decelerate"
        )
    return HIGH_LEVEL_FALLBACK_ACTIONS.get(action, action)


def normalize_intent(intent, default_speed_kmh=25.0):
    """Return a validated, JSON-serializable driving-action dictionary."""
    if isinstance(intent, str):
        intent = json.loads(intent)
    if not isinstance(intent, dict):
        raise TypeError("Driving intent must be a dict or a JSON object string")
    if "parse_result" in intent and "intent" in intent:
        intent = _flatten_driving_intent(intent, default_speed_kmh)

    action = str(intent.get("action", "keep_lane")).strip().lower()
    action = ACTION_ALIASES.get(action, action)
    original_action = action
    action = _compile_high_level_action(action, intent)
    if action not in SUPPORTED_ACTIONS:
        raise ValueError("Unsupported driving action: {0}".format(action))

    target_speed_kmh = intent.get("target_speed_kmh", intent.get("target_speed", default_speed_kmh))
    try:
        target_speed_kmh = float(target_speed_kmh)
    except (TypeError, ValueError):
        target_speed_kmh = float(default_speed_kmh)
    target_speed_kmh = max(0.0, min(target_speed_kmh, 100.0))

    target_location = intent.get("target_location")
    if target_location is not None:
        if not isinstance(target_location, dict):
            raise ValueError("target_location must be a dict with x and y")
        if "x" not in target_location or "y" not in target_location:
            raise ValueError("target_location requires x and y")
        target_location = {
            "x": float(target_location["x"]),
            "y": float(target_location["y"]),
            "z": float(target_location.get("z", 0.0)),
        }
        if "yaw" in intent["target_location"]:
            target_location["yaw"] = float(intent["target_location"]["yaw"])
        reference = intent["target_location"].get("reference")
        if reference is not None:
            if not isinstance(reference, dict) or not all(
                key in reference for key in ("x", "y", "yaw")
            ):
                raise ValueError("target_location.reference requires x, y, and yaw")
            target_location["reference"] = {
                "x": float(reference["x"]),
                "y": float(reference["y"]),
                "z": float(reference.get("z", 0.0)),
                "yaw": float(reference["yaw"]),
            }

    return {
        "action": action,
        "target_speed_kmh": target_speed_kmh,
        "target_lane": intent.get("target_lane"),
        "target_location": target_location,
        "emergency": bool(intent.get("emergency", action == "emergency_brake")),
        "reason": str(intent.get("reason", "")) or (
            "compiled_high_level_{0}".format(original_action)
            if original_action != action else ""
        ),
        "request_id": intent.get("request_id"),
        "parse_status": intent.get("parse_status"),
        "parse_confidence": intent.get("parse_confidence"),
        "source_step_id": intent.get("source_step_id"),
        "source_step_action": intent.get("source_step_action"),
        "source_step_count": intent.get("source_step_count"),
        "command_id": intent.get("command_id"),
        "voice_text": intent.get("voice_text", ""),
        "structured_command": intent.get("structured_command", {}),
        "command_phase": intent.get("command_phase"),
        "audio_file": intent.get("audio_file"),
        "route_target_trusted": bool(intent.get("route_target_trusted", False)),
    }
