"""Deterministic speed-target resolution for one active DrivingIntent step.

Relative ``ADJUST_SPEED`` commands are converted to an absolute setpoint from
the speed observed when the step becomes active.  Stateful callers must store
that setpoint and pass it back on subsequent frames; otherwise a qualitative
"slow down" command would subtract the delta again on every control tick.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


DEFAULT_RELATIVE_SPEED_DELTA_KMH = 5.0
MAX_TARGET_SPEED_KMH = 100.0


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _bounded_speed(value: Any, label: str) -> float:
    result = _finite_number(value)
    if result is None or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return round(min(result, MAX_TARGET_SPEED_KMH), 6)


def is_speed_step(step: Mapping[str, Any]) -> bool:
    return str(step.get("action", "")).strip().upper() in {
        "SET_SPEED",
        "ADJUST_SPEED",
    }


def resolve_step_speed_target(
    step: Mapping[str, Any],
    reference_speed_kmh: float,
    *,
    resolved_target_speed_kmh: float | None = None,
) -> tuple[str, float]:
    """Return the controller action and one absolute speed setpoint.

    ``resolved_target_speed_kmh`` is the persisted setpoint for an already
    active step.  Supplying it prevents per-frame accumulation while still
    retaining the original INCREASE/DECREASE action semantics.
    """

    reference = _bounded_speed(reference_speed_kmh, "reference_speed_kmh")
    parser_action = str(step.get("action", "")).strip().upper()
    parameters = step.get("parameters") or {}
    if not isinstance(parameters, Mapping):
        raise ValueError("speed-step parameters must be an object")

    if parser_action == "SET_SPEED":
        if resolved_target_speed_kmh is not None:
            target = _bounded_speed(
                resolved_target_speed_kmh,
                "resolved_target_speed_kmh",
            )
        elif parameters.get("target_speed_mps") is not None:
            target = _bounded_speed(
                _bounded_speed(
                    parameters["target_speed_mps"],
                    "target_speed_mps",
                )
                * 3.6,
                "target_speed_kmh",
            )
        elif parameters.get("target_speed_kmh") is not None:
            target = _bounded_speed(
                parameters["target_speed_kmh"],
                "target_speed_kmh",
            )
        else:
            raise ValueError(
                "SET_SPEED requires finite non-negative target_speed_mps"
            )
        return "keep_lane", target

    if parser_action != "ADJUST_SPEED":
        raise ValueError(f"unsupported speed action {parser_action!r}")

    change = str(parameters.get("change", "")).strip().upper()
    explicit_target: float | None = None
    if parameters.get("target_speed_mps") is not None:
        value = _bounded_speed(
            parameters["target_speed_mps"],
            "target_speed_mps",
        )
        explicit_target = _bounded_speed(value * 3.6, "target_speed_kmh")
    elif parameters.get("target_speed_kmh") is not None:
        explicit_target = _bounded_speed(
            parameters["target_speed_kmh"],
            "target_speed_kmh",
        )

    if change not in {"INCREASE", "DECREASE", "HOLD"}:
        if explicit_target is None:
            raise ValueError(
                "ADJUST_SPEED requires INCREASE, DECREASE, or HOLD"
            )
        if explicit_target > reference + 1e-6:
            change = "INCREASE"
        elif explicit_target < reference - 1e-6:
            change = "DECREASE"
        else:
            change = "HOLD"

    if resolved_target_speed_kmh is not None:
        target = _bounded_speed(
            resolved_target_speed_kmh,
            "resolved_target_speed_kmh",
        )
    elif explicit_target is not None:
        target = explicit_target
    else:
        delta_kmh = DEFAULT_RELATIVE_SPEED_DELTA_KMH
        if parameters.get("speed_delta_mps") is not None:
            delta_mps = _finite_number(parameters["speed_delta_mps"])
            if delta_mps is None or delta_mps <= 0.0:
                raise ValueError("speed_delta_mps must be a finite positive number")
            delta_kmh = delta_mps * 3.6
        elif parameters.get("speed_delta_kmh") is not None:
            value = _finite_number(parameters["speed_delta_kmh"])
            if value is None or value <= 0.0:
                raise ValueError("speed_delta_kmh must be a finite positive number")
            delta_kmh = value

        if change == "INCREASE":
            target = min(reference + delta_kmh, MAX_TARGET_SPEED_KMH)
        elif change == "DECREASE":
            target = max(reference - delta_kmh, 0.0)
        else:
            target = reference
        target = round(target, 6)

    action = {
        "INCREASE": "accelerate",
        "DECREASE": "decelerate",
        "HOLD": "keep_lane",
    }[change]
    return action, target
