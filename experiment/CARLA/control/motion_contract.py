"""Normalize high-level decisions into actuator-ready motion constraints."""

from __future__ import annotations

MOTION_ACTIONS = {
    "keep_lane", "accelerate", "lane_change_left", "lane_change_right",
    "turn_left", "turn_right",
}


def apply_motion_constraints(decision, context):
    """Return a copy with one deterministic speed contract for every action.

    Decisions remain responsible for safety overrides.  This boundary only
    supplies a scenario cruise speed for geometric actions that otherwise
    cannot move from rest.
    """
    result = dict(decision)
    action = result.get("action")
    requested = float(result.get("target_speed_kmh", 0.0))
    cruise = float((context or {}).get("default_speed_kmh", 0.0))
    if action in MOTION_ACTIONS and requested <= 1.0 and cruise > 0.0:
        result["target_speed_kmh"] = min(cruise, 100.0)
    return result
