"""Resolve high-level ControlDecision actions against a CARLA route."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence


def _road_option_name(value: Any) -> str:
    return str(getattr(value, "name", value)).split(".")[-1].strip().upper()


def _location_dict(location: Any) -> dict[str, float]:
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
    }


def _distance(left: Any, right: Any) -> float:
    return math.hypot(float(left.x) - float(right.x), float(left.y) - float(right.y))


def route_target_location(
    route: Sequence[tuple[Any, Any]],
    route_index: int,
    *,
    action: str,
    lookahead_m: float = 12.0,
    maneuver_search_m: float = 120.0,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    if not route:
        return None, {"status": "UNAVAILABLE", "reason": "empty_route"}
    start = min(max(0, int(route_index)), len(route) - 1)
    desired = {
        "turn_left": "LEFT",
        "turn_right": "RIGHT",
    }.get(str(action))
    maneuver_index = start
    searched_m = 0.0
    previous = route[start][0].transform.location
    if desired is not None:
        found = None
        for index in range(start, len(route)):
            location = route[index][0].transform.location
            searched_m += _distance(previous, location)
            previous = location
            if _road_option_name(route[index][1]) == desired:
                found = index
                break
            if searched_m > maneuver_search_m:
                break
        if found is None:
            return None, {
                "status": "UNAVAILABLE",
                "reason": f"route_has_no_{desired.lower()}_maneuver_ahead",
                "route_index": start,
                "searched_m": round(searched_m, 3),
            }
        maneuver_index = found

    target_index = maneuver_index
    travelled = 0.0
    previous = route[maneuver_index][0].transform.location
    for index in range(maneuver_index + 1, len(route)):
        location = route[index][0].transform.location
        travelled += _distance(previous, location)
        previous = location
        target_index = index
        if travelled >= lookahead_m:
            break
    target = _location_dict(route[target_index][0].transform.location)
    return target, {
        "status": "RESOLVED",
        "reason": "route_maneuver_target"
        if desired is not None
        else "route_lookahead_target",
        "action": action,
        "route_index": start,
        "maneuver_index": maneuver_index,
        "target_index": target_index,
        "road_option": _road_option_name(route[maneuver_index][1]),
        "lookahead_m": round(travelled, 3),
        "target_location": target,
    }


def attach_route_target(
    decision: Mapping[str, Any],
    route: Sequence[tuple[Any, Any]],
    route_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = copy.deepcopy(dict(decision))
    action = str(result.get("action", ""))
    if action in {
        "stop",
        "emergency_brake",
        "lane_change_left",
        "lane_change_right",
    }:
        return result, {
            "status": "NOT_REQUIRED",
            "reason": f"{action}_owns_lateral_control",
        }
    target, diagnostics = route_target_location(
        route,
        route_index,
        action=action,
    )
    if target is not None:
        result["target_location"] = target
    return result, diagnostics
