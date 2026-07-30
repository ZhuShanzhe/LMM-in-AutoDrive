"""Temporary external following policy used for integration demonstrations.

This module intentionally uses CARLA world-state JSON rather than camera-model
outputs. It demonstrates the process boundary that a real perception, risk,
and planning service will later occupy.
"""

import json
import os
import tempfile
import time


def _front_vehicle(world_state):
    candidates = []
    for vehicle in world_state.get("vehicles", []):
        relative = vehicle.get("relative_position", {})
        if relative.get("x", 0.0) > 0.0 and abs(relative.get("y", 0.0)) < 2.0:
            candidates.append(vehicle)
    return min(candidates, key=lambda item: item.get("distance", float("inf")), default=None)


class PlaceholderFollowingPolicy:
    """Minimal state machine for a safe-following architecture demonstration."""

    def __init__(self, target_speed_kmh=25.0):
        self.target_speed_kmh = float(target_speed_kmh)
        self.state = "NORMAL_DRIVING"

    def decide(self, world_state, frame_id="placeholder"):
        ego = world_state.get("ego", {})
        ego_speed = float(ego.get("speed(km/h)", 0.0))
        front = _front_vehicle(world_state)
        action = "keep_lane"
        target_speed = self.target_speed_kmh
        emergency = False
        risk_level = "low"
        reason = "placeholder_clear_road"
        risk_codes = ["placeholder_no_front_risk"]

        # Once an emergency response starts, do not resume cruise merely
        # because a single sensor snapshot temporarily loses the front actor.
        if self.state == "EMERGENCY_BRAKING":
            action = "emergency_brake"
            target_speed = 0.0
            emergency = True
            risk_level = "high"
            reason = "placeholder_emergency_brake_latched"
            risk_codes = ["placeholder_emergency_latched"]
        elif front is not None:
            distance = float(front.get("distance", float("inf")))
            front_speed = float(front.get("speed_kmh", ego_speed))
            closing_speed = ego_speed - front_speed
            if distance < 12.0 or (distance < 30.0 and closing_speed > 5.0):
                self.state = "EMERGENCY_BRAKING"
                action = "emergency_brake"
                target_speed = 0.0
                emergency = True
                risk_level = "high"
                reason = "placeholder_front_vehicle_braking"
                risk_codes = ["placeholder_close_or_closing_front_vehicle"]
            elif distance < 25.0 and ego_speed >= 8.0:
                self.state = "DECELERATING"
                action = "decelerate"
                target_speed = max(0.0, ego_speed - 2.0)
                risk_level = "medium"
                reason = "placeholder_reduce_following_speed"
                risk_codes = ["placeholder_following_gap_reduced"]
            else:
                self.state = "NORMAL_DRIVING"

        return {
            "schema_version": "1.0.0",
            "request_id": "placeholder-safe-following-001",
            "frame_id": str(frame_id),
            "decision_status": "READY",
            "action": action,
            "target_speed_kmh": round(target_speed, 3),
            "target_lane": None,
            "target_location": None,
            "emergency": emergency,
            "reason": reason,
            "parse_status": "VALID",
            "parse_confidence": None,
            "source_step_id": "step_1",
            "source_step_action": "FOLLOW",
            "source_step_count": 1,
            "matched_entity_id": None if front is None else str(front.get("id", "front_vehicle")),
            "risk_level": risk_level,
            "risk_reason_codes": risk_codes,
            "blocked_reason_codes": [],
        }


def build_control_decision(world_state, frame_id="placeholder", target_speed_kmh=25.0):
    """Return one stateless decision for unit tests and one-shot callers."""
    return PlaceholderFollowingPolicy(target_speed_kmh).decide(world_state, frame_id)


def atomic_write_json(path, document):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".decision-", suffix=".json", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
        for attempt in range(20):
            try:
                os.replace(temporary_path, path)
                temporary_path = None
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.005)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
