"""Generate conservative StepFeedback from metric execution observations."""

from __future__ import annotations

import math
from typing import Any, Mapping

from scene_understanding.src.control_plan_executor import validate_step_feedback


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _active_step(
    driving_intent: Mapping[str, Any], plan_state: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    step_id = plan_state.get("active_step_id")
    steps = driving_intent.get("intent", {}).get("steps", [])
    matches = [step for step in steps if step.get("step_id") == step_id]
    return matches[0] if len(matches) == 1 else None


def _new_tracker(
    request_id: str,
    step_id: str,
    timestamp_s: float | None,
    *,
    world_state: Mapping[str, Any],
    control_decision: Mapping[str, Any],
) -> dict[str, Any]:
    ego = world_state.get("ego", {})
    initial_speed = _number(ego.get("speed_mps")) if isinstance(ego, Mapping) else None
    initial_lane = ego.get("lane_id") if isinstance(ego, Mapping) else None
    target_lane_id = None
    target_lane = control_decision.get("target_lane")
    adjacent = ego.get("adjacent_lanes", {}) if isinstance(ego, Mapping) else {}
    if isinstance(adjacent, Mapping) and target_lane in {"left", "right"}:
        candidate = adjacent.get(target_lane)
        if isinstance(candidate, Mapping) and isinstance(candidate.get("lane_id"), int):
            target_lane_id = candidate["lane_id"]
    return {
        "schema_version": "1.0.0",
        "request_id": request_id,
        "active_step_id": step_id,
        "active_started_timestamp_s": timestamp_s,
        "junction_observed": False,
        "initial_speed_mps": initial_speed,
        "initial_lane_id": initial_lane if isinstance(initial_lane, int) else None,
        "target_lane_id": target_lane_id,
        "matched_entity_id": control_decision.get("matched_entity_id"),
        "target_observed_crossing": False,
        "target_observed_ahead": False,
        "following_stable_frames": 0,
        "stable_frames": 0,
    }


def _feedback(
    request_id: str, frame_id: str, step_id: str, outcome: str, reasons: list[str]
) -> dict[str, Any]:
    value = {
        "schema_version": "1.0.0",
        "request_id": request_id,
        "frame_id": frame_id,
        "step_id": step_id,
        "outcome": outcome,
        "reason_codes": reasons,
    }
    errors = validate_step_feedback(value)
    if errors:
        raise ValueError("invalid generated StepFeedback: " + "; ".join(errors))
    return value


def _world_object(world_state: Mapping[str, Any], object_id: Any) -> Mapping[str, Any] | None:
    if not isinstance(object_id, str) or not object_id:
        return None
    objects = world_state.get("objects")
    if not isinstance(objects, list):
        return None
    matches = [item for item in objects if isinstance(item, Mapping) and item.get("object_id") == object_id]
    return matches[0] if len(matches) == 1 else None


def _relative_longitudinal(object_state: Mapping[str, Any] | None) -> float | None:
    if object_state is None:
        return None
    relative = object_state.get("relative_position_ego_m")
    return _number(relative.get("longitudinal")) if isinstance(relative, Mapping) else None


def _relative_lateral(object_state: Mapping[str, Any] | None) -> float | None:
    if object_state is None:
        return None
    relative = object_state.get("relative_position_ego_m")
    return _number(relative.get("lateral")) if isinstance(relative, Mapping) else None


def _target_cleared_feedback(
    *,
    request_id: str,
    frame_id: str,
    step_id: str,
    step: Mapping[str, Any],
    tracker: dict[str, Any],
    world_state: Mapping[str, Any],
    pedestrian_clearance_lateral_m: float,
    minimum_speed_reduction_mps: float,
    overtake_rear_clearance_m: float,
    required_stable_frames: int,
) -> dict[str, Any] | None:
    """Evaluate only grounded pedestrian yield and vehicle overtake clearance."""

    target = _world_object(world_state, tracker.get("matched_entity_id"))
    longitudinal = _relative_longitudinal(target)
    action = str(step.get("action", "")).upper()
    purpose = str(step.get("purpose", "")).upper()

    if target is not None and target.get("category") == "pedestrian":
        if target.get("lane_relation") == "crossing_ego_path":
            tracker["target_observed_crossing"] = True
        if not tracker.get("target_observed_crossing"):
            return None
        lateral = _relative_lateral(target)
        initial_speed = _number(tracker.get("initial_speed_mps"))
        ego = world_state.get("ego", {})
        current_speed = _number(ego.get("speed_mps")) if isinstance(ego, Mapping) else None
        if (
            target.get("lane_relation") != "crossing_ego_path"
            and lateral is not None
            and abs(lateral) >= pedestrian_clearance_lateral_m
            and initial_speed is not None
            and current_speed is not None
            and initial_speed - current_speed >= minimum_speed_reduction_mps
        ):
            return _feedback(
                request_id,
                frame_id,
                step_id,
                "COMPLETED",
                ["pedestrian_crossing_cleared", "ego_speed_reduced", "collision_free"],
            )
        return None

    if target is not None and target.get("category") == "vehicle" and (action == "OVERTAKE" or purpose == "OVERTAKE"):
        if longitudinal is not None and longitudinal > 0:
            tracker["target_observed_ahead"] = True
        if not tracker.get("target_observed_ahead"):
            return None
        cleared = longitudinal is not None and longitudinal <= -overtake_rear_clearance_m
        tracker["stable_frames"] = int(tracker.get("stable_frames", 0)) + 1 if cleared else 0
        if tracker["stable_frames"] >= required_stable_frames:
            return _feedback(
                request_id,
                frame_id,
                step_id,
                "COMPLETED",
                ["slow_vehicle_passed_with_rear_clearance", "clearance_stable", "collision_free"],
            )
    return None


def _lane_change_feedback(
    *,
    request_id: str,
    frame_id: str,
    step_id: str,
    tracker: dict[str, Any],
    world_state: Mapping[str, Any],
    required_stable_frames: int,
) -> dict[str, Any] | None:
    """Confirm a target lane identity over multiple collision-free frames."""

    matched_entity_id = tracker.get("matched_entity_id")
    if isinstance(matched_entity_id, str) and matched_entity_id:
        # A target-grounded lane-change instruction must not become complete
        # merely because the target left the collector radius.
        if _world_object(world_state, matched_entity_id) is None:
            return None
    ego = world_state.get("ego", {})
    current_lane = ego.get("lane_id") if isinstance(ego, Mapping) else None
    target_lane = tracker.get("target_lane_id")
    if not isinstance(current_lane, int) or not isinstance(target_lane, int):
        return None
    tracker["stable_frames"] = int(tracker.get("stable_frames", 0)) + 1 if current_lane == target_lane else 0
    if tracker["stable_frames"] < required_stable_frames:
        return None
    return _feedback(
        request_id,
        frame_id,
        step_id,
        "COMPLETED",
        ["target_lane_reached", "target_lane_stable", "collision_free"],
    )


def _following_feedback(
    *,
    request_id: str,
    frame_id: str,
    step_id: str,
    tracker: dict[str, Any],
    world_state: Mapping[str, Any],
    following_distance_m: float,
    required_stable_frames: int,
) -> dict[str, Any] | None:
    """Confirm a stable, measurable gap to the grounded lead vehicle."""

    target = _world_object(world_state, tracker.get("matched_entity_id"))
    if target is None or target.get("category") != "vehicle":
        tracker["following_stable_frames"] = 0
        return None
    longitudinal = _relative_longitudinal(target)
    relative_speed = _number(target.get("relative_longitudinal_speed_mps"))
    minimum_gap = max(5.0, 0.70 * following_distance_m)
    maximum_gap = max(following_distance_m + 8.0, 1.60 * following_distance_m)
    stable = (
        longitudinal is not None
        and minimum_gap <= longitudinal <= maximum_gap
        and relative_speed is not None
        and abs(relative_speed) <= 2.0
    )
    tracker["following_stable_frames"] = (
        int(tracker.get("following_stable_frames", 0)) + 1 if stable else 0
    )
    if tracker["following_stable_frames"] < required_stable_frames:
        return None
    return _feedback(
        request_id,
        frame_id,
        step_id,
        "COMPLETED",
        ["following_gap_established", "relative_speed_stable", "collision_free"],
    )


def evaluate_execution_feedback(
    driving_intent: Mapping[str, Any],
    plan_state: Mapping[str, Any],
    control_decision: Mapping[str, Any],
    world_state: Mapping[str, Any],
    *,
    tracker: Mapping[str, Any] | None = None,
    speed_tolerance_mps: float = 0.5,
    stop_speed_threshold_mps: float = 0.2,
    target_tolerance_m: float = 3.0,
    pedestrian_clearance_lateral_m: float = 2.5,
    minimum_speed_reduction_mps: float = 3.0,
    overtake_rear_clearance_m: float = 8.0,
    required_stable_frames: int = 5,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return updated tracker and terminal feedback when a condition is proven.

    TARGET_CLEARED is generated only for an actor that was first observed in
    the expected conflict geometry and remains tracked through a measurable
    clearance.  LANE_CHANGE_COMPLETED requires the map-derived target lane ID
    to remain stable for consecutive collision-free frames.  Other semantic
    conditions remain explicit-feedback only.
    """

    thresholds = (
        speed_tolerance_mps,
        stop_speed_threshold_mps,
        target_tolerance_m,
        pedestrian_clearance_lateral_m,
        minimum_speed_reduction_mps,
        overtake_rear_clearance_m,
    )
    if any(value < 0 for value in thresholds) or required_stable_frames <= 0:
        raise ValueError("execution feedback thresholds must be non-negative")
    if plan_state.get("plan_status") != "ACTIVE":
        return None, None
    request_id = plan_state.get("request_id")
    frame_id = world_state.get("frame_id")
    step = _active_step(driving_intent, plan_state)
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("ControlPlanState request_id is required")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("WorldState frame_id is required")
    if step is None:
        raise ValueError("active ControlPlanState step is missing from DrivingIntent")
    step_id = step["step_id"]
    if control_decision.get("request_id") != request_id:
        raise ValueError("ControlDecision request_id does not match ControlPlanState")
    if control_decision.get("frame_id") != frame_id:
        return None, None
    if control_decision.get("source_step_id") != step_id:
        return None, None

    completion = step.get("completion")
    completion_type = completion.get("type") if isinstance(completion, Mapping) else None
    decision_status = control_decision.get("decision_status")
    if decision_status != "READY":
        # The first valid clearance frame normally stops matching an AHEAD or
        # AHEAD_CROSSING target.  The decision bridge then reports an
        # alignment-only block for that same active step.  Permit the
        # dedicated clearance evaluator to close the step, but never bypass a
        # risk, emergency, stale-frame or unrelated blocked decision.
        blocked_reasons = control_decision.get("blocked_reason_codes")
        alignment_loss_only = (
            decision_status == "BLOCKED"
            and completion_type == "TARGET_CLEARED"
            and isinstance(blocked_reasons, list)
            and "no_matching_entity" in blocked_reasons
            and control_decision.get("action") != "emergency_brake"
        )
        if not alignment_loss_only:
            return None, None

    timestamp_s = _number(world_state.get("timestamp_s"))
    current_tracker = dict(tracker or {})
    if (
        current_tracker.get("request_id") != request_id
        or current_tracker.get("active_step_id") != step_id
    ):
        current_tracker = _new_tracker(
            request_id,
            step_id,
            timestamp_s,
            world_state=world_state,
            control_decision=control_decision,
        )

    collisions = world_state.get("sensor_events", {}).get("collisions", [])
    if isinstance(collisions, list) and collisions:
        return current_tracker, _feedback(
            request_id, frame_id, step_id, "FAILED", ["collision_detected"]
        )

    if not isinstance(completion, Mapping):
        return current_tracker, None
    ego = world_state.get("ego", {})
    speed_mps = _number(ego.get("speed_mps")) if isinstance(ego, Mapping) else None
    parameters = step.get("parameters", {})
    if not isinstance(parameters, Mapping):
        return current_tracker, None

    if completion_type == "TARGET_SPEED_REACHED":
        target_speed = _number(parameters.get("target_speed_mps"))
        if speed_mps is not None and target_speed is not None and abs(speed_mps - target_speed) <= speed_tolerance_mps:
            return current_tracker, _feedback(
                request_id, frame_id, step_id, "COMPLETED", ["target_speed_reached"]
            )
    elif completion_type == "ACTION_REACHED":
        # Atomic observation/route-following steps have no separate metric
        # target. Require the approved action to remain active for several
        # consecutive frames so one transient proposal cannot advance a
        # compound command.
        current_tracker["stable_frames"] = int(
            current_tracker.get("stable_frames", 0)
        ) + 1
        if current_tracker["stable_frames"] >= required_stable_frames:
            return current_tracker, _feedback(
                request_id,
                frame_id,
                step_id,
                "COMPLETED",
                ["approved_action_stable"],
            )
    elif completion_type == "VEHICLE_STOPPED":
        if speed_mps is not None and speed_mps <= stop_speed_threshold_mps:
            return current_tracker, _feedback(
                request_id, frame_id, step_id, "COMPLETED", ["vehicle_stopped"]
            )
    elif completion_type == "STOPPED_BEFORE_TARGET":
        target = _world_object(world_state, current_tracker.get("matched_entity_id"))
        longitudinal = _relative_longitudinal(target)
        minimum_distance = _number(parameters.get("distance_m"))
        minimum_distance = 3.0 if minimum_distance is None else max(0.0, minimum_distance)
        if (
            speed_mps is not None
            and speed_mps <= stop_speed_threshold_mps
            and longitudinal is not None
            and longitudinal >= minimum_distance
        ):
            return current_tracker, _feedback(
                request_id,
                frame_id,
                step_id,
                "COMPLETED",
                ["vehicle_stopped", "target_clearance_preserved"],
            )
    elif completion_type == "DURATION_ELAPSED":
        duration_s = _number(parameters.get("duration_s"))
        started_s = _number(current_tracker.get("active_started_timestamp_s"))
        if duration_s is not None and started_s is not None and timestamp_s is not None and timestamp_s - started_s >= duration_s:
            return current_tracker, _feedback(
                request_id, frame_id, step_id, "COMPLETED", ["duration_elapsed"]
            )
    elif completion_type == "JUNCTION_EXITED":
        is_junction = ego.get("is_junction") if isinstance(ego, Mapping) else None
        if is_junction is True:
            current_tracker["junction_observed"] = True
        elif is_junction is False and current_tracker.get("junction_observed"):
            return current_tracker, _feedback(
                request_id, frame_id, step_id, "COMPLETED", ["junction_exited"]
            )
    elif completion_type == "TARGET_REACHED":
        target = parameters.get("target_location")
        position = ego.get("position_world_m") if isinstance(ego, Mapping) else None
        if isinstance(target, Mapping) and isinstance(position, Mapping):
            coordinates = [
                (_number(target.get(axis)), _number(position.get(axis)))
                for axis in ("x", "y", "z")
            ]
            if all(left is not None and right is not None for left, right in coordinates):
                distance = math.sqrt(sum((left - right) ** 2 for left, right in coordinates))
                if distance <= target_tolerance_m:
                    return current_tracker, _feedback(
                        request_id, frame_id, step_id, "COMPLETED", ["target_location_reached"]
                    )
    elif completion_type == "LANE_CHANGE_COMPLETED":
        feedback = _lane_change_feedback(
            request_id=request_id,
            frame_id=frame_id,
            step_id=step_id,
            tracker=current_tracker,
            world_state=world_state,
            required_stable_frames=required_stable_frames,
        )
        if feedback is not None:
            return current_tracker, feedback
    elif completion_type == "TARGET_CLEARED":
        feedback = _target_cleared_feedback(
            request_id=request_id,
            frame_id=frame_id,
            step_id=step_id,
            step=step,
            tracker=current_tracker,
            world_state=world_state,
            pedestrian_clearance_lateral_m=pedestrian_clearance_lateral_m,
            minimum_speed_reduction_mps=minimum_speed_reduction_mps,
            overtake_rear_clearance_m=overtake_rear_clearance_m,
            required_stable_frames=required_stable_frames,
        )
        if feedback is not None:
            return current_tracker, feedback
    elif completion_type == "FOLLOWING_ESTABLISHED":
        following_distance = _number(parameters.get("following_distance_m"))
        if following_distance is None:
            following_distance = 20.0
        feedback = _following_feedback(
            request_id=request_id,
            frame_id=frame_id,
            step_id=step_id,
            tracker=current_tracker,
            world_state=world_state,
            following_distance_m=following_distance,
            required_stable_frames=required_stable_frames,
        )
        if feedback is not None:
            return current_tracker, feedback
    return current_tracker, None
