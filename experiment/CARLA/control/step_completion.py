"""Evaluate DrivingIntent completion conditions from synchronized runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _step_by_id(
    driving_intent: Mapping[str, Any], step_id: str
) -> Mapping[str, Any]:
    matches = [
        step
        for step in driving_intent["intent"]["steps"]
        if step.get("step_id") == step_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one step {step_id!r}")
    return matches[0]


def _alignment_by_step(
    alignment: Mapping[str, Any], step_id: str
) -> Mapping[str, Any]:
    matches = [
        item
        for item in alignment.get("step_alignments", [])
        if item.get("step_id") == step_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one alignment for {step_id!r}")
    return matches[0]


def _world_object(
    world_state: Mapping[str, Any], object_id: str | None
) -> Mapping[str, Any] | None:
    if object_id is None:
        return None
    return next(
        (
            item
            for item in world_state.get("objects", [])
            if item.get("object_id") == object_id
        ),
        None,
    )


@dataclass
class _StepObservation:
    frames: int = 0
    stable_frames: int = 0
    target_seen: bool = False
    junction_seen: bool = False
    initial_progress_m: float | None = None
    initial_speed_kmh: float | None = None


@dataclass
class StepCompletionEvaluator:
    speed_tolerance_kmh: float = 2.0
    stable_frames_required: int = 8
    action_min_frames: int = 4
    _observations: dict[tuple[str, str], _StepObservation] = field(
        default_factory=dict
    )

    def reset(self, request_id: str | None = None) -> None:
        if request_id is None:
            self._observations.clear()
            return
        self._observations = {
            key: value
            for key, value in self._observations.items()
            if key[0] != request_id
        }

    def evaluate(
        self,
        driving_intent: Mapping[str, Any],
        plan_state: Mapping[str, Any] | None,
        world_state: Mapping[str, Any],
        alignment: Mapping[str, Any],
        *,
        lateral_diagnostics: Mapping[str, Any] | None = None,
        route_progress_m: float = 0.0,
        route_length_m: float | None = None,
    ) -> dict[str, Any] | None:
        if not plan_state or plan_state.get("plan_status") != "ACTIVE":
            return None
        step_id = plan_state.get("active_step_id")
        if not isinstance(step_id, str) or not step_id:
            return None
        request_id = str(driving_intent["request_id"])
        key = (request_id, step_id)
        observation = self._observations.setdefault(key, _StepObservation())
        observation.frames += 1
        if observation.initial_progress_m is None:
            observation.initial_progress_m = float(route_progress_m)
        if observation.initial_speed_kmh is None:
            observation.initial_speed_kmh = (
                float(world_state["ego"]["speed_mps"]) * 3.6
            )

        step = _step_by_id(driving_intent, step_id)
        step_alignment = _alignment_by_step(alignment, step_id)
        matched = step_alignment.get("alignment_success") is True
        observation.target_seen = observation.target_seen or matched
        is_junction = bool(
            world_state.get("ego", {}).get("is_junction")
            or world_state.get("environment", {}).get("is_intersection")
        )
        observation.junction_seen = observation.junction_seen or is_junction
        completion = step.get("completion", {}).get("type", "ACTION_REACHED")

        completed = False
        reason = ""
        if completion == "ACTION_REACHED":
            completed = observation.frames >= self.action_min_frames
            reason = "action_minimum_hold_reached"
        elif completion == "TARGET_SPEED_REACHED":
            target_mps = step.get("parameters", {}).get("target_speed_mps")
            target_kmh = float(target_mps) * 3.6
            speed_kmh = float(world_state["ego"]["speed_mps"]) * 3.6
            initial_speed_kmh = float(observation.initial_speed_kmh)
            if initial_speed_kmh > target_kmh + self.speed_tolerance_kmh:
                stable = speed_kmh <= target_kmh + self.speed_tolerance_kmh
            elif initial_speed_kmh < target_kmh - self.speed_tolerance_kmh:
                stable = speed_kmh >= target_kmh - self.speed_tolerance_kmh
            else:
                stable = (
                    abs(speed_kmh - target_kmh)
                    <= self.speed_tolerance_kmh
                )
            observation.stable_frames = (
                observation.stable_frames + 1 if stable else 0
            )
            completed = (
                observation.stable_frames >= self.stable_frames_required
            )
            reason = "target_speed_stable"
        elif completion == "LANE_CHANGE_COMPLETED":
            phase = str((lateral_diagnostics or {}).get("phase", ""))
            completed = phase in {"COMPLETE", "COMPLETE_HOLD"}
            reason = "target_lane_centered"
        elif completion == "JUNCTION_EXITED":
            completed = observation.junction_seen and not is_junction
            reason = "junction_entered_then_exited"
        elif completion in {"TARGET_CLEARED", "WAIT_CONDITION_MET"}:
            completed = observation.target_seen and not matched
            reason = "aligned_target_cleared"
        elif completion in {"VEHICLE_STOPPED", "STOPPED_BEFORE_TARGET"}:
            completed = (
                float(world_state["ego"]["speed_mps"]) <= 0.35
                and (matched or completion == "VEHICLE_STOPPED")
            )
            reason = "ego_stopped"
        elif completion == "FOLLOWING_ESTABLISHED":
            matched_entity = step_alignment.get("matched_entity")
            entity_id = (
                matched_entity.get("entity_id")
                if isinstance(matched_entity, Mapping)
                else None
            )
            obj = _world_object(world_state, entity_id)
            desired = float(
                step.get("parameters", {}).get("following_distance_m", 18.0)
            )
            distance = float(obj.get("distance_m", 1e9)) if obj else 1e9
            relative_speed = (
                abs(
                    float(
                        obj.get("relative_velocity_ego_mps", {}).get(
                            "longitudinal", 1e9
                        )
                    )
                )
                if obj
                else 1e9
            )
            stable = 5.0 <= distance <= desired + 4.0 and relative_speed <= 2.0
            observation.stable_frames = (
                observation.stable_frames + 1 if stable else 0
            )
            completed = (
                observation.stable_frames >= self.stable_frames_required
            )
            reason = "following_gap_stable"
        elif completion == "DURATION_ELAPSED":
            duration_s = float(step.get("parameters", {}).get("duration_s", 0.0))
            elapsed_s = observation.frames * 0.05
            completed = elapsed_s >= duration_s
            reason = "duration_elapsed"
        elif completion == "TARGET_REACHED":
            completed = (
                route_length_m is not None
                and float(route_progress_m) >= float(route_length_m) - 10.0
            )
            reason = "route_target_reached"
        else:
            progress_delta = float(route_progress_m) - float(
                observation.initial_progress_m
            )
            completed = progress_delta >= 12.0
            reason = "action_progress_reached"

        if not completed:
            return None
        return {
            "schema_version": "1.0.0",
            "request_id": request_id,
            "step_id": step_id,
            "frame_id": str(world_state["frame_id"]),
            "outcome": "COMPLETED",
            "reason_codes": [reason],
        }
