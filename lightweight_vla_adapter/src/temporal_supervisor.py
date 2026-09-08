from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import validate_vla_proposal


LONGITUDINAL_SEVERITY = {
    "accelerate": 0,
    "keep_lane": 1,
    "decelerate": 2,
    "stop": 3,
    "emergency_brake": 4,
}


@dataclass(frozen=True)
class TemporalSupervisorConfig:
    switch_confirm_frames: int = 3
    accelerate_confirm_frames: int = 5
    target_switch_confirm_frames: int = 3
    speed_rise_rate_kmh_s: float = 8.0
    default_frame_interval_s: float = 0.05
    closing_speed_threshold_mps: float = 0.5
    caution_ttc_s: float = 4.0
    caution_distance_m: float = 8.0
    caution_crawl_speed_kmh: float = 10.0
    max_state_gap_s: float = 2.0
    max_streams: int = 128

    def __post_init__(self) -> None:
        integer_fields = (
            self.switch_confirm_frames,
            self.accelerate_confirm_frames,
            self.target_switch_confirm_frames,
            self.max_streams,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("temporal integer settings must be positive")
        numeric_fields = (
            self.speed_rise_rate_kmh_s,
            self.default_frame_interval_s,
            self.closing_speed_threshold_mps,
            self.caution_ttc_s,
            self.caution_distance_m,
            self.caution_crawl_speed_kmh,
            self.max_state_gap_s,
        )
        if any(not math.isfinite(value) or value <= 0 for value in numeric_fields):
            raise ValueError("temporal supervisor thresholds must be positive")


@dataclass
class _StreamState:
    action: str | None = None
    target_speed_kmh: float | None = None
    target_entity_id: str | None = None
    pending_action: str | None = None
    pending_action_frames: int = 0
    pending_target: str | None = None
    pending_target_frames: int = 0
    last_frame_id: str | None = None
    last_timestamp_s: float | None = None


class TemporalProposalSupervisor:
    """Stabilize frame-wise VLA proposals before the deterministic safety gate."""

    def __init__(
        self,
        config: TemporalSupervisorConfig | None = None,
    ) -> None:
        self.config = config or TemporalSupervisorConfig()
        self._states: dict[str, _StreamState] = {}
        self._diagnostics: dict[str, dict[str, Any]] = {}

    def reset(self, stream_id: str | None = None) -> None:
        if stream_id is None:
            self._states.clear()
            self._diagnostics.clear()
            return
        self._states.pop(stream_id, None)
        self._diagnostics.pop(stream_id, None)

    def diagnostics(self, stream_id: str) -> dict[str, Any] | None:
        value = self._diagnostics.get(stream_id)
        return copy.deepcopy(value) if value is not None else None

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    @classmethod
    def _relative_motion(
        cls,
        entity: Mapping[str, Any],
    ) -> tuple[float, float, float]:
        position = (
            entity.get("relative_position_m")
            or entity.get("relative_position_ego_m")
            or entity.get("relative_position")
            or {}
        )
        velocity = (
            entity.get("relative_velocity_mps")
            or entity.get("relative_velocity_ego_mps")
            or entity.get("velocity")
            or {}
        )
        if isinstance(position, Mapping):
            x = cls._number(position.get("x", position.get("longitudinal")))
            y = cls._number(position.get("y", position.get("lateral")))
        elif isinstance(position, (list, tuple)) and len(position) >= 2:
            x = cls._number(position[0])
            y = cls._number(position[1])
        else:
            x = cls._number(entity.get("distance_m"))
            y = 0.0
        if isinstance(velocity, Mapping):
            relative_vx = cls._number(
                velocity.get("x", velocity.get("longitudinal"))
            )
        elif isinstance(velocity, (list, tuple)) and velocity:
            relative_vx = cls._number(velocity[0])
        else:
            relative_vx = 0.0
        return x, y, relative_vx

    @staticmethod
    def _is_vehicle(entity: Mapping[str, Any]) -> bool:
        label = str(
            entity.get("category")
            or entity.get("class")
            or entity.get("type")
            or ""
        ).lower()
        return any(token in label for token in ("vehicle", "car", "truck", "bus"))

    @staticmethod
    def _same_lane(entity: Mapping[str, Any], lateral_m: float) -> bool:
        relation = str(entity.get("lane_relation") or "").lower()
        if relation:
            return "same" in relation or "ego" in relation
        return abs(lateral_m) <= 2.0

    def _lead_vehicle_signal(
        self,
        world_state: Mapping[str, Any],
    ) -> dict[str, float | str | None]:
        nearest: tuple[float, str | None, float] | None = None
        objects = world_state.get("objects", [])
        if not isinstance(objects, list):
            return {
                "entity_id": None,
                "distance_m": None,
                "closing_speed_mps": 0.0,
                "ttc_s": None,
            }
        for entity in objects:
            if not isinstance(entity, Mapping) or not self._is_vehicle(entity):
                continue
            x, y, relative_vx = self._relative_motion(entity)
            if x <= 0.0 or not self._same_lane(entity, y):
                continue
            entity_id = entity.get("entity_id") or entity.get("object_id")
            candidate = (x, str(entity_id) if entity_id is not None else None, relative_vx)
            if nearest is None or candidate[0] < nearest[0]:
                nearest = candidate
        if nearest is None:
            return {
                "entity_id": None,
                "distance_m": None,
                "closing_speed_mps": 0.0,
                "ttc_s": None,
            }
        distance_m, entity_id, relative_vx = nearest
        closing_speed_mps = max(0.0, -relative_vx)
        ttc_s = (
            distance_m / closing_speed_mps
            if closing_speed_mps >= self.config.closing_speed_threshold_mps
            else None
        )
        return {
            "entity_id": entity_id,
            "distance_m": distance_m,
            "closing_speed_mps": closing_speed_mps,
            "ttc_s": ttc_s,
        }

    @staticmethod
    def _timestamp(world_state: Mapping[str, Any]) -> float | None:
        for key in ("timestamp_s", "sim_time_s", "elapsed_seconds"):
            value = world_state.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
        return None

    @staticmethod
    def _visible_entity_ids(world_state: Mapping[str, Any]) -> set[str]:
        result: set[str] = set()
        objects = world_state.get("objects", [])
        if not isinstance(objects, list):
            return result
        for entity in objects:
            if not isinstance(entity, Mapping):
                continue
            entity_id = entity.get("entity_id") or entity.get("object_id")
            if entity_id is not None:
                result.add(str(entity_id))
        return result

    def _constrain_for_scene(
        self,
        proposal: dict[str, Any],
        world_state: Mapping[str, Any],
        risk_assessment: Mapping[str, Any],
        reasons: list[str],
    ) -> tuple[str, float, bool, dict[str, float | str | None]]:
        action = str(proposal["action"])
        target_speed = float(proposal["target_speed_kmh"])
        forced = False
        recommended = str(risk_assessment.get("recommended_action") or "")
        ego = world_state.get("ego", {})
        ego = ego if isinstance(ego, Mapping) else {}
        ego_speed_mps = self._number(ego.get("speed_mps"))
        lead = self._lead_vehicle_signal(world_state)

        if recommended == "emergency_brake":
            action = "emergency_brake"
            target_speed = 0.0
            forced = True
            reasons.append("emergency_risk_preemption")
            return action, target_speed, forced, lead

        if recommended == "decelerate":
            if action not in {"decelerate", "stop", "emergency_brake"}:
                action = "decelerate"
                reasons.append("deceleration_risk_constraint")
            caution_ceiling = max(
                self.config.caution_crawl_speed_kmh,
                ego_speed_mps * 3.6 - 3.6,
            )
            target_speed = min(
                max(target_speed, self.config.caution_crawl_speed_kmh),
                caution_ceiling,
            )
            forced = True

        ttc_s = lead["ttc_s"]
        distance_m = lead["distance_m"]
        closing_speed_mps = float(lead["closing_speed_mps"] or 0.0)
        caution = (
            closing_speed_mps >= self.config.closing_speed_threshold_mps
            and (
                (ttc_s is not None and float(ttc_s) <= self.config.caution_ttc_s)
                or (
                    distance_m is not None
                    and float(distance_m)
                    <= max(
                        self.config.caution_distance_m,
                        ego_speed_mps * 1.5,
                    )
                )
            )
        )
        if caution and action == "accelerate":
            action = "decelerate"
            target_speed = min(
                target_speed,
                max(0.0, ego_speed_mps * 3.6 - 3.6),
            )
            forced = True
            reasons.append("closing_lead_vehicle_constraint")
        return action, target_speed, forced, lead

    def _stabilize_action(
        self,
        state: _StreamState,
        proposed_action: str,
        *,
        forced: bool,
        frame_advanced: bool,
        reasons: list[str],
    ) -> str:
        if state.action is None:
            state.action = proposed_action
            return proposed_action
        if proposed_action == state.action:
            state.pending_action = None
            state.pending_action_frames = 0
            return state.action

        current_severity = LONGITUDINAL_SEVERITY.get(state.action, 1)
        proposed_severity = LONGITUDINAL_SEVERITY.get(proposed_action, 1)
        more_conservative = proposed_severity > current_severity
        if forced or more_conservative:
            state.action = proposed_action
            state.pending_action = None
            state.pending_action_frames = 0
            reasons.append("immediate_safety_transition")
            return state.action

        required_frames = (
            self.config.accelerate_confirm_frames
            if proposed_action == "accelerate"
            else self.config.switch_confirm_frames
        )
        if state.pending_action != proposed_action:
            state.pending_action = proposed_action
            state.pending_action_frames = 1 if frame_advanced else 0
        elif frame_advanced:
            state.pending_action_frames += 1
        if state.pending_action_frames >= required_frames:
            state.action = proposed_action
            state.pending_action = None
            state.pending_action_frames = 0
            reasons.append("confirmed_action_transition")
        else:
            reasons.append("action_hysteresis_hold")
        return state.action

    def _stabilize_speed(
        self,
        state: _StreamState,
        proposed_speed_kmh: float,
        *,
        action: str,
        timestamp_s: float | None,
        risk_limited: bool,
        reasons: list[str],
    ) -> float:
        if action in {"stop", "emergency_brake"}:
            state.target_speed_kmh = 0.0
            return 0.0
        proposed_speed_kmh = min(max(proposed_speed_kmh, 0.0), 100.0)
        if state.target_speed_kmh is None:
            state.target_speed_kmh = proposed_speed_kmh
            return proposed_speed_kmh
        if proposed_speed_kmh <= state.target_speed_kmh:
            state.target_speed_kmh = proposed_speed_kmh
            return proposed_speed_kmh
        if risk_limited or action == "decelerate":
            if (
                action == "decelerate"
                and state.target_speed_kmh
                < self.config.caution_crawl_speed_kmh
                and proposed_speed_kmh
                <= self.config.caution_crawl_speed_kmh
            ):
                state.target_speed_kmh = proposed_speed_kmh
                reasons.append("caution_crawl_resume")
                return proposed_speed_kmh
            reasons.append("speed_increase_blocked_during_deceleration")
            return state.target_speed_kmh
        delta_s = self.config.default_frame_interval_s
        if timestamp_s is not None and state.last_timestamp_s is not None:
            measured = timestamp_s - state.last_timestamp_s
            if math.isfinite(measured) and measured > 0.0:
                delta_s = min(measured, 1.0)
        limit = (
            state.target_speed_kmh
            + self.config.speed_rise_rate_kmh_s * delta_s
        )
        stabilized = min(proposed_speed_kmh, limit)
        if stabilized < proposed_speed_kmh:
            reasons.append("speed_rise_rate_limited")
        state.target_speed_kmh = stabilized
        return stabilized

    def _stabilize_target(
        self,
        state: _StreamState,
        proposed_target: str | None,
        *,
        visible_entity_ids: set[str],
        frame_advanced: bool,
        reasons: list[str],
    ) -> str | None:
        current = state.target_entity_id
        if current is None:
            state.target_entity_id = proposed_target
            return proposed_target
        if current not in visible_entity_ids:
            state.target_entity_id = proposed_target
            state.pending_target = None
            state.pending_target_frames = 0
            reasons.append("missing_target_replaced")
            return proposed_target
        if proposed_target == current:
            state.pending_target = None
            state.pending_target_frames = 0
            return current
        if proposed_target not in visible_entity_ids:
            reasons.append("invalid_target_switch_rejected")
            return current
        if state.pending_target != proposed_target:
            state.pending_target = proposed_target
            state.pending_target_frames = 1 if frame_advanced else 0
        elif frame_advanced:
            state.pending_target_frames += 1
        if state.pending_target_frames >= self.config.target_switch_confirm_frames:
            state.target_entity_id = proposed_target
            state.pending_target = None
            state.pending_target_frames = 0
            reasons.append("confirmed_target_transition")
        else:
            reasons.append("target_hysteresis_hold")
        return state.target_entity_id

    def stabilize(
        self,
        proposal: dict[str, Any],
        world_state: Mapping[str, Any],
        risk_assessment: Mapping[str, Any],
        *,
        stream_id: str | None = None,
    ) -> dict[str, Any]:
        errors = validate_vla_proposal(proposal)
        if errors:
            raise ValueError("invalid VLADecisionProposal: " + "; ".join(errors))
        key = stream_id or str(proposal["request_id"])
        if key not in self._states and len(self._states) >= self.config.max_streams:
            oldest = next(iter(self._states))
            self.reset(oldest)
        state = self._states.setdefault(key, _StreamState())
        frame_id = str(proposal["frame_id"])
        timestamp_s = self._timestamp(world_state)
        reasons: list[str] = []
        if (
            timestamp_s is not None
            and state.last_timestamp_s is not None
            and (
                timestamp_s < state.last_timestamp_s
                or timestamp_s - state.last_timestamp_s
                > self.config.max_state_gap_s
            )
        ):
            state = _StreamState()
            self._states[key] = state
            reasons.append("temporal_state_reset_on_time_gap")
        frame_advanced = frame_id != state.last_frame_id
        action, speed, forced, lead = self._constrain_for_scene(
            proposal,
            world_state,
            risk_assessment,
            reasons,
        )
        action = self._stabilize_action(
            state,
            action,
            forced=forced,
            frame_advanced=frame_advanced,
            reasons=reasons,
        )
        speed = self._stabilize_speed(
            state,
            speed,
            action=action,
            timestamp_s=timestamp_s,
            risk_limited=forced,
            reasons=reasons,
        )
        target = self._stabilize_target(
            state,
            proposal.get("target_entity_id"),
            visible_entity_ids=self._visible_entity_ids(world_state),
            frame_advanced=frame_advanced,
            reasons=reasons,
        )
        result = copy.deepcopy(proposal)
        result["action"] = action
        result["target_speed_kmh"] = round(speed, 6)
        result["target_entity_id"] = target
        if action == "lane_change_left":
            result["target_lane"] = "left"
        elif action == "lane_change_right":
            result["target_lane"] = "right"
        elif not action.startswith("lane_change_"):
            result["target_lane"] = None
        state.last_frame_id = frame_id
        state.last_timestamp_s = timestamp_s
        self._diagnostics[key] = {
            "frame_id": frame_id,
            "raw_action": proposal["action"],
            "stabilized_action": action,
            "raw_target_speed_kmh": proposal["target_speed_kmh"],
            "stabilized_target_speed_kmh": result["target_speed_kmh"],
            "raw_target_entity_id": proposal.get("target_entity_id"),
            "stabilized_target_entity_id": target,
            "lead_vehicle": lead,
            "reasons": reasons,
        }
        final_errors = validate_vla_proposal(result)
        if final_errors:
            raise ValueError(
                "invalid stabilized VLADecisionProposal: "
                + "; ".join(final_errors)
            )
        return result
