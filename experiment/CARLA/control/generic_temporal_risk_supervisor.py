"""Generic temporal risk supervisor for the universal VLA control chain.

The supervisor owns only sensor-derived risk history, vehicle liveness state
and generic instruction semantics.  It never reads command ids, event ids,
scene ids, actor truth or scheduled event mileages.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


HAZARD_STOP_INTENTS = {"YIELD", "STOP", "EMERGENCY_BRAKE"}
LANE_CHANGE_INTENTS = {"CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"}
LANE_CHANGE_ACTIONS = {
    "CHANGE_LANE_LEFT": "lane_change_left",
    "CHANGE_LANE_RIGHT": "lane_change_right",
}

OVERRIDE_CRAWL = "low_risk_deceleration_crawl"
OVERRIDE_HAZARD_CLEARANCE = "temporal_hazard_clearance"
OVERRIDE_CAUTIOUS_RESUME = "cautious_hazard_resume"
OVERRIDE_LANE_CLEARANCE = "target_lane_visual_clearance"
OVERRIDE_RISK_CONFIRMATION = "temporal_risk_confirmation"
OVERRIDE_UNCONFIRMED_STOP = "unconfirmed_stop_crawl_floor"
OVERRIDE_COMMAND_SPEED_FLOOR = "low_risk_command_speed_floor"
OVERRIDE_LANE_CHANGE_TIMEOUT = "lane_change_wait_timeout"
OVERRIDE_RADAR_GUARDED_CRAWL = "stationary_high_risk_radar_guarded_crawl"


@dataclass(frozen=True)
class TemporalRiskSupervisorConfig:
    history_size: int = 20
    hold_seconds: float = 20.0
    low_fraction: float = 0.8
    high_free_tail: int = 5
    crawl_speed_kmh: float = 15.0
    min_samples: int = 20
    max_stop_gap_s: float = 2.0
    min_move_s: float = 2.0
    decision_history_size: int = 40
    high_confirm_frames: int = 3
    stopped_speed_kmh: float = 0.5
    lane_change_wait_timeout_s: float = 8.0
    radar_guarded_crawl_speed_kmh: float = 6.0
    immediate_high_probability: float = 0.80
    immediate_risk_score: float = 0.70
    immediate_horizon_probability: float = 0.65

    def __post_init__(self) -> None:
        if self.history_size < 1:
            raise ValueError("history_size must be positive")
        if not math.isfinite(self.hold_seconds) or self.hold_seconds <= 0:
            raise ValueError("hold_seconds must be positive")
        if not 0.0 < self.low_fraction <= 1.0:
            raise ValueError("low_fraction must be in (0, 1]")
        if self.high_free_tail < 1:
            raise ValueError("high_free_tail must be positive")
        if not math.isfinite(self.crawl_speed_kmh) or self.crawl_speed_kmh <= 0:
            raise ValueError("crawl_speed_kmh must be positive")
        if not math.isfinite(self.min_move_s) or self.min_move_s <= 0:
            raise ValueError("min_move_s must be positive")
        if self.high_confirm_frames < 2:
            raise ValueError("high_confirm_frames must be at least 2")
        if (
            not math.isfinite(self.stopped_speed_kmh)
            or self.stopped_speed_kmh < 0.0
        ):
            raise ValueError("stopped_speed_kmh must be non-negative")
        if (
            not math.isfinite(self.lane_change_wait_timeout_s)
            or self.lane_change_wait_timeout_s <= 0.0
        ):
            raise ValueError("lane_change_wait_timeout_s must be positive")
        if (
            not math.isfinite(self.radar_guarded_crawl_speed_kmh)
            or self.radar_guarded_crawl_speed_kmh <= 0.0
        ):
            raise ValueError(
                "radar_guarded_crawl_speed_kmh must be positive"
            )
        for name in (
            "immediate_high_probability",
            "immediate_risk_score",
            "immediate_horizon_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


class GenericTemporalRiskSupervisor:
    """Maintain risk history and generic liveness/resume gates."""

    def __init__(
        self,
        config: TemporalRiskSupervisorConfig | None = None,
    ) -> None:
        self.config = config or TemporalRiskSupervisorConfig()
        self._global_risk_history: deque[str] = deque(maxlen=self.config.history_size)
        self._lane_risk_history: dict[str, deque[str]] = {
            direction: deque(maxlen=self.config.history_size)
            for direction in ("left", "right")
        }
        self._stopped_since_frame: int | None = None
        self._resume_intent: str | None = None
        self._crawl_pending = False
        self._high_confirm_count = 0
        self._consecutive_decelerate_frames = 0
        self._radar_guarded_crawl_active = False
        self._moving_since_frame: int | None = None
        self._decision_history: deque[dict[str, Any]] = deque(
            maxlen=self.config.decision_history_size
        )
        self._last_decision_frame: int | None = None
        self._last_frame: int | None = None
        self._last_timestamp_s: float | None = None
        self._override_counts: Counter[str] = Counter()

    def reset(self) -> None:
        self._global_risk_history.clear()
        for history in self._lane_risk_history.values():
            history.clear()
        self._stopped_since_frame = None
        self._resume_intent = None
        self._crawl_pending = False
        self._high_confirm_count = 0
        self._consecutive_decelerate_frames = 0
        self._radar_guarded_crawl_active = False
        self._moving_since_frame = None
        self._decision_history.clear()
        self._last_decision_frame = None
        self._last_frame = None
        self._last_timestamp_s = None

    def observe(
        self,
        *,
        frame: int,
        timestamp_s: float | None,
        parsed_intent: str,
        risk_level: str,
        target_lane_risk_level: str | None,
        ego_speed_kmh: float,
        requested_lane_direction: str | None,
    ) -> None:
        """Update histories and the stopped-state liveness tracker."""

        risk_level = str(risk_level).lower()
        self._global_risk_history.append(risk_level)
        if (
            requested_lane_direction in {"left", "right"}
            and target_lane_risk_level is not None
        ):
            self._lane_risk_history[requested_lane_direction].append(
                str(target_lane_risk_level).lower()
            )

        gap_s = None
        if (
            self._last_frame is not None
            and self._last_timestamp_s is not None
            and timestamp_s is not None
        ):
            gap_s = max(0.0, float(timestamp_s) - float(self._last_timestamp_s))
        if (
            self._last_frame is not None
            and int(frame) - int(self._last_frame) > 20
        ):
            # A long gap means the decision stream restarted; the ordinary
            # 3-6 frame decision cadence must NOT reset the stopped tracker.
            self._stopped_since_frame = None
            self._resume_intent = None

        if ego_speed_kmh >= 1.0 and self._moving_since_frame is None:
            self._moving_since_frame = int(frame)
        if ego_speed_kmh < 0.5:
            self._moving_since_frame = None
        if ego_speed_kmh < 0.5:
            if self._stopped_since_frame is None:
                self._stopped_since_frame = int(frame)
        elif ego_speed_kmh >= 1.0:
            self._stopped_since_frame = None
            self._resume_intent = None

        if parsed_intent != self._resume_intent:
            self._resume_intent = None

        self._last_frame = int(frame)
        self._last_timestamp_s = (
            float(timestamp_s) if timestamp_s is not None else None
        )

    def record_decision(
        self,
        *,
        frame: int,
        risk_level: str,
        action: str,
        target_speed_kmh: float,
        override: str | None,
    ) -> None:
        """Record the recent risk/action timeline for the temporal gate."""

        self._decision_history.append(
            {
                "frame": int(frame),
                "risk_level": str(risk_level).lower(),
                "action": action,
                "target_speed_kmh": float(target_speed_kmh),
                "override": override,
            }
        )
        self._last_decision_frame = int(frame)

    def moving_elapsed_s(
        self,
        frame: int,
        fixed_delta_seconds: float,
    ) -> float:
        if self._moving_since_frame is None:
            return 0.0
        return max(
            0.0,
            (int(frame) - int(self._moving_since_frame))
            * float(fixed_delta_seconds),
        )

    def stationary_elapsed_s(
        self,
        frame: int,
        fixed_delta_seconds: float,
    ) -> float:
        if self._stopped_since_frame is None:
            return 0.0
        return max(
            0.0,
            (int(frame) - int(self._stopped_since_frame))
            * float(fixed_delta_seconds),
        )

    @property
    def resume_intent(self) -> str | None:
        return self._resume_intent

    def clearance_evidence(
        self,
        risk_levels: Sequence[str],
    ) -> bool:
        levels = [str(level).lower() for level in risk_levels]
        if len(levels) < self.config.min_samples:
            return False
        window = levels[-self.config.min_samples :]
        required_low = math.ceil(self.config.min_samples * self.config.low_fraction)
        tail = window[-min(self.config.high_free_tail, len(window)) :]
        return (
            window.count("low") >= required_low and "high" not in tail
        ) or ("high" not in window)

    def apply(
        self,
        final_decision: Mapping[str, Any],
        canonical: Mapping[str, Any],
        risk: Mapping[str, Any],
        *,
        parsed_intent: str,
        requested_lane_direction: str | None,
        target_lane_risk: Mapping[str, Any] | None,
        stationary_elapsed_s: float,
        resume_active: bool,
        resume_speed_kmh: float,
        hold_seconds: float | None = None,
        frame: int | None = None,
        fixed_delta_seconds: float = 0.05,
    ) -> tuple[dict[str, Any], str | None]:
        """Apply the generic crawl/resume/lane-clearance gates."""

        decision = dict(final_decision)
        current_frame = (
            int(frame)
            if frame is not None
            else (self._last_decision_frame or 0)
        )
        hold = (
            float(hold_seconds)
            if hold_seconds is not None
            else self.config.hold_seconds
        )
        risk_level = str(risk.get("risk_level", "high")).lower()
        low_risk = risk_level == "low"
        stop_like = str(decision.get("action")) in {"stop", "emergency_brake"}
        stopped_target = float(decision.get("target_speed_kmh", 0.0)) <= 0.1
        moving = self._moving_since_frame is not None
        probabilities = risk.get("probabilities") or {}
        high_probability = float(probabilities.get("high", 0.0) or 0.0)
        risk_score = float(risk.get("risk_score", 0.0) or 0.0)
        horizons = risk.get("horizon_probabilities") or []
        imminent_horizon = max(
            [float(value) for value in horizons[:2]] or [0.0]
        )
        immediate_high = (
            high_probability >= self.config.immediate_high_probability
            or risk_score >= self.config.immediate_risk_score
            or imminent_horizon >= self.config.immediate_horizon_probability
        )

        radar = risk.get("physical_forward_radar") or {}
        radar_distance = radar.get("nearest_distance_m")
        radar_emergency_distance = radar.get("emergency_distance_m")
        radar_caution_distance = radar.get("caution_distance_m")
        radar_guarded_clearance = False
        if all(
            isinstance(value, (int, float))
            for value in (
                radar_distance,
                radar_emergency_distance,
                radar_caution_distance,
            )
        ):
            distance_m = float(radar_distance)
            emergency_m = float(radar_emergency_distance)
            caution_m = float(radar_caution_distance)
            radar_guarded_clearance = (
                math.isfinite(distance_m)
                and math.isfinite(emergency_m)
                and math.isfinite(caution_m)
                and distance_m > emergency_m + 0.25
                and distance_m <= caution_m + 4.0
            )
        if risk_level != "high" or not radar_guarded_clearance:
            self._radar_guarded_crawl_active = False

        # Generic unconfirmed-stop crawl floor: when the learned risk head says
        # low/medium and the text envelope does not request a stop, a model
        # stop action is treated as an unconfirmed flicker and converted to a
        # cautious crawl.  High risk still stops immediately.
        if (
            stop_like
            and stopped_target
            and risk_level in {"low", "medium"}
            and str(canonical.get("action")) not in {"stop", "emergency_brake"}
        ):
            decision.update(
                action="decelerate",
                target_speed_kmh=self.config.crawl_speed_kmh,
                target_lane=None,
                emergency=False,
                reason="unconfirmed_stop_crawl_floor",
                blocked_reason_codes=[],
            )
            self._override_counts[OVERRIDE_UNCONFIRMED_STOP] += 1
            return decision, OVERRIDE_UNCONFIRMED_STOP

        # Generic lane-change liveness: never block the road indefinitely
        # while waiting for a safe lane-change gap.  After a configurable
        # stationary timeout, fall back to a cautious crawl in the current
        # lane so the ego keeps moving instead of deadlocking.
        if (
            parsed_intent in LANE_CHANGE_INTENTS
            and stationary_elapsed_s >= self.config.lane_change_wait_timeout_s
            and str(decision.get("action"))
            in {"decelerate", "lane_change_left", "lane_change_right"}
        ):
            decision.update(
                action="keep_lane",
                target_speed_kmh=self.config.crawl_speed_kmh,
                target_lane=None,
                emergency=False,
                reason="lane_change_wait_timeout",
                blocked_reason_codes=[],
            )
            self._override_counts[OVERRIDE_LANE_CHANGE_TIMEOUT] += 1
            return decision, OVERRIDE_LANE_CHANGE_TIMEOUT

        # Generic stationary high-risk liveness: a learned high signal while
        # the ego has been stopped for a while without an explicit text stop
        # is treated as a view false-positive and downgraded to a cautious
        # crawl, so the vehicle never deadlocks on an empty scene.
        stationary_high = (
            risk_level == "high"
            and (
                stationary_elapsed_s >= 2.0
                or self._radar_guarded_crawl_active
            )
            and str(decision.get("action")) == "emergency_brake"
            and parsed_intent not in {"STOP", "EMERGENCY_BRAKE"}
        )
        unconfirmed_high = (
            high_probability < self.config.immediate_high_probability
            and risk_score < self.config.immediate_risk_score
            and imminent_horizon < self.config.immediate_horizon_probability
        )
        if stationary_high and (unconfirmed_high or radar_guarded_clearance):
            radar_guarded = radar_guarded_clearance
            self._radar_guarded_crawl_active = radar_guarded
            speed_kmh = (
                self.config.radar_guarded_crawl_speed_kmh
                if radar_guarded
                else 12.0
            )
            reason = (
                OVERRIDE_RADAR_GUARDED_CRAWL
                if radar_guarded
                else "stationary_high_risk_liveness"
            )
            decision.update(
                action="decelerate",
                target_speed_kmh=speed_kmh,
                emergency=False,
                reason=reason,
                blocked_reason_codes=[],
            )
            self._override_counts[reason] += 1
            return decision, reason

        # Generic low-risk command-speed floor: when the text command defines
        # an explicit speed envelope and the learned risk head says low or
        # medium (no hazard), a model over-deceleration below that envelope is
        # treated as a cautious flicker and floored back to the commanded
        # speed.  Explicit stop/yield commands are never floored.
        canonical_speed = float(canonical.get("target_speed_kmh", 0.0) or 0.0)
        model_decelerates = str(decision.get("action")) == "decelerate"
        model_speed = float(decision.get("target_speed_kmh", 0.0) or 0.0)
        if model_decelerates:
            self._consecutive_decelerate_frames += 1
        else:
            self._consecutive_decelerate_frames = 0
        if (
            low_risk
            and model_decelerates
            and self._consecutive_decelerate_frames <= 1
            and not stopped_target
            and str(canonical.get("action"))
            not in {"stop", "emergency_brake"}
            and canonical_speed > 0.0
            and model_speed < canonical_speed
        ):
            decision.update(
                target_speed_kmh=canonical_speed,
                reason="low_risk_command_speed_floor",
            )
            self._override_counts[OVERRIDE_COMMAND_SPEED_FLOOR] += 1
            return decision, OVERRIDE_COMMAND_SPEED_FLOOR

        # Generic temporal confirmation of a high-risk signal.  A noisy high
        # frame does not slam the brakes while the ego is moving: the first
        # N-1 consecutive high decisions are held as a cautious crawl, and
        # only the Nth consecutive high decision escalates to an emergency
        # stop.  Text that explicitly demands a stop, or a high risk while
        # the ego is already stationary, is never delayed.
        if risk_level == "high":
            stop_intent = parsed_intent in HAZARD_STOP_INTENTS
            if (
                moving
                and not stop_intent
                and not immediate_high
                and self._high_confirm_count + 1
                < self.config.high_confirm_frames
            ):
                self._high_confirm_count += 1
                self._crawl_pending = True
                decision.update(
                    action="decelerate",
                    target_speed_kmh=self.config.crawl_speed_kmh,
                    target_lane=None,
                    emergency=False,
                    reason="temporal_risk_confirmation_hold_crawl",
                    blocked_reason_codes=[],
                )
                self._override_counts[OVERRIDE_RISK_CONFIRMATION] += 1
                return decision, OVERRIDE_RISK_CONFIRMATION
            # Keep a confirmed high latched until risk actually clears; the
            # old reset alternated persistent hazards between emergency and
            # crawl every decision interval and increased stopping distance.
            self._high_confirm_count = self.config.high_confirm_frames
            self._crawl_pending = False
        else:
            crawl_decision = (
                str(decision.get("action")) == "decelerate"
                and float(decision.get("target_speed_kmh", 0.0))
                <= self.config.crawl_speed_kmh + 0.01
            )
            if crawl_decision:
                self._crawl_pending = True
                self._high_confirm_count = 0
            else:
                self._crawl_pending = False
                self._high_confirm_count = 0

        # 1) Generic low-risk deceleration crawl floor.
        model_decelerates = str(decision.get("action")) == "decelerate"
        envelope_decelerates = str(canonical.get("action")) == "decelerate"
        if (
            low_risk
            and model_decelerates
            and envelope_decelerates
            and stopped_target
        ):
            decision.update(
                target_speed_kmh=self.config.crawl_speed_kmh,
                emergency=False,
                reason="low_risk_deceleration_crawl_floor",
            )
            self._override_counts[OVERRIDE_CRAWL] += 1
            return decision, OVERRIDE_CRAWL

        # 2) Generic hazard clearance / cautious resume.
        if parsed_intent in HAZARD_STOP_INTENTS:
            hazard_clear = self.clearance_evidence(
                list(self._global_risk_history)
            )
            initial_clearance = (
                risk_level in {"low", "medium"}
                and hazard_clear
                and stop_like
                and stopped_target
            )
            cautious_continuation = (
                resume_active
                and risk_level == "medium"
                and stop_like
                and stopped_target
            )
            if cautious_continuation or (
                initial_clearance
                and risk_level == "medium"
                and stationary_elapsed_s >= hold
            ):
                decision.update(
                    action="decelerate",
                    target_speed_kmh=self.config.crawl_speed_kmh,
                    target_lane=None,
                    emergency=False,
                    reason="cautious_resume_after_hazard_cleared",
                    blocked_reason_codes=[],
                )
                self._override_counts[OVERRIDE_CAUTIOUS_RESUME] += 1
                self._resume_intent = parsed_intent
                return decision, OVERRIDE_CAUTIOUS_RESUME
            if (
                initial_clearance
                and risk_level == "low"
                and (
                    resume_active
                    or stationary_elapsed_s >= hold
                )
            ):
                decision.update(
                    action="keep_lane",
                    target_speed_kmh=max(
                        self.config.crawl_speed_kmh,
                        float(resume_speed_kmh),
                    ),
                    target_lane=None,
                    emergency=False,
                    reason="temporal_resume_after_hazard_cleared",
                    blocked_reason_codes=[],
                )
                self._override_counts[OVERRIDE_HAZARD_CLEARANCE] += 1
                self._resume_intent = parsed_intent
                return decision, OVERRIDE_HAZARD_CLEARANCE

        # 3) Generic target-lane visual clearance for instructed lane changes.
        if (
            parsed_intent in LANE_CHANGE_INTENTS
            and requested_lane_direction in {"left", "right"}
            and str(decision.get("action"))
            != LANE_CHANGE_ACTIONS[parsed_intent]
        ):
            lane_risk_level = str(
                (target_lane_risk or risk).get("risk_level", "high")
            ).lower()
            if lane_risk_level != "high":
                lane_history = self._lane_risk_history[requested_lane_direction]
                lane_clear = self.clearance_evidence(list(lane_history))
                if (
                    resume_active
                    or (
                        stationary_elapsed_s >= hold
                        and lane_clear
                    )
                ):
                    decision.update(
                        action=LANE_CHANGE_ACTIONS[parsed_intent],
                        target_speed_kmh=self.config.crawl_speed_kmh,
                        target_lane=requested_lane_direction,
                        emergency=False,
                        reason="sensor_clearance_resume_lane_change",
                        blocked_reason_codes=[],
                    )
                    self._override_counts[OVERRIDE_LANE_CLEARANCE] += 1
                    self._resume_intent = parsed_intent
                    return decision, OVERRIDE_LANE_CLEARANCE
        return decision, None

    def override_counts(self) -> dict[str, int]:
        return dict(self._override_counts)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "global_visual_risk_history": list(self._global_risk_history),
            "left_lane_visual_risk_history": list(
                self._lane_risk_history["left"]
            ),
            "right_lane_visual_risk_history": list(
                self._lane_risk_history["right"]
            ),
            "stopped_since_frame": self._stopped_since_frame,
            "resume_active_intent": self._resume_intent,
            "crawl_pending": self._crawl_pending,
            "high_confirm_count": self._high_confirm_count,
            "override_counts": dict(self._override_counts),
            "decision_history": list(self._decision_history),
            "moving_since_frame": self._moving_since_frame,
        }
