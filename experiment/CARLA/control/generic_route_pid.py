"""Generic route-following PID used by every CARLA scene.

The PID consumes one high-level decision dict and converts it into a
``carla.VehicleControl``.  It owns no scene/event/command semantics: lane
changes, target speed and hazard responses always come from the decision
produced by the universal VLA chain.
"""

from __future__ import annotations

import bisect
import math
from typing import Any, Mapping, Sequence

import carla

from control.pid_controller import EgoPIDController, _clamp
from control.protocol import normalize_intent


class GenericRoutePID:
    """Low-level route follower over a CARLA waypoint or point route."""

    def __init__(
        self,
        world: Any,
        ego: Any,
        *,
        target_speed_kmh: float,
        fixed_delta_seconds: float,
        route_context: Any | None = None,
        route_plan: Sequence[tuple[Any, Any]] | None = None,
        route_manager: Any | None = None,
        logical_lane_adapter: Any | None = None,
    ) -> None:
        if route_context is not None and route_plan is None:
            raise ValueError("route_plan is required with route_context")
        self.world = world
        self.ego = ego
        self.route_context = route_context
        self.route_plan = list(route_plan) if route_plan is not None else None
        self.route_manager = route_manager
        self.logical_lane_adapter = logical_lane_adapter
        self.fixed_delta_seconds = float(fixed_delta_seconds)
        self.target_speed_kmh = float(target_speed_kmh)
        self._default_speed_kmh = float(target_speed_kmh)
        self._action = "keep_lane"
        self._target_speed = float(target_speed_kmh)
        self._target_lane: str | None = None
        self._emergency = False
        self._pid = EgoPIDController(
            ego,
            world.get_map(),
            float(target_speed_kmh),
        )

    def set_high_level_decision(self, decision: Mapping[str, Any]) -> None:
        """Store one gated VLA decision for the current control step."""

        self._action = str(decision.get("action", "keep_lane"))
        try:
            speed = float(decision.get("target_speed_kmh", 0.0))
        except (TypeError, ValueError):
            speed = 0.0
        self._target_speed = max(
            0.0, min(self._default_speed_kmh, speed)
        )
        if self._action in {"stop", "emergency_brake"}:
            self._target_speed = 0.0
        self._target_lane = decision.get("target_lane")
        self._emergency = bool(decision.get("emergency", False))

    def progress_m(self) -> float:
        if self.route_context is not None:
            return float(
                self.route_context.distances_m[
                    self.route_context.tracker.index
                ]
            )
        if self.route_manager is not None:
            return float(self.route_manager.progress_m)
        return 0.0

    def _update_progress(self) -> float:
        if self.route_context is not None:
            self.route_context.tracker.update(self.ego.get_location())
            return float(
                self.route_context.distances_m[
                    self.route_context.tracker.index
                ]
            )
        if self.route_manager is not None:
            return float(self.route_manager.update(self.ego.get_location()))
        return 0.0

    def _current_logical_lane(self, waypoint: Any) -> int | None:
        if self.logical_lane_adapter is None:
            return None
        progress_m = self.progress_m()
        for logical_lane in (-1, -2, -3):
            if self.logical_lane_adapter.waypoint_matches_logical_lane(
                waypoint,
                logical_lane,
                progress_m,
            ):
                return logical_lane
        return None

    def _plan_logical_lane(self, progress_m: float) -> int | None:
        """Resolve the plan waypoint's logical lane at a route position."""

        if self.logical_lane_adapter is None:
            return None
        try:
            planned = self.route_context.adapter.route_waypoint(progress_m)
        except Exception:
            return None
        for logical_lane in (-1, -2, -3):
            if self.logical_lane_adapter.waypoint_matches_logical_lane(
                planned,
                logical_lane,
                progress_m,
            ):
                return logical_lane
        return None

    def _target_logical_lane(self, progress_m: float) -> int | None:
        """Choose the corridor lane generically.

        Follow the planned lane when it is stable over the next 30 m and
        differs from the current lane by at most one lane.  When the planned
        lane cannot be mapped to the logical lane set (for example a positive
        connector lane), return ``None`` so the caller follows the plan
        waypoint exactly.  This keeps the ego on the corridor without
        multi-lane plan jumps or event-derived lane windows.
        """

        if self.logical_lane_adapter is None:
            return None
        try:
            current_waypoint = self.logical_lane_adapter.get_waypoint(
                self.ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            return None
        if current_waypoint is None:
            return None
        current_logical = self._current_logical_lane(current_waypoint)
        if current_logical is None:
            return None
        planned_lanes = {
            self._plan_logical_lane(progress_m + offset)
            for offset in (0.0, 15.0, 30.0)
        }
        if None in planned_lanes or len(planned_lanes) != 1:
            return None
        planned = next(iter(planned_lanes))
        if abs(planned - current_logical) <= 1:
            return planned
        return current_logical

    def _waypoint_target(self, lookahead_m: float) -> dict[str, float] | None:
        if self.route_context is None or self.route_plan is None:
            return None
        progress_m = self._update_progress()
        target_index = bisect.bisect_left(
            self.route_context.distances_m,
            progress_m + lookahead_m,
        )
        target_index = min(target_index, len(self.route_plan) - 1)
        planned = self.route_plan[target_index][0]

        lane_change_active = self._action in {
            "lane_change_left",
            "lane_change_right",
        }
        if self.logical_lane_adapter is not None:
            current_waypoint = self.logical_lane_adapter.get_waypoint(
                self.ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            current_logical = (
                self._current_logical_lane(current_waypoint)
                if current_waypoint is not None
                else None
            )
            target_logical = self._target_logical_lane(progress_m)
            if (
                lane_change_active
                and self._target_lane in {"left", "right"}
                and current_logical is not None
            ):
                # A commanded lane change is only executed when the planned
                # corridor agrees with the commanded direction.  This keeps
                # text-driven changes aligned with the route geometry (and
                # avoids desynchronising the progress tracker by merging
                # early into a lane the plan has not reached yet).
                if self._target_lane == "left":
                    intended = (
                        current_logical
                        if current_logical == -1
                        else current_logical + 1
                    )
                else:
                    intended = (
                        current_logical
                        if current_logical == -3
                        else current_logical - 1
                    )
                plan_lane = self._plan_logical_lane(
                    progress_m + lookahead_m
                )
                if plan_lane == intended:
                    target_logical = intended
                else:
                    target_logical = current_logical
            if target_logical is not None:
                translated = self.logical_lane_adapter.logical_waypoint(
                    target_logical,
                    progress_m + lookahead_m,
                )
                if translated is not None:
                    planned = translated
        transform = planned.transform
        return {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
            "yaw": float(transform.rotation.yaw),
        }

    def _route_target(self) -> dict[str, Any] | None:
        speed_kmh = self._current_speed_kmh()
        # Speed-adaptive lookahead with a moderate floor.  Junction turns use
        # a shorter horizon derived from the current speed instead of a fixed
        # constant, so slow turns keep a tight arc while faster approaches
        # still look far enough ahead.
        lookahead_m = max(4.0, min(12.0, 3.5 + speed_kmh * 0.30))
        if self._junction_or_road_transition_ahead() and speed_kmh >= 3.0:
            # Short speed-adaptive horizon: tight junction turns need a close
            # pure-pursuit point, otherwise the approach cuts the inside edge.
            lookahead_m = min(
                lookahead_m,
                max(2.8, min(4.5, 2.2 + speed_kmh * 0.10)),
            )
        if self.route_context is not None:
            target = self._waypoint_target(lookahead_m)
            if target is None:
                return None
            reference = self._waypoint_target(0.0) or target
            return {
                **target,
                "reference": reference,
            }
        if self.route_manager is not None:
            target = self.route_manager.target_point(lookahead_m)
            if target is None:
                return None
            reference = self.route_manager.target_point(0.0) or target
            result = dict(target)
            result["reference"] = reference
            return result
        return None

    def _ahead_heading_change_deg(
        self, start_m: float = 8.0, span_m: float = 26.0
    ) -> float:
        """Absolute heading change of the planned corridor ahead (degrees)."""

        try:
            if self.route_context is not None and self.route_plan is not None:
                progress_m = self.progress_m()

                def _yaw(offset: float) -> float:
                    index = bisect.bisect_left(
                        self.route_context.distances_m,
                        progress_m + offset,
                    )
                    index = max(0, min(index, len(self.route_plan) - 1))
                    return math.radians(
                        float(
                            self.route_plan[index][0]
                            .transform.rotation.yaw
                        )
                    )

                yaw0 = _yaw(start_m)
                yaw1 = _yaw(start_m + span_m)
            elif self.route_manager is not None:
                point0 = self.route_manager.target_point(start_m)
                point1 = self.route_manager.target_point(start_m + span_m)
                if point0 is None or point1 is None:
                    return 0.0
                yaw0 = math.radians(float(point0["yaw"]))
                yaw1 = math.radians(float(point1["yaw"]))
            else:
                return 0.0
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return 0.0
        return abs(math.degrees(self._pid._angle_delta(yaw1, yaw0)))

    def _current_speed_kmh(self) -> float:
        velocity = self.ego.get_velocity()
        return 3.6 * math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )

    def _junction_or_road_transition_ahead(
        self,
        lookahead_m: float | None = None,
    ) -> bool:
        """Detect junctions or road-id transitions in the planned corridor."""

        if lookahead_m is None:
            lookahead_m = _clamp(
                20.0 + self._current_speed_kmh() * 0.40,
                20.0,
                50.0,
            )
        lookahead_m = float(lookahead_m)
        if self.route_context is None or self.route_plan is None:
            if self.route_manager is not None:
                # Route-manager corridors expose the exact planned waypoints
                # with junction flags; scanning them is faster and more
                # reliable than walking the official map lanes.
                route = getattr(self.route_manager, "route", None)
                if route:
                    progress_m = float(
                        getattr(self.route_manager, "progress_m", 0.0) or 0.0
                    )
                    horizon = progress_m + float(lookahead_m)
                    for point in route:
                        distance_m = float(point.get("distance_m", -1.0))
                        if distance_m < progress_m - 2.0:
                            continue
                        if distance_m > horizon:
                            break
                        if bool(point.get("is_junction", False)):
                            return True
                return False
            return False
        progress_m = self.progress_m()
        index = bisect.bisect_left(
            self.route_context.distances_m,
            progress_m,
        )
        end = bisect.bisect_left(
            self.route_context.distances_m,
            progress_m + float(lookahead_m),
        )
        end = min(end, len(self.route_plan) - 1)
        window = [
            waypoint
            for waypoint, _road_option in self.route_plan[index : end + 1]
        ]
        if any(
            bool(getattr(waypoint, "is_junction", False))
            for waypoint in window
        ):
            return True
        road_ids = {int(waypoint.road_id) for waypoint in window}
        return len(road_ids) > 1

    def _lane_transition_ahead(self) -> bool:
        """True while the planned corridor moves to a different logical lane."""

        if self.logical_lane_adapter is None or self.route_context is None:
            return False
        try:
            current_waypoint = self.logical_lane_adapter.get_waypoint(
                self.ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except RuntimeError:
            return False
        if current_waypoint is None:
            return False
        current_logical = self._current_logical_lane(current_waypoint)
        if current_logical is None:
            return False
        planned = self._plan_logical_lane(self.progress_m() + 15.0)
        return planned is not None and planned != current_logical

    def autonomous_lane_change_legal(self, direction: str) -> bool:
        """Conservatively validate an uncommanded collision-escape lane.

        Unlike instructed lane changes, uncertainty is fail-closed: the
        target lane must exist, be a driving lane and be reachable across a
        marking that explicitly permits the requested direction.
        """

        if direction not in {"left", "right"}:
            raise ValueError("direction must be 'left' or 'right'")
        try:
            waypoint = self.world.get_map().get_waypoint(
                self.ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except (RuntimeError, AttributeError, TypeError):
            return False
        if waypoint is None:
            return False
        try:
            permission = int(waypoint.lane_change)
            target = (
                waypoint.get_left_lane()
                if direction == "left"
                else waypoint.get_right_lane()
            )
            if target is None or target.lane_type != carla.LaneType.Driving:
                return False
            # Adjacent lanes travelling in the opposite direction are never
            # valid evasive targets even if malformed map metadata exposes
            # a lane-change bit.
            if int(target.lane_id) * int(waypoint.lane_id) <= 0:
                return False
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False
        mask = 1 if direction == "left" else 2
        return bool(permission & mask)

    def _lane_change_legal(self) -> bool:
        """Map-legality check for the requested lane change.

        Uses only static map geometry (waypoint lane-change permissions), not
        actor truth.  A change across a solid marking is never executed; the
        intent remains active and proceeds when the corridor reaches a legal
        window (or the supervisor timeout keeps the ego crawling).
        """

        if self._action not in ("lane_change_left", "lane_change_right"):
            return True
        direction = self._action.removeprefix("lane_change_")
        try:
            waypoint = self.world.get_map().get_waypoint(
                self.ego.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except (RuntimeError, AttributeError, TypeError):
            return True
        if waypoint is None:
            return True
        try:
            permission = int(waypoint.lane_change)
        except (AttributeError, TypeError, ValueError):
            return True
        if direction == "left":
            return bool(permission & 1)
        return bool(permission & 2)

    def run_step(self) -> carla.VehicleControl:
        if not self._lane_change_legal():
            # Hold the lateral action in the current lane; the high-level
            # intent stays untouched so it can resume at a legal window.
            self._target_lane = None
            self._action = "keep_lane"
        intent = {
            "schema_version": "1.0.0",
            "action": self._action,
            "target_speed_kmh": self._target_speed,
            "target_lane": self._target_lane,
            "target_location": None,
            "emergency": self._emergency,
            "route_target_trusted": False,
        }
        if self._junction_or_road_transition_ahead():
            # Sharpness-aware junction ceiling: the cap falls with the actual
            # heading change of the planned corridor and rises slightly with
            # the approach speed.  It only limits physical speed and never
            # changes lane/turn selection.
            sharpness_deg = self._ahead_heading_change_deg()
            junction_cap = _clamp(
                11.0 - 0.05 * sharpness_deg
                + self._current_speed_kmh() * 0.05,
                7.0,
                14.0,
            )
            intent["target_speed_kmh"] = min(
                float(intent["target_speed_kmh"]),
                junction_cap,
            )
        if self._lane_transition_ahead():
            # Speed-adaptive lane-transition ceiling: faster approaches get a
            # lower cap so the lateral manoeuvre stays stable and cannot clip
            # lane boundaries; slow crawls keep enough momentum to finish the
            # change.
            transition_cap = _clamp(
                16.0 - self._current_speed_kmh() * 0.10,
                9.0,
                16.0,
            )
            intent["target_speed_kmh"] = min(
                float(intent["target_speed_kmh"]),
                transition_cap,
            )
        route_target = self._route_target()
        if route_target is not None:
            intent["target_location"] = route_target
            intent["route_target_trusted"] = True
        # Route-manager corridors (e.g. scene 1) sample corners coarsely and
        # need additional steering authority while approaching or crossing a
        # junction; lane-centred route contexts keep the conservative
        # validated limit.
        intent["junction_steering_authority"] = bool(
            self.route_manager is not None
            and self._junction_or_road_transition_ahead()
        )
        control, _ = self._pid.run_step(intent, self.fixed_delta_seconds)
        return control

    def execution_state(self) -> dict[str, Any]:
        return {
            "action": self._action,
            "target_speed_kmh": self._target_speed,
            "target_lane": self._target_lane,
            "speed_kmh": round(self._current_speed_kmh(), 3),
            "progress_m": round(self.progress_m(), 3),
            "pid": self._pid.get_execution_state(),
        }
