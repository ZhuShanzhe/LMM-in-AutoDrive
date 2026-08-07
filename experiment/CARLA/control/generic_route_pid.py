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

from control.pid_controller import EgoPIDController
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
        # Longer lookahead keeps the ego centred on sweeping curves, where a
        # short pure-pursuit horizon cuts the inside edge and clips solid
        # lane boundaries.  Junction turns still use the short horizon below.
        lookahead_m = max(5.0, min(10.0, 3.0 + speed_kmh * 0.25))
        if self._junction_or_road_transition_ahead() and speed_kmh >= 3.0:
            # Short pure-pursuit horizon keeps tight Town05 junction turns
            # centred without cutting into an adjacent branch.
            lookahead_m = min(lookahead_m, 3.2)
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

    def _current_speed_kmh(self) -> float:
        velocity = self.ego.get_velocity()
        return 3.6 * math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )

    def _junction_or_road_transition_ahead(
        self,
        lookahead_m: float = 40.0,
    ) -> bool:
        """Detect junctions or road-id transitions in the planned corridor."""

        if self.route_context is None or self.route_plan is None:
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

    def run_step(self) -> carla.VehicleControl:
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
            # Generic actuator-safety ceiling before a junction or a road-id
            # transition.  It only limits physical speed and never changes
            # lane/turn selection.
            intent["target_speed_kmh"] = min(
                float(intent["target_speed_kmh"]),
                9.0,
            )
        if self._lane_transition_ahead():
            # Generic cautious speed during any lane transition: keep the
            # lateral manoeuvre stable and avoid clipping lane boundaries.
            intent["target_speed_kmh"] = min(
                float(intent["target_speed_kmh"]),
                15.0,
            )
        route_target = self._route_target()
        if route_target is not None:
            intent["target_location"] = route_target
            intent["route_target_trusted"] = True
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
