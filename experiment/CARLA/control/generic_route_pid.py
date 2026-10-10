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
        self._requested_cruise_speed_kmh = float(target_speed_kmh)
        self._explicit_cruise_speed_kmh = None
        self._action = "keep_lane"
        self._target_speed = float(target_speed_kmh)
        self._target_lane: str | None = None
        self._emergency = False
        self._sequence_command = {}
        self._sequence_expiry = None
        self._requested_decision_speed_kmh = float(target_speed_kmh)
        self._speed_constraint_codes: list[str] = []
        self._pid = EgoPIDController(
            ego,
            world.get_map(),
            float(target_speed_kmh),
        )

    def set_cruise_target_kmh(self, target: float | None) -> float:
        """Keep the requested cruise goal distinct from the current road limit."""
        if target is not None:
            if isinstance(target,bool) or not math.isfinite(float(target)) or not 0 <= float(target) <= 120:
                raise ValueError('Cruise target must be a finite speed from 0 to 120 km/h')
            self._requested_cruise_speed_kmh=float(target)
            self._explicit_cruise_speed_kmh=float(target)
        requested=getattr(self,'_requested_cruise_speed_kmh',self._default_speed_kmh)
        getter=getattr(getattr(self,'ego',None),'get_speed_limit',None)
        try:limit=float(getter()) if callable(getter) else float('nan')
        except (TypeError,ValueError,RuntimeError):limit=float('nan')
        if not math.isfinite(limit) or limit<=0:
            limit=self._default_speed_kmh
        self.target_speed_kmh=max(0.,min(requested,limit,120.))
        return self.target_speed_kmh

    def explicit_cruise_target_kmh(self) -> float | None:
        """A driver-set goal, unlike the default route speed ceiling."""
        return getattr(self,'_explicit_cruise_speed_kmh',None)

    def configured_speed_cap_kmh(self) -> float:
        """Route ceiling, independent of temporary or requested speed goals."""
        value = self._default_speed_kmh
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 120:
            raise ValueError('Configured route ceiling must be finite and nonnegative')
        return float(value)

    def clear_explicit_cruise_target(self):
        self._explicit_cruise_speed_kmh=None
        self._requested_cruise_speed_kmh=self._default_speed_kmh

    def set_high_level_decision(self, decision: Mapping[str, Any]) -> None:
        """Store one gated VLA decision for the current control step."""

        self._action = str(decision.get("action", "keep_lane"))
        if self._action in {'lane_change_left','lane_change_right','turn_left','turn_right'}:
            self._completed_lane_navigation=self._action in {'turn_left','turn_right'}
            maneuver_key = (self._action, decision.get('command_id'), decision.get('source_step_id'))
            if maneuver_key != getattr(self, '_navigation_maneuver_key', None):
                self._navigation_branch_samples = None
                self._navigation_maneuver_key = maneuver_key
        self._source_step_id=decision.get('source_step_id')
        permission=decision.get('risk_motion_permission')
        self._risk_motion_permission=dict(permission) if isinstance(permission,Mapping) else None
        self._command_id=decision.get('command_id') or self._action+':'+str(self._source_step_id)
        self._environment_speed_constraint = None
        constraint = decision.get('environment_speed_constraint')
        if isinstance(constraint, Mapping):
            requested = constraint.get('requested_target_speed_kmh')
            effective = constraint.get('effective_target_speed_kmh')
            codes = constraint.get('speed_constraint_codes')
            if (all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (requested, effective)) and 0 <= effective < requested <= 120
                    and isinstance(codes, list) and codes
                    and all(code in {'road_speed_limit', 'route_speed_cap', 'rainy_night_speed_cap'}
                            for code in codes)):
                self._environment_speed_constraint = dict(constraint)
        self._allow_positive_acceleration=bool(decision.get('allow_positive_acceleration',True))
        self._longitudinal_control_mode=decision.get('longitudinal_control_mode','ABSOLUTE_SPEED')
        self._speed_tracking_mode=decision.get('speed_tracking_mode')
        try:
            speed = float(decision.get("target_speed_kmh", 0.0))
        except (TypeError, ValueError):
            speed = 0.0
        self._requested_decision_speed_kmh = max(0.0, speed)
        cruise_cap = self.set_cruise_target_kmh(None)
        self._target_speed = max(0.0, min(cruise_cap, speed))
        self._speed_constraint_codes = []
        if self._target_speed < self._requested_decision_speed_kmh - 1e-6:
            self._speed_constraint_codes.append("road_speed_limit")
        if self._action in {"stop", "emergency_brake"}:
            self._target_speed = 0.0
            self._speed_constraint_codes = ["stop_requested"]
        self._target_lane = decision.get("target_lane")
        self._emergency = bool(decision.get("emergency", False))
        self._sequence_command = {}
        self._sequence_expiry = None
        control_expiry = decision.get('control_valid_until_s')
        if control_expiry is not None:
            try:
                control_expiry = float(control_expiry)
                if not math.isfinite(control_expiry):raise ValueError('Nonfinite expiry')
                self._sequence_expiry = control_expiry
            except (TypeError,ValueError):
                self._action, self._target_speed = 'stop', 0.
        if decision.get('longitudinal_sequence_schema') == 'longitudinal_sequence/1.0':
            try:
                acceleration = float(decision['target_acceleration_mps2'])
                expiry = float(decision['sequence_valid_until_s'])
                valid = math.isfinite(speed) and math.isfinite(acceleration) and -8.0001 <= acceleration <= 3.0001 and math.isfinite(expiry)
            except (KeyError,TypeError,ValueError):
                valid = False
            if not valid:
                self._action, self._target_speed = 'stop', 0.
            else:
                # A speed cap may cancel feedforward, but cannot extend command validity.
                self._sequence_expiry = min(expiry,self._sequence_expiry) if self._sequence_expiry is not None else expiry
                acceleration = max(-8.,min(3.,acceleration))
                if abs(self._target_speed-speed) < 1e-6 and not self._emergency:
                    self._sequence_command = dict(longitudinal_sequence_schema='longitudinal_sequence/1.0',
                        target_acceleration_mps2=acceleration,sequence_valid_until_s=expiry)

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

    def road_step_readiness(self, document, *, frame_id):
        """Publish fresh turn targets from the navigation route, not actor truth."""
        if not any(step.get('action') == 'TURN' for step in document['intent']['steps']):
            return {}
        from control.road_target_resolver import RoadTargetResolver
        resolver = getattr(self, '_road_target_resolver', None)
        if resolver is None:
            try:
                if self.route_context is not None and self.route_plan is not None:
                    from benchmark.turn_fixture import waypoint_route
                    route = waypoint_route([waypoint for waypoint, _ in self.route_plan],
                                           self.route_context.distances_m)
                elif self.route_manager is not None:
                    route = self.route_manager.route
                else:
                    raise ValueError('road_target_route_unavailable')
                resolver = RoadTargetResolver(route)
                self._road_target_resolver = resolver
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                return {step['step_id']: dict(request_id=document['request_id'],
                    step_id=step['step_id'], frame_id=frame_id, source='planned_route_topology',
                    valid=False, reachable=False, ambiguous=False, reason=str(error))
                    for step in document['intent']['steps'] if step.get('action') == 'TURN'}
        return resolver.resolve(document, frame_id=frame_id, progress_m=self.progress_m())

    def pin_road_step_target(self, step, evidence, *, frame_id):
        resolver = getattr(self, '_road_target_resolver', None)
        if resolver is not None:
            resolver.pin(step, evidence, frame_id=frame_id)

    def _sensor_road_map(self):
        road_map = getattr(self, '_corridor_road_map', None)
        if road_map is None:
            road_map = self.world.get_map()
            self._corridor_road_map = road_map
        return road_map

    def _corridor_transition_width(self, first, last, heading):
        """Verify an oblique lane connector against static, same-level topology."""
        world = getattr(self, 'world', None)
        if world is None or heading is None:
            return None
        try:
            road_map = self._sensor_road_map()
            ego_location = self.ego.get_location()
            positions = [(ego_location.x, ego_location.y), first[:2], last[:2]]
            waypoints = [road_map.get_waypoint(carla.Location(x=float(x), y=float(y),
                z=float(getattr(ego_location, 'z', 0.))), project_to_road=True,
                lane_type=carla.LaneType.Driving) for x, y in positions]
            if any(w is None or w.is_junction for w in waypoints):
                return None
            current, start, end = waypoints
            if start.lane_id * end.lane_id <= 0 or abs(start.lane_id-end.lane_id) > 1:
                return None
            for waypoint, point in zip(waypoints, positions):
                yaw = math.radians(float(waypoint.transform.rotation.yaw))
                if math.cos(yaw)*heading[0]+math.sin(yaw)*heading[1] < .85:
                    return None
                center = waypoint.transform.location
                if math.hypot(point[0]-center.x, point[1]-center.y) > waypoint.lane_width/2.+.25:
                    return None
            if len({(w.road_id, w.section_id) for w in waypoints}) != 1:
                # A road-ID seam is not necessarily a junction. Verify the
                # actual unique lane successor instead of assuming IDs match.
                for waypoint, point in zip((start,end),positions[1:]):
                    along=sum((point[i]-positions[0][i])*heading[i] for i in range(2))
                    distance=abs(along)
                    if not .1<=distance<=10.:
                        return None
                    origin, destination=(waypoint,current) if along<0 else (current,waypoint)
                    successors=origin.next(distance)
                    if len(successors)!=1:
                        return None
                    successor=successors[0]
                    if (successor.road_id,successor.section_id,successor.lane_id)!=(
                            destination.road_id,destination.section_id,destination.lane_id):
                        return None
                    a=successor.transform.location;b=destination.transform.location
                    if math.hypot(a.x-b.x,a.y-b.y)>.5 or abs(a.z-b.z)>.25:
                        return None
            return min(float(w.lane_width) for w in waypoints)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def entity_surface_polyline(self, behind_m=30., ahead_m=55.):
        """Static, uniquely connected lane surface for multi-view metric lifting."""
        current=self._sensor_road_map().get_waypoint(self.ego.get_location(),
            project_to_road=True,lane_type=carla.LaneType.Driving)
        if current is None:return []
        def walk(method,distance):
            points=[];cursor=current
            for _ in range(int(distance/2.)):
                candidates=getattr(cursor,method)(2.)
                if len(candidates)!=1:break
                candidate=candidates[0];a=cursor.transform.location;b=candidate.transform.location
                length=math.hypot(a.x-b.x,a.y-b.y)
                if not .1<=length<=3. or abs(a.z-b.z)>length*.3:break
                points.append(candidate);cursor=candidate
            return points
        waypoints=list(reversed(walk('previous',behind_m)))+[current]+walk('next',ahead_m)
        return [(float(w.transform.location.x),float(w.transform.location.y),
                 float(w.transform.location.z)) for w in waypoints]

    def radar_corridor_polyline(
        self,
        horizon_m: float = 85.0,
        *,
        include_height: bool = False,
    ) -> list[tuple[float, ...]]:
        """Return the planner-owned centreline ahead for sensor gating.

        The result contains only route geometry and the ego pose.  It does not
        enumerate actors or scenario events, so all three scenes can use the
        same physical radar corridor filter.
        """

        def point_at(x, y, z=None):
            if not include_height:
                return (float(x), float(y))
            if z is None or not math.isfinite(float(z)):
                raise ValueError('Selected route height is unavailable')
            return (float(x), float(y), float(z))

        location = self.ego.get_location()
        maneuver = getattr(self, '_sensor_maneuver_corridor_world', None)
        if maneuver:
            from control.maneuver_sensor_corridor import maneuver_sensor_corridor
            ground = self._sensor_road_map().get_waypoint(location,
                project_to_road=True, lane_type=carla.LaneType.Driving)
            if ground is not None:
                swept = maneuver_sensor_corridor(maneuver,
                    (float(location.x), float(location.y), float(ground.transform.location.z)),
                    transition_m=10., horizon_m=min(85., max(5., float(horizon_m))))
                if swept:
                    return swept if include_height else [(p[0], p[1]) for p in swept]
        ego_height = getattr(location, 'z', None)
        if include_height:
            # Anchor on the current road level, never reproject future XY at z=0.
            waypoint = self._sensor_road_map().get_waypoint(location,
                project_to_road=True, lane_type=carla.LaneType.Driving)
            if waypoint is None:
                raise ValueError('Current road surface is unavailable')
            ego_height = waypoint.transform.location.z
        points = [point_at(location.x, location.y, ego_height)]
        progress_m = self.progress_m()
        horizon = progress_m + max(5.0, float(horizon_m))
        if getattr(self,'_completed_lane_navigation',False):
            ego_yaw_deg=float(self.ego.get_transform().rotation.yaw)
            for offset in range(0,int(horizon-progress_m)+1,5):
                lane_point=self._current_lane_navigation_point(float(offset),
                    ego_location=location,ego_yaw_deg=ego_yaw_deg)
                if lane_point is None:break
                lane_location=lane_point.transform.location
                point=point_at(lane_location.x,lane_location.y,lane_location.z)
                if math.hypot(point[0]-points[-1][0],point[1]-points[-1][1])>.05:
                    points.append(point)
        elif self.route_context is not None and self.route_plan is not None:
            start = bisect.bisect_left(
                self.route_context.distances_m, progress_m
            )
            end = bisect.bisect_right(
                self.route_context.distances_m, horizon
            )
            for index in range(start,end):
                waypoint, _road_option = self.route_plan[index]
                if getattr(self,'logical_lane_adapter',None) is not None:
                    target=self._waypoint_target(max(0.,self.route_context.distances_m[index]-progress_m),
                        progress_m=progress_m,ego_location=location)
                    if target is None:
                        break
                    point=point_at(target['x'],target['y'],target.get('z'))
                else:
                    waypoint_location = waypoint.transform.location
                    point = point_at(waypoint_location.x,waypoint_location.y,
                        getattr(waypoint_location,'z',None))
                if math.hypot(
                    point[0] - points[-1][0], point[1] - points[-1][1]
                ) > 0.05:
                    points.append(point)
        elif self.route_manager is not None:
            route = getattr(self.route_manager, "route", None) or []
            for route_point in route:
                try:
                    distance_m = float(route_point["distance_m"])
                    point = point_at(route_point['x'],route_point['y'],route_point.get('z'))
                except (KeyError, TypeError, ValueError):
                    continue
                if distance_m < progress_m:
                    continue
                if distance_m > horizon:
                    break
                if math.hypot(
                    point[0] - points[-1][0], point[1] - points[-1][1]
                ) > 0.05:
                    points.append(point)
        points=self._extend_observation_after_goal(points,horizon-progress_m,
            progress_m=progress_m,include_height=include_height)
        # A quantized route cursor may retain the waypoint just passed.
        # Trim only when the ego is on or just past its next short segment;
        # otherwise a required corner must remain in the swept-path evidence.
        heading = None
        read_transform = getattr(self.ego, 'get_transform', None)
        if callable(read_transform):
            yaw = math.radians(float(read_transform().rotation.yaw))
            heading = (math.cos(yaw), math.sin(yaw))
        while len(points)>2:
            first,next_point=points[1:3]
            dx,dy=next_point[0]-first[0],next_point[1]-first[1]
            length2=dx*dx+dy*dy
            if not .01<=length2<=225.:break
            if heading is not None:
                along = [sum((p[i]-points[0][i])*heading[i] for i in range(2))
                         for p in (first, next_point)]
                if all(-3. <= value < -.1 for value in along):
                    width = self._corridor_transition_width(first, next_point, heading)
                    across = [abs((p[0]-points[0][0])*heading[1]
                                  -(p[1]-points[0][1])*heading[0])
                              for p in (first, next_point)]
                    # Quantized lane targets can retain two passed points in
                    # reversed order. Only same-level, aligned road geometry
                    # proves this is a stale prefix rather than a real corner.
                    if width is not None and max(across) <= width/2.:
                        points.pop(1)
                        continue
            fraction=((points[0][0]-first[0])*dx+(points[0][1]-first[1])*dy)/length2
            lateral=math.hypot(points[0][0]-first[0]-fraction*dx,
                               points[0][1]-first[1]-fraction*dy)
            tolerance = .75
            just_passed = False
            if heading is not None:
                behind = sum((first[i]-points[0][i])*heading[i] for i in range(2)) < -.1
                aligned = (dx*heading[0]+dy*heading[1])/math.sqrt(length2) > .5
                width = self._corridor_transition_width(first, next_point, heading) if behind else None
                if width is not None and dx*heading[0]+dy*heading[1] > 0:
                    # Road tangents disambiguate an oblique lane stitch from
                    # a true corner; this changes neither lane permission nor target.
                    aligned = True
                    extent = getattr(getattr(self.ego, 'bounding_box', None), 'extent', None)
                    tolerance = min(4.25, width/2.+float(getattr(extent, 'y', 1.))+.35)
                just_passed = behind and aligned and 0 < (fraction-1)*math.sqrt(length2) <= 3.
                extent = getattr(getattr(self.ego, 'bounding_box', None), 'extent', None)
                if behind and aligned and extent is not None and width is None:
                    # Allow a modest tracking offset, not an off-lane shortcut.
                    tolerance = min(2., max(.75, float(extent.y)+.75))
            if not (0<fraction and (fraction<=1 or just_passed) and lateral<=tolerance):break
            points.pop(1)
        if len(points) < 2:
            target = self._route_target()
            if target is not None:
                point = point_at(target['x'],target['y'],target.get('z'))
                if math.hypot(
                    point[0] - points[-1][0], point[1] - points[-1][1]
                ) > 0.05:
                    points.append(point)
        return points

    def _update_progress(self) -> float:
        if self.route_context is not None:
            self.route_context.tracker.update(self.ego.get_location())
            return float(
                self.route_context.distances_m[
                    self.route_context.tracker.index
                ]
            )
        if self.route_manager is not None:
            return float(self.route_manager.update(self.ego))
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

    def _target_logical_lane(self, progress_m: float, *, ego_location=None) -> int | None:
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
                self.ego.get_location() if ego_location is None else ego_location,
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

    def _waypoint_target(self, lookahead_m: float, *, progress_m=None, ego_location=None) -> dict[str, float] | None:
        if self.route_context is None or self.route_plan is None:
            return None
        if getattr(self,'_completed_lane_navigation',False):
            current=self._current_lane_navigation_point(lookahead_m,ego_location=ego_location)
            if current is None:return None
            transform=current.transform
            return dict(x=float(transform.location.x),y=float(transform.location.y),
                z=float(transform.location.z),yaw=float(transform.rotation.yaw))
        progress_m = self._update_progress() if progress_m is None else float(progress_m)
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
                self.ego.get_location() if ego_location is None else ego_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            current_logical = (
                self._current_logical_lane(current_waypoint)
                if current_waypoint is not None
                else None
            )
            target_logical = self._target_logical_lane(progress_m,ego_location=ego_location)
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

    def handoff_completed_lane(self):
        """Continue from the current lane, not the old route's lane centre."""
        self._completed_lane_navigation=True

    def _junction_approach_lane(self, waypoint):
        """Prefer the imminent connector over a temporary upstream lane stitch."""
        if self.route_context is None or not self.route_plan:
            return False, None
        progress = self.progress_m()
        distances = self.route_context.distances_m
        start = bisect.bisect_right(distances, progress)
        end = bisect.bisect_right(distances, progress + 150.)
        entry = next((index for index in range(start, end)
                      if getattr(self.route_plan[index][0], 'is_junction', False)), None)
        if entry is None:
            return False, None
        planned = self.route_plan[entry][0]
        gap = float(distances[entry]) - progress

        def reaches(lane):
            if lane is None:
                return False
            # Route lane stitches and parallel curved lanes differ in length.
            for delta in (-8., -4., 0., 4., 8.):
                for candidate in lane.next(max(.1, gap + delta)):
                    if (candidate.road_id == planned.road_id
                            and candidate.lane_id == planned.lane_id
                            and getattr(candidate, 'section_id', 0) == getattr(planned, 'section_id', 0)
                            and abs(candidate.transform.location.z - planned.transform.location.z) < .5
                            and abs(self._pid._angle_delta(
                                math.radians(candidate.transform.rotation.yaw),
                                math.radians(planned.transform.rotation.yaw))) < math.radians(45.)):
                        return True
            return False

        if reaches(waypoint):
            return True, None
        options = [direction for direction in ('left', 'right')
                   if self.autonomous_lane_change_legal(direction) and reaches(
                       EgoPIDController._adjacent_driving_lane(waypoint, 'lane_change_' + direction))]
        if len(options) == 1:
            return True, options[0]
        return False, None

    def route_lane_recovery_direction(self):
        """Find a legal navigation lane before a turn-only branch diverges."""
        if not getattr(self,'_completed_lane_navigation',False):return None
        waypoint=self._sensor_road_map().get_waypoint(self.ego.get_location(),
            project_to_road=True,lane_type=carla.LaneType.Driving)
        if waypoint is None or waypoint.is_junction:return None
        resolved, direction = self._junction_approach_lane(waypoint)
        if resolved:
            return direction
        for distance in (15.,20.,30.,45.,60.,90.,120.):
            if self.route_manager is not None:
                reference=self.route_manager.target_point(distance)
                if reference is None:continue
                planned=self._sensor_road_map().get_waypoint(carla.Location(
                    x=reference['x'],y=reference['y'],z=reference.get('z',waypoint.transform.location.z)),
                    project_to_road=True,lane_type=carla.LaneType.Driving)
            elif self.route_context is not None and self.route_plan:
                index=min(len(self.route_plan)-1,bisect.bisect_left(
                    self.route_context.distances_m,self.progress_m()+distance))
                planned=self.route_plan[index][0]
            else:return None
            if planned is None:continue
            def reaches(lane):
                return lane is not None and any(candidate.road_id==planned.road_id
                    and candidate.lane_id==planned.lane_id
                    and abs(math.degrees(self._pid._angle_delta(
                        math.radians(candidate.transform.rotation.yaw),
                        math.radians(planned.transform.rotation.yaw))))<15.
                    and abs(candidate.transform.location.z-planned.transform.location.z)<.5
                    for candidate in lane.next(distance))
            if reaches(waypoint):continue
            options=[direction for direction in ('left','right')
                if self.autonomous_lane_change_legal(direction) and reaches(
                    EgoPIDController._adjacent_driving_lane(waypoint,'lane_change_'+direction))]
            if len(options)==1:return options[0]
        return None

    def _extend_observation_after_goal(self,points,horizon,*,progress_m,include_height):
        """The sensor horizon may outlive the mission; never extend execution."""
        if len(points)<2 or not math.isfinite(horizon) or horizon<=0:return points
        if self.route_context is not None and self.route_plan:
            end=float(self.route_context.distances_m[-1])
            transform=getattr(self.route_plan[-1][0],'transform',None)
            height=getattr(getattr(transform,'location',None),'z',None)
            if height is None:return points
            height=float(height)
        elif self.route_manager is not None and getattr(self.route_manager,'route',None):
            terminal=self.route_manager.route[-1]
            try:end=float(terminal['distance_m']);height=float(terminal['z'])
            except (KeyError,TypeError,ValueError):return points
        else:return points
        if not math.isfinite(end) or not math.isfinite(height) or end-progress_m>=horizon:return points
        try:
            last=points[-1]
            road_map=self._sensor_road_map()
            waypoint=road_map.get_waypoint(carla.Location(x=last[0],y=last[1],
                z=last[2] if include_height else height),project_to_road=True,lane_type=carla.LaneType.Driving)
            if waypoint is None:return points
            location=waypoint.transform.location
            if (math.hypot(location.x-last[0],location.y-last[1])>.5
                    or abs(location.z-(last[2] if include_height else height))>.25):return points
            dx,dy=last[0]-points[-2][0],last[1]-points[-2][1]
            length=math.hypot(dx,dy);yaw=math.radians(waypoint.transform.rotation.yaw)
            if length<.01 or (dx*math.cos(yaw)+dy*math.sin(yaw))/length<.85:return points
            result=list(points)
            covered=sum(math.hypot(b[0]-a[0],b[1]-a[1]) for a,b in zip(result,result[1:]))
            while covered<horizon:
                choices=waypoint.next(5.)
                if len(choices)!=1:break
                candidate=choices[0];next_location=candidate.transform.location
                step=math.hypot(next_location.x-location.x,next_location.y-location.y)
                if (getattr(candidate,'lane_type',carla.LaneType.Driving)!=carla.LaneType.Driving
                        or not .1<step<=6. or abs(next_location.z-location.z)/step>.3):break
                result.append((float(next_location.x),float(next_location.y),float(next_location.z))
                    if include_height else (float(next_location.x),float(next_location.y)))
                covered+=step;waypoint=candidate;location=next_location
            return result
        except (AttributeError,RuntimeError,TypeError,ValueError):
            return points

    def _current_lane_navigation_point(self,distance,*,ego_location=None,ego_yaw_deg=None):
        epoch=getattr(self,'_observation_epoch',None)
        if epoch is None or ego_location is None or ego_yaw_deg is None:
            return self._uncached_current_lane_navigation_point(distance,
                ego_location=ego_location,ego_yaw_deg=ego_yaw_deg)
        pose=tuple(float(getattr(ego_location,k)) for k in ('x','y','z'))+(float(ego_yaw_deg),)
        offset=float(distance)
        if not all(math.isfinite(v) for v in (*pose,offset)):
            return self._uncached_current_lane_navigation_point(distance,
                ego_location=ego_location,ego_yaw_deg=ego_yaw_deg)
        def context():
            return (epoch,pose,self.progress_m(),id(self.route_plan),id(self.route_manager),
                id(getattr(self,'_navigation_branch_samples',None)),
                getattr(self,'_action',None),getattr(self,'_completed_lane_navigation',False))
        before=context();cache=getattr(self,'_navigation_point_cache',None)
        if cache is None or cache['context']!=before:
            cache={'context':before,'points':{}};self._navigation_point_cache=cache
        if offset in cache['points']:return cache['points'][offset]
        value=self._uncached_current_lane_navigation_point(distance,
            ego_location=ego_location,ego_yaw_deg=ego_yaw_deg)
        # Branch selection has side effects; never cache a mutating query.
        if value is not None and context()==before and len(cache['points'])<64:
            cache['points'][offset]=value
        return value

    def _uncached_current_lane_navigation_point(self,distance,*,ego_location=None,ego_yaw_deg=None):
        road_map=self._sensor_road_map()
        location=self.ego.get_location() if ego_location is None else ego_location
        waypoint=road_map.get_waypoint(location,project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if waypoint is None:return None
        pinned=getattr(self,'_navigation_branch_samples',None)
        if pinned:
            closest=min(pinned,key=lambda w:math.hypot(w.transform.location.x-location.x,
                w.transform.location.y-location.y))
            transform=closest.transform;yaw=math.radians(transform.rotation.yaw)
            dx,dy=location.x-transform.location.x,location.y-transform.location.y
            lateral=abs(-math.sin(yaw)*dx+math.cos(yaw)*dy)
            longitudinal=abs(math.cos(yaw)*dx+math.sin(yaw)*dy)
            ego_yaw=math.radians(self.ego.get_transform().rotation.yaw
                if ego_yaw_deg is None else ego_yaw_deg)
            if (lateral<=max(1.,float(getattr(closest,'lane_width',3.5))*.5)
                    and longitudinal<=1.5 and abs(transform.location.z-location.z)<=1.
                    and abs(self._pid._angle_delta(ego_yaw,yaw))<=math.radians(25.)):
                waypoint=closest
            else:self._navigation_branch_samples=None
        if distance<=.01:return waypoint
        choices=waypoint.next(distance)
        if not choices:return None
        if len(choices)==1:return choices[0]
        # Local headings can be almost identical at the start of a junction
        # fork. Resolve by future route reachability before geometric scoring.
        for further in (30.,60.,90.):
            if self.route_manager is not None:
                future=self.route_manager.target_point(distance+further)
                planned=None if future is None else road_map.get_waypoint(carla.Location(
                    x=future['x'],y=future['y'],z=future.get('z',waypoint.transform.location.z)),
                    project_to_road=True,lane_type=carla.LaneType.Driving)
            elif self.route_context is not None and self.route_plan:
                index=min(len(self.route_plan)-1,bisect.bisect_left(
                    self.route_context.distances_m,self.progress_m()+distance+further))
                planned=self.route_plan[index][0]
            else:planned=None
            if planned is None:continue
            viable=[candidate for candidate in choices if any(
                successor.road_id==planned.road_id
                and abs(math.degrees(self._pid._angle_delta(
                    math.radians(successor.transform.rotation.yaw),
                    math.radians(planned.transform.rotation.yaw))))<15.
                and abs(successor.transform.location.z-planned.transform.location.z)<.5
                for successor in candidate.next(further))]
            if len(viable)==1:
                selected=viable[0]
                if distance<=15.:
                    samples=[]
                    for offset in range(-20,41):
                        linked=(selected.previous(-offset) if offset<0 else selected.next(offset)
                            if offset>0 else [selected])
                        if len(linked)==1:samples.append(linked[0])
                    self._navigation_branch_samples=samples
                return selected
            if viable:choices=viable
        if self.route_manager is not None:
            reference=self.route_manager.target_point(distance)
        elif self.route_context is not None and self.route_plan:
            index=min(len(self.route_plan)-1,bisect.bisect_left(
                self.route_context.distances_m,self.progress_m()+distance))
            transform=self.route_plan[index][0].transform
            reference=dict(x=float(transform.location.x),y=float(transform.location.y),
                yaw=float(transform.rotation.yaw))
        else:reference=None
        if reference is None:return None
        def score(candidate):
            transform=candidate.transform
            return (abs(self._pid._angle_delta(math.radians(transform.rotation.yaw),
                math.radians(float(reference['yaw'])))),
                math.hypot(transform.location.x-reference['x'],transform.location.y-reference['y']))
        ordered=sorted(choices,key=score)
        first,second=score(ordered[0]),score(ordered[1])
        if abs(first[0]-second[0])<math.radians(1.) and abs(first[1]-second[1])<.1:
            return None
        return ordered[0]

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
            if getattr(self,'_completed_lane_navigation',False):
                ahead=self._current_lane_navigation_point(lookahead_m)
                current=self._current_lane_navigation_point(0.)
                if ahead is None or current is None:return None
                def point(waypoint):
                    transform=waypoint.transform
                    return dict(x=float(transform.location.x),y=float(transform.location.y),
                        z=float(transform.location.z),yaw=float(transform.rotation.yaw))
                return dict(point(ahead),reference=point(current))
            target = self.route_manager.target_point(lookahead_m)
            if target is None:
                return None
            reference = self.route_manager.target_point(0.0) or target
            result = dict(target)
            result["reference"] = reference
            return result
        return None

    def _ahead_heading_change_deg(
        self, start_m: float = 8.0, span_m: float = 26.0, *, unknown_value: float = 0.0
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
                    return unknown_value
                yaw0 = math.radians(float(point0["yaw"]))
                yaw1 = math.radians(float(point1["yaw"]))
            else:
                return unknown_value
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return unknown_value
        return abs(math.degrees(self._pid._angle_delta(yaw1, yaw0)))

    @staticmethod
    def _junction_speed_ceiling_kmh(sharpness_deg, current_speed_kmh):
        speed=float(current_speed_kmh)
        if not math.isfinite(speed) or speed<0:
            raise ValueError('Finite nonnegative speed required')
        if not math.isfinite(sharpness_deg) or sharpness_deg<0:
            return _clamp(11.0+speed*.05,7.0,14.0)
        turn_cap=_clamp(11.0-.05*sharpness_deg+speed*.05,7.0,14.0)
        # Straight connectors do not imply a tight turn. Curvature, lamp,
        # obstacle and legal speed bounds remain separate downstream caps.
        straight_weight=_clamp((45.0-sharpness_deg)/35.0,0.0,1.0)
        return turn_cap+(30.0-turn_cap)*straight_weight

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
            target = EgoPIDController._adjacent_driving_lane(waypoint,'lane_change_'+direction)
            if target is None or target.lane_type != carla.LaneType.Driving:
                return False
            # Adjacent lanes travelling in the opposite direction are never
            # valid evasive targets even if malformed map metadata exposes
            # a lane-change bit.
            if int(target.lane_id) * int(waypoint.lane_id) <= 0:
                return False
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False
        mask = int(carla.LaneChange.Left if direction == 'left' else carla.LaneChange.Right)
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
            return False
        if waypoint is None:
            return False
        pid = getattr(self, '_pid', None)
        command = getattr(self, '_command_id', None)
        if (pid is not None and command is not None
                and getattr(pid, '_lane_change_command_id', None) == command
                and getattr(pid, '_lane_change_target_lane_id', None) == waypoint.lane_id):
            # Already inside the latched target: finishing centering does not
            # request another boundary crossing. Sensor clearance still gates it.
            return True
        try:
            permission = int(waypoint.lane_change)
        except (AttributeError, TypeError, ValueError):
            return False
        if direction == "left":
            return bool(permission & int(carla.LaneChange.Left))
        return bool(permission & int(carla.LaneChange.Right))

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
            "allow_positive_acceleration": getattr(self,'_allow_positive_acceleration',True),
            "longitudinal_control_mode": getattr(self,'_longitudinal_control_mode','ABSOLUTE_SPEED'),
            "speed_tracking_mode": getattr(self,'_speed_tracking_mode',None),
            "command_id": getattr(self,'_command_id',self._action),
            "requested_step_speed_kmh": getattr(
                self, '_requested_decision_speed_kmh', self._target_speed
            ),
            "speed_constraint_codes": list(
                getattr(self, '_speed_constraint_codes', [])
            ),
        }
        if self._junction_or_road_transition_ahead():
            # Sharpness-aware junction ceiling: the cap falls with the actual
            # heading change of the planned corridor and rises slightly with
            # the approach speed.  It only limits physical speed and never
            # changes lane/turn selection.
            sharpness_deg = self._ahead_heading_change_deg(unknown_value=math.nan)
            junction_cap = self._junction_speed_ceiling_kmh(
                sharpness_deg,self._current_speed_kmh())
            intent["target_speed_kmh"] = min(
                float(intent["target_speed_kmh"]),
                junction_cap,
            )
            if (
                intent["target_speed_kmh"]
                < float(intent["requested_step_speed_kmh"]) - 1e-6
                and "junction_speed_cap" not in intent["speed_constraint_codes"]
            ):
                intent["speed_constraint_codes"].append("junction_speed_cap")
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
            if (
                intent["target_speed_kmh"]
                < float(intent["requested_step_speed_kmh"]) - 1e-6
                and "lane_transition_speed_cap" not in intent["speed_constraint_codes"]
            ):
                intent["speed_constraint_codes"].append(
                    "lane_transition_speed_cap"
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
        expiry = self._sequence_expiry
        if expiry is not None and self.world.get_snapshot().timestamp.elapsed_seconds > expiry:
            intent.update(action='stop',target_speed_kmh=0.,emergency=False)
        elif self._sequence_command and abs(intent['target_speed_kmh']-self._target_speed)<1e-6:
            intent.update(self._sequence_command)
        control, _ = self._pid.run_step(intent, self.fixed_delta_seconds)
        from control.road_speed_guard import apply_road_speed_guard
        try:
            velocity = self.ego.get_velocity()
            acceleration = self.ego.get_acceleration()
            speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            along_acceleration = ((velocity.x * acceleration.x + velocity.y * acceleration.y
                + velocity.z * acceleration.z) / speed_mps) if speed_mps > .1 else 0.
            road_limit = float(self.ego.get_speed_limit())
            self._road_speed_guard = apply_road_speed_guard(control, speed_kmh=speed_mps * 3.6,
                speed_limit_kmh=road_limit, acceleration_mps2=along_acceleration,
                dt_s=self.fixed_delta_seconds)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            self._road_speed_guard = dict(source='measured_ego_road_speed_guard/1.0',
                status='UNAVAILABLE', changed=False)
        self._last_route_target=route_target
        return control

    def execution_state(self) -> dict[str, Any]:
        pid_state = self._pid.get_execution_state()
        if (pid_state.get('lane_change_completed') is True
                and self._action in {'keep_lane','accelerate','decelerate','lane_change_left','lane_change_right'}):
            self.handoff_completed_lane()
        constraint = getattr(self, '_environment_speed_constraint', None)
        if constraint is not None:
            pid_state = dict(pid_state, **constraint, speed_target_status='CONSTRAINED')
        return {
            "source_step_id": getattr(self,'_source_step_id',None),
            "risk_motion_permission": getattr(self,'_risk_motion_permission',None),
            "action": self._action,
            "target_speed_kmh": self._target_speed,
            "target_lane": self._target_lane,
            "speed_kmh": round(self._current_speed_kmh(), 3),
            "progress_m": round(self.progress_m(), 3),
            "requested_target_speed_kmh": (constraint['requested_target_speed_kmh'] if constraint else getattr(
                self, '_requested_decision_speed_kmh', self._target_speed
            )),
            "effective_target_speed_kmh": pid_state.get(
                "effective_target_speed_kmh", self._target_speed
            ),
            "speed_target_status": pid_state.get(
                "speed_target_status", "UNKNOWN"
            ),
            "speed_constraint_codes": pid_state.get(
                "speed_constraint_codes",
                list(getattr(self, '_speed_constraint_codes', [])),
            ),
              "pid": pid_state,
              "road_speed_guard": getattr(self, '_road_speed_guard', None),
              "navigation_reference": dict(completed_lane_navigation=bool(
                  getattr(self,'_completed_lane_navigation',False)),
                  branch_latched=bool(getattr(self,'_navigation_branch_samples',None)),
                  target=getattr(self,'_last_route_target',None)),
        }

    def navigation_goal_observation(self, *, frame_id):
        """Report arrival from measured ego geometry and the public route endpoint."""
        if self.route_context is not None and self.route_plan:
            transform = self.route_plan[-1][0].transform
            end = float(self.route_context.distances_m[-1])
            goal = transform.location
        elif self.route_manager is not None and getattr(self.route_manager, 'route', None):
            point = self.route_manager.route[-1]
            end = float(point['distance_m'])
            goal = carla.Location(x=point['x'], y=point['y'], z=point['z'])
        else:
            return dict(valid=False, source='measured_ego_public_route_endpoint')
        location = self.ego.get_location()
        progress = self.progress_m()
        distance = math.sqrt((location.x-goal.x)**2 + (location.y-goal.y)**2 +
                             (location.z-goal.z)**2)
        return dict(valid=all(math.isfinite(value) for value in (end, progress, distance)),
                    source='measured_ego_public_route_endpoint', frame_id=frame_id,
                    route_end_m=end, route_progress_m=progress, endpoint_distance_m=distance,
                    reached=progress >= end-3. and distance <= 3.)
