"""Route progress tracking for continuous CARLA experiments."""

import json
import math


class RouteManager:
    def __init__(self, world):
        self.world = world
        self.map = world.get_map()
        self.route = []
        self.route_length_m = 0.0
        self.progress_m = 0.0
        self.current_index = 0
        self.applied_directives = []
        self.unapplied_directives = []
        self.cross_track_error_m = 0.0

    def build_route(self, start_location=None, length_m=8000.0, step_m=5.0, directives=None):
        """Build a route and apply map-relative lane/turn directives.

        A directive is applied at the first suitable map waypoint after its
        ``distance_m``.  The resulting route, not the directive itself, is
        then used by the controller as the geometric reference.
        """
        if step_m <= 0.0:
            raise ValueError("step_m must be positive")
        if start_location is None:
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                raise RuntimeError("CARLA map has no spawn points")
            start_location = spawn_points[0].location
        waypoint = self.map.get_waypoint(start_location, project_to_road=True)
        self.route = []
        pending_directives = [dict(item) for item in (directives or [])]
        pending_directives.sort(key=lambda item: float(item.get("distance_m", 0.0)))
        self.applied_directives = []
        self.unapplied_directives = []
        distance_m = 0.0
        while waypoint is not None and distance_m < float(length_m):
            transform = waypoint.transform
            self.route.append({
                "x": round(transform.location.x, 3),
                "y": round(transform.location.y, 3),
                "z": round(transform.location.z, 3),
                "yaw": round(transform.rotation.yaw, 3),
                "distance_m": round(distance_m, 3),
                "road_id": int(waypoint.road_id),
                "section_id": int(waypoint.section_id),
                "lane_id": int(waypoint.lane_id),
                "is_junction": bool(waypoint.is_junction),
            })
            next_waypoints = waypoint.next(step_m)
            directive = self._next_due_directive(pending_directives, distance_m)
            if directive is not None:
                entry_waypoint = waypoint
                selected = self._apply_directive(waypoint, next_waypoints, directive)
                if selected is not None:
                    waypoint = selected
                    pending_directives.pop(0)
                    applied = dict(directive)
                    applied["applied_distance_m"] = round(distance_m, 3)
                    applied["entry_road_id"] = int(entry_waypoint.road_id)
                    applied["selected_road_id"] = int(selected.road_id)
                    applied["entry_yaw_deg"] = round(
                        float(entry_waypoint.transform.rotation.yaw), 3
                    )
                    self.applied_directives.append(applied)
                else:
                    waypoint = self._choose_straight(waypoint, next_waypoints)
            else:
                waypoint = self._choose_straight(waypoint, next_waypoints)
            distance_m += float(step_m)
        self.route_length_m = self.route[-1]["distance_m"] if self.route else 0.0
        self.progress_m = 0.0
        self.current_index = 0
        self.unapplied_directives = pending_directives
        return list(self.route)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        waypoints = payload.get("waypoints", [])
        self.route = [self._normalize_waypoint(item) for item in waypoints]
        self.route_length_m = float(payload.get(
            "length_m", payload.get("length", self.route[-1]["distance_m"] if self.route else 0.0)
        ))
        self.progress_m = 0.0
        self.current_index = 0

    def target_point(self, lookahead_m=20.0):
        """Return a continuously interpolated route point ahead of the ego."""
        if not self.route:
            return None
        target_distance = self.progress_m + max(0.0, float(lookahead_m))
        start_index = max(0, min(self.current_index, len(self.route) - 1))
        for index in range(start_index + 1, len(self.route)):
            first = self.route[index - 1]
            second = self.route[index]
            if second["distance_m"] < target_distance:
                continue
            span = float(second["distance_m"]) - float(first["distance_m"])
            if span <= 1e-9:
                return dict(second)
            ratio = max(
                0.0,
                min(
                    1.0,
                    (target_distance - float(first["distance_m"])) / span,
                ),
            )
            result = dict(second)
            for axis in ("x", "y", "z"):
                result[axis] = (
                    float(first[axis])
                    + ratio * (float(second[axis]) - float(first[axis]))
                )
            dx = float(second["x"]) - float(first["x"])
            dy = float(second["y"]) - float(first["y"])
            # The route polyline is the controller's geometric authority.
            # CARLA junction waypoint rotations can jump to a connector's
            # terminal heading before its coordinates turn, which creates a
            # fictitious curvature spike and drives the ego over lane lines.
            if math.hypot(dx, dy) > 1e-6:
                result["yaw"] = math.degrees(math.atan2(dy, dx))
            else:
                result["yaw"] = float(first["yaw"])
            result["distance_m"] = target_distance
            return result
        return dict(self.route[-1])

    def seek(self, progress_m):
        """Set route tracking to the waypoint nearest to a saved progress value."""
        if not self.route:
            self.current_index = 0
            self.progress_m = 0.0
            return self.progress_m
        target = max(0.0, min(float(progress_m), self.route_length_m))
        self.current_index = min(
            range(len(self.route)),
            key=lambda index: abs(self.route[index]["distance_m"] - target),
        )
        self.progress_m = self.route[self.current_index]["distance_m"]
        return self.progress_m

    def update(self, ego_vehicle):
        if not self.route:
            return self.progress_m
        location = ego_vehicle.get_location()
        transform_getter = getattr(ego_vehicle, "get_transform", None)
        if callable(transform_getter):
            yaw_deg = float(transform_getter().rotation.yaw)
        else:
            # Keep route progress usable for lightweight callers that only
            # expose position. Real CARLA actors always take the heading-aware
            # path above; this fallback simply avoids rejecting valid legacy
            # route-manager integrations and tests.
            yaw_deg = float(self.route[self.current_index].get("yaw", 0.0))
        start = max(0, self.current_index - 5)
        maximum_progress = self.progress_m + 20.0
        end = self.current_index + 1
        while (
            end < len(self.route)
            and self.route[end]["distance_m"] <= maximum_progress
        ):
            end += 1
        end = min(len(self.route), max(end, self.current_index + 2))
        closest_index = min(
            range(start, end),
            key=lambda index: self._tracking_cost(location, yaw_deg, self.route[index]),
        )
        projected_progress, cross_track_error = self._project_progress(
            location, closest_index
        )
        self.cross_track_error_m = cross_track_error
        self.progress_m = max(
            self.progress_m,
            min(projected_progress, maximum_progress),
        )
        while (
            self.current_index + 1 < len(self.route)
            and self.route[self.current_index + 1]["distance_m"] <= self.progress_m
        ):
            self.current_index += 1
        return self.progress_m

    def _tracking_cost(self, location, yaw_deg, route_point):
        distance = math.hypot(
            location.x - route_point["x"],
            location.y - route_point["y"],
        )
        heading_error = abs(math.degrees(self._angle_delta(
            math.radians(yaw_deg),
            math.radians(route_point["yaw"]),
        )))
        return distance + min(heading_error, 120.0) * 0.08

    def _project_progress(self, location, closest_index):
        best = None
        for start_index in (closest_index - 1, closest_index):
            if start_index < 0 or start_index + 1 >= len(self.route):
                continue
            first = self.route[start_index]
            second = self.route[start_index + 1]
            dx = second["x"] - first["x"]
            dy = second["y"] - first["y"]
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-9:
                continue
            ratio = max(0.0, min(1.0, (
                (location.x - first["x"]) * dx
                + (location.y - first["y"]) * dy
            ) / length_sq))
            projected_x = first["x"] + ratio * dx
            projected_y = first["y"] + ratio * dy
            error = math.hypot(location.x - projected_x, location.y - projected_y)
            progress = (
                float(first["distance_m"])
                + ratio * (
                    float(second["distance_m"]) - float(first["distance_m"])
                )
            )
            candidate = (error, progress)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            return best[1], best[0]
        point = self.route[closest_index]
        return float(point["distance_m"]), math.hypot(
            location.x - point["x"], location.y - point["y"]
        )

    def is_finished(self, tolerance_m=10.0):
        return self.progress_m >= max(0.0, self.route_length_m - float(tolerance_m))

    @staticmethod
    def _next_due_directive(directives, distance_m):
        if directives and float(directives[0].get("distance_m", 0.0)) <= distance_m:
            return directives[0]
        return None

    def _apply_directive(self, waypoint, next_waypoints, directive):
        action = str(directive.get("action", "")).strip().lower()
        if action in ("lane_change_left", "lane_change_right"):
            candidate = (
                waypoint.get_left_lane()
                if action == "lane_change_left"
                else waypoint.get_right_lane()
            )
            if self._is_same_direction_driving_lane(waypoint, candidate):
                return candidate
            return None
        if action in ("turn_left", "turn_right", "u_turn"):
            return self._choose_turn(waypoint, next_waypoints, action)
        raise ValueError("Unsupported route directive: {0}".format(action))

    def _choose_straight(self, waypoint, candidates):
        if not candidates:
            return None
        reference_yaw = math.radians(waypoint.transform.rotation.yaw)
        return min(
            candidates,
            key=lambda candidate: abs(self._angle_delta(
                math.radians(candidate.transform.rotation.yaw),
                reference_yaw,
            )),
        )

    def _choose_turn(self, waypoint, candidates, action):
        if not candidates:
            return None
        reference_yaw = math.radians(waypoint.transform.rotation.yaw)
        choices = []
        for candidate in candidates:
            endpoint = self._trace_branch_endpoint(candidate, 40.0, 5.0)
            delta = self._angle_delta(
                math.radians(endpoint.transform.rotation.yaw),
                reference_yaw,
            )
            if action == "u_turn":
                if abs(delta) < math.radians(90.0):
                    continue
            elif abs(delta) < math.radians(12.0) or abs(delta) > math.radians(150.0):
                continue
            choices.append((delta, candidate))
        if not choices:
            return None
        if action == "u_turn":
            return max(choices, key=lambda item: abs(item[0]))[1]
        if action == "turn_right":
            desired = [item for item in choices if item[0] > 0.0]
            return max(desired, key=lambda item: item[0])[1] if desired else None
        desired = [item for item in choices if item[0] < 0.0]
        return min(desired, key=lambda item: item[0])[1] if desired else None

    def _trace_branch_endpoint(self, waypoint, horizon_m, step_m):
        """Follow a connector far enough to distinguish a gradual turn."""
        current = waypoint
        travelled = 0.0
        while current is not None and travelled < float(horizon_m):
            candidates = current.next(step_m)
            if not candidates:
                break
            current = self._choose_straight(current, candidates)
            travelled += float(step_m)
        return current or waypoint

    @staticmethod
    def _is_same_direction_driving_lane(reference, candidate):
        if candidate is None:
            return False
        if candidate.lane_type != reference.lane_type:
            return False
        if candidate.road_id != reference.road_id:
            return False
        return reference.lane_id * candidate.lane_id > 0

    @staticmethod
    def _angle_delta(current, reference):
        return math.atan2(math.sin(current - reference), math.cos(current - reference))

    @staticmethod
    def _normalize_waypoint(waypoint):
        return {
            "x": float(waypoint["x"]),
            "y": float(waypoint["y"]),
            "z": float(waypoint.get("z", 0.0)),
            "yaw": float(waypoint.get("yaw", 0.0)),
            "distance_m": float(waypoint.get("distance_m", waypoint.get("distance", 0.0))),
            "road_id": waypoint.get("road_id"),
            "section_id": waypoint.get("section_id"),
            "lane_id": waypoint.get("lane_id"),
            "is_junction": waypoint.get("is_junction"),
        }
