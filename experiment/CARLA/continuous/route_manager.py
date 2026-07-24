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
            })
            next_waypoints = waypoint.next(step_m)
            directive = self._next_due_directive(pending_directives, distance_m)
            if directive is not None:
                selected = self._apply_directive(waypoint, next_waypoints, directive)
                if selected is not None:
                    waypoint = selected
                    pending_directives.pop(0)
                    applied = dict(directive)
                    applied["applied_distance_m"] = round(distance_m, 3)
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
        """Return the first route point at least ``lookahead_m`` ahead."""
        if not self.route:
            return None
        target_distance = self.progress_m + max(0.0, float(lookahead_m))
        for waypoint in self.route[self.current_index:]:
            if waypoint["distance_m"] >= target_distance:
                return dict(waypoint)
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
        start = max(0, self.current_index - 50)
        end = min(len(self.route), self.current_index + 201)
        closest_index = min(
            range(start, end),
            key=lambda index: math.hypot(
                location.x - self.route[index]["x"],
                location.y - self.route[index]["y"],
            ),
        )
        self.current_index = closest_index
        self.progress_m = self.route[closest_index]["distance_m"]
        return self.progress_m

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
        if action in ("turn_left", "turn_right"):
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
            delta = self._angle_delta(
                math.radians(candidate.transform.rotation.yaw),
                reference_yaw,
            )
            if abs(delta) < math.radians(12.0) or abs(delta) > math.radians(150.0):
                continue
            choices.append((delta, candidate))
        if not choices:
            return None
        if action == "turn_right":
            return max(choices, key=lambda item: item[0])[1]
        return min(choices, key=lambda item: item[0])[1]

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
        }
