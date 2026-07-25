"""PID-based low-level controller for the ego vehicle."""

import math

import carla

from control.protocol import normalize_intent


def _clamp(value, lower, upper):
    return max(lower, min(value, upper))


class EgoPIDController:
    """Convert a normalized action into ``carla.VehicleControl``.

    It intentionally owns only low-level control.  A planner/agent can still
    supply ``target_location`` for turns and junction navigation.
    """

    def __init__(self, vehicle, world_map, target_speed_kmh=25.0):
        self.vehicle = vehicle
        self.world_map = world_map
        self.default_speed_kmh = float(target_speed_kmh)
        self._speed_integral = 0.0
        self._previous_speed_error = 0.0

    def run_step(self, intent, dt):
        intent = normalize_intent(intent, self.default_speed_kmh)
        if intent["emergency"] or intent["action"] == "emergency_brake":
            # 紧急制动只接管纵向控制，横向车道保持继续生效，避免
            # 车辆在路口已产生转向后继续偏离当前车道。
            hold_lane_intent = dict(intent)
            hold_lane_intent["action"] = "keep_lane"
            steer = self._lateral_control(hold_lane_intent)
            return carla.VehicleControl(throttle=0.0, brake=1.0, steer=steer), intent

        target_speed = self._resolve_target_speed(intent)
        current_speed = self._get_speed_kmh()
        throttle, brake = self._longitudinal_control(target_speed, current_speed, dt)
        steer = self._lateral_control(intent)
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer), intent

    def _resolve_target_speed(self, intent):
        action = intent["action"]
        current_speed = self._get_speed_kmh()
        requested = intent["target_speed_kmh"]
        if action == "accelerate":
            return max(requested, current_speed + 5.0)
        if action == "decelerate":
            return min(requested, max(0.0, current_speed - 5.0))
        if action == "stop":
            return 0.0
        return requested

    def _longitudinal_control(self, target_speed, current_speed, dt):
        error = target_speed - current_speed
        safe_dt = max(float(dt), 1e-3)
        self._speed_integral = _clamp(self._speed_integral + error * safe_dt, -30.0, 30.0)
        derivative = (error - self._previous_speed_error) / safe_dt
        self._previous_speed_error = error
        command = 0.045 * error + 0.003 * self._speed_integral + 0.004 * derivative

        if error >= 0.0:
            return _clamp(command, 0.0, 0.75), 0.0
        return 0.0, _clamp(-command, 0.0, 0.8)

    def _lateral_control(self, intent):
        vehicle_transform = self.vehicle.get_transform()
        waypoint = self.world_map.get_waypoint(
            vehicle_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return 0.0

        target_location = self._target_location(waypoint, intent, vehicle_transform)
        if target_location is None:
            return 0.0

        dx = target_location.x - vehicle_transform.location.x
        dy = target_location.y - vehicle_transform.location.y
        desired_yaw = math.atan2(dy, dx)
        current_yaw = math.radians(vehicle_transform.rotation.yaw)
        angle = math.atan2(math.sin(desired_yaw - current_yaw), math.cos(desired_yaw - current_yaw))
        return _clamp(1.35 * angle, -0.7, 0.7)

    def _target_location(self, waypoint, intent, vehicle_transform):
        requested = intent.get("target_location")
        if requested is not None:
            return carla.Location(requested["x"], requested["y"], requested["z"])

        target_waypoint = self._target_waypoint(waypoint, intent)
        return target_waypoint.transform.location if target_waypoint is not None else None

    def _target_waypoint(self, waypoint, intent):
        action = intent["action"]
        if action == "lane_change_left":
            candidate = waypoint.get_left_lane()
            if candidate is not None and candidate.lane_type == carla.LaneType.Driving:
                waypoint = candidate
        elif action == "lane_change_right":
            candidate = waypoint.get_right_lane()
            if candidate is not None and candidate.lane_type == carla.LaneType.Driving:
                waypoint = candidate

        next_waypoints = waypoint.next(8.0)
        if not next_waypoints:
            return waypoint

        reference_yaw = math.radians(waypoint.transform.rotation.yaw)
        return min(
            next_waypoints,
            key=lambda candidate: abs(self._angle_delta(
                math.radians(candidate.transform.rotation.yaw),
                reference_yaw,
            )),
        )

    @staticmethod
    def _angle_delta(current, reference):
        return math.atan2(math.sin(current - reference), math.cos(current - reference))

    def _get_speed_kmh(self):
        velocity = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
