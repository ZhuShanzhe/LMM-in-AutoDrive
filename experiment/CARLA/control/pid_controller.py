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
        self._lane_change_command_id = None
        self._lane_change_target_lane_id = None
        self._last_lane_change_intent = None
        self._lane_change_settle_frames = 0
        self._emergency_latched = False
        self._last_control = None
        # Tuned for CARLA passenger vehicles on level arterial roads. Keep
        # these controller-owned so high-level decisions remain unit-agnostic.
        self.speed_kp = 0.075
        self.speed_ki = 0.012
        self.speed_kd = 0.004
        self.speed_integral_limit = 45.0

    def run_step(self, intent, dt):
        intent = normalize_intent(intent, self.default_speed_kmh)
        emergency_requested = (
            intent["emergency"] or intent["action"] == "emergency_brake"
        )
        if emergency_requested:
            self._emergency_latched = True
        elif self._emergency_latched and self._get_speed_kmh() <= 0.5:
            self._emergency_latched = False
        if self._emergency_latched:
            # 紧急制动只接管纵向控制，横向车道保持继续生效，避免
            # 车辆在路口已产生转向后继续偏离当前车道。
            hold_lane_intent = dict(intent)
            hold_lane_intent["action"] = "keep_lane"
            steer = self._lateral_control(hold_lane_intent)
            self._reset_longitudinal_state()
            control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=steer)
            self._last_control = control
            return control, intent

        target_speed = self._resolve_target_speed(intent)
        current_speed = self._get_speed_kmh()
        throttle, brake = self._longitudinal_control(target_speed, current_speed, dt)
        steer = self._lateral_control(self._lateral_intent_for_control(intent))
        control = self._smooth_control(throttle, brake, steer)
        self._last_control = control
        return control, intent

    def _resolve_target_speed(self, intent):
        action = intent["action"]
        current_speed = self._get_speed_kmh()
        requested = intent["target_speed_kmh"]
        if action == "accelerate":
            return max(requested, current_speed + 5.0)
        if action == "decelerate":
            # SET_SPEED/ADJUST_SPEED carries an absolute target when the
            # parser provides one.  A relative decrement here can collapse
            # the target to zero when a newly activated command starts while
            # the vehicle is already slow, leaving the PID permanently stopped.
            return requested
        if action == "stop":
            return 0.0
        return requested

    def _longitudinal_control(self, target_speed, current_speed, dt):
        error = target_speed - current_speed
        safe_dt = max(float(dt), 1e-3)
        # Avoid throttle/brake chatter around the target speed.
        if abs(error) <= 0.8:
            self._speed_integral *= 0.8
            self._previous_speed_error = error
            return 0.0, 0.0
        if error < 0.0 < self._previous_speed_error:
            self._speed_integral = 0.0
        self._speed_integral = _clamp(
            self._speed_integral + error * safe_dt,
            -self.speed_integral_limit,
            self.speed_integral_limit,
        )
        derivative = (error - self._previous_speed_error) / safe_dt
        self._previous_speed_error = error
        command = (
            self.speed_kp * error
            + self.speed_ki * self._speed_integral
            + self.speed_kd * derivative
        )

        if error >= 0.0:
            return _clamp(command, 0.0, 0.85), 0.0
        return 0.0, _clamp(-command, 0.0, 0.8)

    def _smooth_control(self, throttle, brake, steer):
        """Limit nominal actuator changes; emergency braking bypasses this path."""
        if self._last_control is None:
            return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
        previous = self._last_control
        throttle = _clamp(throttle, max(0.0, previous.throttle - 0.18), previous.throttle + 0.18)
        brake = _clamp(brake, max(0.0, previous.brake - 0.24), previous.brake + 0.24)
        steer = _clamp(steer, previous.steer - 0.06, previous.steer + 0.06)
        if brake > 0.01:
            throttle = 0.0
        elif throttle > 0.01:
            brake = 0.0
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)

    def _reset_longitudinal_state(self):
        self._speed_integral = 0.0
        self._previous_speed_error = 0.0

    def _lateral_intent_for_control(self, intent):
        """Hold the target-lane heading briefly after a merge is reported done."""
        if intent["action"] in ("lane_change_left", "lane_change_right"):
            self._last_lane_change_intent = dict(intent)
            self._lane_change_settle_frames = 20
            return intent
        if self._last_lane_change_intent is not None and self._lane_change_settle_frames > 0:
            self._lane_change_settle_frames -= 1
            settling = dict(self._last_lane_change_intent)
            # The stored action only owns lateral guidance; current intent
            # continues to own target speed and longitudinal safety handling.
            settling["target_speed_kmh"] = intent["target_speed_kmh"]
            return settling
        self._last_lane_change_intent = None
        return intent

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
        lane_yaw = math.radians(waypoint.transform.rotation.yaw)
        lateral_error_m = (
            (vehicle_transform.location.x - waypoint.transform.location.x) * -math.sin(lane_yaw)
            + (vehicle_transform.location.y - waypoint.transform.location.y) * math.cos(lane_yaw)
        )
        # The route target supplies heading; the nearest lane center prevents
        # the vehicle from cutting through the inside of a tight curve.
        lane_change_active = intent["action"] in ("lane_change_left", "lane_change_right")
        # During a lane change ``waypoint`` belongs to the target lane. Applying
        # its full centreline correction before the ego reaches that lane creates
        # a saturated steering impulse and can cross multiple lanes.
        centerline_correction = 0.0 if lane_change_active else -0.22 * _clamp(
            lateral_error_m, -2.5, 2.5
        )
        # Keep the low-level command smooth on Town04's tight ramps.  The
        # previous 0.7 limit could steer through a roadside guardrail or
        # vegetation when the nearest waypoint changed at a junction.
        steering_limit = 0.16 if lane_change_active else 0.45
        return _clamp(1.35 * angle + centerline_correction, -steering_limit, steering_limit)

    def _target_location(self, waypoint, intent, vehicle_transform):
        requested = intent.get("target_location")
        if requested is not None:
            return carla.Location(requested["x"], requested["y"], requested["z"])

        target_waypoint = self._target_waypoint(waypoint, intent)
        return target_waypoint.transform.location if target_waypoint is not None else None

    def _target_waypoint(self, waypoint, intent):
        action = intent["action"]
        if action in ("lane_change_left", "lane_change_right"):
            waypoint = self._lane_change_waypoint(waypoint, intent)
            if waypoint.lane_id == self._lane_change_target_lane_id:
                # A lane centre at the ego's current longitudinal position is
                # not a drivable reference. Look ahead on the target lane so
                # the lateral controller follows a shallow, continuous merge.
                return self._forward_waypoint(waypoint, self._lookahead_distance(2.0, 32.0, 45.0))

        next_waypoints = waypoint.next(self._lookahead_distance(1.5, 10.0, 24.0))
        if not next_waypoints:
            return waypoint

        # 在规划模块提供路线前，变道和转向暂使用该兜底逻辑；
        # 保持车道则使用上方的初始航向参考线。
        if action in ("turn_left", "turn_right"):
            turn_waypoint = self._turn_waypoint(waypoint, next_waypoints, action)
            if turn_waypoint is not None:
                return turn_waypoint
        return min(
            next_waypoints,
            key=lambda candidate: abs(self._angle_delta(
                math.radians(candidate.transform.rotation.yaw),
                math.radians(waypoint.transform.rotation.yaw),
            )),
        )

    def _turn_waypoint(self, waypoint, candidates, action):
        """Pick the requested branch when no route target is available."""
        reference_yaw = math.radians(waypoint.transform.rotation.yaw)
        choices = []
        for candidate in candidates:
            delta = self._angle_delta(
                math.radians(candidate.transform.rotation.yaw), reference_yaw
            )
            if math.radians(12.0) <= abs(delta) <= math.radians(150.0):
                choices.append((delta, candidate))
        if not choices:
            return None
        selector = max if action == "turn_right" else min
        return selector(choices, key=lambda item: item[0])[1]

    def _lane_change_waypoint(self, waypoint, intent):
        command_id = intent.get("command_id") or intent["action"]
        if command_id != self._lane_change_command_id:
            candidate = self._adjacent_driving_lane(waypoint, intent["action"])
            self._lane_change_command_id = command_id
            self._lane_change_target_lane_id = (
                candidate.lane_id
                if candidate is not None and candidate.lane_type == carla.LaneType.Driving
                else None
            )

        if self._lane_change_target_lane_id is None:
            return waypoint
        if waypoint.lane_id == self._lane_change_target_lane_id:
            return waypoint

        candidate = self._adjacent_driving_lane(waypoint, intent["action"])
        if (
            candidate is not None
            and candidate.lane_type == carla.LaneType.Driving
            and candidate.lane_id == self._lane_change_target_lane_id
        ):
            return candidate
        return waypoint

    @staticmethod
    def _adjacent_driving_lane(waypoint, action):
        """Return the adjacent lane on the vehicle-relative requested side.

        CARLA's OpenDRIVE lane accessors use the road reference direction. That
        direction can differ from a lane's travel direction, so choosing an API
        named ``get_left_lane`` alone can select the physical right side.
        """
        candidates = []
        for candidate in (waypoint.get_left_lane(), waypoint.get_right_lane()):
            if (
                candidate is None
                or candidate.lane_type != carla.LaneType.Driving
                or candidate.road_id != waypoint.road_id
                or candidate.lane_id * waypoint.lane_id <= 0
            ):
                continue
            yaw = math.radians(waypoint.transform.rotation.yaw)
            dx = candidate.transform.location.x - waypoint.transform.location.x
            dy = candidate.transform.location.y - waypoint.transform.location.y
            # CARLA uses a left-handed road frame: at yaw=0, +Y is the
            # vehicle's right. This projection is therefore a *rightward*
            # offset, not a leftward offset as in a conventional XY plane.
            right_offset = -math.sin(yaw) * dx + math.cos(yaw) * dy
            candidates.append((right_offset, candidate))
        if action == "lane_change_left":
            choices = [item for item in candidates if item[0] < -0.1]
            return min(choices, key=lambda item: item[0])[1] if choices else None
        choices = [item for item in candidates if item[0] > 0.1]
        return max(choices, key=lambda item: item[0])[1] if choices else None

    @staticmethod
    def _angle_delta(current, reference):
        return math.atan2(math.sin(current - reference), math.cos(current - reference))

    def _get_speed_kmh(self):
        velocity = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def get_execution_state(self):
        """Return controller-side facts used for plan-step feedback."""
        waypoint = self.world_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        current_lane_id = getattr(waypoint, "lane_id", None)
        centered_in_target_lane = False
        heading_aligned_with_target_lane = False
        if waypoint is not None and current_lane_id == self._lane_change_target_lane_id:
            vehicle_transform = self.vehicle.get_transform()
            lane_yaw = math.radians(waypoint.transform.rotation.yaw)
            lateral_error_m = (
                (vehicle_transform.location.x - waypoint.transform.location.x) * -math.sin(lane_yaw)
                + (vehicle_transform.location.y - waypoint.transform.location.y) * math.cos(lane_yaw)
            )
            centered_in_target_lane = abs(lateral_error_m) <= 0.45
            heading_error = self._angle_delta(
                math.radians(vehicle_transform.rotation.yaw), lane_yaw
            )
            # Do not hand control back to normal lane keeping while the ego is
            # still yawed into the merge. That hand-off was the source of the
            # visible late steering kick at the end of a lane change.
            heading_aligned_with_target_lane = abs(heading_error) <= math.radians(3.0)
        return {
            "speed_kmh": self._get_speed_kmh(),
            "current_lane_id": current_lane_id,
            "target_lane_id": self._lane_change_target_lane_id,
            "lane_change_completed": (
                self._lane_change_target_lane_id is not None
                and centered_in_target_lane
                and heading_aligned_with_target_lane
            ),
            "emergency_latched": self._emergency_latched,
        }

    def _forward_waypoint(self, waypoint, distance_m):
        candidates = waypoint.next(distance_m)
        if not candidates:
            return waypoint
        reference_yaw = math.radians(waypoint.transform.rotation.yaw)
        return min(
            candidates,
            key=lambda candidate: abs(self._angle_delta(
                math.radians(candidate.transform.rotation.yaw), reference_yaw
            )),
        )

    def _lookahead_distance(self, speed_gain, minimum_m, maximum_m):
        speed_mps = self._get_speed_kmh() / 3.6
        return _clamp(minimum_m + speed_gain * speed_mps, minimum_m, maximum_m)
