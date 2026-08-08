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
        self._lane_change_stable_frames = 0
        self._last_lane_change_intent = None
        self._lane_change_settle_frames = 0
        self._emergency_latched = False
        self._emergency_clear_frames = 0
        self._last_control = None
        self._filtered_steer = 0.0
        self._last_lateral_debug = {}
        self._yaw_history = []
        self._last_yaw_time_s = 0.0
        self._last_yaw_rate_rad_s = 0.0
        self._last_applied_steer = 0.0
        self._turn_unsafe_frames = 0
        self._turn_unsafe_speed_cap_kmh = 100.0
        # Tuned for CARLA passenger vehicles on level arterial roads. Keep
        # these controller-owned so high-level decisions remain unit-agnostic.
        self.speed_kp = 0.075
        self.speed_ki = 0.012
        self.speed_kd = 0.004
        self.speed_integral_limit = 45.0
        self.speed_kp_crawl = 0.22
        self._crawl_active = False

    def run_step(self, intent, dt):
        intent = normalize_intent(intent, self.default_speed_kmh)
        emergency_requested = (
            intent["emergency"] or intent["action"] == "emergency_brake"
        )
        if emergency_requested:
            self._emergency_latched = True
            self._emergency_clear_frames = 0
        elif self._emergency_latched:
            self._emergency_clear_frames += 1
            if self._emergency_clear_frames >= 2:
                self._emergency_latched = False
                self._emergency_clear_frames = 0
        if self._emergency_latched:
            # 紧急制动只接管纵向控制，横向车道保持继续生效，避免
            # 车辆在路口已产生转向后继续偏离当前车道。
            hold_lane_intent = dict(intent)
            hold_lane_intent["action"] = "keep_lane"
            steer = self._lateral_control(hold_lane_intent, dt)
            self._reset_longitudinal_state()
            control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=steer)
            self._last_applied_steer = steer
            self._last_control = control
            return control, intent

        target_speed = self._resolve_target_speed(intent)
        target_speed = min(target_speed, self._curvature_speed_cap(intent))
        if self._turn_unsafe_frames > 0:
            target_speed = min(target_speed, self._turn_unsafe_speed_cap_kmh)
            self._turn_unsafe_frames -= 1
        current_speed = self._get_speed_kmh()
        self._crawl_active = (
            target_speed <= 16.0 and current_speed < 2.0
        )
        throttle, brake = self._longitudinal_control(target_speed, current_speed, dt)
        lateral_intent = self._lateral_intent_for_control(intent)
        steer = self._lateral_control(lateral_intent, dt)
        control = self._smooth_control(
            throttle, brake, steer, dt, lateral_intent["action"]
        )
        self._last_control = control
        return control, intent

    def _resolve_target_speed(self, intent):
        action = intent["action"]
        current_speed = self._get_speed_kmh()
        requested = intent["target_speed_kmh"]
        if action == "accelerate":
            # A scheduled SET_SPEED command carries an absolute target.
            # Using ``current + 5`` here makes the setpoint recede forever once
            # the vehicle passes the requested speed, causing full-throttle
            # overspeed instead of convergence.
            return requested
        if action == "decelerate":
            # SET_SPEED/ADJUST_SPEED carries an absolute target when the
            # parser provides one.  A relative decrement here can collapse
            # the target to zero when a newly activated command starts while
            # the vehicle is already slow, leaving the PID permanently stopped.
            return requested
        if action == "stop":
            return 0.0
        return requested

    def _wheelbase_m(self):
        """Estimate the vehicle wheelbase from its actual bounding box."""

        try:
            half_length = float(self.vehicle.bounding_box.extent.x)
        except Exception:
            half_length = 2.4
        return max(2.3, min(3.2, 1.15 * half_length))

    def _max_wheel_angle_rad(self):
        """Passenger-car steering geometry (normalized steer = 1 at ~40deg)."""

        return math.radians(40.0)

    def _route_curvature_of(self, route_reference, target_location):
        """Curvature (1/m) implied by the heading change over the path."""

        if route_reference is None or target_location is None:
            return 0.0
        try:
            reference_yaw = math.radians(float(route_reference["yaw"]))
            target_yaw = math.radians(float(target_location["yaw"]))
            distance = max(
                1.0,
                math.hypot(
                    float(target_location["x"]) - float(route_reference["x"]),
                    float(target_location["y"]) - float(route_reference["y"]),
                ),
            )
        except (KeyError, TypeError, ValueError):
            return 0.0
        return self._angle_delta(target_yaw, reference_yaw) / distance

    def _dynamic_steering_limit(
        self,
        *,
        speed_kmh,
        heading_error_rad,
        curvature,
        lane_width_m=3.5,
    ):
        """Steering authority computed from kinematics, speed and lane width.

        Straight gentle driving keeps a small limit; sharp curves derive
        their authority from the bicycle-model steering angle plus a bounded
        heading-feedback term, scaled down at higher speeds.
        """

        wheelbase = self._wheelbase_m()
        kinematic = math.atan(wheelbase * max(0.0, abs(curvature)))
        feedback = 0.60 * min(
            1.0, abs(heading_error_rad) / math.radians(35.0)
        )
        width_factor = max(0.85, min(1.15, float(lane_width_m) / 3.5))
        speed_factor = max(0.30, 1.0 - float(speed_kmh) / 70.0)
        limit = (kinematic * 1.35 + feedback) * width_factor * speed_factor
        return _clamp(limit, 0.10, 0.85)

    def _dynamic_curve_speed_cap(self, curvature, requested_kmh):
        """Speed ceiling from the lateral-acceleration limit on the arc."""

        if abs(curvature) <= 1e-4:
            return float(requested_kmh)
        lateral_accel_mps2 = 2.2  # comfort ceiling, physically meaningful
        dynamic_kmh = math.sqrt(lateral_accel_mps2 / abs(curvature)) * 3.6
        return min(float(requested_kmh), max(6.0, dynamic_kmh))

    def _update_yaw_history(self, dt):
        """Track recent yaw so the actual path curvature can be estimated."""

        try:
            yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        except Exception:
            return
        now = getattr(self, "_last_yaw_time_s", 0.0) + max(float(dt), 1e-3)
        if self._yaw_history:
            last_t, last_yaw = self._yaw_history[-1]
            d_t = max(1e-3, now - last_t)
            self._last_yaw_rate_rad_s = self._angle_delta(
                yaw, last_yaw
            ) / d_t
        self._yaw_history.append((now, yaw))
        if len(self._yaw_history) > 8:
            self._yaw_history.pop(0)
        self._last_yaw_time_s = now

    def _estimate_curvature_per_m(self, speed_kmh, dt):
        """Actual path curvature from the yaw rate, else the bicycle model."""

        speed_mps = max(0.5, float(speed_kmh) / 3.6)
        self._update_yaw_history(dt)
        if abs(self._last_yaw_rate_rad_s) > 1e-4:
            return self._last_yaw_rate_rad_s / speed_mps
        steer_angle = (
            float(self._last_applied_steer)
            * self._max_wheel_angle_rad()
        )
        return math.tan(steer_angle) / self._wheelbase_m()

    def _predict_arc_points(
        self, origin, yaw, curvature_per_m, horizon_m, step_m=1.0
    ):
        """Sample the predicted circular path (bicycle kinematics).

        The sampled arc is the multi-point trajectory estimate used to reason
        about the turn before it is fully entered.
        """

        points = []
        distance = step_m
        while distance <= horizon_m + 1e-9:
            if abs(curvature_per_m) < 1e-6:
                px = origin.x + math.cos(yaw) * distance
                py = origin.y + math.sin(yaw) * distance
            else:
                px = origin.x + (
                    math.sin(yaw + curvature_per_m * distance)
                    - math.sin(yaw)
                ) / curvature_per_m
                py = origin.y + (
                    math.cos(yaw)
                    - math.cos(yaw + curvature_per_m * distance)
                ) / curvature_per_m
            points.append((distance, px, py))
            distance += step_m
        return points

    def _turn_trajectory_adjust(
        self,
        raw_steer,
        speed_kmh,
        dt,
        curvature_req,
        lateral_error_m,
        action="keep_lane",
    ):
        """Continuously compare the actual trajectory arc with the route arc.

        The ego's actual path curvature is estimated every control step from
        the recent yaw rate (fallback: bicycle model from the last applied
        steer).  The difference against the planned corridor curvature closes
        the gap with a bounded steering correction, and the cruise speed is
        capped while the mismatch is large so the turn is negotiated with both
        speed and steering coordinated.
        """

        if action in ("lane_change_left", "lane_change_right"):
            return raw_steer
        speed_mps = max(0.0, float(speed_kmh) / 3.6)
        if speed_mps < 0.8:
            return raw_steer
        curvature_actual = self._estimate_curvature_per_m(speed_kmh, dt)
        error = float(curvature_req) - curvature_actual
        lateral_error = float(lateral_error_m or 0.0)
        inside_curve = (
            abs(float(curvature_req)) > 1e-4
            and curvature_req * lateral_error > 0.0
        )
        if inside_curve:
            # The ego is already offset toward the centre of the bend:
            # adding more curvature-direction steering would cut the inside
            # edge.  Gently unwind and let the cross-track correction return
            # the vehicle to the corridor centre.
            unwind = _clamp(
                0.02 + 0.10 * min(1.0, abs(lateral_error) / 2.0),
                0.0,
                0.12,
            )
            adjust = -math.copysign(unwind, float(curvature_req))
            self._last_lateral_debug["trajectory_adjust"] = round(
                float(adjust), 6
            )
            self._last_lateral_debug["curvature_error_per_m"] = round(
                float(error), 6
            )
            if abs(lateral_error) > 0.35:
                urgency = _clamp(
                    (abs(lateral_error) - 0.35) / 0.8, 0.0, 1.0
                )
                self._turn_unsafe_frames = 3 + int(4.0 * urgency)
                self._turn_unsafe_speed_cap_kmh = max(
                    12.0, 24.0 - 10.0 * urgency
                )
            return _clamp(raw_steer + adjust, -0.85, 0.85)
        if abs(error) < 0.005:
            return raw_steer
        adjust = _clamp(
            0.5
            * error
            * self._wheelbase_m()
            / self._max_wheel_angle_rad(),
            -0.15,
            0.15,
        )
        self._last_lateral_debug["curvature_error_per_m"] = round(
            float(error), 6
        )
        self._last_lateral_debug["trajectory_adjust"] = round(
            float(adjust), 6
        )
        if abs(error) > 0.015:
            urgency = _clamp(
                (abs(error) - 0.015) / 0.05, 0.0, 1.0
            )
            self._turn_unsafe_frames = 3 + int(4.0 * urgency)
            self._turn_unsafe_speed_cap_kmh = max(
                12.0, 24.0 - 12.0 * urgency
            )
        return _clamp(raw_steer + adjust, -0.85, 0.85)

    def _curvature_speed_cap(self, intent):
        """Apply a speed ceiling derived from the actual road curvature.

        The high-level rule keeps ownership of the requested cruise speed.
        This is a local actuator-safety cap: it only reduces speed on an
        upcoming curved lane and never changes lane/turn selection.
        """

        if intent["action"] in {"emergency_brake", "stop"}:
            return 0.0
        requested = float(intent.get("target_speed_kmh", 100.0) or 100.0)
        target = intent.get("target_location")
        reference = target.get("reference") if isinstance(target, dict) else None
        curvature = self._route_curvature_of(reference, target)
        if abs(curvature) <= 1e-4:
            waypoint = self.world_map.get_waypoint(
                self.vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if waypoint is not None:
                candidates = waypoint.next(30.0)
                if candidates:
                    current_yaw = math.radians(waypoint.transform.rotation.yaw)
                    heading_change_deg = min(
                        abs(math.degrees(self._angle_delta(
                            math.radians(candidate.transform.rotation.yaw),
                            current_yaw,
                        )))
                        for candidate in candidates
                    )
                    curvature = math.radians(heading_change_deg) / 30.0
        return self._dynamic_curve_speed_cap(curvature, requested)

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
        gain = (
            self.speed_kp_crawl
            if self._crawl_active
            else self.speed_kp
        )
        command = (
            gain * error
            + self.speed_ki * self._speed_integral
            + self.speed_kd * derivative
        )

        if error >= 0.0:
            return _clamp(command, 0.0, 0.85), 0.0
        return 0.0, _clamp(-command, 0.0, 0.8)

    def _smooth_control(self, throttle, brake, steer, dt=0.05, lateral_action="keep_lane"):
        """Limit nominal actuator changes; emergency braking bypasses this path.

        Steering is rate-limited in physical time rather than by a fixed value
        per simulation tick.  That keeps a replay with a slower fixed delta
        from becoming more aggressive than the nominal 20 Hz control loop.
        """
        if self._last_control is None:
            self._last_applied_steer = steer
            return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
        previous = self._last_control
        throttle_rate = 0.45 if self._crawl_active else 0.18
        throttle = _clamp(
            throttle,
            max(0.0, previous.throttle - throttle_rate),
            previous.throttle + throttle_rate,
        )
        brake = _clamp(brake, max(0.0, previous.brake - 0.24), previous.brake + 0.24)
        steer_rate = {
            "turn_left": 0.85,
            "turn_right": 0.85,
            "lane_change_left": 0.48,
            "lane_change_right": 0.48,
        }.get(lateral_action, 0.75)
        steer_rate *= self._steering_response_scale()
        steer_delta = steer_rate * _clamp(float(dt), 0.01, 0.10)
        steer = _clamp(steer, previous.steer - steer_delta, previous.steer + steer_delta)
        if brake > 0.01:
            throttle = 0.0
        elif throttle > 0.01:
            brake = 0.0
        self._last_applied_steer = steer
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

    def _lateral_control(self, intent, dt=0.05):
        vehicle_transform = self.vehicle.get_transform()
        waypoint = self.world_map.get_waypoint(
            vehicle_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return 0.0

        local_lane_hold = (
            intent["action"] in ("keep_lane", "accelerate", "decelerate")
            and not bool(intent.get("route_target_trusted", False))
        )
        if local_lane_hold:
            if self._lane_change_target_lane_id is not None:
                held_waypoint = (
                    waypoint
                    if waypoint.lane_id == self._lane_change_target_lane_id
                    else self._lane_with_id(
                        waypoint, self._lane_change_target_lane_id
                    )
                )
                if held_waypoint is not None:
                    waypoint = held_waypoint
            target_waypoint = self._target_waypoint(
                waypoint,
                {**intent, "target_location": None},
            )
            if target_waypoint is not None:
                lane_yaw = math.radians(waypoint.transform.rotation.yaw)
                lateral_error_m = (
                    (vehicle_transform.location.x - waypoint.transform.location.x)
                    * -math.sin(lane_yaw)
                    + (vehicle_transform.location.y - waypoint.transform.location.y)
                    * math.cos(lane_yaw)
                )
                speed_kmh = self._get_speed_kmh()
                hold_reference = {
                    "x": waypoint.transform.location.x,
                    "y": waypoint.transform.location.y,
                    "yaw": waypoint.transform.rotation.yaw,
                }
                hold_target = {
                    "x": target_waypoint.transform.location.x,
                    "y": target_waypoint.transform.location.y,
                    "yaw": target_waypoint.transform.rotation.yaw,
                }
                raw_steer = self._planned_route_steer(
                    route_reference=hold_reference,
                    target_location=hold_target,
                    current_yaw=math.radians(vehicle_transform.rotation.yaw),
                    lateral_error_m=lateral_error_m,
                    speed_kmh=speed_kmh,
                )
                steering_limit = self._dynamic_steering_limit(
                    speed_kmh=speed_kmh,
                    heading_error_rad=self._angle_delta(
                        math.radians(float(hold_target["yaw"])),
                        math.radians(vehicle_transform.rotation.yaw),
                    ),
                    curvature=self._route_curvature_of(
                        hold_reference, hold_target
                    ),
                    lane_width_m=float(
                        getattr(waypoint, "lane_width", 3.5) or 3.5
                    ),
                )
                raw_steer = self._turn_trajectory_adjust(
                    raw_steer,
                    speed_kmh,
                    dt,
                    self._route_curvature_of(hold_reference, hold_target),
                    lateral_error_m,
                    intent["action"],
                )
                return self._filter_steering(
                    _clamp(raw_steer, -steering_limit, steering_limit),
                    intent["action"],
                    dt,
                )

        target_location = self._target_location(waypoint, intent, vehicle_transform)
        if target_location is None:
            return 0.0

        dx = target_location.x - vehicle_transform.location.x
        dy = target_location.y - vehicle_transform.location.y
        desired_yaw = math.atan2(dy, dx)
        current_yaw = math.radians(vehicle_transform.rotation.yaw)
        angle = math.atan2(math.sin(desired_yaw - current_yaw), math.cos(desired_yaw - current_yaw))
        trusted_route_active = (
            bool(intent.get("route_target_trusted", False))
            and isinstance(intent.get("target_location"), dict)
        )
        route_yaw_deg = (
            intent["target_location"].get("yaw")
            if trusted_route_active
            else None
        )
        route_reference = (
            intent["target_location"].get("reference")
            if trusted_route_active
            else None
        )
        route_reference_active = isinstance(route_reference, dict)
        route_curve_active = False
        if route_reference_active:
            lane_yaw = math.radians(float(route_reference["yaw"]))
            if route_yaw_deg is not None:
                route_curve_active = abs(self._angle_delta(
                    math.radians(float(route_yaw_deg)),
                    lane_yaw,
                )) >= math.radians(3.0)
            lateral_error_m = (
                (vehicle_transform.location.x - float(route_reference["x"]))
                * -math.sin(lane_yaw)
                + (vehicle_transform.location.y - float(route_reference["y"]))
                * math.cos(lane_yaw)
            )
        elif route_yaw_deg is not None:
            lane_yaw = math.radians(float(route_yaw_deg))
            tangent_error = abs(self._angle_delta(lane_yaw, desired_yaw))
            if tangent_error <= math.radians(8.0):
                # On a locally straight segment, track the planner-owned route
                # centreline. CARLA's nearest-lane projection can switch to an
                # adjacent lane near a marking and self-reinforce the error.
                reference_location = target_location
                lateral_error_m = (
                    (vehicle_transform.location.x - reference_location.x)
                    * -math.sin(lane_yaw)
                    + (vehicle_transform.location.y - reference_location.y)
                    * math.cos(lane_yaw)
                )
            else:
                # A look-ahead point around a bend lies on an arc, not on the
                # target point's tangent line. Pure-pursuit heading owns that
                # interval; tangent-line centring would oppose the turn.
                lateral_error_m = 0.0
        else:
            lane_yaw = math.radians(waypoint.transform.rotation.yaw)
            reference_location = waypoint.transform.location
            lateral_error_m = (
                (vehicle_transform.location.x - reference_location.x) * -math.sin(lane_yaw)
                + (vehicle_transform.location.y - reference_location.y) * math.cos(lane_yaw)
            )
        # CARLA's projected waypoint can move by a few centimetres from one
        # frame to the next even on a straight road.  Treat that movement as
        # measurement noise rather than repeatedly reversing the wheel.
        if abs(angle) < math.radians(0.6):
            angle = 0.0
        if abs(lateral_error_m) < 0.08:
            lateral_error_m = 0.0
        # The route target supplies heading; the nearest lane center prevents
        # the vehicle from cutting through the inside of a tight curve.
        lane_change_active = (
            intent["action"] in ("lane_change_left", "lane_change_right")
            and waypoint.lane_id != self._lane_change_target_lane_id
        )
        lane_change_settling = (
            intent["action"] in ("lane_change_left", "lane_change_right")
            and not lane_change_active
        )
        # During a lane change ``waypoint`` belongs to the target lane. Applying
        # its full centreline correction before the ego reaches that lane creates
        # a saturated steering impulse and can cross multiple lanes.
        route_turn_active = intent["action"] in ("turn_left", "turn_right")
        current_speed_kmh = self._get_speed_kmh()
        planned_route_active = (
            route_reference_active
            and not route_turn_active
            and intent["action"] not in ("lane_change_left", "lane_change_right")
        )
        if planned_route_active:
            raw_steer = self._planned_route_steer(
                route_reference=route_reference,
                target_location=intent["target_location"],
                current_yaw=current_yaw,
                lateral_error_m=lateral_error_m,
                speed_kmh=current_speed_kmh,
                suppress_cross_track=(
                    bool(getattr(waypoint, "is_junction", False))
                    and abs(lateral_error_m) > 1.0
                ),
            )
            steering_limit = self._dynamic_steering_limit(
                speed_kmh=current_speed_kmh,
                heading_error_rad=angle,
                curvature=self._route_curvature_of(
                    route_reference, intent["target_location"]
                ),
                lane_width_m=float(
                    getattr(waypoint, "lane_width", 3.5) or 3.5
                ),
            )
            raw_steer = self._turn_trajectory_adjust(
                raw_steer,
                current_speed_kmh,
                dt,
                self._route_curvature_of(
                    route_reference, intent["target_location"]
                ),
                lateral_error_m,
                intent["action"],
            )
            return self._filter_steering(
                _clamp(raw_steer, -steering_limit, steering_limit),
                intent["action"],
                dt,
            )
        route_center_gain = _clamp(
            0.55 - 0.005 * max(0.0, current_speed_kmh - 20.0),
            0.30,
            0.55,
        )
        junction_route_active = (
            trusted_route_active
            and bool(getattr(waypoint, "is_junction", False))
            and not route_turn_active
            and abs(lateral_error_m) > 1.0
        )
        centerline_correction = (
            0.0
            if (
                lane_change_active
                or junction_route_active
                or (route_turn_active and not trusted_route_active)
            )
            else (
                -0.24 * _clamp(lateral_error_m, -2.5, 2.5)
                if lane_change_settling
                else -0.07 * _clamp(lateral_error_m, -2.5, 2.5)
                if route_curve_active and not route_turn_active
                # A shallow arterial bend can keep a long look-ahead point
                # on the inside of the curve while the ego drifts toward the
                # outer edge. Increase centre recovery only after a tangible
                # offset, preserving the small-error deadband above.
                else -(
                    (
                        route_center_gain
                        if abs(lateral_error_m) < 0.45
                        else min(0.70, route_center_gain + 0.20)
                    )
                    if route_reference_active
                    else (
                        0.38 if abs(lateral_error_m) < 0.45 else 0.62
                    )
                ) * _clamp(lateral_error_m, -2.5, 2.5)
            )
        )
        base_curvature = self._route_curvature_of(
            route_reference, intent.get("target_location")
        )
        steering_limit = self._dynamic_steering_limit(
            speed_kmh=current_speed_kmh,
            heading_error_rad=angle,
            curvature=base_curvature,
            lane_width_m=float(
                getattr(waypoint, "lane_width", 3.5) or 3.5
            ),
        )
        if lane_change_settling:
            steering_limit = min(steering_limit, 0.22)
        elif lane_change_active:
            steering_limit = min(steering_limit, 0.30)
        if route_curve_active and not route_turn_active:
            reference_yaw = math.radians(float(route_reference["yaw"]))
            current_heading_error = self._angle_delta(
                reference_yaw, current_yaw
            )
            route_yaw = math.radians(float(route_yaw_deg))
            reference_to_target_m = max(
                1.0,
                math.hypot(
                    target_location.x - float(route_reference["x"]),
                    target_location.y - float(route_reference["y"]),
                ),
            )
            route_curvature = (
                self._angle_delta(route_yaw, reference_yaw)
                / reference_to_target_m
            )
            # Approximate bicycle-model feed-forward in CARLA's normalized
            # steering space, then use heading/cross-track terms only to
            # reject tracking error instead of aiming directly through the arc.
            wheelbase = self._wheelbase_m()
            max_wheel = self._max_wheel_angle_rad()
            feed_forward = math.atan(wheelbase * route_curvature) / max_wheel
            heading_command = (
                feed_forward
                + 0.90 * current_heading_error
            )
        else:
            heading_gain = (
                1.00
                if route_turn_active
                else (0.88 if route_reference_active else 1.20)
            )
            heading_command = heading_gain * angle
        if (
            trusted_route_active
            and not route_curve_active
            and heading_command * centerline_correction < 0.0
            and abs(angle) > math.radians(2.0)
        ):
            # Around ramps and junction connectors CARLA can project the ego
            # onto a nearby lane whose centre correction points away from the
            # planned route. Retain a bounded centring influence, but never
            # let that uncertain map projection reverse a clear route-heading
            # command.
            correction_limit = 0.10 * abs(heading_command)
            centerline_correction = _clamp(
                centerline_correction,
                -correction_limit,
                correction_limit,
            )
        raw_steer = heading_command + centerline_correction
        raw_steer = self._turn_trajectory_adjust(
            raw_steer,
            current_speed_kmh,
            dt,
            base_curvature,
            lateral_error_m,
            intent["action"],
        )
        raw_steer = _clamp(raw_steer, -steering_limit, steering_limit)
        return self._filter_steering(raw_steer, intent["action"], dt)

    def _planned_route_steer(
        self,
        *,
        route_reference,
        target_location,
        current_yaw,
        lateral_error_m,
        speed_kmh,
        suppress_cross_track=False,
    ):
        """Track a planner path with curvature feed-forward and Stanley error terms."""
        reference_yaw = math.radians(float(route_reference["yaw"]))
        target_yaw = math.radians(float(target_location.get(
            "yaw", route_reference["yaw"]
        )))
        path_distance_m = max(
            1.0,
            math.hypot(
                float(target_location["x"]) - float(route_reference["x"]),
                float(target_location["y"]) - float(route_reference["y"]),
            ),
        )
        curvature = self._angle_delta(target_yaw, reference_yaw) / path_distance_m
        wheelbase_m = self._wheelbase_m()
        max_wheel_angle_rad = self._max_wheel_angle_rad()
        feed_forward = math.atan(wheelbase_m * curvature)
        heading_error = self._angle_delta(reference_yaw, current_yaw)
        speed_mps = max(0.0, float(speed_kmh) / 3.6)
        cross_track = 0.0 if suppress_cross_track else math.atan2(
            -1.0 * float(lateral_error_m),
            speed_mps + 2.0,
        )
        wheel_angle = feed_forward + 0.85 * heading_error + cross_track
        normalized = wheel_angle / max_wheel_angle_rad
        self._last_lateral_debug = {
            "mode": "stanley_route",
            "curvature_per_m": round(curvature, 7),
            "feed_forward": round(feed_forward / max_wheel_angle_rad, 6),
            "heading_term": round(
                0.85 * heading_error / max_wheel_angle_rad, 6
            ),
            "cross_track_term": round(
                cross_track / max_wheel_angle_rad, 6
            ),
            "lateral_error_m": round(float(lateral_error_m), 6),
            "raw_normalized_steer": round(normalized, 6),
        }
        return normalized

    def _steering_response_scale(self):
        """Speed-adaptive steering response.

        At low speed (junction turns, crawling) the wheel can respond faster;
        at cruising speed the same command is low-passed harder so high-speed
        corrections stay smooth and do not excite the vehicle dynamics.
        """
        return _clamp(1.6 - self._get_speed_kmh() / 70.0, 0.65, 1.6)

    def _filter_steering(self, raw_steer, action, dt):
        """Apply mode-aware low-pass filtering before actuator rate limiting.

        Normal lane keeping receives the strongest filtering to reject waypoint
        quantization noise.  Turns retain a faster response, but still cannot
        create a one-frame steering spike when the route target switches at a
        junction.
        """
        base_response_per_s = {
            "turn_left": 5.5,
            "turn_right": 5.5,
            "lane_change_left": 3.4,
            "lane_change_right": 3.4,
        }.get(action, 3.0)
        response_per_s = (
            base_response_per_s * self._steering_response_scale()
        )
        alpha = 1.0 - math.exp(-response_per_s * _clamp(float(dt), 0.01, 0.10))
        self._filtered_steer += alpha * (raw_steer - self._filtered_steer)
        if abs(self._filtered_steer) < 0.008:
            return 0.0
        return self._filtered_steer

    def _target_location(self, waypoint, intent, vehicle_transform):
        requested = intent.get("target_location")
        # A scene decision may carry the route lookahead together with a lane
        # change. The route point still lies on the pre-change lane and must not
        # override the latched adjacent-lane target.
        if intent["action"] in ("lane_change_left", "lane_change_right"):
            requested = None
        # CARLA changes road/lane ids across junction connector segments, so id
        # equality cannot determine whether a planned point remains on the
        # current driving corridor. Accept a forward route point when its
        # lateral offset from the current lane centre is small. After a lane
        # change the old-route point is about one lane width away and is
        # deliberately rejected, preventing the controller from pulling back.
        if requested is not None:
            requested_location = carla.Location(requested["x"], requested["y"], requested["z"])
            if intent["action"] in ("turn_left", "turn_right"):
                return requested_location
            if bool(intent.get("route_target_trusted", False)):
                # The basic route point is planner-owned. Rejecting it after
                # a cross-track error traps the ego on a neighbouring map
                # corridor and prevents a smooth route recovery.
                return requested_location
            lane_yaw = math.radians(waypoint.transform.rotation.yaw)
            dx = requested_location.x - waypoint.transform.location.x
            dy = requested_location.y - waypoint.transform.location.y
            longitudinal_m = dx * math.cos(lane_yaw) + dy * math.sin(lane_yaw)
            lateral_m = -dx * math.sin(lane_yaw) + dy * math.cos(lane_yaw)
            lane_width = max(2.5, float(getattr(waypoint, "lane_width", 3.5)))
            if longitudinal_m > 0.0 and abs(lateral_m) <= min(2.5, 0.65 * lane_width):
                return requested_location
            if longitudinal_m > 0.0:
                # The route may remain referenced to the pre-change lane.
                # Preserve the ego's current lane, but use the planned point's
                # heading to choose the matching branch at a junction.
                requested_waypoint = self.world_map.get_waypoint(
                    requested_location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                lookahead_m = _clamp(
                    math.hypot(
                        requested_location.x - vehicle_transform.location.x,
                        requested_location.y - vehicle_transform.location.y,
                    ),
                    10.0,
                    24.0,
                )
                candidates = waypoint.next(lookahead_m)
                if requested_waypoint is not None and candidates:
                    requested_yaw = math.radians(
                        requested_waypoint.transform.rotation.yaw
                    )
                    selected = min(
                        candidates,
                        key=lambda candidate: abs(self._angle_delta(
                            math.radians(candidate.transform.rotation.yaw),
                            requested_yaw,
                        )),
                    )
                    return selected.transform.location

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
                # On a curved arterial, a distant adjacent-lane point can be
                # almost collinear with the ego heading and produce no lateral
                # motion. A 10-14 m pursuit point retains a shallow merge while
                # still finishing before the next junction.
                return self._forward_waypoint(waypoint, self._lookahead_distance(0.35, 10.0, 14.0))

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
            self._lane_change_stable_frames = 0

        if self._lane_change_target_lane_id is None:
            return waypoint
        if waypoint.lane_id == self._lane_change_target_lane_id:
            return waypoint

        candidate = self._lane_with_id(waypoint, self._lane_change_target_lane_id)
        if candidate is None:
            candidate = self._adjacent_driving_lane(waypoint, intent["action"])
        if (
            candidate is not None
            and candidate.lane_type == carla.LaneType.Driving
            and candidate.lane_id == self._lane_change_target_lane_id
        ):
            return candidate
        return waypoint

    @staticmethod
    def _lane_with_id(waypoint, lane_id):
        """Find the latched target lane even if the ego briefly overshoots it."""
        if lane_id is None:
            return None
        for getter_name in ("get_left_lane", "get_right_lane"):
            candidate = getattr(waypoint, getter_name)()
            while (
                candidate is not None
                and candidate.lane_type == carla.LaneType.Driving
                and candidate.road_id == waypoint.road_id
                and candidate.lane_id * waypoint.lane_id > 0
            ):
                if candidate.lane_id == lane_id:
                    return candidate
                candidate = getattr(candidate, getter_name)()
        return None

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
        in_target_lane = (
            self._lane_change_target_lane_id is not None
            and current_lane_id == self._lane_change_target_lane_id
        )
        if in_target_lane:
            self._lane_change_stable_frames += 1
        else:
            self._lane_change_stable_frames = 0
        return {
            "speed_kmh": self._get_speed_kmh(),
            "current_lane_id": current_lane_id,
            "target_lane_id": self._lane_change_target_lane_id,
            "lane_change_completed": (
                self._lane_change_target_lane_id is not None
                and in_target_lane
                # At the fixed 20 Hz CARLA step, ten frames provide a short
                # 0.5 s settling phase before normal lane keeping resumes.
                and self._lane_change_stable_frames >= 10
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
