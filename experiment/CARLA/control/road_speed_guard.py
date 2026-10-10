"""Short-horizon actuator ceiling from measured ego speed and acceleration."""
import math


def apply_road_speed_guard(control, *, speed_kmh, speed_limit_kmh,
                          acceleration_mps2, dt_s):
    values = (speed_kmh, speed_limit_kmh, acceleration_mps2, dt_s)
    if (any(isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in values)
            or speed_kmh < 0 or speed_limit_kmh <= 0 or not 0 < dt_s <= .25):
        return dict(source='measured_ego_road_speed_guard/1.0', status='UNAVAILABLE', changed=False)
    horizon = min(.25, .1 + dt_s)
    predicted = speed_kmh + max(0., acceleration_mps2) * 3.6 * horizon
    before = (control.throttle, control.brake)
    if predicted >= speed_limit_kmh - .2:
        control.throttle = 0.
    if predicted > speed_limit_kmh:
        control.brake = max(control.brake, min(.8, max(.1, .2 * (predicted - speed_limit_kmh))))
    return dict(source='measured_ego_road_speed_guard/1.0', status='MEASURED',
        speed_kmh=speed_kmh, speed_limit_kmh=speed_limit_kmh,
        acceleration_mps2=acceleration_mps2, horizon_s=horizon,
        predicted_speed_kmh=predicted, changed=before != (control.throttle, control.brake))
