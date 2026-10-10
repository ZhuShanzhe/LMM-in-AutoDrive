"""Explicit engineering completion, distinct from exact-speed acceptance."""
import math


GEOMETRIC_CAPS={'road_speed_limit','route_speed_cap','rainy_night_speed_cap',
    'curvature_speed_cap','junction_speed_cap','trajectory_stability_speed_cap'}


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def safety_adjusted_speed_evidence(execution, *, frame_id, timestamp_s,
        requested_kmh, actual_kmh, tolerance_kmh=2., acceleration_mps2=None):
    """Accept tracking or measured braking under a verified lower speed bound.

    A risk class alone, a stop, missing geometry, or a stale permission cannot
    satisfy this criterion. The requested cruise setpoint is not overwritten.
    """
    if not (isinstance(execution,dict) and execution.get('observation_frame_id')==frame_id
            and all(finite(v) for v in (timestamp_s,requested_kmh,actual_kmh,tolerance_kmh))
            and 5.<=actual_kmh<=requested_kmh+tolerance_kmh and requested_kmh>=5.
            and execution.get('action') in {'accelerate','decelerate','keep_lane','turn_left','turn_right'}):
        return None
    pid=execution.get('pid') or {}
    if not (all(finite(pid.get(k)) for k in ('lateral_error_m','heading_error_deg'))
            and abs(pid['lateral_error_m'])<=.35 and abs(pid['heading_error_deg'])<=5.
            and not pid.get('emergency_latched')):
        return None
    effective=execution.get('effective_target_speed_kmh')
    codes=set(execution.get('speed_constraint_codes') or [])
    permission=execution.get('risk_motion_permission')
    sensor=False
    if isinstance(permission,dict):
        observed=permission.get('observed_at_s');until=permission.get('valid_until_s')
        bound=permission.get('max_speed_kmh')
        sensor=(permission.get('schema_version')=='cautious_motion/2.0'
            and permission.get('source')=='sensor_risk_behavior_policy'
            and all(finite(v)for v in(observed,until,bound))
            and 0<=timestamp_s-observed<=.25 and timestamp_s<=until and 5.<=bound<=60.)
        if sensor:
            effective=min(effective,bound)if finite(effective)else bound
            codes.add('sensor_verified_cautious_speed_cap')
    allowed=GEOMETRIC_CAPS|{'sensor_verified_cautious_speed_cap'}
    if not (codes and codes<=allowed and finite(effective) and 5.<=effective<requested_kmh
            and ('sensor_verified_cautious_speed_cap'not in codes or sensor)):
        return None
    tracking=abs(actual_kmh-effective)<=tolerance_kmh
    braking=(actual_kmh>effective and finite(acceleration_mps2) and acceleration_mps2<=-.05)
    if not (tracking or braking):return None
    return dict(mode='SAFETY_ADJUSTED',requested_target_speed_kmh=requested_kmh,
        effective_target_speed_kmh=effective,actual_speed_kmh=actual_kmh,
        constraint_codes=sorted(codes),response='TRACKING_BOUND'if tracking else 'BRAKING_TO_BOUND')
