"""Defer an unstarted navigation merge without authorizing blind lane entry."""
from copy import deepcopy
import math


def rear_departure_window(current, target, *, horizon_m=10.0):
    result = dict(allowed=False, source='static_rear_lane_departure_topology',
                  horizon_m=horizon_m, reason='rear_departure_geometry_unavailable')
    if (current is None or target is None or isinstance(horizon_m, bool)
            or not isinstance(horizon_m, (int, float))
            or not math.isfinite(horizon_m) or horizon_m < 1.):
        return result
    try:
        for distance in range(int(math.ceil(horizon_m)) + 1):
            for initial in (current, target):
                choices = initial.previous(float(distance)) if distance else [initial]
                if len(choices) != 1:
                    return dict(result, reason='rear_departure_branch_or_endpoint', distance_m=distance)
                waypoint = choices[0]
                if waypoint.is_junction:
                    return dict(result, reason='rear_departure_junction_horizon', distance_m=distance)
                if int(waypoint.lane_id) * int(initial.lane_id) <= 0:
                    return dict(result, reason='rear_departure_direction_discontinuity', distance_m=distance)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return result
    return dict(result, allowed=True, reason='nonjunction_unique_rear_lane_horizon')


def defer_navigation_merge(decision, risk, direction, *, navigation_recovery=False,
                           maneuver_started=False, corridor_evidence=None):
    """Move only in a freshly verified current lane while a rear window forms."""
    departure = (risk.get('target_lane_contract') or {}).get('departure_window') or {}
    window = departure.get('rear_window') or {}
    topology_wait = (window.get('allowed') is False and window.get('reason') in {
        'rear_departure_branch_or_endpoint', 'rear_departure_junction_horizon',
        'rear_departure_direction_discontinuity'})
    topology_wait = topology_wait or (departure.get('allowed') is False and departure.get('reason') in {
        'departure_branch_or_endpoint', 'departure_junction_horizon',
        'departure_direction_discontinuity'})
    if (not navigation_recovery or maneuver_started or direction not in {'left', 'right'}
            or decision.get('reason') != 'requested_lane_wait_for_sensor_clearance'
            or decision.get('emergency') or decision.get('action') == 'emergency_brake'
            or not topology_wait):
        return deepcopy(decision), None
    policy = risk.get('behavior_policy') or {}
    cap = policy.get('max_speed_kmh')
    if cap is None:
        cap = policy.get('speed_cap_kmh')
    if (isinstance(cap, bool) or not isinstance(cap, (int, float))
            or not math.isfinite(cap) or cap <= 0.):
        return deepcopy(decision), None
    result = deepcopy(decision)
    result.update(decision_status='READY', action='keep_lane', target_lane=None,
        target_location=None, target_speed_kmh=min(10., cap), emergency=False,
        allow_positive_acceleration=True, longitudinal_control_mode='CAUTIOUS_SPEED_CAP',
        reason='navigation_merge_deferred_current_lane', deferred_lane_change=True,
        deferred_current_lane_evidence=deepcopy(corridor_evidence or {}))
    for key in ('longitudinal_sequence_schema', 'target_acceleration_mps2', 'sequence_valid_until_s'):
        result.pop(key, None)
    from .lane_risk_contract import enforce_requested_lane
    checked, override = enforce_requested_lane(result, risk, direction)
    if override != 'deferred_current_lane_only':
        return deepcopy(decision), None
    return checked, 'navigation_merge_deferred_current_lane'
