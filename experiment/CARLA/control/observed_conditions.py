"""Fresh sensor predicates and controller feedback, without scenario truth."""
import math
from collections.abc import Mapping

from lightweight_vla_adapter.src.risk_behavior import cautious_permission, finite


def read_execution_feedback(controller):
    for name in ('execution_state', 'get_execution_state'):
        reader = getattr(controller, name, None)
        if callable(reader):
            value = reader()
            if isinstance(value, Mapping):
                return dict(value)
    return {}


def observed_condition_readiness(document, risk, *, frame_id, timestamp_s):
    policy = risk.get('behavior_policy') or {}
    lidar = policy.get('lidar') or {}
    radar = risk.get('physical_forward_radar') or {}
    evidence = risk.get('risk_evidence') or {}
    observed_at = lidar.get('observed_at_s')
    age = timestamp_s-observed_at if finite(observed_at) else math.inf
    current_required = policy.get('current_speed_required_clearance_m')
    semantic_required = max(10., current_required) if finite(current_required) else math.inf
    fresh = (risk.get('frame_id') == frame_id and evidence.get('is_fresh') is True
        and 0 <= age <= .25 and lidar.get('sensor_frame') == radar.get('sensor_frame')
        == evidence.get('sensor_frame') and lidar.get('valid') is True)
    allowed = (risk.get('risk_level') == 'low' or cautious_permission(risk, frame_id=frame_id) is not None)
    from lightweight_vla_adapter.src.task_feasibility import task_permission
    allowed=allowed or task_permission(risk,'PATH_CLEAR',frame_id=frame_id)
    nearest = radar.get('nearest_distance_m')
    result = {}
    for step in document['intent']['steps']:
        condition = (step.get('parameters') or {}).get('condition')
        if step.get('action') not in {'WAIT', 'CHECK', 'CONFIRM', 'KEEP_SAFE_DISTANCE'}:
            continue
        # Braking-distance clearance cannot prove that a named person left
        # the lane or a whole crosswalk is empty. Those predicates must come
        # from the semantic observer through the same step-bound contract.
        unbound_distance=(step.get('action')=='KEEP_SAFE_DISTANCE'
            and not any(step.get(key) for key in ('target','target_ref'))
            and not any((step.get('parameters') or {}).get(key) for key in ('target','target_ref','condition')))
        if unbound_distance:condition='PATH_CLEAR'
        bounded_motion=condition=='CURRENT_LANE_MOTION'
        required=(current_required if finite(current_required) and current_required>=.5 else math.inf) if bounded_motion else semantic_required
        supported = condition in {'PATH_CLEAR','CURRENT_LANE_MOTION'}
        clear = (fresh and allowed and not policy.get('physical_stop') and not policy.get('absolute_stop_active')
            and policy.get('disposition') in {'CLEAR', 'CAUTIOUS_PROCEED'}
            and lidar.get('clear_distance_m', 0.) >= required
            and (radar.get('obstacle_candidate_count') == 0 and nearest is None
                 or finite(nearest) and nearest > required))
        satisfied = bool(supported and clear)
        if condition in {'LEFT_LANE_SAFE','RIGHT_LANE_SAFE'}:
            contract=risk.get('target_lane_contract') or {}
            direction=condition.split('_')[0].lower()
            supported=True
            satisfied=bool(fresh and contract.get('direction')==direction
                and contract.get('frame')==lidar.get('sensor_frame')
                and contract.get('can_execute') is True and not policy.get('physical_stop'))
        result[step['step_id']] = dict(request_id=document['request_id'], step_id=step['step_id'],
            frame_id=frame_id, condition=condition, satisfied=satisfied, valid=bool(fresh and supported),
            can_proceed=satisfied,
            observed_at_s=observed_at, valid_until_s=observed_at+.25 if finite(observed_at) else None,
            source='fresh_target_lane_contract' if condition in {'LEFT_LANE_SAFE','RIGHT_LANE_SAFE'}
                else 'fresh_lidar_radar_driving_corridor', sensor_frame=lidar.get('sensor_frame'),
            status='CLEAR' if satisfied else 'UNKNOWN',
            reason='measured_corridor_clear' if satisfied else 'identity_evidence_required' if not supported
            else 'corridor_clearance_not_confirmed', required_clearance_m=required if finite(required) else None,
            semantic_scope='bounded_current_lane_motion_not_semantic_path_clear' if bounded_motion
                else 'requested_adjacent_lane' if condition in {'LEFT_LANE_SAFE','RIGHT_LANE_SAFE'}
                else 'measured_driving_corridor_only')
    return result
