"""Separate requested speed setpoints from risk-imposed upper bounds."""

import math
from .risk_behavior import cautious_permission


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def resolve_cruise_instruction_contract(parsed_intent,target_speed_kmh,explicit_cruise_kmh):
    """Keep a driver-set cruise goal across plain lane keeping, never infer one."""
    inherited=(parsed_intent in ('KEEP_LANE','PROCEED','RESUME')
        and target_speed_kmh is None and _finite_number(explicit_cruise_kmh)
        and 0 <= explicit_cruise_kmh <= 120.)
    return (('SET_SPEED',float(explicit_cruise_kmh)) if inherited
            else (parsed_intent,target_speed_kmh))


def guarded_crawl_permission(risk, speed_kmh):
    """Bind a supervisor-approved crawl to the current physical observation."""
    evidence = risk.get('risk_evidence') or {}
    radar = risk.get('physical_forward_radar') or {}
    now = evidence.get('decision_timestamp_s')
    observed = evidence.get('observed_at_s')
    until = evidence.get('valid_until_s')
    emergency = radar.get('emergency_distance_m')
    distance = radar.get('nearest_distance_m')
    clearance = (
        distance is None and radar.get('obstacle_candidate_count') == 0
    ) or (_finite_number(distance) and _finite_number(emergency) and distance > emergency + .25)
    closing = radar.get('nearest_relative_velocity_mps')
    closing_safe = closing is None or (_finite_number(closing) and closing >= -2.
        and (closing >= 0. or (_finite_number(distance) and distance / -closing >= 5.)))
    frame = radar.get('sensor_frame')
    if not (evidence.get('is_fresh') is True
            and all(_finite_number(value) for value in (now, observed, until, emergency, speed_kmh))
            and emergency > 0. and 0. < speed_kmh <= 6.
            and observed <= now <= until and 0. <= now - observed <= .5
            and isinstance(frame, int) and not isinstance(frame, bool)
            and frame == evidence.get('sensor_frame')
            and str(radar.get('schema_version', '')).startswith('physical_front_radar/')
            and clearance and closing_safe
            and 'physical_forward_radar_emergency_distance' not in (risk.get('reason_codes') or [])):
        return None
    return dict(schema_version='guarded_crawl/1.0', source='temporal_risk_supervisor',
                sensor_frame=frame, observed_at_s=observed, valid_until_s=until,
                max_speed_kmh=float(speed_kmh))


def _approved_crawl(decision, risk):
    permission = decision.get('risk_motion_permission')
    if not isinstance(permission, dict) or permission.get('source') != 'temporal_risk_supervisor':
        return False
    expected = guarded_crawl_permission(risk, permission.get('max_speed_kmh'))
    return bool(expected and permission == expected
        and decision.get('action') == 'decelerate'
        and decision.get('decision_status', 'READY') == 'READY'
        and not decision.get('blocked_reason_codes') and not decision.get('emergency'))


def enforce_active_instruction_contract(decision, parsed_intent, target_speed_kmh, ego_speed_kmh, *, risk=None):
    """Keep an active stop or explicit speed setpoint authoritative.

    ``SET_SPEED`` is an absolute execution goal, not merely an upper bound for
    the learned proposal.  The goal is only promoted to the issued setpoint
    while the already-gated decision is ready for ordinary forward motion.
    Safety deceleration, stops, blocked decisions and downstream risk/traffic/
    road caps remain authoritative. A learned speed adjustment is not itself
    a safety constraint when fresh fused observations explicitly permit motion.
    """
    result = dict(decision)
    before = dict(result)
    speed = float(ego_speed_kmh)
    if not math.isfinite(speed) or speed < 0:
        raise ValueError('Finite nonnegative ego speed required')
    cap = None if target_speed_kmh is None else float(target_speed_kmh)
    if cap is not None and (not math.isfinite(cap) or cap < 0):
        raise ValueError('Finite nonnegative instruction speed required')
    active_stop = parsed_intent in ('STOP', 'EMERGENCY_BRAKE')
    if active_stop:
        result.pop('risk_motion_permission', None)
        emergency = parsed_intent == 'EMERGENCY_BRAKE' or result.get('emergency') or result.get('action') == 'emergency_brake'
        result.update(action='emergency_brake' if emergency else 'stop',
            emergency=bool(emergency), target_speed_kmh=0.,
            longitudinal_control_mode='STOP', allow_positive_acceleration=False)
    elif cap is not None:
        requested = float(result.get('target_speed_kmh', 0.))
        if not math.isfinite(requested):
            raise ValueError('Finite decision target speed required')
        evidence=(risk or {}).get('risk_evidence') or {}
        policy=(risk or {}).get('behavior_policy') or {}
        now=evidence.get('decision_timestamp_s')
        observed=evidence.get('observed_at_s')
        until=evidence.get('valid_until_s')
        policy_until=policy.get('valid_until_s')
        fresh=(all(_finite_number(v) for v in (now,observed,until,policy_until))
            and observed<=now<=min(until,policy_until) and now-observed<=.5)
        learned_adjustment=(
            result.get('action')=='decelerate'
            and str(result.get('reason','')).startswith('vla_accepted_')
            and (risk or {}).get('frame_id')==result.get('frame_id')
            and (risk or {}).get('frame_id') is not None
            and (risk or {}).get('risk_level')=='low'
            and not (risk or {}).get('reason_codes')
            and evidence.get('is_fresh') is True
            and fresh
            and policy.get('disposition')=='CLEAR'
            and policy.get('confirmed') is True
            and policy.get('frame_id')==result.get('frame_id')
            and not policy.get('physical_stop',False)
        )
        setpoint_ready = (
            parsed_intent == 'SET_SPEED'
            and result.get('decision_status', 'READY') == 'READY'
            and (result.get('action') in ('accelerate', 'keep_lane') or learned_adjustment)
            and not result.get('emergency', False)
            and not result.get('blocked_reason_codes')
            and result.get('allow_positive_acceleration', True) is not False
        )
        result['target_speed_kmh'] = (
            cap if setpoint_ready else max(0., min(requested, cap))
        )
        if setpoint_ready and learned_adjustment:
            result['action']='keep_lane'
        if setpoint_ready:
            result['speed_tracking_mode']='EXPLICIT_CRUISE'
        if speed >= cap:
            result['allow_positive_acceleration'] = False
    if active_stop or result.get('allow_positive_acceleration') is False or (
        result.get('target_speed_kmh') != before.get('target_speed_kmh')
    ) or result.get('action')!=before.get('action'):
        for key in ('longitudinal_sequence_schema', 'target_acceleration_mps2', 'sequence_valid_until_s'):
            result.pop(key, None)
    changed = result != before
    if changed:
        result['reason'] = (
            'active_instruction_stop' if active_stop else
            'active_instruction_speed_setpoint'
            if parsed_intent == 'SET_SPEED' else
            'active_instruction_speed_bound'
        )
    return result, dict(schema_version='active_instruction_contract/1.0',
        parsed_intent=parsed_intent, active_stop=active_stop, speed_bound_kmh=cap,
        speed_contract_mode=(
            'STOP' if active_stop else
            'ABSOLUTE_SETPOINT' if cap is not None and parsed_intent == 'SET_SPEED' else
            'UPPER_BOUND' if cap is not None else 'NONE'
        ),
        input_action=before.get('action'), input_target_speed_kmh=before.get('target_speed_kmh'),
        issued_action=result.get('action'), issued_target_speed_kmh=result.get('target_speed_kmh'),
        changed=changed, policy_safety_credit=False)


def enforce_longitudinal_contract(decision, risk, ego_speed_kmh):
    result=dict(decision)
    speed=float(ego_speed_kmh)
    requested=float(result.get('target_speed_kmh',0.))
    if not math.isfinite(speed) or speed<0 or not math.isfinite(requested):
        raise ValueError('Finite nonnegative ego speed and finite target required')
    reasons=result.get('blocked_reason_codes') or []
    risk_cap=(risk.get('recommended_action') in ('decelerate','emergency_brake','stop')
        and risk.get('risk_level') in ('medium','high')) or any(
            code in reasons for code in ('vla_overridden_by_deceleration_risk','vla_overridden_by_emergency_risk'))
    stop=result.get('action') in ('stop','emergency_brake') or bool(result.get('emergency'))
    from .task_feasibility import task_permission
    lane_motion=(result.get('action') in {'lane_change_left','lane_change_right'}
        and task_permission(risk,'CHANGE_LANE',frame_id=result.get('frame_id')))
    cautious=cautious_permission(risk,frame_id=result.get('frame_id'))
    cautious=bool(cautious and not stop and result.get('decision_status','READY')=='READY'
        and not reasons and (result.get('action') in ('accelerate','decelerate','keep_lane','turn_left','turn_right') or lane_motion)
        and result.get('risk_motion_permission')==cautious)
    crawl = not stop and _approved_crawl(result, risk)
    if not crawl and not cautious:
        result.pop('risk_motion_permission', None)
    mode='STOP' if stop else 'CAUTIOUS_SPEED_CAP' if cautious else 'GUARDED_CRAWL' if crawl else 'RISK_SPEED_CAP' if risk_cap else 'ABSOLUTE_SPEED'
    target=(0. if stop else max(0., min(requested, result['risk_motion_permission']['max_speed_kmh']))
            if crawl or cautious else max(0.,min(requested,speed)) if risk_cap else max(0.,requested))
    changed=abs(target-requested)>1e-6
    result.update(target_speed_kmh=target,longitudinal_control_mode=mode,
        allow_positive_acceleration=not stop and (crawl or cautious or not risk_cap)
        and (not cautious or result['risk_motion_permission'].get('allow_positive_acceleration',True))
        and result.get('allow_positive_acceleration',True))
    if risk_cap or stop or crawl or cautious:
        for key in ('longitudinal_sequence_schema','target_acceleration_mps2','sequence_valid_until_s'):
            result.pop(key,None)
    if risk_cap and not cautious and result.get('action') not in ('stop','emergency_brake','lane_change_left','lane_change_right'):
        result['action']='decelerate'
    return result,dict(schema_version='longitudinal_control_contract/1.0',mode=mode,
        input_action=decision.get('action'),input_target_speed_kmh=requested,
        issued_action=result.get('action'),issued_target_speed_kmh=target,
        reference_ego_speed_kmh=speed,speed_clamped=changed,risk_constraint_active=bool(risk_cap or cautious),
        cautious_motion_authorized=bool(cautious),
        guarded_crawl_authorized=bool(crawl), policy_safety_credit=False)
