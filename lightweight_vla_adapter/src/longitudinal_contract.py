"""Separate requested speed setpoints from risk-imposed upper bounds."""

import math


def enforce_active_instruction_contract(decision, parsed_intent, target_speed_kmh, ego_speed_kmh):
    """Keep an active stop or speed bound authoritative over recovery policies."""
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
        emergency = parsed_intent == 'EMERGENCY_BRAKE' or result.get('emergency') or result.get('action') == 'emergency_brake'
        result.update(action='emergency_brake' if emergency else 'stop',
            emergency=bool(emergency), target_speed_kmh=0.,
            longitudinal_control_mode='STOP', allow_positive_acceleration=False)
    elif cap is not None:
        requested = float(result.get('target_speed_kmh', 0.))
        if not math.isfinite(requested):
            raise ValueError('Finite decision target speed required')
        result['target_speed_kmh'] = max(0., min(requested, cap))
        if speed >= cap:
            result['allow_positive_acceleration'] = False
    if active_stop or result.get('allow_positive_acceleration') is False or (
        result.get('target_speed_kmh') != before.get('target_speed_kmh')
    ):
        for key in ('longitudinal_sequence_schema', 'target_acceleration_mps2', 'sequence_valid_until_s'):
            result.pop(key, None)
    changed = result != before
    if changed:
        result['reason'] = 'active_instruction_stop' if active_stop else 'active_instruction_speed_bound'
    return result, dict(schema_version='active_instruction_contract/1.0',
        parsed_intent=parsed_intent, active_stop=active_stop, speed_bound_kmh=cap,
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
    mode='STOP' if stop else 'RISK_SPEED_CAP' if risk_cap else 'ABSOLUTE_SPEED'
    target=0. if stop else max(0.,min(requested,speed)) if risk_cap else max(0.,requested)
    changed=abs(target-requested)>1e-6
    result.update(target_speed_kmh=target,longitudinal_control_mode=mode,
        allow_positive_acceleration=not (risk_cap or stop) and result.get('allow_positive_acceleration',True))
    if risk_cap or stop:
        for key in ('longitudinal_sequence_schema','target_acceleration_mps2','sequence_valid_until_s'):
            result.pop(key,None)
    if risk_cap and result.get('action') not in ('stop','emergency_brake','lane_change_left','lane_change_right'):
        result['action']='decelerate'
    return result,dict(schema_version='longitudinal_control_contract/1.0',mode=mode,
        input_action=decision.get('action'),input_target_speed_kmh=requested,
        issued_action=result.get('action'),issued_target_speed_kmh=target,
        reference_ego_speed_kmh=speed,speed_clamped=changed,risk_constraint_active=bool(risk_cap),
        policy_safety_credit=False)
