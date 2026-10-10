"""Keep transient manoeuvre speeds separate from a persistent cruise goal."""

import math


def qualitative_instruction_bound(document):
    policy = ((document or {}).get('parse_result') or {}).get('speed_target_policy') or {}
    if (policy.get('type') != 'QUALITATIVE_CRUISE_BOUND'
            or policy.get('user_supplied_numeric_target') is not False):
        return None
    target = policy.get('target_kmh')
    if (isinstance(target, bool) or not isinstance(target, (int, float))
            or not math.isfinite(target) or not 0. <= target <= 120.):
        raise ValueError('Invalid qualitative instruction speed bound')
    return float(target)


def active_speed_contract(parsed_intent, target_speed_kmh, qualitative_bound_kmh):
    if qualitative_bound_kmh is None or parsed_intent in {'STOP', 'EMERGENCY_BRAKE'}:
        return parsed_intent, target_speed_kmh
    cap = qualitative_bound_kmh if target_speed_kmh is None else min(
        target_speed_kmh, qualitative_bound_kmh)
    return 'DECELERATE', cap


def overtake_nominal_goal(decision, step, cruise_speed_kmh):
    """Supply a nominal goal before safety gates, never an issued override."""
    parameters=(step or {}).get('parameters') or {}
    if ((step or {}).get('action') != 'OVERTAKE'
            or parameters.get('direction') is not None
            or any(parameters.get(key) is not None for key in ('target_speed_mps', 'target_speed_kmh'))
            or decision.get('source_step_id') != (step or {}).get('step_id')
            or decision.get('source_step_action') != 'OVERTAKE'
            or decision.get('action') != 'accelerate'
            or decision.get('decision_status') != 'READY'
            or decision.get('emergency') or decision.get('blocked_reason_codes')):
        return decision
    if (isinstance(cruise_speed_kmh,bool) or not isinstance(cruise_speed_kmh,(int,float))
            or not math.isfinite(cruise_speed_kmh) or not 0. < cruise_speed_kmh <= 100.):
        return decision
    return dict(decision,target_speed_kmh=float(cruise_speed_kmh),
                reason='active_overtake_nominal_cruise_goal')


def cruise_goal_update(parsed_intent, target_speed_kmh, *, planned_action=None, qualitative_bound=False):
    if qualitative_bound:return None
    if planned_action is not None:
        return target_speed_kmh if planned_action == 'SET_SPEED' else None
    return target_speed_kmh if parsed_intent in {'SET_SPEED', 'DECELERATE', 'RESUME'} else None
