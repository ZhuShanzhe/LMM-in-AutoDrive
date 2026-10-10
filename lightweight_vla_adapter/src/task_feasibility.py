"""Task authority is a predicate; learned risk grades remain speed advisories."""
from copy import deepcopy


def predicate(task, allowed, *, known=True, reasons=(), evidence=None):
    return dict(schema_version='task_feasibility/1.0', task=task,
                can_execute=bool(known and allowed),
                status='ALLOWED' if known and allowed else 'BLOCKED' if known else 'UNKNOWN',
                reason_codes=list(reasons), evidence=evidence or {}, policy_truth_access=False)


def requested_lane_for_task(document, state, requested_direction):
    if requested_direction in {'left','right'}:
        return requested_direction
    if not document:return None
    index=(state or {}).get('active_step_index',0)
    steps=document.get('intent',{}).get('steps',[])
    if index is None or index>=len(steps):return None
    step=steps[index];params=step.get('parameters') or {}
    condition=params.get('condition')
    if condition in {'LEFT_LANE_SAFE','RIGHT_LANE_SAFE'}:
        return condition.split('_')[0].lower()
    if 'TARGET_LANE_SAFE' in (step.get('preconditions') or []):
        direction=str(params.get('direction','')).lower()
        return direction if direction in {'left','right'} else None
    return None


def task_permission(risk, action, *, frame_id=None):
    if frame_id is not None and risk.get('frame_id') != frame_id:
        return False
    policy = risk.get('behavior_policy') or {}
    if (policy.get('physical_stop') or policy.get('absolute_stop_active')
            or risk.get('recommended_action') == 'emergency_brake'):
        return False
    if action == 'CHANGE_LANE':
        contract = risk.get('target_lane_contract') or {}
        return contract.get('can_execute') is True
    contract = (risk.get('task_feasibility') or {}).get(action) or {}
    return contract.get('can_execute') is True


def attach_motion_predicates(risk):
    from .risk_behavior import cautious_permission
    result = deepcopy(risk)
    policy = risk.get('behavior_policy') or {}
    fresh = (risk.get('risk_evidence') or {}).get('is_fresh') is True
    known = fresh and policy.get('disposition') in {'CLEAR', 'CAUTIOUS_PROCEED', 'MUST_STOP'}
    allowed = known and policy.get('disposition') in {'CLEAR', 'CAUTIOUS_PROCEED'}
    allowed = allowed and not policy.get('physical_stop') and not policy.get('absolute_stop_active')
    allowed = allowed and (risk.get('risk_level') == 'low'
                          or cautious_permission(risk, frame_id=risk.get('frame_id')) is not None)
    tasks = result.setdefault('task_feasibility', {})
    for task in ('PATH_CLEAR', 'KEEP_LANE', 'FOLLOW', 'TURN', 'PROCEED', 'SET_SPEED', 'ADJUST_SPEED'):
        tasks[task] = predicate(task, allowed, known=known,
            reasons=() if allowed else ('motion_clearance_not_verified',),
            evidence=dict(frame_id=risk.get('frame_id'), disposition=policy.get('disposition'),
                semantic_scope='physical_motion_clearance_only;step_target_trigger_and_map_gates_remain_required'))
    result['learned_risk_advisory'] = dict(level=risk.get('risk_level'),
        probabilities=deepcopy(risk.get('probabilities')), affects='speed_not_task_authority')
    return result
