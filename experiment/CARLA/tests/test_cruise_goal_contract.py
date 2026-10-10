import pytest

from control.cruise_goal_contract import (cruise_goal_update,
    qualitative_instruction_bound, active_speed_contract, overtake_nominal_goal)


@pytest.mark.parametrize('target', [0., 5., 20.])
def test_temporary_decrease_cannot_destroy_the_following_cruise_goal(target):
    assert cruise_goal_update('DECELERATE', target, planned_action='ADJUST_SPEED') is None


@pytest.mark.parametrize('action', ['CHANGE_LANE', 'OVERTAKE', 'YIELD', 'STOP', 'KEEP_LANE'])
def test_non_speed_steps_do_not_replace_persistent_speed(action):
    assert cruise_goal_update('RESUME', 5., planned_action=action) is None


def test_explicit_absolute_speed_is_retained():
    assert cruise_goal_update('DECELERATE', 20., planned_action='SET_SPEED') == 20.
    assert cruise_goal_update('SET_SPEED', 0., planned_action='SET_SPEED') == 0.


def test_standalone_speed_contract_is_unchanged():
    assert cruise_goal_update('DECELERATE', 20.) == 20.
    assert cruise_goal_update('KEEP_LANE', 20.) is None


@pytest.mark.parametrize('action',['SET_SPEED','ADJUST_SPEED',None])
def test_qualitative_adapter_setpoint_is_not_a_new_persistent_cruise_goal(action):
    assert cruise_goal_update('DECELERATE',18.,planned_action=action,qualitative_bound=True) is None


def qualitative(target):
    return {'parse_result': {'speed_target_policy': {
        'type': 'QUALITATIVE_CRUISE_BOUND', 'target_kmh': target,
        'user_supplied_numeric_target': False}}}


def test_completed_plan_keeps_active_bound_without_overwriting_cruise():
    base = 60.
    bound = qualitative_instruction_bound(qualitative(45.))
    assert active_speed_contract('SET_SPEED', base, bound) == ('DECELERATE', 45.)
    assert active_speed_contract('KEEP_LANE', None, bound) == ('DECELERATE', 45.)
    assert cruise_goal_update('SET_SPEED', 45., planned_action='SET_SPEED', qualitative_bound=True) is None
    assert base == 60.
    assert active_speed_contract('SET_SPEED', base, qualitative_instruction_bound(None)) == ('SET_SPEED', 60.)


@pytest.mark.parametrize('intent', ['STOP', 'EMERGENCY_BRAKE'])
def test_temporary_speed_bound_never_replaces_stop(intent):
    assert active_speed_contract(intent, 0., 45.) == (intent, 0.)


def test_existing_lower_speed_constraint_remains_authoritative():
    assert active_speed_contract('SET_SPEED', 20., 45.) == ('DECELERATE', 20.)


@pytest.mark.parametrize('target', [True, float('nan'), float('inf'), -1., 121., '45'])
def test_invalid_qualitative_bound_fails_closed(target):
    with pytest.raises(ValueError):
        qualitative_instruction_bound(qualitative(target))


def test_explicit_numeric_policy_is_not_reinterpreted_as_temporary():
    document=qualitative(45.)
    document['parse_result']['speed_target_policy']['user_supplied_numeric_target']=True
    assert qualitative_instruction_bound(document) is None


def passing_step():
    return {'step_id':'passing','action':'OVERTAKE','parameters':{}}


def passing_decision(**changes):
    return dict(dict(source_step_id='passing',source_step_action='OVERTAKE',
        action='accelerate',decision_status='READY',target_speed_kmh=0.,
        emergency=False,blocked_reason_codes=[]),**changes)


def test_stopped_overtake_receives_nominal_speed_before_the_safety_gates():
    original=passing_decision()
    result=overtake_nominal_goal(original,passing_step(),30.)
    assert result['target_speed_kmh']==30.
    assert original['target_speed_kmh']==0.


@pytest.mark.parametrize('changes', [dict(action='stop'),dict(action='emergency_brake'),
    dict(decision_status='BLOCKED'),dict(emergency=True),dict(blocked_reason_codes=['target_missing']),
    dict(source_step_id='other'),dict(source_step_action='ADJUST_SPEED')])
def test_overtake_nominal_goal_cannot_replace_a_block_or_another_step(changes):
    decision=passing_decision(**changes)
    assert overtake_nominal_goal(decision,passing_step(),30.) is decision


@pytest.mark.parametrize('parameters', [dict(direction='LEFT'),dict(target_speed_mps=0.),
    dict(target_speed_kmh=20.)])
def test_overtake_respects_directional_and_explicit_speed_parameters(parameters):
    step=passing_step();step['parameters']=parameters
    decision=passing_decision()
    assert overtake_nominal_goal(decision,step,30.) is decision


@pytest.mark.parametrize('cruise', [0.,-1.,float('nan'),True,101.])
def test_overtake_does_not_invent_missing_cruise_goals(cruise):
    decision=passing_decision()
    assert overtake_nominal_goal(decision,passing_step(),cruise) is decision
