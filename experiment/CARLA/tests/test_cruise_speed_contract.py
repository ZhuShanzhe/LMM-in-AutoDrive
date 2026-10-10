from types import SimpleNamespace
import pytest
from control.generic_route_pid import GenericRoutePID


def controller(limit):
    value=object.__new__(GenericRoutePID)
    value._default_speed_kmh=45.
    value._requested_cruise_speed_kmh=45.
    value.ego=SimpleNamespace(get_speed_limit=lambda:limit)
    return value


def test_requested_speed_can_exceed_default_but_not_road_limit():
    value=controller(60.)
    assert value.set_cruise_target_kmh(50.)==50.
    value.set_high_level_decision(dict(action='accelerate',target_speed_kmh=50.))
    assert value._target_speed==50.
    value.ego.get_speed_limit=lambda:30.
    value.set_high_level_decision(dict(action='accelerate',target_speed_kmh=50.))
    assert value._target_speed==30.
    value.ego.get_speed_limit=lambda:60.
    assert value.set_cruise_target_kmh(None)==50.


def test_cruise_goal_does_not_replace_model_stop_or_lower_speed():
    value=controller(60.)
    value.set_cruise_target_kmh(50.)
    value.set_high_level_decision(dict(action='decelerate',target_speed_kmh=12.))
    assert value._target_speed==12.
    value.set_high_level_decision(dict(action='stop',target_speed_kmh=50.))
    assert value._target_speed==0.


def test_road_limited_setpoint_is_reported_as_an_explicit_constraint():
    value=controller(30.)
    value._pid=SimpleNamespace(get_execution_state=lambda: {
        'effective_target_speed_kmh':30.,'speed_target_status':'CONSTRAINED',
        'speed_constraint_codes':['road_speed_limit']})
    value._current_speed_kmh=lambda:30.
    value.progress_m=lambda:1.
    value.set_high_level_decision(dict(action='accelerate',target_speed_kmh=45.,source_step_id='speed'))
    state=value.execution_state()
    assert state['requested_target_speed_kmh']==45.
    assert state['effective_target_speed_kmh']==30.
    assert state['speed_target_status']=='CONSTRAINED'
    assert state['speed_constraint_codes']==['road_speed_limit']


@pytest.mark.parametrize('limit',[0.,float('nan'),float('inf')])
def test_unknown_road_limit_retains_conservative_default(limit):
    assert controller(limit).set_cruise_target_kmh(80.)==45.


@pytest.mark.parametrize('target',[-1,121,float('nan'),True])
def test_invalid_goal_rejected(target):
    with pytest.raises(ValueError):controller(60.).set_cruise_target_kmh(target)


def test_environment_constraint_keeps_goal_separate_from_temporary_stop():
    from lightweight_vla_adapter.src.environment_speed_policy import environment_speed_bound, apply_environment_speed_bound
    value=controller(30.)
    value._pid=SimpleNamespace(get_execution_state=lambda: {
        'effective_target_speed_kmh':0.,'speed_target_status':'TRACKING'})
    value._current_speed_kmh=lambda:0.
    value.progress_m=lambda:1.
    bound=environment_speed_bound(30.,45.,precipitation=0.,wetness=0.,sun_altitude=30.)
    decision,_=apply_environment_speed_bound(dict(action='stop',target_speed_kmh=0.,source_step_id='s1'),bound,0.,45.)
    value.set_high_level_decision(decision)
    state=value.execution_state()
    assert state['target_speed_kmh']==0.
    assert state['requested_target_speed_kmh']==45.
    assert state['effective_target_speed_kmh']==30.
    assert state['speed_target_status']=='CONSTRAINED'
    assert 'road_speed_limit' in state['speed_constraint_codes']
    value.set_high_level_decision(dict(action='keep_lane',target_speed_kmh=20.,source_step_id='s2'))
    assert value.execution_state()['speed_target_status']=='TRACKING'


def test_only_explicit_cruise_goal_is_inherited_and_survives_road_limit():
    value=controller(20.)
    assert value.explicit_cruise_target_kmh() is None
    assert value.set_cruise_target_kmh(30.)==20.
    assert value.explicit_cruise_target_kmh()==30.
    value.set_cruise_target_kmh(None)
    assert value.explicit_cruise_target_kmh()==30.
    value.set_cruise_target_kmh(15.)
    assert value.explicit_cruise_target_kmh()==15.


def test_stop_or_cancel_can_discard_previous_explicit_cruise_goal():
    value=controller(60.)
    value.set_cruise_target_kmh(30.)
    value.clear_explicit_cruise_target()
    assert value.explicit_cruise_target_kmh() is None
    assert value.set_cruise_target_kmh(None)==45.


@pytest.mark.parametrize('intent',['KEEP_LANE','PROCEED','RESUME'])
def test_plain_driving_inherits_driver_set_goal(intent):
    from lightweight_vla_adapter.src.longitudinal_contract import resolve_cruise_instruction_contract
    assert resolve_cruise_instruction_contract(intent,None,30.)==('SET_SPEED',30.)


@pytest.mark.parametrize('intent',['FOLLOW','TURN','STOP','EMERGENCY_BRAKE','ADJUST_SPEED','DECELERATE'])
def test_cruise_does_not_replace_other_instruction_types(intent):
    from lightweight_vla_adapter.src.longitudinal_contract import resolve_cruise_instruction_contract
    assert resolve_cruise_instruction_contract(intent,None,30.)==(intent,None)


@pytest.mark.parametrize('goal',[None,True,-1.,121.,float('nan'),float('inf')])
def test_invalid_or_absent_goal_does_not_create_setpoint(goal):
    from lightweight_vla_adapter.src.longitudinal_contract import resolve_cruise_instruction_contract
    assert resolve_cruise_instruction_contract('PROCEED',None,goal)==('PROCEED',None)


def test_new_explicit_target_has_precedence_over_previous_cruise():
    from lightweight_vla_adapter.src.longitudinal_contract import resolve_cruise_instruction_contract
    assert resolve_cruise_instruction_contract('PROCEED',15.,30.)==('PROCEED',15.)


@pytest.mark.parametrize('angle',[0.,5.,10.])
def test_straight_junction_does_not_force_crawl(angle):
    assert GenericRoutePID._junction_speed_ceiling_kmh(angle,10.)==30.


@pytest.mark.parametrize('angle',[45.,60.,90.,180.])
def test_sharp_turn_retains_previous_junction_ceiling(angle):
    old=max(7.,min(14.,11.-.05*angle+20.*.05))
    assert GenericRoutePID._junction_speed_ceiling_kmh(angle,20.)==old


@pytest.mark.parametrize('angle',[float('nan'),float('inf'),-1.])
def test_unknown_junction_geometry_retains_conservative_ceiling(angle):
    assert GenericRoutePID._junction_speed_ceiling_kmh(angle,20.)==12.


def test_junction_ceiling_is_monotonic_and_bounded():
    caps=[GenericRoutePID._junction_speed_ceiling_kmh(angle,20.) for angle in range(181)]
    assert all(7.<=v<=30. for v in caps)
    assert all(a>=b for a,b in zip(caps,caps[1:]))


def ready_speed_proposal():
    return dict(frame_id='f1',decision_status='READY',action='decelerate',
        target_speed_kmh=12.,reason='vla_accepted_model',blocked_reason_codes=[],
        allow_positive_acceleration=True,target_acceleration_mps2=-2.,
        longitudinal_sequence_schema='longitudinal_sequence/1.0'),dict(
        frame_id='f1',risk_level='low',reason_codes=[],
        risk_evidence=dict(is_fresh=True,decision_timestamp_s=10.,observed_at_s=9.95,valid_until_s=10.2),
        behavior_policy=dict(frame_id='f1',disposition='CLEAR',confirmed=True,
            physical_stop=False,valid_until_s=10.2))


def test_fresh_clear_learned_adjustment_cannot_erase_explicit_setpoint():
    from lightweight_vla_adapter.src.longitudinal_contract import enforce_active_instruction_contract
    decision,risk=ready_speed_proposal()
    result,_=enforce_active_instruction_contract(decision,'SET_SPEED',30.,20.,risk=risk)
    assert result['action']=='keep_lane' and result['target_speed_kmh']==30.
    assert 'target_acceleration_mps2' not in result
    assert decision['target_speed_kmh']==12.
    assert result['speed_tracking_mode']=='EXPLICIT_CRUISE'


def test_explicit_cruise_has_faster_bounded_pid_response():
    from control.pid_controller import EgoPIDController
    normal=EgoPIDController(None,None)
    explicit=EgoPIDController(None,None)
    normal._previous_speed_error=explicit._previous_speed_error=5.
    ordinary=normal._longitudinal_control(30.,25.,.05)
    goal=explicit._longitudinal_control(30.,25.,.05,explicit_goal=True)
    assert 0.<ordinary[0]<goal[0]<=.85
    assert ordinary[1]==goal[1]==0.


def test_explicit_cruise_integral_converges_faster_without_exceeding_limits():
    from control.pid_controller import EgoPIDController
    normal=EgoPIDController(None,None);explicit=EgoPIDController(None,None)
    for _ in range(40):
        ordinary=normal._longitudinal_control(30.,27.,.05)
        goal=explicit._longitudinal_control(30.,27.,.05,explicit_goal=True)
    assert 0.<ordinary[0]<goal[0]<=.85
    assert ordinary[1]==goal[1]==0.
    assert explicit._speed_integral<=explicit.speed_integral_limit


@pytest.mark.parametrize('explicit',[False,True])
def test_pid_speed_response_preserves_deadband_braking_and_actuator_limits(explicit):
    from control.pid_controller import EgoPIDController
    pid=EgoPIDController(None,None)
    assert pid._longitudinal_control(30.,29.5,.05,explicit_goal=explicit)==(0.,0.)
    throttle,brake=pid._longitudinal_control(30.,40.,.05,explicit_goal=explicit)
    assert throttle==0. and 0.<brake<=.8
    throttle,brake=pid._longitudinal_control(30.,0.,.05,explicit_goal=explicit)
    assert 0.<throttle<=.85 and brake==0.


@pytest.mark.parametrize('fault',[
    'medium','high','stale','wrong_frame','risk_reason','unconfirmed','expired',
    'physical_stop','safety_deceleration','blocked','no_acceleration','no_risk'])
def test_safety_deceleration_never_promoted_to_cruise(fault):
    from lightweight_vla_adapter.src.longitudinal_contract import enforce_active_instruction_contract
    decision,risk=ready_speed_proposal()
    if fault in ('medium','high'):risk['risk_level']=fault
    elif fault=='stale':risk['risk_evidence']['is_fresh']=False
    elif fault=='wrong_frame':risk['frame_id']='old'
    elif fault=='risk_reason':risk['reason_codes']=['front_ttc_low']
    elif fault=='unconfirmed':risk['behavior_policy']['confirmed']=False
    elif fault=='expired':risk['risk_evidence']['valid_until_s']=9.
    elif fault=='physical_stop':risk['behavior_policy']['physical_stop']=True
    elif fault=='safety_deceleration':decision['reason']='risk_bounded_instruction_execution'
    elif fault=='blocked':decision['blocked_reason_codes']=['front_ttc_low']
    elif fault=='no_acceleration':decision['allow_positive_acceleration']=False
    elif fault=='no_risk':risk=None
    result,_=enforce_active_instruction_contract(decision,'SET_SPEED',30.,20.,risk=risk)
    assert result['action']=='decelerate' and result['target_speed_kmh']==12.
