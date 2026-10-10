from copy import deepcopy
from types import SimpleNamespace

import pytest

from control.road_target_resolver import RoadTargetResolver
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.tests.fixtures import integration_documents


def route():
    return [dict(distance_m=s, x=x, y=y, z=0., yaw=yaw, is_junction=junction,
                 road_id=road, section_id=0, lane_id=-1)
            for s,x,y,yaw,junction,road in [
                (0,0,0,0,False,1), (10,10,0,0,True,2),
                (20,20,0,45,True,2), (30,20,10,90,False,3),
                (40,20,20,90,True,4), (50,20,30,90,False,5),
                (60,20,40,90,True,6), (70,10,40,180,False,7),
                (80,0,40,180,False,7)]]


def document():
    return dict(request_id='r1',intent=dict(steps=[dict(step_id='turn1',action='TURN',
        parameters=dict(direction='RIGHT')),
        dict(step_id='turn2',action='TURN',parameters=dict(direction='RIGHT'))]))


def test_future_target_is_available_without_entering_junction():
    resolver=RoadTargetResolver(route())
    result=resolver.resolve(document(),frame_id='f1',progress_m=0.)['turn1']
    assert result['reachable'] and result['valid']
    assert result['target_location']==dict(x=20.,y=10.,z=0.,yaw=90.)
    assert result['junction']['start_m']==10.
    assert result['frame_id']=='f1' and result['step_id']=='turn1'


def test_pinned_active_target_survives_exit_future_steps_are_not_pinned():
    resolver=RoadTargetResolver(route());doc=document()
    ready=resolver.resolve(doc,frame_id='f1',progress_m=0.)
    resolver.pin(doc['intent']['steps'][0],ready['turn1'],frame_id='f1')
    after=resolver.resolve(doc,frame_id='f2',progress_m=31.)
    assert after['turn1']['target_location']==ready['turn1']['target_location']
    assert after['turn1']['frame_id']=='f2'
    assert after['turn2']['reason']=='road_target_direction_conflict'
    assert not after['turn2']['reachable']
    assert resolver.resolve(doc,frame_id='f3',progress_m=51.)['turn2']['reachable']


@pytest.mark.parametrize('fault',['stale','wrong_step','wrong_request','wrong_source','invalid','fake_junction'])
def test_invalid_evidence_cannot_pin(fault):
    resolver=RoadTargetResolver(route());doc=document()
    ready=resolver.resolve(doc,frame_id='f1',progress_m=0.)['turn1']
    if fault=='stale':ready['frame_id']='old'
    if fault=='wrong_step':ready['step_id']='other'
    if fault=='wrong_request':ready['request_id']='other'
    if fault=='wrong_source':ready['source']='scenario_truth'
    if fault=='invalid':ready['valid']=False
    if fault=='fake_junction':ready['junction']['start_m']=12345.
    resolver.pin(doc['intent']['steps'][0],ready,frame_id='f1')
    assert not resolver.pinned


def test_request_change_resets_target_and_does_not_skip_mismatch():
    resolver=RoadTargetResolver(route());doc=document()
    ready=resolver.resolve(doc,frame_id='f1',progress_m=0.)
    resolver.pin(doc['intent']['steps'][0],ready['turn1'],frame_id='f1')
    doc['request_id']='r2'
    after=resolver.resolve(doc,frame_id='f2',progress_m=31.)
    assert not resolver.pinned and after['turn1']['reason']=='road_target_direction_conflict'


def test_route_without_junction_and_discontinuous_route_fail_closed():
    pts=route()[:2];pts[-1]['is_junction']=False
    resolver=RoadTargetResolver(pts)
    assert resolver.resolve(document(),frame_id='f1',progress_m=0.)['turn1']['reason']=='road_target_no_next_junction'
    pts[-1]['x']=1000.
    with pytest.raises(ValueError):RoadTargetResolver(pts)


def test_route_pid_publishes_readiness_from_point_route():
    from control.generic_route_pid import GenericRoutePID
    pid=GenericRoutePID.__new__(GenericRoutePID)
    pid.route_context=None;pid.route_plan=None
    pid.route_manager=SimpleNamespace(route=route(),progress_m=0.)
    ready=pid.road_step_readiness(document(),frame_id='f1')
    assert ready['turn1']['reachable']
    pid.pin_road_step_target(document()['intent']['steps'][0],ready['turn1'],frame_id='f1')
    pid.route_manager.progress_m=31.
    assert pid.road_step_readiness(document(),frame_id='f2')['turn1']['reachable']


def test_point_route_update_receives_actor_not_location():
    from control.generic_route_pid import GenericRoutePID
    pid=GenericRoutePID.__new__(GenericRoutePID)
    pid.route_context=None;pid.ego=object()
    observed=[]
    def update(actor):
        observed.append(actor)
        return 12.
    pid.route_manager=SimpleNamespace(update=update)
    assert pid._update_progress()==12. and observed==[pid.ego]


def test_single_turn_enters_grounded_plan_without_falsely_claiming_model_parse():
    from control.generic_instruction_fsm import GenericInstructionFSM
    fsm=GenericInstructionFSM()
    command=dict(id='single-turn',text='\u524d\u65b9\u8def\u53e3\u53f3\u8f6c')
    parsed=fsm.parse(command,use_parser_model=False)
    doc=fsm.driving_intent(command,parsed=parsed)
    assert doc['request_id']=='single-turn'
    assert doc['intent']['steps'][0]['parameters']['direction']=='RIGHT'
    assert doc['intent']['steps'][0]['completion']['type']=='JUNCTION_EXITED'
    assert not doc['parse_result']['model_prediction']
    assert RoadTargetResolver(route()).resolve(doc,frame_id='f1',progress_m=0.)['step_1']['reachable']


def test_unresolved_compound_cannot_be_replaced_by_single_turn_adapter():
    from control.generic_instruction_fsm import GenericInstructionFSM
    fsm=GenericInstructionFSM()
    command=dict(text='Turn right then turn left.')
    assert fsm.driving_intent(command,parsed=fsm.parse(command,use_parser_model=False)) is None


def test_geometry_readiness_reaches_runtime_and_preserves_emergency_priority():
    doc,world,_,risk=integration_documents(parser_action='TURN')
    doc['intent']['steps'][0].update(parameters=dict(direction='RIGHT'),
        completion=dict(type='JUNCTION_EXITED'),on_blocked='SAFE_STOP')
    resolver=RoadTargetResolver(route());runtime=DrivingPlanRuntime()
    ready=resolver.resolve(doc,frame_id=world['frame_id'],progress_m=0.)
    step=runtime.prepare(doc,frame_id=world['frame_id'],timestamp_s=0.,speed_mps=2.,readiness=ready)
    assert step['parameters']['target_location']
    decision=runtime.advance(world,risk,readiness=ready)
    assert decision['decision_status']=='READY' and decision['target_location']
    assert 'target_location' not in doc['intent']['steps'][0]['parameters']
    world['frame_id']=risk['frame_id']='next'
    risk.update(risk_level='high',recommended_action='emergency_brake')
    ready=resolver.resolve(doc,frame_id='next',progress_m=1.)
    runtime.prepare(doc,frame_id='next',timestamp_s=.1,speed_mps=2.,readiness=ready)
    assert runtime.advance(world,risk,readiness=ready)['action']=='emergency_brake'


@pytest.mark.parametrize('action',['accelerate','keep_lane','turn_left'])
def test_downstream_motion_cannot_drop_active_turn_target(action):
    runtime=DrivingPlanRuntime()
    canonical=dict(decision_status='READY',action='turn_right',target_location=dict(x=2.,y=3.))
    result,changed=runtime.enforce_execution(dict(decision_status='READY',action=action,
        target_location=None,target_speed_kmh=5.,allow_positive_acceleration=False),canonical)
    assert changed and result['action']=='turn_right'
    assert result['target_location']==canonical['target_location']
    assert result['target_speed_kmh']==5. and not result['allow_positive_acceleration']


@pytest.mark.parametrize('action',['stop','emergency_brake','decelerate'])
def test_turn_target_never_overrides_braking(action):
    runtime=DrivingPlanRuntime()
    result,changed=runtime.enforce_execution(dict(action=action,target_location=None,target_speed_kmh=0.),
        dict(decision_status='READY',action='turn_right',target_location=dict(x=2.,y=3.)))
    assert not changed and result['action']==action


@pytest.mark.parametrize('action',['turn_left','lane_change_left'])
def test_vla_cannot_replace_requested_turn_with_another_navigation(action):
    from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
    from lightweight_vla_adapter.tests.fixtures import proposal
    from scene_understanding.src.control_decision import build_control_decision
    doc,world,alignment,risk=integration_documents(parser_action='TURN')
    doc['intent']['steps'][0]['parameters']=dict(direction='RIGHT',target_location=dict(x=20.,y=10.))
    canonical=build_control_decision(doc,world,alignment,risk)
    result=gate_vla_proposal(proposal(action=action),canonical,risk)
    assert result['action']=='turn_right' and result['target_location']
    assert 'vla_incompatible_with_active_intent' in result['blocked_reason_codes']


@pytest.mark.parametrize('action',['accelerate','keep_lane'])
def test_longitudinal_proposal_keeps_turn_target_without_latching_zero_speed(action):
    from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
    from lightweight_vla_adapter.tests.fixtures import proposal
    from scene_understanding.src.control_decision import build_control_decision
    doc,world,alignment,risk=integration_documents(parser_action='TURN')
    world['ego']['speed_mps']=0.
    doc['intent']['steps'][0]['parameters']=dict(direction='RIGHT',target_location=dict(x=20.,y=10.))
    canonical=build_control_decision(doc,world,alignment,risk)
    result=gate_vla_proposal(proposal(action=action,target_speed_kmh=10.),canonical,risk)
    assert result['action']=='turn_right' and result['target_location']==canonical['target_location']
    assert result['target_speed_kmh']==10.
    assert result['reason']=='vla_longitudinal_only_turn_target_preserved'
    risk['recommended_action']='emergency_brake'
    # Safety precedence is tested on a freshly safety-gated canonical command.
    canonical=build_control_decision(doc,world,alignment,risk)
    assert gate_vla_proposal(proposal(action=action),canonical,risk)['action']=='emergency_brake'
