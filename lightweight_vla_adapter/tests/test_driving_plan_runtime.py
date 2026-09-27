from copy import deepcopy
import pytest
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime


def test_waiting_plan_cannot_be_restarted_by_liveness_or_sequence():
    runtime=DrivingPlanRuntime()
    result,changed=runtime.enforce_execution(dict(action='accelerate',target_speed_kmh=20.,target_acceleration_mps2=1.),
        dict(decision_status='BLOCKED',blocked_reason_codes=['await_observed_step_condition']))
    assert changed and result['action']=='stop' and result['target_speed_kmh']==0.
    assert 'target_acceleration_mps2' not in result
from lightweight_vla_adapter.tests.fixtures import integration_documents


def documents():
    d,w,a,r=integration_documents(parser_action='ADJUST_SPEED')
    d['intent']['steps'][0]['parameters']={'target_speed_mps':4.}
    d['intent']['steps'][0]['completion']={'type':'TARGET_SPEED_REACHED'}
    second=deepcopy(d['intent']['steps'][0]);second.update(step_id='step_2',action='STOP',parameters={},
        depends_on=['step_1'],trigger={'type':'AFTER_STEP','step_id':'step_1'},completion={'type':'VEHICLE_STOPPED'})
    d['intent']['steps'].append(second)
    return d,w,r


@pytest.mark.parametrize('fault',[None,'stale','wrong_step','off_lane','stopped','gap','unhandled_condition'])
def test_keep_lane_requires_fresh_centered_moving_execution(fault):
    d,w,r=documents();runtime=DrivingPlanRuntime()
    d['intent']['steps'][0].update(action='KEEP_LANE',parameters={},completion={'type':'ACTION_REACHED'})
    if fault=='unhandled_condition':d['intent']['steps'][0]['parameters']={'until':'bus_stop'}
    for i in range(12):
        frame=f'f{i}';w['frame_id']=r['frame_id']=frame
        evidence=dict(source_step_id='step_1',observation_frame_id=frame,
            pid=dict(current_lane_id=-2,in_junction=False,lateral_error_m=.1,heading_error_deg=1.))
        if fault=='stale':evidence['observation_frame_id']='old'
        if fault=='wrong_step':evidence['source_step_id']='other'
        if fault=='off_lane':evidence['pid']['lateral_error_m']=1.
        runtime.prepare(d,frame_id=frame,timestamp_s=i*(.5 if fault=='gap' else .1),
            speed_mps=0. if fault=='stopped' else 2.,execution_state=evidence)
        runtime.advance(w,r)
    assert (runtime.state['step_states'][0]['status']=='COMPLETED') == (fault is None)


@pytest.mark.parametrize('exit_yaw,entered,expected',[(2.,True,True),(90.,True,False),(0.,False,False)])
def test_proceed_requires_observed_straight_junction_exit(exit_yaw,entered,expected):
    d,w,r=documents();runtime=DrivingPlanRuntime()
    d['intent']['steps'][0].update(action='PROCEED',parameters={'condition':'STRAIGHT_THROUGH_JUNCTION'},completion={'type':'ACTION_REACHED'})
    for i in range(14):
        frame=f'f{i}';w['frame_id']=r['frame_id']=frame
        evidence=dict(source_step_id='step_1',observation_frame_id=frame,ego_heading_deg=0. if i<4 else exit_yaw,
            pid=dict(current_lane_id=-2,in_junction=entered and i<4,lateral_error_m=.1,heading_error_deg=1.))
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=2.,execution_state=evidence)
        runtime.advance(w,r)
    assert (runtime.state['step_states'][0]['status']=='COMPLETED') == expected


def test_graph_advances_only_after_actual_speed_completion():
    d,w,r=documents();runtime=DrivingPlanRuntime()
    for i,speed in enumerate([1.,4.,4.,4.,4.,4.,4.,4.]):
        frame=f'f{i}';w['frame_id']=frame;r['frame_id']=frame;w['ego']['speed_mps']=speed
        active=runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=speed)
        decision=runtime.advance(w,r)
        if i==0:assert active['step_id']=='step_1'
        if i==1:assert active['step_id']=='step_1'
        if i==6:assert active['step_id']=='step_2'
    assert runtime.state['step_states'][0]['status']=='COMPLETED'
    assert decision['source_step_id']=='step_2' and decision['action']=='stop'


@pytest.mark.parametrize('direction,yaw,expected',[
    ('RIGHT',90.,True),('LEFT',-90.,True),('RIGHT',-90.,False),
    ('LEFT',90.,False),('RIGHT',0.,False)])
def test_turn_completion_requires_observed_matching_direction(direction,yaw,expected):
    d,w,r=documents();runtime=DrivingPlanRuntime()
    d['intent']['steps'][0].update(action='TURN',parameters={'direction':direction,'target_location':{'x':10.,'y':10.}},
                                 completion={'type':'JUNCTION_EXITED'})
    for i in range(14):
        frame=f'f{i}';w['frame_id']=r['frame_id']=frame
        evidence=dict(source_step_id='step_1',observation_frame_id=frame,
            ego_heading_deg=0. if i<4 else yaw,
            pid=dict(in_junction=i<4,current_lane_id=-2,lateral_error_m=.1,heading_error_deg=1.))
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=2.,execution_state=evidence)
        runtime.advance(w,r)
    assert (runtime.state['step_states'][0]['status']=='COMPLETED') == expected


@pytest.mark.parametrize('condition,fault,expected',[
    ('PATH_CLEAR',None,True),('LEFT_LANE_SAFE',None,True),('RIGHT_LANE_SAFE',None,True),
    ('PATH_CLEAR','gap',False),('PATH_CLEAR','risk',False),
    ('LEFT_LANE_SAFE','unsafe_lane',False),('PEDESTRIAN_CLEAR',None,False)])
def test_check_completion_uses_fresh_sustained_explicit_evidence(condition,fault,expected):
    d,w,r=documents();runtime=DrivingPlanRuntime()
    d['intent']['steps'][0].update(action='CHECK',parameters={'condition':condition},
                                 completion={'type':'ACTION_REACHED'})
    for i in range(14):
        frame=f'f{i}';w['frame_id']=frame;r['frame_id']='old' if fault=='stale' else frame
        r['risk_level']='high' if fault=='risk' else 'low'
        r['recommended_action']='emergency_brake' if fault=='risk' else 'maintain_speed'
        r['lane_change']={side:dict(is_safe=fault!='unsafe_lane',reason_codes=[]) for side in ('left','right')}
        runtime.prepare(d,frame_id=frame,timestamp_s=i*(.5 if fault=='gap' else .1),speed_mps=0.)
        runtime.advance(w,r)
    assert (runtime.state['step_states'][0]['status']=='COMPLETED') == expected


def test_unobserved_condition_cannot_be_flattened_into_immediate_action():
    d,w,r=documents();d['intent']['steps'][0]['trigger']={'type':'TARGET_VISIBLE','target_ref':'bus_stop'}
    runtime=DrivingPlanRuntime();runtime.prepare(d,frame_id=w['frame_id'],timestamp_s=0.,speed_mps=1.)
    decision=runtime.advance(w,r)
    assert decision['action']=='stop' and runtime.state['step_states'][0]['status']=='WAITING'


def test_waiting_does_not_count_as_completed_even_when_target_speed_happens_to_match():
    d,w,r=documents();d['intent']['steps'][0]['trigger']={'type':'TARGET_VISIBLE','target_ref':'bus_stop'}
    runtime=DrivingPlanRuntime()
    for i in range(10):
        w['frame_id']=r['frame_id']=f'f{i}'
        runtime.prepare(d,frame_id=w['frame_id'],timestamp_s=i*.1,speed_mps=4.)
        runtime.advance(w,r)
    assert runtime.state['active_step_id']=='step_1'


def test_new_request_resets_old_completion():
    d,w,r=documents();runtime=DrivingPlanRuntime()
    runtime.prepare(d,frame_id=w['frame_id'],timestamp_s=0.,speed_mps=0.);runtime.advance(w,r)
    new=deepcopy(d);new['request_id']='new'
    assert runtime.prepare(new,frame_id=w['frame_id'],timestamp_s=1.,speed_mps=0.)['step_id']=='step_1'
    assert runtime.state is None


def test_path_clear_uses_current_risk_evidence_and_recovers():
    d,w,r=documents();d['intent']['steps'][0]['preconditions']=['PATH_CLEAR']
    runtime=DrivingPlanRuntime()
    for i,level in enumerate(['high','low']):
        w['frame_id']=r['frame_id']=f'f{i}';r['risk_level']=level
        runtime.prepare(d,frame_id=w['frame_id'],timestamp_s=i*.1,speed_mps=0.)
        runtime.advance(w,r)
        assert runtime.state['step_states'][0]['status']==('WAITING' if i==0 else 'ACTIVE')


def test_transient_emergency_waits_then_recovers_same_step_and_speed():
    d,w,r=documents()
    d['intent']['steps'][0]['on_blocked']='SAFE_STOP'
    runtime=DrivingPlanRuntime()
    targets=[]
    for i in range(10):
        w['frame_id']=r['frame_id']=f'risk-{i}'
        r['risk_level']='high' if i==1 else 'low'
        r['recommended_action']='emergency_brake' if i==1 else 'maintain_speed'
        runtime.prepare(d,frame_id=w['frame_id'],timestamp_s=i*.1,speed_mps=0.)
        decision=runtime.advance(w,r)
        targets.append(runtime.state['step_states'][0]['resolved_target_speed_kmh'])
        assert runtime.state['active_step_id']=='step_1'
        assert runtime.state['plan_status']=='ACTIVE'
        if i==1:
            assert decision['action']=='emergency_brake'
            enforced,_=runtime.enforce_execution(dict(action='accelerate',emergency=False),decision)
            assert enforced['action']=='emergency_brake' and enforced['emergency']
        if 1<=i<7:
            assert runtime.state['step_states'][0]['status']=='WAITING'
            assert decision['decision_status']=='BLOCKED'
        if i>=8:
            assert decision['decision_status']=='READY'
    assert len(set(targets))==1


def test_risk_recovery_never_bypasses_missing_trigger_or_advances_step():
    d,w,r=documents()
    d['intent']['steps'][0]['trigger']={'type':'TARGET_VISIBLE','target_ref':'bus'}
    runtime=DrivingPlanRuntime()
    for i in range(20):
        w['frame_id']=r['frame_id']=f'condition-{i}'
        r['risk_level']='high' if i==0 else 'low'
        r['recommended_action']='emergency_brake' if i==0 else 'maintain_speed'
        runtime.prepare(d,frame_id=w['frame_id'],timestamp_s=i*.1,speed_mps=4.)
        decision=runtime.advance(w,r)
        assert decision['decision_status']=='BLOCKED'
        assert runtime.state['active_step_id']=='step_1'
        if i==0:assert decision['action']=='emergency_brake'


def execution(frame, step_id, **pid):
    defaults=dict(current_lane_id=-2,in_junction=False,
        lateral_error_m=.1,heading_error_deg=1.)
    defaults.update(pid)
    return dict(source_step_id=step_id,observation_frame_id=frame,pid=defaults)


def test_constrained_speed_target_completes_at_observed_safe_cap():
    d,w,r=documents();runtime=DrivingPlanRuntime()
    for i in range(9):
        frame=f'cap-{i}';w['frame_id']=r['frame_id']=frame
        observed=execution(frame,'step_1',speed_target_status='CONSTRAINED',
            effective_target_speed_kmh=10.8,
            speed_constraint_codes=['road_speed_limit'])
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=3.,
            execution_state=observed)
        runtime.advance(w,r)
    first=runtime.state['step_states'][0]
    assert first['status']=='COMPLETED'
    assert 'observed_constrained_speed_completion' in first['reason_codes']
    assert runtime.state['active_step_id']=='step_2'


def test_explicit_unreachable_speed_fails_instead_of_deadlocking():
    d,w,r=documents();runtime=DrivingPlanRuntime()
    frame='start';w['frame_id']=r['frame_id']=frame
    runtime.prepare(d,frame_id=frame,timestamp_s=0.,speed_mps=0.,
        execution_state=execution(frame,'step_1'))
    runtime.advance(w,r)
    frame='unreachable';w['frame_id']=r['frame_id']=frame
    observed=execution(frame,'step_1',speed_target_status='UNREACHABLE',
        speed_constraint_codes=['road_target_unreachable'])
    runtime.prepare(d,frame_id=frame,timestamp_s=.1,speed_mps=0.,execution_state=observed)
    decision=runtime.advance(w,r)
    assert runtime.state['plan_status']=='FAILED'
    assert runtime.state['active_step_id'] is None
    assert decision['action']=='stop'
    assert 'target_speed_unreachable' in runtime.state['reason_codes']


def predicate(d,frame,step_id,condition,**changes):
    result=dict(request_id=d['request_id'],step_id=step_id,frame_id=frame,
        condition=condition,satisfied=True,valid=True,source='scene_observer')
    result.update(changes)
    return {step_id:result}


@pytest.mark.parametrize('fault',[None,'wrong_request','wrong_step','stale','invalid'])
def test_non_risk_condition_requires_fresh_bound_predicate(fault):
    d,w,r=documents();runtime=DrivingPlanRuntime()
    d['intent']['steps'][0].update(action='WAIT',parameters={'condition':'PEDESTRIAN_CLEAR'},
        completion={'type':'ACTION_REACHED'})
    for i in range(9):
        frame=f'predicate-{i}';w['frame_id']=r['frame_id']=frame
        ready=predicate(d,frame,'step_1','PEDESTRIAN_CLEAR')
        item=ready['step_1']
        if fault=='wrong_request':item['request_id']='old-request'
        if fault=='wrong_step':item['step_id']='old-step'
        if fault=='stale':item['frame_id']='old-frame'
        if fault=='invalid':item['valid']=False
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=0.)
        runtime.advance(w,r,readiness=ready)
    assert (runtime.state['step_states'][0]['status']=='COMPLETED') == (fault is None)


def return_document():
    d,w,r=documents()
    d['intent']['steps'][0].update(action='CHANGE_LANE',
        parameters={'return_to':'ORIGINAL_LANE'},
        completion={'type':'LANE_CHANGE_COMPLETED'})
    return d,w,r


@pytest.mark.parametrize('current,left,right,expected',[
    (-1,None,-2,'RIGHT'),(-3,-2,None,'LEFT')])
def test_return_direction_is_resolved_from_original_lane_topology(current,left,right,expected):
    d,w,r=return_document();runtime=DrivingPlanRuntime()
    runtime.lane_change_history=[dict(source=dict(lane_id=-2,road_id=1,section_id=0),
        target=dict(lane_id=current,road_id=1,section_id=0),returned=False)]
    frame='return';w['frame_id']=r['frame_id']=frame
    observed=execution(frame,'step_1',current_lane_ref=dict(lane_id=current,road_id=1,section_id=0),
        left_lane_ref=None if left is None else dict(lane_id=left,road_id=1,section_id=0),
        right_lane_ref=None if right is None else dict(lane_id=right,road_id=1,section_id=0),
        left_lane_change_allowed=True,right_lane_change_allowed=True)
    step=runtime.prepare(d,frame_id=frame,timestamp_s=0.,speed_mps=2.,execution_state=observed)
    assert step['parameters']['direction']==expected
    decision=runtime.advance(w,r)
    assert decision['action']=='lane_change_'+expected.lower()


def test_return_waits_for_lane_safety_then_resumes_same_step():
    d,w,r=return_document();runtime=DrivingPlanRuntime()
    runtime.lane_change_history=[dict(source=dict(lane_id=-2),target=dict(lane_id=-1),returned=False)]
    for i,safe in enumerate((False,True)):
        frame=f'safe-{i}';w['frame_id']=r['frame_id']=frame
        r['lane_change']['right']={'is_safe':safe,'reason_codes':[] if safe else ['rear_ttc_low']}
        observed=execution(frame,'step_1',current_lane_ref=dict(lane_id=-1),
            right_lane_ref=dict(lane_id=-2),right_lane_change_allowed=True)
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=2.,execution_state=observed)
        decision=runtime.advance(w,r)
        assert runtime.state['active_step_id']=='step_1'
        assert decision['action']==('lane_change_right' if safe else 'decelerate')


def test_unreachable_original_lane_waits_without_inventing_a_direction():
    d,w,r=return_document();runtime=DrivingPlanRuntime()
    runtime.lane_change_history=[dict(source=dict(lane_id=-2),target=dict(lane_id=-1),returned=False)]
    frame='return';w['frame_id']=r['frame_id']=frame
    observed=execution(frame,'step_1',current_lane_ref=dict(lane_id=-1),
        left_lane_ref=None,right_lane_ref=dict(lane_id=-3),right_lane_change_allowed=True)
    step=runtime.prepare(d,frame_id=frame,timestamp_s=0.,speed_mps=2.,execution_state=observed)
    assert 'direction' not in step['parameters']
    decision=runtime.advance(w,r)
    assert decision['action']=='stop' and decision['decision_status']=='BLOCKED'
    assert decision['blocked_reason_codes']==['original_lane_not_adjacent']


def test_lane_change_history_drives_later_return_step():
    d,w,r=documents();runtime=DrivingPlanRuntime()
    d['intent']['steps'][0].update(action='CHANGE_LANE',parameters={'direction':'LEFT'},
        completion={'type':'LANE_CHANGE_COMPLETED'})
    d['intent']['steps'][1].update(action='CHANGE_LANE',
        parameters={'return_to':'ORIGINAL_LANE'},
        completion={'type':'LANE_CHANGE_COMPLETED'})
    for i in range(9):
        frame=f'outbound-{i}';w['frame_id']=r['frame_id']=frame
        pid=dict(current_lane_ref=dict(road_id=1,section_id=0,lane_id=-2 if i==0 else -1),
            left_lane_ref=dict(road_id=1,section_id=0,lane_id=-1),
            right_lane_ref=dict(road_id=1,section_id=0,lane_id=-2),
            left_lane_change_allowed=True,right_lane_change_allowed=True,
            lane_change_completed=i>0)
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=2.,
            execution_state=execution(frame,'step_1',**pid))
        decision=runtime.advance(w,r)
    assert runtime.lane_change_history[-1]['source']['lane_id']==-2
    assert runtime.state['active_step_id']=='step_2'
    assert decision['action']=='lane_change_right'


def test_already_in_original_lane_completes_without_lateral_command():
    d,w,r=return_document();runtime=DrivingPlanRuntime()
    runtime.lane_change_history=[dict(source=dict(lane_id=-2),target=dict(lane_id=-1),returned=False)]
    decisions=[]
    for i in range(9):
        frame=f'already-{i}';w['frame_id']=r['frame_id']=frame
        observed=execution(frame,'step_1',current_lane_ref=dict(lane_id=-2),
            left_lane_ref=dict(lane_id=-1),right_lane_ref=dict(lane_id=-3))
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=2.,execution_state=observed)
        decisions.append(runtime.advance(w,r))
    assert all(not item['action'].startswith('lane_change_') for item in decisions)
    assert runtime.state['step_states'][0]['reason_codes']==['already_in_original_lane']
    assert runtime.lane_change_history[-1]['returned'] is True


def test_turn_consumes_only_fresh_reachable_road_target():
    d,w,r=documents();runtime=DrivingPlanRuntime()
    d['intent']['steps'][0].update(action='TURN',parameters={'direction':'RIGHT'},
        completion={'type':'JUNCTION_EXITED'})
    frame='turn';w['frame_id']=r['frame_id']=frame
    ready={'step_1':dict(request_id=d['request_id'],step_id='step_1',frame_id=frame,
        reachable=True,source='road_topology',target_location={'x':10.,'y':5.,'z':0.})}
    step=runtime.prepare(d,frame_id=frame,timestamp_s=0.,speed_mps=2.,
        execution_state=execution(frame,'step_1'),readiness=ready)
    assert step['parameters']['target_location']=={'x':10.,'y':5.,'z':0.}
    assert runtime.advance(w,r,readiness=ready)['decision_status']=='READY'


def test_turn_rejects_stale_road_target():
    d,w,r=documents();runtime=DrivingPlanRuntime()
    d['intent']['steps'][0].update(action='TURN',parameters={'direction':'RIGHT'},
        completion={'type':'JUNCTION_EXITED'},on_blocked='WAIT_FOR_SAFE')
    frame='turn';w['frame_id']=r['frame_id']=frame
    ready={'step_1':dict(request_id=d['request_id'],step_id='step_1',frame_id='old',
        reachable=True,source='road_topology',target_location={'x':10.,'y':5.})}
    step=runtime.prepare(d,frame_id=frame,timestamp_s=0.,speed_mps=2.,
        execution_state=execution(frame,'step_1'),readiness=ready)
    assert 'target_location' not in step['parameters']
    decision=runtime.advance(w,r,readiness=ready)
    assert decision['decision_status']=='BLOCKED'
    assert 'turn_target_location_missing' in decision['blocked_reason_codes']
