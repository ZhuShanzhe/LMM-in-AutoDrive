from copy import deepcopy
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


def test_graph_advances_only_after_actual_speed_completion():
    d,w,r=documents();runtime=DrivingPlanRuntime()
    for i,speed in enumerate([1.,4.,4.,4.,4.,4.,4.,4.]):
        frame=f'f{i}';w['frame_id']=frame;r['frame_id']=frame;w['ego']['speed_mps']=speed
        active=runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=speed)
        decision=runtime.advance(w,r)
        if i==0:assert active['step_id']=='step_1'
        if i==1:assert active['step_id']=='step_2'
    assert runtime.state['step_states'][0]['status']=='COMPLETED'
    assert decision['source_step_id']=='step_2' and decision['action']=='stop'


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
