from copy import deepcopy

import pytest

from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.tests.fixtures import integration_documents


def documents():
    document,world,_,risk=integration_documents(parser_action='ADJUST_SPEED')
    first=document['intent']['steps'][0]
    first.update(action='KEEP_LANE',parameters={})
    first.pop('completion',None)
    second=deepcopy(first)
    second.update(step_id='step_2',action='SET_SPEED',parameters={'target_speed_mps':30/3.6},
                  depends_on=[first['step_id']],trigger={'type':'AFTER_STEP','step_id':first['step_id']})
    document['intent']['steps'].append(second)
    return document,world,risk


@pytest.mark.parametrize('fault',[None,'stale','wrong_step','off_lane','stopped','gap','unhandled_condition'])
def test_implicit_nonterminal_lane_hold_keeps_all_physical_requirements(fault):
    document,world,risk=documents()
    if fault=='unhandled_condition':document['intent']['steps'][0]['parameters']={'until':'bus_stop'}
    runtime=DrivingPlanRuntime()
    for i in range(12):
        frame=f'f{i}';world['frame_id']=risk['frame_id']=frame
        evidence=dict(source_step_id=document['intent']['steps'][0]['step_id'],observation_frame_id=frame,
            pid=dict(current_lane_id=-2,in_junction=False,lateral_error_m=.1,heading_error_deg=1.))
        if fault=='stale':evidence['observation_frame_id']='old'
        if fault=='wrong_step':evidence['source_step_id']='other'
        if fault=='off_lane':evidence['pid']['lateral_error_m']=1.
        runtime.prepare(document,frame_id=frame,timestamp_s=i*(.5 if fault=='gap' else .1),
                        speed_mps=0 if fault=='stopped' else 20/3.6,execution_state=evidence)
        runtime.advance(world,risk)
    assert (runtime.state['step_states'][0]['status']=='COMPLETED')==(fault is None)


def test_keep_lane_then_absolute_speed_reaches_the_terminal_setpoint():
    document,world,risk=documents();runtime=DrivingPlanRuntime()
    for i in range(25):
        frame=f'f{i}';world['frame_id']=risk['frame_id']=frame
        index=(runtime.state or {}).get('active_step_index',0)
        if index is None:break
        active=document['intent']['steps'][index]['step_id']
        speed=20/3.6 if index==0 else 30/3.6
        world['ego']['speed_mps']=speed
        runtime.prepare(document,frame_id=frame,timestamp_s=i*.1,speed_mps=speed,
            execution_state=dict(source_step_id=active,observation_frame_id=frame,
                pid=dict(current_lane_id=-2,in_junction=False,lateral_error_m=.1,heading_error_deg=1.)))
        runtime.advance(world,risk)
    assert runtime.state['plan_status']=='COMPLETED'
    assert all(step['status']=='COMPLETED' for step in runtime.state['step_states'])


def test_terminal_lane_hold_is_not_silently_converted_to_short_completion():
    document,world,risk=documents();document['intent']['steps'].pop()
    runtime=DrivingPlanRuntime()
    for i in range(12):
        frame=f'f{i}';world['frame_id']=risk['frame_id']=frame
        runtime.prepare(document,frame_id=frame,timestamp_s=i*.1,speed_mps=5,
            execution_state=dict(source_step_id=document['intent']['steps'][0]['step_id'],observation_frame_id=frame,
                pid=dict(current_lane_id=-2,in_junction=False,lateral_error_m=.1,heading_error_deg=1.)))
        runtime.advance(world,risk)
    assert runtime.state['plan_status']=='ACTIVE'


def test_explicit_completion_is_not_overridden_by_the_default():
    document,world,risk=documents()
    document['intent']['steps'][0]['completion']={'type':'VEHICLE_STOPPED'}
    runtime=DrivingPlanRuntime()
    for i in range(12):
        frame=f'f{i}';world['frame_id']=risk['frame_id']=frame
        runtime.prepare(document,frame_id=frame,timestamp_s=i*.1,speed_mps=5,
            execution_state=dict(source_step_id=document['intent']['steps'][0]['step_id'],observation_frame_id=frame,
                pid=dict(current_lane_id=-2,in_junction=False,lateral_error_m=.1,heading_error_deg=1.)))
        runtime.advance(world,risk)
    assert runtime.state['step_states'][0]['status']=='ACTIVE'
