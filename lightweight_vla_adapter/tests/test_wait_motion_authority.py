import pytest
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
from lightweight_vla_adapter.tests.test_driving_plan_runtime import documents,predicate
from lightweight_vla_adapter.tests.fixtures import proposal


@pytest.mark.parametrize('action',['WAIT','CHECK','CONFIRM'])
def test_unresolved_predicate_blocks_positive_network_motion(action):
    d,w,r=documents()
    d['intent']['steps'][0].update(action=action,parameters={'condition':'PEDESTRIAN_CLEAR'},
        completion={'type':'ACTION_REACHED'})
    runtime=DrivingPlanRuntime()
    for i in range(12):
        frame=f'unknown-{i}';w['frame_id']=r['frame_id']=frame
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=0.)
        canonical=runtime.advance(w,r)
        assert canonical['decision_status']=='BLOCKED'
        assert canonical['action']=='stop' and canonical['target_speed_kmh']==0.
        proposed=proposal(action='accelerate',target_speed_kmh=20.)
        proposed.update(request_id=d['request_id'],frame_id=frame)
        final=gate_vla_proposal(proposed,canonical,r)
        assert final['action']=='stop' and final['target_speed_kmh']==0.
    assert runtime.state['step_states'][0]['status']=='ACTIVE'


def test_verified_predicate_can_complete_while_vehicle_is_held():
    d,w,r=documents()
    d['intent']['steps'][0].update(action='WAIT',parameters={'condition':'PEDESTRIAN_CLEAR'},
        completion={'type':'ACTION_REACHED'})
    runtime=DrivingPlanRuntime()
    for i in range(10):
        frame=f'clear-{i}';w['frame_id']=r['frame_id']=frame
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=0.)
        runtime.advance(w,r,readiness=predicate(d,frame,'step_1','PEDESTRIAN_CLEAR'))
    assert runtime.state['step_states'][0]['status']=='COMPLETED'
