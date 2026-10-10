"""Step-bound sensor feedback, including missing and expired observations."""
import pytest
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from control.passing_conditions import PassingMemory


@pytest.mark.parametrize('action,condition', [('OVERTAKE','SLOW_VEHICLE'),('PASS_BY','CYCLIST')])
@pytest.mark.parametrize('failure',[None,'wrong_request','expired','missing','wrong_condition'])
def test_passing_completion_requires_fresh_matching_proof(action,condition,failure):
    runtime=DrivingPlanRuntime()
    runtime.document=dict(request_id='command',intent=dict(steps=[dict(step_id='pass',action=action,
        parameters=dict(condition=condition))]))
    runtime.state=dict(plan_status='ACTIVE',active_step_index=0)
    for i in range(14):
        now=i*.05;frame=f'f{i}'
        runtime.frame_id=frame;runtime.timestamp_s=now
        evidence=dict(request_id='command',step_id='pass',frame_id=frame,condition=condition,
            satisfied=True,valid=True,source='rgb_lidar_passing_observer',observed_at_s=now,valid_until_s=now+.25)
        if failure=='wrong_request':evidence['request_id']='other'
        if failure=='expired':evidence['observed_at_s']=now-1.
        if failure=='wrong_condition':evidence['condition']='PATH_CLEAR'
        runtime._observe_condition_completion(dict(frame_id=frame),dict(frame_id=frame,risk_level='low'),
            {} if failure=='missing' else {'pass':evidence})
    assert bool(runtime.feedback)==(failure is None)


def test_cyclist_confirmation_cannot_consume_an_unrelated_pass():
    memory=PassingMemory();memory.last_frame=100;memory.last_time=5.
    memory.facts[('old','pass')]=dict(category='cyclist',target_id='one',completed_at_s=4.9,
        source_request_id='old',source_step_id='pass')
    document=dict(request_id='new',intent=dict(steps=[dict(step_id='confirm',action='CONFIRM',
        depends_on=['pass'],parameters=dict(condition='CYCLIST_PASSED'))]))
    kwargs=dict(frame_id='f100',frame=100,timestamp_s=5.,observed_at_s=5.,ego_xy=[0.,0.],
        heading=[1.,0.],ego_rear_m=2.4,corridor_clear=True)
    assert not memory.proofs(document,**kwargs)['confirm']['satisfied']
    document['request_id']='old'
    assert memory.proofs(document,**kwargs)['confirm']['satisfied']


def test_standalone_confirmation_uses_completed_fact_not_position_prediction():
    memory=PassingMemory();memory.last_frame=100;memory.last_time=100.
    memory.facts[('old','pass')]=dict(category='cyclist',target_id='one',completed_at_s=4.9,
        source_request_id='old',source_step_id='pass')
    document=dict(request_id='new',intent=dict(steps=[dict(step_id='confirm',action='CONFIRM',
        depends_on=[],parameters=dict(condition='CYCLIST_PASSED'))]))
    kwargs=dict(frame_id='f100',frame=100,timestamp_s=100.,observed_at_s=100.,ego_xy=[0.,0.],
        heading=[1.,0.],ego_rear_m=2.4,corridor_clear=True)
    proof=memory.proofs(document,**kwargs)['confirm']
    assert proof['satisfied'] and not memory.tracks
    assert proof['completed_event_fact']['source_request_id']=='old'
    document['request_id']='named';document['intent']['steps'][0]['target_ref']='different-cyclist'
    assert not memory.proofs(document,**kwargs)['confirm']['satisfied']
