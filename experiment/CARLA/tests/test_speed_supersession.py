from copy import deepcopy

import pytest

from control.command_dispatch import CompletionCommandQueue


def commands():
    def command(identity, progress, actions):
        return dict(announce_at_m=progress, driving_intent=dict(request_id=identity,
            intent=dict(steps=[dict(step_id=f'{identity}_{i}', action=action)
                              for i, action in enumerate(actions)])))
    return [command('old', 0., ['WAIT', 'CHANGE_LANE', 'SET_SPEED']),
            command('new', 10., ['KEEP_LANE', 'SET_SPEED', 'WAIT', 'PROCEED'])]


def feedback():
    steps=[dict(s, status='COMPLETED' if i < 2 else 'ACTIVE')
           for i,s in enumerate(commands()[0]['driving_intent']['intent']['steps'])]
    return dict(request_id='old', observed_at_s=1., plan_status='ACTIVE',
        plan_state=dict(request_id='old', plan_status='ACTIVE', active_step_index=2,
                        active_step_id='old_2', step_states=steps))


def test_later_speed_target_supersedes_only_old_final_speed_goal():
    queue=CompletionCommandQueue(commands(),speed_target_supersession=True)
    queue.select(0.,1.)
    selected=queue.select(10.,1.1,feedback())
    assert selected['driving_intent']['request_id']=='new'
    status=queue.status()
    assert status['completed_commands']==0 and status['superseded_commands']==1
    old=status['events'][1]
    assert old['status']=='SUPERSEDED' and old['prior_plan_state']['step_states'][-1]['status']=='ACTIVE'


@pytest.mark.parametrize('fault', ['stale','identity','prior_wait','prior_lane','final_wait','not_final',
    'no_new_speed','early','disabled','missing_state','wrong_step_identity'])
def test_supersession_cannot_skip_required_steps_or_use_stale_feedback(fault):
    items=deepcopy(commands()); fb=feedback(); progress=10.
    if fault=='stale':fb['observed_at_s']=0.
    if fault=='identity':fb['request_id']='other'
    if fault=='prior_wait':fb['plan_state']['step_states'][0]['status']='WAITING'
    if fault=='prior_lane':fb['plan_state']['step_states'][1]['status']='ACTIVE'
    if fault=='final_wait':
        items[0]['driving_intent']['intent']['steps'][-1]['action']='WAIT'
        fb['plan_state']['step_states'][-1]['action']='WAIT'
    if fault=='not_final':fb['plan_state']['active_step_index']=1
    if fault=='no_new_speed':items[1]['driving_intent']['intent']['steps']=[dict(step_id='new_0',action='PROCEED')]
    if fault=='early':progress=9.
    if fault=='missing_state':fb.pop('plan_state')
    if fault=='wrong_step_identity':fb['plan_state']['step_states'][0]['step_id']='unrelated'
    queue=CompletionCommandQueue(items,speed_target_supersession=fault!='disabled')
    queue.select(0.,1.)
    assert queue.select(progress,1.1,fb)['driving_intent']['request_id']=='old'
    assert queue.status()['superseded_commands']==0
