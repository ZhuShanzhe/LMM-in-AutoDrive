import pytest
from control.command_dispatch import CompletionCommandQueue, resolve_command_dispatch_mode


def command(identity, trigger):
    return dict(id=identity,activate_at_m=trigger,
                driving_intent=dict(request_id=identity,intent=dict(steps=[dict(action='KEEP_LANE')])) )


def test_dispatch_mode_auto_enables_completion_for_structured_commands():
    assert resolve_command_dispatch_mode([command('a', 0), command('b', 10)]) == 'completion_serial'


def test_dispatch_mode_auto_preserves_route_latest_for_unstructured_commands():
    assert resolve_command_dispatch_mode([dict(text='keep going')]) == 'route_latest'


def test_dispatch_mode_explicit_override_is_respected():
    assert resolve_command_dispatch_mode([command('a', 0)], 'route_latest') == 'route_latest'


def test_later_route_trigger_cannot_overwrite_running_plan():
    queue=CompletionCommandQueue([command('a',0),command('b',10)])
    assert queue.select(0,0)['id']=='a'
    assert queue.select(20,1)['id']=='a'
    assert queue.select(20,2,dict(request_id='other',plan_status='COMPLETED'))['id']=='a'
    assert queue.select(20,3,dict(request_id='a',plan_status='COMPLETED'))['id']=='b'
    assert queue.status()['completed_commands']==1


def test_blocked_is_not_completion_and_timeout_does_not_advance():
    queue=CompletionCommandQueue([command('a',0),command('b',10)],timeout_s=5)
    queue.select(0,0)
    assert queue.select(20,1,dict(request_id='a',plan_status='BLOCKED'))['id']=='a'
    with pytest.raises(ValueError,match='TIMEOUT'):
        queue.select(20,5)
    assert queue.index==0


def test_expired_pending_command_not_executed_at_wrong_location():
    later=command('b',10)
    later['end_progress_m']=15
    queue=CompletionCommandQueue([command('a',0),later])
    queue.select(0,0)
    with pytest.raises(ValueError,match='MISSED_COMMAND_WINDOW'):
        queue.select(20,1,dict(request_id='a',plan_status='COMPLETED'))


def test_requires_structured_unique_requests():
    with pytest.raises(ValueError,match='distinct structured'):
        CompletionCommandQueue([dict(text='hello')])


@pytest.mark.parametrize('end',[0,-1,float('nan'),float('inf'),True,'10'])
def test_invalid_end_rejected_before_driving(end):
    item=command('a',0)
    item['end_progress_m']=end
    with pytest.raises(ValueError,match='end progress'):
        CompletionCommandQueue([item])


def test_active_window_expires_even_with_late_completion():
    item=command('a',0)
    item['deactivate_at_m']=20
    queue=CompletionCommandQueue([item,command('b',21)])
    assert '_dispatch_end_m' not in queue.select(0,0)
    with pytest.raises(ValueError,match='ACTIVE_COMMAND_WINDOW_EXPIRED'):
        queue.select(20,1,dict(request_id='a',plan_status='COMPLETED'))
    assert queue.index==0
    assert queue.status()['events'][-1]['progress_m']==20
    with pytest.raises(ValueError,match='halted'):
        queue.select(21,2)


def test_late_completion_cannot_override_timeout():
    queue=CompletionCommandQueue([command('a',0)],timeout_s=5)
    queue.select(0,0)
    with pytest.raises(ValueError,match='TIMEOUT'):
        queue.select(1,5,dict(request_id='a',plan_status='COMPLETED'))
    assert queue.index==0
