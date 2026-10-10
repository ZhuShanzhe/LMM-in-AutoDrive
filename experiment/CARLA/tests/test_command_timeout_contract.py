import pytest
from control.command_dispatch import CompletionCommandQueue, CommandDispatchHalted


def command(**extra):
    return dict(activate_at_m=0,driving_intent=dict(request_id='request-1',
        intent=dict(steps=[dict(action='KEEP_LANE')])),**extra)


def test_per_command_budget_is_enforced_not_default_timeout():
    queue=CompletionCommandQueue([command(execution_timeout_s=180)],timeout_s=120)
    queue.select(0,0)
    queue.select(10,179.9)
    assert queue.status()['execution_timeout_s']==180
    with pytest.raises(CommandDispatchHalted):queue.select(10,180)
    assert queue.status()['failure']=='TIMEOUT'


@pytest.mark.parametrize('budget',[True,None,0,-1,float('nan'),float('inf'),'180'])
def test_invalid_per_command_budgets_are_rejected(budget):
    with pytest.raises(ValueError):CompletionCommandQueue([command(execution_timeout_s=budget)])


def test_other_command_sources_still_use_their_explicit_queue_budget():
    queue=CompletionCommandQueue([command()],timeout_s=12)
    queue.select(0,0)
    with pytest.raises(CommandDispatchHalted):queue.select(0,12)
