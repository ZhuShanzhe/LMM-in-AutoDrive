from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime


def runtime():
    plan=DrivingPlanRuntime()
    plan.document=dict(request_id='active')
    plan.state=dict(plan_status='ACTIVE',active_step_index=0,active_step_id='step_1',
        step_states=[dict(step_id='step_1',status='ACTIVE')])
    return plan


def test_dispatcher_failure_closes_only_matching_active_plan():
    plan=runtime()
    assert not plan.terminate('other','TIMEOUT')
    assert plan.state['plan_status']=='ACTIVE'
    assert plan.terminate('active','ACTIVE_COMMAND_WINDOW_EXPIRED')
    assert plan.state['plan_status']=='FAILED'
    assert plan.state['active_step_index'] is None
    assert plan.state['step_states'][0]['status']=='FAILED'
    assert not plan.terminate('active','TIMEOUT')


def test_completed_plan_is_not_rewritten_as_failure():
    plan=runtime()
    plan.state['plan_status']='COMPLETED'
    assert not plan.terminate('active','TIMEOUT')
    assert plan.state['plan_status']=='COMPLETED'
