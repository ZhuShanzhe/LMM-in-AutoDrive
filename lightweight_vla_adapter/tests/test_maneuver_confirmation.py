"""Historical completion must be physical, bound and not a future event."""
import pytest
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from scene2_runtime_interface import _scene2_step_contract


@pytest.mark.parametrize('kind', ['JUNCTION', 'LANE_CHANGE'])
def test_configured_complete_is_an_explicit_confirmation(kind):
    action, parameters, _, _ = _scene2_step_contract('COMPLETE:' + kind)
    assert action == 'CONFIRM'
    assert parameters['condition'] == kind + '_COMPLETED'


@pytest.mark.parametrize('kind', ['JUNCTION_COMPLETED', 'LANE_CHANGE_COMPLETED'])
@pytest.mark.parametrize('failure', [None, 'missing', 'future', 'wrong_source', 'wrong_kind'])
def test_completion_confirmation_has_bound_physical_fact(kind, failure):
    runtime = DrivingPlanRuntime()
    parameters = dict(condition=kind)
    if failure == 'wrong_source':
        parameters['source_request_id'] = 'unrelated'
    runtime.document = dict(request_id='new', intent=dict(steps=[dict(
        step_id='confirm', action='CONFIRM', parameters=parameters)]))
    runtime.state = dict(plan_status='ACTIVE', active_step_index=0)
    if failure != 'missing':
        runtime.completed_maneuver_facts.append(dict(request_id='previous', step_id='maneuver',
            condition='OTHER' if failure == 'wrong_kind' else kind, frame_id='f-old',
            observed_at_s=100. if failure == 'future' else 1., source='held_physical_execution_feedback'))
    for i in range(14):
        now=2. + i*.05
        frame=f'f{i}'
        runtime.frame_id=frame
        runtime.timestamp_s=now
        runtime._observe_condition_completion(dict(frame_id=frame), dict(frame_id=frame, risk_level='low'))
    assert bool(runtime.feedback) == (failure is None)


def test_missing_confirmation_cannot_bind_an_event_that_happens_later():
    runtime = DrivingPlanRuntime()
    runtime.document = dict(request_id='new', intent=dict(steps=[dict(
        step_id='confirm', action='CONFIRM', parameters=dict(condition='JUNCTION_COMPLETED'))]))
    runtime.state = dict(plan_status='ACTIVE', active_step_index=0)
    for i in range(20):
        now=i*.05
        frame=f'f{i}'
        runtime.frame_id=frame
        runtime.timestamp_s=now
        if i==2:
            runtime.completed_maneuver_facts.append(dict(request_id='future',step_id='junction',
                condition='JUNCTION_COMPLETED',frame_id=frame,observed_at_s=now,
                source='held_physical_execution_feedback'))
        runtime._observe_condition_completion(dict(frame_id=frame),dict(frame_id=frame,risk_level='low'))
    assert runtime.feedback is None
