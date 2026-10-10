import pytest

from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.tests.fixtures import integration_documents


@pytest.mark.parametrize('failure', [None, 'missing', 'wrong_frame', 'wrong_source',
                                   'too_far', 'early_progress', 'off_corridor', 'not_reached'])
def test_finish_requires_current_measured_endpoint_and_lane_geometry(failure):
    document, _, _, _ = integration_documents(parser_action='PROCEED')
    step = document['intent']['steps'][0]
    step.update(parameters=dict(condition='FINISH'), completion=dict(type='ACTION_REACHED'))
    runtime = DrivingPlanRuntime()
    runtime.prepare(document, frame_id='f0', timestamp_s=0., speed_mps=5.)
    runtime.state = dict(plan_status='ACTIVE', active_step_index=0,
                         step_states=[dict(status='ACTIVE')])
    goal = dict(valid=True, source='measured_ego_public_route_endpoint', frame_id='f1',
                reached=True, route_end_m=8002., route_progress_m=8000., endpoint_distance_m=2.)
    execution = dict(observation_frame_id='f1', source_step_id=step['step_id'],
                     navigation_goal=goal, pid=dict(lateral_error_m=.1,
                     heading_error_deg=1., emergency_latched=False))
    if failure == 'missing': execution.pop('navigation_goal')
    if failure == 'wrong_frame': goal['frame_id'] = 'old'
    if failure == 'wrong_source': goal['source'] = 'estimated_from_time'
    if failure == 'too_far': goal['endpoint_distance_m'] = 4.
    if failure == 'early_progress': goal['route_progress_m'] = 7998.
    if failure == 'off_corridor': execution['pid']['lateral_error_m'] = .4
    if failure == 'not_reached': goal['reached'] = False
    runtime.prepare(document, frame_id='f1', timestamp_s=.05, speed_mps=5.,
                    execution_state=execution)
    assert bool(runtime.feedback) == (failure is None)
