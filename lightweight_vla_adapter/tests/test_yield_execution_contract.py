"""Yield approach and completion need current physical observation."""
from copy import deepcopy
import pytest

from scene_understanding.src.control_decision import _map_step_action
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.tests.fixtures import integration_documents


@pytest.mark.parametrize('speed', [0., 5., 30.])
def test_yield_approach_does_not_inherit_a_traffic_stop(speed):
    action, target, lane, location = _map_step_action(
        dict(action='YIELD', parameters={'condition': 'PEDESTRIAN_CLEAR'}), speed)
    assert (action, target, lane, location) == ('decelerate', 10., None, None)


@pytest.mark.parametrize('parameters,target', [
    ({'target_speed_mps': 2.}, 7.2), ({'target_speed_kmh': 5.}, 5.),
    ({'target_speed_kmh': 0.}, 0.), ({'target_speed_kmh': 50.}, 10.)])
def test_explicit_yield_speed_is_respected(parameters, target):
    assert _map_step_action(dict(action='YIELD', parameters=parameters), 0.)[1] == target


@pytest.mark.parametrize('fault', [None, 'not_clear', 'stale', 'wrong_step', 'gap', 'unsafe'])
def test_yield_completes_only_after_continuous_step_bound_clearance(fault):
    d, w, a, r = integration_documents(parser_action='YIELD')
    first = d['intent']['steps'][0]
    first.update(parameters={'condition': 'PEDESTRIAN_CLEAR'}, completion={'type': 'ACTION_REACHED'})
    second = deepcopy(first)
    second.update(step_id='step_2', action='STOP', parameters={},
                  depends_on=[first['step_id']], completion={'type': 'VEHICLE_STOPPED'})
    d['intent']['steps'].append(second)
    runtime = DrivingPlanRuntime()
    for i in range(10):
        now = i * (.5 if fault == 'gap' else .1)
        frame = f'f{i}'
        w['frame_id'] = r['frame_id'] = frame
        evidence = dict(request_id=d['request_id'], step_id=first['step_id'], frame_id=frame,
            condition='PEDESTRIAN_CLEAR', satisfied=fault != 'not_clear', valid=True,
            observed_at_s=now, valid_until_s=now+.25, source='rgb_identity_exit')
        if fault == 'stale': evidence['observed_at_s'] = now-1.
        if fault == 'wrong_step': evidence['step_id'] = 'wrong'
        if fault == 'unsafe':
            r.update(risk_level='high', recommended_action='emergency_brake')
        runtime.prepare(d, frame_id=frame, timestamp_s=now, speed_mps=0.)
        runtime.advance(w, r, readiness={first['step_id']: evidence})
    assert (runtime.state['step_states'][0]['status'] == 'COMPLETED') == (fault is None)
