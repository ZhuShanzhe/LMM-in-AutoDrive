"""A new yield cannot consume an unrelated historical crossing."""
from types import SimpleNamespace as NS
import numpy as np
from control.pedestrian_conditions import PedestrianClearanceMemory, PedestrianConditionObserver
from scene2_runtime_interface import _scene2_step_contract


def test_configured_yield_uses_supported_pedestrian_predicate():
    action, params, policy, _ = _scene2_step_contract('YIELD:PEDESTRIAN')
    assert action == 'YIELD' and params['condition'] == 'PEDESTRIAN_CLEAR'
    assert policy == 'WAIT_FOR_SAFE'


def test_configured_relative_speed_requires_speed_completion():
    action, params, _, completion = _scene2_step_contract('ADJUST_SPEED:DECREASE')
    assert action == 'ADJUST_SPEED' and params == {'change': 'DECREASE'}
    assert completion['type'] == 'TARGET_SPEED_REACHED'


def test_new_yield_requires_a_current_observed_target():
    observer = PedestrianConditionObserver.__new__(PedestrianConditionObserver)
    observer.detector = NS(detect=lambda rgb: ([], 0.))
    observer.tracker = NS(update=lambda values: [])
    observer.memory = PedestrianClearanceMemory()
    observer.memory.completed_events = {'old': dict(fact='OBSERVED_CORRIDOR_EXIT')}
    observer.last_sensor_frame = None
    observer.observations = []
    observer.condition_targets = {}
    doc = dict(request_id='new-yield', intent=dict(steps=[dict(step_id='yield', action='YIELD',
        parameters=dict(condition='PEDESTRIAN_CLEAR'))]))
    result = observer.observe(doc, np.zeros((100, 100, 3)), np.eye(4),
        [[0., 0., 0.], [50., 0., 0.]], fov=90., frame_id='f10', sensor_frame=10,
        observed_at_s=1., timestamp_s=1., half_width=1.,
        corridor_evidence=dict(satisfied=True, valid=True, frame_id='f10'))['yield']
    assert result['valid'] and not result['satisfied']
    assert result['reason'] == 'pedestrian_identity_not_observed'
    assert not result['completed_event_facts']
