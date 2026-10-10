from unittest.mock import Mock
import numpy as np
from control.passing_conditions import PassingConditionObserver, PassingMemory


def test_historical_pass_confirmation_does_not_rerun_visual_classification():
    observer=PassingConditionObserver.__new__(PassingConditionObserver)
    observer.detector=Mock();observer.trackers={};observer.memory=PassingMemory()
    observer.last_sensor_frame=None;observer.artifact_dir=None
    observer.sparse_cyclist_support={}
    observer.memory.facts[('previous','pass')]=dict(category='cyclist',target_id='c1',
        completed_at_s=1.,source_request_id='previous',source_step_id='pass')
    document=dict(request_id='confirm',intent=dict(steps=[dict(step_id='check',
        action='CONFIRM',parameters=dict(condition='CYCLIST_PASSED'),depends_on=[])]))
    kwargs=dict(frame_id='f100',sensor_frame=100,observed_at_s=5.,timestamp_s=5.,
        ego_xy=[0.,0.],heading=[1.,0.],ego_rear_m=2.4,corridor_clear=True,scan=False)
    proof=observer.observe(document,{},np.empty((0,3)),[],**kwargs)['check']
    assert proof['satisfied'] and proof['valid']
    observer.detector.detect.assert_not_called()
    kwargs.update(sensor_frame=101,observed_at_s=5.05,timestamp_s=5.05)
    proof=observer.observe(document,{},None,[],**kwargs)['check']
    assert not proof['satisfied'] and not proof['valid']


def test_no_scan_cannot_create_a_new_pass_completion():
    observer=PassingConditionObserver.__new__(PassingConditionObserver)
    observer.detector=Mock();observer.trackers={};observer.memory=PassingMemory()
    observer.last_sensor_frame=None;observer.artifact_dir=None
    observer.sparse_cyclist_support={}
    document=dict(request_id='new',intent=dict(steps=[dict(step_id='pass',
        action='PASS_BY',parameters=dict(condition='CYCLIST'))]))
    proof=observer.observe(document,{},np.empty((0,3)),[],frame_id='f100',sensor_frame=100,
        observed_at_s=5.,timestamp_s=5.,ego_xy=[0.,0.],heading=[1.,0.],
        ego_rear_m=2.4,corridor_clear=True,scan=False)['pass']
    assert not proof['satisfied'] and not observer.memory.facts
