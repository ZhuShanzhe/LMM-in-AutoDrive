import numpy as np
import pytest
from control.pedestrian_conditions import PedestrianClearanceMemory, ground_footpoint


ROAD = [[0., 0., 0.], [50., 0., 0.]]


def update(memory, time, people, request='request'):
    return memory.update(request, [dict(track_id=k, position=v) for k, v in people], ROAD,
        half_width=1., timestamp_s=time, sensor_frame=round(time*100))


def test_empty_image_cannot_prove_identity_left():
    m = PedestrianClearanceMemory()
    for t in np.arange(0, 1, .1):
        assert not update(m, float(t), [])['satisfied']


def test_positive_tracked_exit_is_required():
    m = PedestrianClearanceMemory()
    assert not update(m, 0., [('person', [10., 0., 0.])])['satisfied']
    for t in [.1, .2, .3, .4, .5]:
        assert not update(m, t, [('person', [10., 3., 0.])])['satisfied']
    assert update(m, .7, [('person', [10., 3., 0.])])['satisfied']


def test_occlusion_and_expiration_are_not_completion():
    m = PedestrianClearanceMemory()
    update(m, 0., [('person', [10., 0., 0.])])
    assert not update(m, .1, [])['satisfied']
    result = update(m, 31., [])
    assert not result['satisfied']
    assert 'position' not in result['target_evidence']['person']


def test_new_person_and_new_request_reset_clearance():
    m = PedestrianClearanceMemory()
    update(m, 0., [('person', [10., 0., 0.])])
    for t in [.1, .2, .3, .4, .5, .7]:
        update(m, t, [('person', [10., 3., 0.])])
    assert not update(m, .8, [('person', [10., 3., 0.]), ('other', [12., 0., 0.])])['satisfied']
    assert not update(m, .9, [], request='new')['satisfied']


def test_completed_event_outlives_position_but_requires_current_clearance():
    m=PedestrianClearanceMemory()
    update(m,0.,[('person',[10.,0.,0.])])
    for t in [.1,.2,.3,.4,.5,.7]:
        update(m,t,[('person',[10.,3.,0.])])
    update(m,1.,[])
    assert not m.result(1.)['satisfied']
    assert m.result(1.,identities=['person'],corridor_clear=True)['satisfied']
    update(m,31.,[])
    assert 'person' not in m.targets
    assert m.result(31.,identities=['person'],corridor_clear=True)['satisfied']
    fact=m.completed_events['person']
    assert 'position' not in fact and fact['event_status']=='COMPLETED'
    update(m,1.1,[('person',[10.,0.,0.])])
    assert not m.result(1.1,identities=['person'],corridor_clear=True)['satisfied']
    assert 'person' not in m.completed_events


def test_new_unresolved_crossing_blocks_even_with_old_completed_fact():
    m=PedestrianClearanceMemory()
    update(m,0.,[('person',[10.,0.,0.])])
    for t in [.1,.2,.3,.4,.5,.7]:update(m,t,[('person',[10.,3.,0.])])
    update(m,31.,[])
    assert m.result(31.,corridor_clear=True)['satisfied']
    update(m,31.1,[('new',[10.,0.,0.])])
    assert not m.result(31.1,corridor_clear=True)['satisfied']
    assert not m.result(31.1,identities=['person','never_observed'],corridor_clear=True)['satisfied']


def test_observer_does_not_age_tracker_on_repeated_sensor_frames():
    from types import SimpleNamespace as NS
    from control.pedestrian_conditions import PedestrianConditionObserver
    observer=PedestrianConditionObserver.__new__(PedestrianConditionObserver)
    calls=[]
    observer.detector=NS(detect=lambda rgb:([],0.))
    observer.tracker=NS(update=lambda values:calls.append('tracked') or [])
    observer.memory=PedestrianClearanceMemory()
    observer.last_sensor_frame=None;observer.observations=[];observer.condition_targets={}
    document=dict(request_id='wait',intent=dict(steps=[dict(step_id='step',action='WAIT',
        parameters=dict(condition='PEDESTRIAN_CLEAR'))]))
    for t in [0.,.05,.1]:
        observer.observe(document,np.zeros((100,100,3)),np.eye(4),ROAD,fov=90.,
            frame_id=str(t),sensor_frame=10,observed_at_s=0.,timestamp_s=t,half_width=1.)
    assert calls==['tracked']


def test_crossing_completion_cannot_prove_passenger_boarding():
    from types import SimpleNamespace as NS
    from control.pedestrian_conditions import PedestrianConditionObserver
    observer=PedestrianConditionObserver.__new__(PedestrianConditionObserver)
    observer.detector=NS(detect=lambda rgb:([],0.))
    observer.tracker=NS(update=lambda values:[])
    observer.memory=PedestrianClearanceMemory()
    observer.memory.completed_events={'person':dict(event_id='crossing',fact='OBSERVED_CORRIDOR_EXIT')}
    observer.last_sensor_frame=None;observer.observations=[];observer.condition_targets={}
    document=dict(request_id='bus',intent=dict(steps=[dict(step_id='wait',action='WAIT',
        parameters=dict(condition='PASSENGERS_CLEAR'))]))
    result=observer.observe(document,np.zeros((100,100,3)),np.eye(4),ROAD,fov=90.,
        frame_id='current',sensor_frame=10,observed_at_s=31.,timestamp_s=31.,half_width=1.,
        corridor_evidence=dict(satisfied=True,valid=True,frame_id='current'))['wait']
    assert not result['valid'] and not result['satisfied']
    assert result['reason']=='role_or_region_binding_required'
    assert not result['completed_event_facts']


@pytest.mark.parametrize('box', [[10., 10., 20., 49.], [10., 10., 20., 100.], [0., 0., 100., 99.]])
def test_horizon_or_clipped_footpoint_is_unknown(box):
    camera = np.eye(4)
    camera[2, 3] = -1.5
    assert ground_footpoint(box, camera, (100, 100), 90., ROAD) is None


def test_ground_ray_gives_metric_position_and_rejects_large_grade():
    camera = np.eye(4)
    camera[2, 3] = -1.5
    point = ground_footpoint([40., 60., 60., 75.], camera, (100, 100), 90., ROAD)
    assert point == pytest.approx([3., 0., 0.])
    slope_point = ground_footpoint([40., 60., 60., 75.], camera, (100, 100), 90., [[0., 0., 0.], [50., 0., 5.]])
    assert slope_point == pytest.approx([2.5, 0., .25], abs=.03)
    assert ground_footpoint([40., 60., 60., 75.], camera, (100, 100), 90., [[0., 0., 0.], [5., 0., 5.]]) is None
