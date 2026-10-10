import numpy as np
from control.passing_conditions import PassingMemory


def observe(memory, frame, foreground_speed, foreground_present=True):
    now=frame*.05
    measurements=[dict(view='front',local_id='far',category='vehicle',
        position=np.array([38.+3.*now,0.,1.]))]
    if foreground_present:
        measurements.append(dict(view='front',local_id='near',category='vehicle',
            position=np.array([18.+foreground_speed*now,0.,1.])))
    memory.update(measurements,frame=frame,timestamp_s=now,ego_xy=[0,0],heading=[1,0])
    document=dict(request_id='request',intent=dict(steps=[dict(step_id='pass',action='OVERTAKE',
        parameters=dict(condition='SLOW_VEHICLE'))]))
    return memory.proofs(document,frame_id=str(frame),frame=frame,timestamp_s=now,
        observed_at_s=now,ego_xy=[0,0],heading=[1,0],ego_rear_m=2.4,
        corridor_clear=True,target_binding_allowed=True)['pass']


def test_stopped_foreground_does_not_select_occluded_distant_vehicle():
    memory=PassingMemory()
    for frame in range(1,36):proof=observe(memory,frame,0.)
    assert proof['target_id'] is None
    assert not proof['satisfied']


def test_fast_foreground_does_not_select_distant_slow_vehicle():
    memory=PassingMemory()
    for frame in range(1,36):proof=observe(memory,frame,8.)
    assert proof['target_id'] is None


def test_moving_slow_foreground_is_selected():
    memory=PassingMemory()
    for frame in range(1,36):proof=observe(memory,frame,3.)
    assert proof['target_id']
    assert memory.tracks[proof['target_id']]['ahead_m']<30.
    assert not proof['satisfied']


def test_distant_target_available_when_no_foreground_is_observed():
    memory=PassingMemory()
    for frame in range(1,36):proof=observe(memory,frame,0.,False)
    assert proof['target_id']
    assert not proof['satisfied']
