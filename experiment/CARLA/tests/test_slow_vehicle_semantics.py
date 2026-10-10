import numpy as np
import pytest

from control.passing_conditions import PassingMemory


def acquire(speed_kmh, cruise_kmh):
    memory=PassingMemory()
    doc=dict(request_id='generic',intent=dict(steps=[dict(step_id='pass',action='OVERTAKE',
        parameters=dict(condition='SLOW_VEHICLE'))]))
    for frame in range(1,35):
        time=frame*.05;x=20.+speed_kmh/3.6*time
        point=dict(view='front',local_id='vehicle',category='vehicle',position=np.array([x,0.,1.]),
            surface=np.array([[x,0.,1.],[x+.3,0.,1.],[x,0.,1.1],[x+.3,0.,1.1]]))
        memory.update([point],frame=frame,timestamp_s=time,ego_xy=[0.,0.],heading=[1.,0.])
        proof=memory.proofs(doc,frame_id=f'f{frame}',frame=frame,timestamp_s=time,
            observed_at_s=time,ego_xy=[0.,0.],heading=[1.,0.],ego_rear_m=2.4,
            corridor_clear=True,cruise_speed_kmh=cruise_kmh)['pass']
    return proof


@pytest.mark.parametrize('cruise', [20.,30.,40.,60.,80.])
@pytest.mark.parametrize('ratio,expected', [(.4,True),(.65,True),(.69,True),(.75,False),(.9,False),(1.1,False)])
def test_slow_is_relative_to_desired_cruise_not_fixed_actor_speed(cruise,ratio,expected):
    proof=acquire(cruise*ratio,cruise)
    assert bool(proof['target_id']) is expected
    assert not proof['satisfied']


@pytest.mark.parametrize('ratio', [0.,1.,-1.,float('nan'),float('inf'),True,None])
def test_invalid_slow_ratio_rejected(ratio):
    with pytest.raises(ValueError):PassingMemory(slow_speed_ratio=ratio)


def test_stationary_vehicle_not_bound_as_slow_moving_vehicle():
    assert acquire(0.,30.)['target_id'] is None
