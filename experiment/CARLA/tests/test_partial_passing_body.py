import numpy as np
import pytest
from control.passing_conditions import PassingMemory,lift_box_body


def step(memory,frame,*,partial,view,ego_speed=5.):
    now=frame*.05;x=20.+3.*now
    body=np.array([[x,0.,1.],[x+1.,0.,1.],[x,.2,1.],[x+1.,.2,1.]])
    memory.update([dict(view=view,local_id=view,category='vehicle',position=body.mean(0),
        surface=body,box_cropped=partial)],frame=frame,timestamp_s=now,
        ego_xy=[ego_speed*now,0.],heading=[1.,0.])
    return memory.proofs(dict(request_id='test',intent=dict(steps=[
        dict(step_id='pass',action='OVERTAKE',parameters=dict(condition='SLOW_VEHICLE'))])),
        frame_id=f'f{frame}',frame=frame,timestamp_s=now,observed_at_s=now,
        ego_xy=[ego_speed*now,0.],heading=[1.,0.],ego_rear_m=2.4,corridor_clear=True)['pass']


def test_partial_body_does_not_acquire_a_new_referent():
    m=PassingMemory()
    for frame in range(1,100):p=step(m,frame,partial=True,view='front',ego_speed=0.)
    assert p['target_id'] is None and not p['satisfied']


def test_partial_body_maintains_identity_without_proving_completion():
    m=PassingMemory()
    for frame in range(1,100):p=step(m,frame,partial=False,view='front')
    original=p['target_id'];assert original
    for frame in range(100,450):
        view='right' if frame<240 else 'rear'
        p=step(m,frame,partial=True,view=view)
        assert p['target_id']==original and p['target_observation']['measured_in_current_frame']
        assert p['target_observation']['measured_rear_clearance_m'] is None
        assert not p['satisfied']
    for frame in range(450,480):p=step(m,frame,partial=False,view='rear')
    assert p['satisfied'] and p['completed_event_fact']['target_id']==original


@pytest.mark.parametrize('allow,expected', [(False,False),(True,True)])
def test_partial_projection_requires_explicit_tracking_permission(allow,expected):
    body=np.array([[10.,y,z] for y in np.linspace(-8.,-4.,16) for z in np.linspace(.5,2.,8)])
    value=lift_box_body(body,np.eye(4),(640,640),90.,[0,250,200,320],allow_partial=allow)
    assert (value is not None)==expected


@pytest.mark.parametrize('box', [[-1,200,40,300],[0,200,641,300],[0,float('nan'),40,300],[0,200,0,300]])
def test_partial_projection_does_not_accept_invalid_box(box):
    assert lift_box_body(np.ones((20,3)),np.eye(4),(640,640),90.,box,allow_partial=True) is None


def test_partial_empty_or_ambiguous_measurements_are_not_supported_bodies():
    assert lift_box_body(np.empty((0,3)),np.eye(4),(640,640),90.,[0,200,200,320],allow_partial=True) is None
    first=np.array([[10.,-6.,1.] for _ in range(8)])
    second=np.array([[30.,-18.,3.] for _ in range(8)])
    assert lift_box_body(np.r_[first,second],np.eye(4),(640,640),90.,[0,250,200,320],allow_partial=True) is None
