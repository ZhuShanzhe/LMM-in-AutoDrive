import numpy as np
from control.passing_conditions import PassingMemory,road_relative_reference


def test_bicycle_ahead_on_curve_is_not_an_off_road_target():
    road=np.array([[0.,0.,0.],[20.,0.,0.],[20.,20.,0.]])
    reference=road_relative_reference([20.,15.],[10.,0.],[1.,0.],road)
    assert reference['ahead_m']==25.
    assert reference['lateral_m']==0.
    memory=PassingMemory()
    measured=dict(category='cyclist',position=np.array([20.,15.,1.]),view='front',local_id='observed',
        box_cropped=False,bbox_xyxy=[20,20,40,60],surface=np.array([[20.,15.,1.]]*4))
    memory.update([measured],frame=10,timestamp_s=1.,ego_xy=[10.,0.],heading=[1.,0.],road=road)
    document=dict(request_id='r',intent=dict(steps=[dict(step_id='p',action='PASS_BY',parameters={'condition':'CYCLIST'})]))
    proof=memory.proofs(document,frame_id='f',frame=10,timestamp_s=1.,observed_at_s=1.,
        ego_xy=[10.,0.],heading=[1.,0.],ego_rear_m=2.4,corridor_clear=True)['p']
    assert proof['target_id']=='measured-body-1'
    assert not proof['satisfied']


def test_ambiguous_repeated_corridor_does_not_make_a_road_reference():
    road=np.array([[-20.,0.],[20.,0.],[20.,20.],[-20.,20.],[-20.,0.],[20.,0.]])
    assert road_relative_reference([0.,0.],[-10.,0.],[1.,0.],road) is None


def test_opposite_heading_or_unobserved_route_extension_is_rejected():
    road=np.array([[0.,0.],[20.,0.],[40.,0.]])
    assert road_relative_reference([20.,0.],[10.,0.],[-1.,0.],road) is None
    assert road_relative_reference([43.,0.],[10.,0.],[1.,0.],road) is None
