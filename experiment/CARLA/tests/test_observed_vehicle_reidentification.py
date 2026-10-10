from collections import deque
import numpy as np
import pytest
from control.observed_vehicle_reidentification import body_appearance, occlusion_score, road_prediction, road_coordinates
from control.passing_conditions import PassingMemory


def fixtures():
    road=np.column_stack([np.arange(-30., 100.),np.zeros(130),np.zeros(130)])
    appearance=np.zeros(144);appearance[20]=1
    track=dict(category='vehicle',position=np.array([10.,3.]),velocity=np.array([4.,0.]),
        last_seen_s=1.,velocity_valid=True,history=deque([(i/10,np.array([6+4*i/10,3.])) for i in range(11)]),
        appearances=deque([appearance]))
    measurement=dict(category='vehicle',position=np.array([22.,3.]),box_cropped=False,
        appearance=appearance,view='front',local_id='new-identity',surface=np.array([[22,3,1]]*4))
    return road,track,measurement


def test_unique_fresh_appearance_and_road_support_can_recover_after_occlusion():
    road,t,m=fixtures()
    assert occlusion_score(t,m,4.,road)==pytest.approx(0)


@pytest.mark.parametrize('change', ['age','crop','color','position','velocity','history','category','missing_color'])
def test_recovery_rejects_insufficient_or_contradictory_evidence(change):
    road,t,m=fixtures();timestamp=4.
    if change=='age':timestamp=9.01
    if change=='crop':m['box_cropped']=True
    if change=='color':m['appearance']=np.roll(m['appearance'],20)
    if change=='position':m['position']+=np.array([0,2.])
    if change=='velocity':t['velocity_valid']=False
    if change=='history':t['history']=deque([(1.,t['position'])])
    if change=='category':m['category']='cyclist'
    if change=='missing_color':m['appearance']=None
    assert occlusion_score(t,m,timestamp,road) is None


def test_no_extrapolation_past_route_end():
    road,t,m=fixtures();t['position']=np.array([95.,3.])
    assert road_prediction(t,road,3.) is None


def test_bounded_acceleration_during_occlusion_requires_matching_appearance():
    road,t,m=fixtures();m['position']=np.array([40.,3.])
    assert occlusion_score(t,m,7.,road) is not None
    m['position']=np.array([60.,3.])
    assert occlusion_score(t,m,7.,road) is None


def test_established_identity_survives_two_short_reappearances_but_velocity_must_be_fresh_valid():
    road,t,m=fixtures();memory=PassingMemory();memory.tracks['old']=t
    memory.update([m],frame=100,timestamp_s=4.,ego_xy=(0,0),heading=(1,0),road=road)
    recovered=memory.tracks['old']
    assert recovered['motion_history_established'] is True
    assert len(recovered['history'])==1
    recovered['velocity_valid']=True;recovered['velocity']=np.array([4.,0.])
    second=dict(m,position=np.array([34.,3.]),local_id='another-view-identity')
    assert occlusion_score(recovered,second,7.,road) is not None
    recovered['velocity_valid']=False
    assert occlusion_score(recovered,second,7.,road) is None


def test_recovered_identity_needs_new_measurement_and_does_not_create_completion_fact():
    road,t,m=fixtures();memory=PassingMemory();memory.tracks['old']=t
    memory.update([m],frame=100,timestamp_s=4.,ego_xy=(0,0),heading=(1,0),road=road)
    assert memory.aliases[('front','new-identity')]=='old'
    assert memory.tracks['old']['last_seen_s']==4.
    assert memory.tracks['old']['reidentified_gap_s']==3.
    assert memory.facts=={}
    assert not memory.tracks['old']['velocity_valid']


def test_ambiguous_same_color_objects_are_not_reidentified():
    road,t,m=fixtures();memory=PassingMemory();memory.tracks['a']=t
    other=dict(t);other['position']=t['position']+np.array([.1,0])
    memory.tracks['b']=other
    memory.update([m],frame=100,timestamp_s=4.,ego_xy=(0,0),heading=(1,0),road=road)
    assert ('front','new-identity') not in memory.aliases
    assert memory.facts=={}


def test_color_uses_only_neighborhoods_of_measured_body_pixels():
    image=np.zeros((100,100,3),dtype=np.uint8);image[:]=[50,150,30]
    points=np.array([[10., y, z] for y in (-1,0,1) for z in (-1,0,1)])
    descriptor=body_appearance(image,points,np.eye(4),90.)
    assert descriptor is not None and np.linalg.norm(descriptor)==pytest.approx(1.)
    assert body_appearance(image,points[:3],np.eye(4),90.) is None


def test_route_prediction_follows_observed_road_turn_not_world_constant_velocity():
    road=np.array([[0,0,0],[10,0,0],[20,0,0],[20,10,0],[20,20,0]])
    t=dict(position=np.array([15.,0.]),velocity=np.array([5.,0.]))
    point,tangent=road_prediction(t,road,2.)
    np.testing.assert_allclose(point,[20,5]);np.testing.assert_allclose(tangent,[0,1])


def test_far_adjacent_vehicle_on_curve_is_not_ego_tangent_lateral_distance():
    angles=np.linspace(-.4,1.,100);radius=60.
    road=np.column_stack([radius*np.sin(angles),radius*(1-np.cos(angles)),np.zeros(len(angles))])
    angle=.5
    point=np.array([radius*np.sin(angle),radius*(1-np.cos(angle))])+4*np.array([-np.sin(angle),np.cos(angle)])
    geometry=road_coordinates(point,road)
    assert geometry is not None
    assert point[1]>10
    assert geometry[-1]==pytest.approx(4.,abs=.02)


def test_passing_proof_is_json_serializable_with_numpy_motion_validity():
    import json
    memory=PassingMemory()
    document={'request_id':'serialization','intent':{'steps':[{'step_id':'pass','action':'OVERTAKE','parameters':{'condition':'SLOW_VEHICLE'}}]}}
    for frame in range(1,35):
        now=frame*.05;x=20.+3.*now
        measurement=dict(view='front',local_id='one',category='vehicle',position=np.array([x,0.,1.]),
            surface=np.array([[x+1,-.2,1],[x+1,.2,1],[x,.2,1],[x,-.2,1]]))
        memory.update([measurement],frame=frame,timestamp_s=now,ego_xy=[0,0],heading=[1,0])
        proofs=memory.proofs(document,frame_id=f'f{frame}',frame=frame,timestamp_s=now,observed_at_s=now,
            ego_xy=[0,0],heading=[1,0],ego_rear_m=2.4,corridor_clear=True)
        json.dumps(proofs,allow_nan=False)
    assert isinstance(proofs['pass']['target_observation']['velocity_valid'],bool)
