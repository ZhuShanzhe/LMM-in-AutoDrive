import numpy as np
import pytest
from control.passing_conditions import (PassingMemory,lift_box_body,object_surface_points,
    sparse_body_consensus)


def document(action='OVERTAKE',condition='SLOW_VEHICLE',request='test'):
    return dict(request_id=request,intent=dict(steps=[dict(step_id='pass',action=action,
        parameters=dict(condition=condition))]))


def measurement(x, *, view='front',category='vehicle',identity='one'):
    return dict(view=view,local_id=identity,category=category,position=np.array([x,0.,1.]),
        surface=np.array([[x+1.,-.2,1.],[x+1.,.2,1.],[x,.2,1.],[x,-.2,1.]]))


def update(memory,frame,objects,*,ego_x=0.,document=None,clear=True):
    now=frame*.05
    memory.update(objects,frame=frame,timestamp_s=now,ego_xy=[ego_x,0.],heading=[1.,0.])
    return memory.proofs(document or globals()['document'](),frame_id=f'f{frame}',frame=frame,
        timestamp_s=now,observed_at_s=now,ego_xy=[ego_x,0.],heading=[1.,0.],
        ego_rear_m=2.4,corridor_clear=clear)['pass']


def test_real_ahead_to_rear_body_measurements_complete_after_continuous_hold():
    m=PassingMemory();proof=None
    for frame in range(1,420):
        now=frame*.05;target=20.+3*now;ego=5*now
        proof=update(m,frame,[measurement(target,view='rear' if target<ego-4 else 'front')],ego_x=ego)
    assert proof['satisfied']
    assert proof['completed_event_fact']['measured_rear_clearance_m']>=8
    assert len({fact['target_id'] for fact in m.facts.values()})==1


@pytest.mark.parametrize('origin,expected',[(None,False),('LEFT',False),('RIGHT',True)])
def test_adjacent_slow_target_requires_an_observed_origin_lane(origin,expected):
    memory=PassingMemory();doc=document()
    for frame in range(1,35):
        now=frame*.05;m=measurement(20.+3.*now)
        m['position'][1]=3.5;m['surface'][:,1]+=3.5
        memory.update([m],frame=frame,timestamp_s=now,ego_xy=[0.,0.],heading=[1.,0.])
        proof=memory.proofs(doc,frame_id=f'f{frame}',frame=frame,timestamp_s=now,
            observed_at_s=now,ego_xy=[0.,0.],heading=[1.,0.],ego_rear_m=2.4,
            corridor_clear=True,passing_origin_side=origin)['pass']
    assert bool(memory.bound)==expected
    assert not proof['satisfied']


@pytest.mark.parametrize('failure',['disappeared','rear_only','too_close','unsafe','ambiguous'])
def test_missing_or_unsafe_passing_evidence_never_completes(failure):
    m=PassingMemory();proof=None
    for frame in range(1,45):
        objects=[measurement(20+frame*.15)]
        if failure=='rear_only':objects=[measurement(-20.,view='rear')]
        if failure=='ambiguous':objects.append(measurement(22+frame*.15,identity='two'))
        proof=update(m,frame,objects)
    for frame in range(45,80):
        objects=[] if failure=='disappeared' else [measurement(20+frame*.15,view='rear')]
        ego=60. if failure in {'unsafe','rear_only','ambiguous'} else 20+frame*.15+8
        proof=update(m,frame,objects,ego_x=ego,clear=failure!='unsafe')
    assert not proof['satisfied']


def test_identity_is_not_rebound_after_thirty_second_occlusion():
    m=PassingMemory()
    for frame in range(1,35):update(m,frame,[measurement(20+frame*.15)])
    target=m.bound[('test','pass')]['target_id']
    for frame in range(35,680):update(m,frame,[])
    assert target not in m.tracks
    proof=update(m,681,[measurement(-20.,view='rear',identity='new')],ego_x=0.)
    assert not proof['satisfied']


def test_duplicate_sensor_frame_does_not_destroy_real_confirmation_history():
    m=PassingMemory()
    for frame in range(1,440):
        now=frame*.05;target=20+3*now;ego=5*now
        values=[measurement(target,view='rear' if target<ego-4 else 'front')]
        update(m,frame,values,ego_x=ego)
        proof=update(m,frame,values,ego_x=ego)
    assert proof['satisfied']


@pytest.mark.parametrize('acceleration',[1.,2.])
def test_startup_traffic_is_not_a_persistent_slow_passing_target(acceleration):
    memory=PassingMemory()
    for frame in range(1,32):
        now=frame*.05
        proof=update(memory,frame,[measurement(20.+now+.5*acceleration*now*now)])
    assert not memory.bound and not proof['satisfied']


@pytest.mark.parametrize('angle',[0.,.4,1.2])
def test_persistent_slow_qualification_is_not_world_axis_specific(angle):
    memory=PassingMemory();heading=np.array([np.cos(angle),np.sin(angle)])
    for frame in range(1,35):
        now=frame*.05;m=measurement(20.+3.*now)
        m['position'][:2]=heading*(20.+3.*now)
        memory.update([m],frame=frame,timestamp_s=now,ego_xy=[0.,0.],heading=heading)
    assert memory.stable_slow_motion(next(iter(memory.tracks.values())),now,heading,30.)


def test_short_or_gapped_motion_cannot_qualify_slow_target():
    memory=PassingMemory()
    for frame in range(1,10):update(memory,frame,[measurement(20.+frame*.15)])
    assert not memory.bound
    update(memory,19,[measurement(20.+19*.15)])
    assert not memory.bound


def test_real_image_box_requires_lidar_body_not_ground_or_empty_space():
    road=np.array([[-30.,0.,0.],[60.,0.,0.]])
    ground=np.array([[x,0.,0.] for x in np.linspace(5,30,40)])
    body=np.array([[10.,y,z] for y in np.linspace(-.8,.8,8) for z in np.linspace(.5,1.5,8)])
    surface=object_surface_points(np.r_[ground,body],road)
    assert len(surface)==len(body)
    result=lift_box_body(surface,np.eye(4),(640,640),90.,[280,260,360,325])
    assert result is not None
    assert lift_box_body(np.empty((0,3)),np.eye(4),(640,640),90.,[280,260,360,325]) is None
    assert lift_box_body(surface,np.eye(4),(640,640),90.,[0,260,360,325]) is None


def test_sparse_cyclist_lidar_support_is_available_only_as_an_explicit_mode():
    points=np.array([[10.,-.05,1.],[10.,.05,1.]])
    assert lift_box_body(points,np.eye(4),(640,640),90.,[300,275,340,305]) is None
    body=lift_box_body(points,np.eye(4),(640,640),90.,[300,275,340,305],
        allow_partial=True,sparse_cyclist=True)
    assert body is not None and len(body)==2


def test_sparse_body_requires_three_consecutive_consistent_scans():
    history=[]
    result=False
    for index in range(3):
        center=index*.15
        points=np.array([[center,-.1,1.],[center,.1,1.]])
        history,result=sparse_body_consensus(history,points,.05*(index+1))
    assert result and len(history)==3
    history,result=sparse_body_consensus(history,
        np.array([[.3,-.1,1.],[.3,.1,1.]]),.3)
    assert not result and len(history)==1


def test_sparse_body_consensus_rejects_unstable_returns():
    history=[]
    for index,center in enumerate((0.,.1,1.2)):
        history,result=sparse_body_consensus(history,
            np.array([[center,-.1,1.],[center,.1,1.]]),.05*(index+1))
    assert not result


def test_two_point_rear_body_never_proves_cyclist_pass_completion():
    memory=PassingMemory();doc=document('PASS_BY','CYCLIST');proof=None
    for frame in range(1,420):
        now=frame*.05;target=20.+3*now;ego=5*now
        view='rear' if target<ego-4 else 'front'
        item=measurement(target,view=view,category='cyclist')
        if view=='rear':item['surface']=item['surface'][:2]
        proof=update(memory,frame,[item],ego_x=ego,document=doc)
    assert memory.bound
    assert not proof['satisfied']
    assert not memory.facts


def test_three_frame_confirmed_sparse_cyclist_rear_body_can_complete_with_ten_meter_clearance():
    memory=PassingMemory();doc=document('PASS_BY','CYCLIST');proof=None
    for frame in range(1,460):
        now=frame*.05;target=20.+3*now;ego=5*now
        view='rear' if target<ego-4 else 'front'
        item=measurement(target,view=view,category='cyclist')
        if view=='rear':
            item['surface']=item['surface'][:2]
            item['body_support_mode']='temporally_confirmed_sparse'
            item['body_support_frames']=3
        proof=update(memory,frame,[item],ego_x=ego,document=doc)
    fact=proof['completed_event_fact']
    assert proof['satisfied']
    assert fact['rear_body_support']=='temporally_confirmed_sparse'
    assert fact['required_rear_clearance_m']==10.
    assert fact['measured_rear_clearance_m']>=10.


def test_sparse_cyclist_clearance_between_eight_and_ten_meters_does_not_complete():
    memory=PassingMemory();doc=document('PASS_BY','CYCLIST');proof=None
    for frame in range(1,350):
        now=frame*.05;target=20.+3*now;ego=5*now
        view='rear' if target<ego-4 else 'front'
        item=measurement(target,view=view,category='cyclist')
        if view=='rear':
            item['surface']=item['surface'][:2]
            item['body_support_mode']='temporally_confirmed_sparse'
            item['body_support_frames']=3
        proof=update(memory,frame,[item],ego_x=ego,document=doc)
    measured=proof['target_observation']['measured_rear_clearance_m']
    assert measured is None or measured<10.
    assert not proof['satisfied']
    assert not memory.facts
