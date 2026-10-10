from types import SimpleNamespace as NS

import pytest

from benchmark.catalog import ConfigError
from benchmark.episode import pack_actor
from benchmark.role_journal import RoleJournal
from benchmark.role_journal import configured_role_anchors
from benchmark.truth_capture import RouteProjector


def actor(identity=2, role='pedestrian', kind='walker.pedestrian.0001'):
    return NS(id=identity,attributes={'role_name':role},type_id=kind,
              bounding_box=NS(extent=NS(x=.3,y=.3),location=NS(x=0,y=0),rotation=NS(yaw=0)))


def capture(journal, metadata, identities):
    frozen=NS(get_transform=lambda:NS(location=NS(x=10,y=0,z=0),rotation=NS(yaw=0)),
              get_velocity=lambda:NS(x=1,y=0,z=0))
    snapshot=NS(find=lambda identity:frozen if identity in identities else None)
    route=[dict(x=0,y=0,z=0,distance_m=0),dict(x=100,y=0,z=0,distance_m=100)]
    return journal.capture(snapshot,metadata,RouteProjector(route),0,pack_actor)


def journal():
    return RoleJournal([{'steps':[{'kind':'yield_pedestrian','target_role':'pedestrian'}]}])


def test_delayed_spawn_and_frozen_kinematics():
    recorder=journal()
    states, packed=capture(recorder,[],[])
    assert states['pedestrian']['status']=='NOT_SPAWNED'
    assert not packed
    states, packed=capture(recorder,[actor()],[2])
    assert states['pedestrian']['status']=='BOUND'
    assert states['pedestrian']['binding']['initial_route_s_m']==10
    assert packed['2']['vx']==1
    # Metadata contains no live get_transform/get_velocity methods.


def test_missing_actor_is_not_silently_rebound():
    recorder=journal()
    capture(recorder,[actor()],[2])
    states,_=capture(recorder,[],[])
    assert states['pedestrian']['reason']=='bound_actor_missing'
    states,_=capture(recorder,[actor(3)],[3])
    assert states['pedestrian']['status']=='INVALID'
    assert states['pedestrian']['binding']['actor_id']==2


@pytest.mark.parametrize('actors,reason',[
    ([actor(),actor(3)],'duplicate_role'),
    ([actor(kind='vehicle.audi.a2')],'wrong_actor_type')])
def test_bad_role_metadata_is_explicit_and_sticky(actors,reason):
    recorder=journal()
    states,_=capture(recorder,actors,[a.id for a in actors])
    assert states['pedestrian']['reason']==reason
    states,_=capture(recorder,[actor()],[2])
    assert states['pedestrian']['reason']==reason


def test_actor_created_after_snapshot_is_not_bound_to_earlier_frame():
    recorder=journal()
    states,_=capture(recorder,[actor()],[])
    assert states['pedestrian']['status']=='NOT_SPAWNED'
    assert not recorder.bindings


def test_direct_actor_replacement_is_rejected():
    recorder=journal()
    capture(recorder,[actor()],[2])
    states,_=capture(recorder,[actor(3)],[3])
    assert states['pedestrian']['reason']=='actor_identity_changed'


def test_incompatible_use_of_same_role_is_rejected():
    with pytest.raises(ConfigError,match='inconsistent'):
        RoleJournal([{'steps':[{'kind':'yield_pedestrian','target_role':'target'},
                              {'kind':'overtake','target_role':'target'}]}])


def test_group_roles_registered_as_walkers():
    recorder=RoleJournal([{'steps':[{'kind':'wait_clear','target_roles':['a','b']}]}])
    states,packed=capture(recorder,[actor(2,'a'),actor(3,'b')],[2,3])
    assert all(states[k]['status']=='BOUND' for k in ('a','b'))
    assert set(packed)=={'2','3'}


def test_guarded_lane_roles_registered_as_vehicles():
    recorder=RoleJournal([{'steps':[{'kind':'guarded_lane_change','target_roles':['front','rear','obstacle']}]}])
    assert recorder.expected=={key:'vehicle.' for key in ('front','rear','obstacle')}


def test_scene3_static_roles_use_their_actual_configured_route_anchor():
    from benchmark.catalog import load_catalog
    anchors=configured_role_anchors(load_catalog('scene_3').events)
    assert anchors['scene3_crossing_worker']==3345
    assert 'scene3_maintenance_vehicle' not in anchors
    # These actors are spawned relative to ego, not at the static obstacle.
    assert 'scene3_gap_front_vehicle' not in anchors


def test_distant_actor_uses_own_anchor_and_tracks_current_progress():
    recorder=RoleJournal([{'steps':[{'kind':'overtake','target_role':'lead'}]}],{'lead':1000})
    position=NS(x=1020,y=0,z=0)
    frozen=NS(get_transform=lambda:NS(location=position,rotation=NS(yaw=0)),
              get_velocity=lambda:NS(x=5,y=0,z=0))
    snapshot=NS(find=lambda identity:frozen if identity==2 else None)
    route=[dict(x=x,y=0,z=0,distance_m=x) for x in range(0,1201,5)]
    metadata=[actor(role='lead',kind='vehicle.audi.tt')]
    states,_=recorder.capture(snapshot,metadata,RouteProjector(route),0,pack_actor)
    assert states['lead']['binding']['initial_route_s_m']==1020
    assert states['lead']['route_s_m']==1020
    position.x=1040
    states,_=recorder.capture(snapshot,metadata,RouteProjector(route),10,pack_actor)
    assert states['lead']['binding']['initial_route_s_m']==1020
    assert states['lead']['route_s_m']==1040


def crossing_route():
    points=[(-20,0),(20,0),(20,20),(0,20),(0,-20)]
    distance=0.
    result=[]
    for i,(x,y) in enumerate(points):
        if i:
            old=points[i-1]
            distance+=((x-old[0])**2+(y-old[1])**2)**.5
        result.append(dict(x=x,y=y,z=0.,distance_m=distance))
    return result


def test_crossing_projection_requires_measured_unique_heading():
    projector=RouteProjector(crossing_route())
    position=NS(x=0.,y=0.,z=0.)
    with pytest.raises(ConfigError,match='ambiguous'):
        projector.project(position,60.)
    assert projector.project(position,60.,heading_deg=0.)['route_s_m']==20.
    assert projector.project(position,60.,heading_deg=-90.)['route_s_m']==100.
    with pytest.raises(ConfigError,match='measured branch evidence'):
        projector.project(position,60.,heading_deg=90.)


def test_same_heading_repeated_route_needs_motion_evidence():
    route=[dict(x=x,y=y,z=0.,distance_m=s) for x,y,s in
        [(-20,0,0),(20,0,40),(20,20,60),(-20,20,100),(-20,0,120),(20,0,160)]]
    projector=RouteProjector(route)
    position=NS(x=0.,y=0.,z=0.)
    with pytest.raises(ConfigError,match='remains'):
        projector.project(position,70.,window=160.,heading_deg=0.)
    assert projector.project(position,20.,window=160.,heading_deg=0.,maximum_offset_m=2.)['route_s_m']==20.
    with pytest.raises(ConfigError,match='without measured'):
        projector.project(position,70.,window=160.,heading_deg=0.,maximum_offset_m=2.)


def test_actor_projection_same_snapshot_reuses_verified_branch():
    projector=RouteProjector(crossing_route())
    state={}
    frozen=NS(get_transform=lambda:NS(location=NS(x=0.,y=0.,z=0.),rotation=NS(yaw=0.)),
              get_velocity=lambda:NS(x=2.,y=0.,z=0.))
    result=projector.project_actor(frozen,60.,1.,motion_state=state)
    assert projector.project_actor(frozen,20.,1.,motion_state=state)==result


def test_unresolved_projection_is_recorded_without_rebinding_or_crashing():
    recorder=journal()
    capture(recorder,[actor()],[2])
    class Ambiguous(RouteProjector):
        def project(self,*args,**kwargs):
            raise ConfigError('ambiguous route projection')
    frozen=NS(get_transform=lambda:NS(location=NS(x=10.,y=0.,z=0.),rotation=NS(yaw=0.)),
              get_velocity=lambda:NS(x=1.,y=0.,z=0.))
    snapshot=NS(find=lambda identity:frozen if identity==2 else None)
    states,packed=recorder.capture(snapshot,[actor()],Ambiguous(crossing_route()),10.,pack_actor)
    assert states['pedestrian']['status']=='INVALID'
    assert states['pedestrian']['reason']=='actor_route_projection_unresolved'
    assert states['pedestrian']['binding']['actor_id']==2
    assert packed['2']['vx']==1.
    states,_=capture(recorder,[actor()],[2])
    assert states['pedestrian']['status']=='BOUND'
    assert states['pedestrian']['binding']['actor_id']==2
