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
    assert anchors['scene3_maintenance_vehicle']==4850
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
