from benchmark.catalog import load_catalog
from benchmark.event_dependencies import audit_event_dependencies
from benchmark.planning import build_plan
from scenarios.complex.town05_scene2 import DeterministicSceneEvents, ScriptedWalker, crosswalk_polygon_endpoints
from types import SimpleNamespace as NS
import sys
import pytest


def test_actual_scene2_uses_separate_long_lived_compound_role():
    plan=build_plan(load_catalog('scene_2'),'s2_t05_cmd_03')
    audit=plan['event_dependency_audits']['s2_t05_cmd_03']
    assert audit['compatible']
    assert 'scene2_compound_slow_vehicle' in audit['required_preserved_roles']
    assert 'event_lifetime_conflict' not in plan['blockers']


def test_old_scene2_lifetime_conflict_still_detected():
    spec={'activate_m':800,'steps':[{'target_role':'scene2_crosswalk_pedestrian'},
                                   {'target_role':'scene2_slow_vehicle'}]}
    audit=audit_event_dependencies(spec,load_catalog('scene_2').events)
    assert not audit['compatible']
    finding=audit['findings'][0]
    assert finding['role']=='scene2_slow_vehicle'
    assert finding['resolve_m']==820 and finding['earliest_start_m']==830


def test_no_role_is_not_reported_as_conflict():
    audit=audit_event_dependencies({'activate_m':0,'steps':[{'kind':'speed'}]},[])
    assert audit['compatible']


def test_retained_cyclist_available_to_followup_assessment():
    plan=build_plan(load_catalog('scene_2'),'s2_t05_cmd_08')
    audit=plan['event_dependency_audits']['s2_t05_cmd_08']
    assert audit['compatible']
    assert 'scene2_slow_cyclist' in audit['required_preserved_roles']


def test_retention_requires_explicit_boolean():
    spec={'activate_m':100,'steps':[{'target_role':'lead'}]}
    event={'id':'lead_event','activate_at_m':0,'resolve_after_m':50,
           'ground_truth':{'actor_roles':['lead']},'retain_after_completion':'false'}
    assert not audit_event_dependencies(spec,[event])['compatible']
    event['retain_after_completion']=True
    assert audit_event_dependencies(spec,[event])['compatible']


def test_missing_event_for_role_is_not_silently_ignored():
    audit=audit_event_dependencies({'activate_m':0,'steps':[{'target_role':'lead'}]},[])
    assert audit['findings'][0]['reason']=='missing_or_ambiguous_event_binding'


def test_retained_role_is_not_hidden_after_event_resolves():
    events=DeterministicSceneEvents(None,None,None,[],[],[],42,
                                  preserve_roles=['scene2_slow_vehicle'])
    calls=[]
    events._retire_actor=lambda role,actor:calls.append(role) or True
    assert events._retire_event_actors('slow_vehicle')==0
    assert calls==[] and 'slow_vehicle' not in events._retired_events
    assert events.spawn_diagnostics['scene2_slow_vehicle']['retirement']=='deferred_for_independent_evaluation'


def test_default_retirement_behavior_unchanged():
    events=DeterministicSceneEvents(None,None,None,[],[],[],42)
    calls=[]
    events._retire_actor=lambda role,actor:calls.append(role) or True
    assert events._retire_event_actors('slow_vehicle')==1
    assert calls==['scene2_slow_vehicle']
    assert events._retire_event_actors('slow_vehicle')==0


def test_route_slow_vehicle_parks_and_releases_without_transform(monkeypatch):
    monkeypatch.setitem(sys.modules,'carla',NS(VehicleControl=lambda **kw:NS(**kw)))
    calls=[]
    actor=NS(is_alive=True,
             set_autopilot=lambda *args:calls.append(('autopilot',args)),
             apply_control=lambda control:calls.append(('control',vars(control))))
    tm=NS(get_port=lambda:8000,set_desired_speed=lambda *args:calls.append(('speed',args[1])),
          auto_lane_change=lambda *args:None,update_vehicle_lights=lambda *args:None)
    manager=DeterministicSceneEvents(None,tm,None,[],[],[],42)
    event=dict(id='compound',actor_role='lead',anchor_progress_m=1020,target_speed_kmh=20)
    manager._waypoint=lambda s:s
    def spawn(blueprints,waypoint,role,hidden_staging):
        assert waypoint==1020 and role=='lead' and hidden_staging is False
        manager.spawn_diagnostics[role]={}
        return actor
    manager._spawn_vehicle=spawn
    manager._spawn_route_slow_vehicle(event)
    assert calls[-1]==('control',{'brake':1.0,'hand_brake':True})
    manager._activate_route_slow_vehicle(event)
    assert ('control',{}) in calls and ('speed',20.0) in calls
    # No set_transform/set_simulate_physics methods exist on this actor.
    actor.is_alive=False
    with pytest.raises(RuntimeError,match='disappeared'):
        manager._activate_route_slow_vehicle(event)


def test_route_slow_event_completes_without_hiding_actor():
    event=next(e for e in load_catalog('scene_2').events if e['kind']=='route_slow_vehicle')
    manager=DeterministicSceneEvents(None,None,None,[],[],[event],42)
    manager.states['bus_stop_passengers']='STAGED'
    calls=[]
    manager._activate_route_slow_vehicle=lambda e:calls.append('release')
    manager._retire_event_actors=lambda e:calls.append('hide')
    assert manager.update(949.9)==[]
    manager.update(950)
    manager.update(1100)
    manager.update(1350)
    manager.update(1400)
    assert calls==['release']
    assert manager.states[event['id']]=='RESOLVED'


def test_source_crossing_is_physical_and_not_retired_underground():
    events=load_catalog('scene_2').events
    crossing=next(e for e in events if e['kind']=='crossing_pedestrian')
    assert crossing['physical_staging'] and crossing['retain_after_completion']
    manager=DeterministicSceneEvents(None,None,None,[],[],events,42)
    calls=[]
    manager._retire_actor=lambda role,actor:calls.append(role)
    assert manager._retire_event_actors('crosswalk_pedestrian')==0
    assert not calls


def test_physically_staged_walker_starts_without_transform_or_physics_toggle():
    walker=ScriptedWalker(NS(is_alive=True),NS(x=5,y=0,z=0),1.4)
    walker.start()
    assert walker.active and not walker._activation_pending
    walker.start()
    assert not walker._activation_validation_pending


def test_crosswalk_clearance_places_both_endpoints_outside_polygon(monkeypatch):
    monkeypatch.setitem(sys.modules,'carla',NS(Location=lambda **kw:NS(**kw)))
    points=[NS(x=x,y=y,z=0) for x,y in [(0,-1),(20,-1),(20,1),(0,1),(0,-1)]]
    world_map=NS(get_crosswalks=lambda:points)
    start,target=crosswalk_polygon_endpoints(world_map,0,clearance_m=1.5)
    assert start.x==-1.5 and target.x==21.5
    assert start.y==target.y==0
    start,target=crosswalk_polygon_endpoints(world_map,0)
    assert start.x==pytest.approx(.35) and target.x==pytest.approx(19.65)
    with pytest.raises(ValueError,match='clearance'):
        crosswalk_polygon_endpoints(world_map,0,clearance_m=-1)
