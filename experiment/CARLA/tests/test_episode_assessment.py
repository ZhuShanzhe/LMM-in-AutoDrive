import json
from dataclasses import asdict, replace
from types import SimpleNamespace as NS

import pytest

from benchmark.catalog import CONFIG_ROOT, ConfigError
from benchmark.episode import (EpisodeAssessment, pack_actor, unpack_snapshot,
                               finish_episode, defer_dynamic_role_entry)
from benchmark.truth_capture import ActorBinding


class Map:
    name='Town04_Opt'
    def get_waypoint(self,location):
        position=NS(x=location.x,y=0,z=0)
        position.distance=lambda other: ((position.x-other.x)**2+other.y**2+other.z**2)**.5
        return NS(road_id=1,section_id=0,lane_id=-1,is_junction=False,lane_type='Driving',
                  transform=NS(location=position,rotation=NS(yaw=0)),
                  next=lambda distance:[Map().get_waypoint(NS(x=location.x+distance))])


def snapshot(frame,x):
    pose=NS(location=NS(x=x,y=0,z=0),rotation=NS(yaw=0))
    actor=NS(get_transform=lambda:pose,get_velocity=lambda:NS(x=12.5,y=0,z=0))
    return NS(frame=frame,timestamp=NS(elapsed_seconds=frame*.05),find=lambda identity:actor if identity==1 else None)


def episode(tmp_path,initial=0):
    ego=NS(id=1,bounding_box=NS(extent=NS(x=2,y=1),location=NS(x=0,y=0),rotation=NS(yaw=0)))
    route=[dict(x=x,y=0,z=0,distance_m=x,road_id=1,section_id=0,lane_id=-1) for x in range(0,201,5)]
    return EpisodeAssessment('scene_1',route,Map(),ego,tmp_path/'capture',CONFIG_ROOT/'basic_voice_urban_5km.json',initial)


def test_snapshot_journal_reconstructs_kinematics():
    snap=snapshot(10,5)
    row=dict(frame=10,sim_time_s=.5,actors={'1':pack_actor(snap.find(1))})
    restored=unpack_snapshot(json.loads(json.dumps(row)))
    assert restored.find(1).get_transform().location.x==5
    assert restored.find(1).get_velocity().x==12.5
    assert restored.find(2) is None


def test_dynamic_role_entry_grace_is_bounded():
    profile=dict(activate_m=1250,steps=[dict(kind='yield_cut_in',target_role='lead')])
    row=dict(sim_time_s=10.0,route_s_m=1250.0,roles={'lead':dict(status='NOT_SPAWNED')})
    assert defer_dynamic_role_entry(profile,row,10.0)
    row.update(sim_time_s=10.1,route_s_m=1250.8,
               roles={'lead':dict(status='BOUND')})
    assert not defer_dynamic_role_entry(profile,row,10.0)
    row.update(sim_time_s=10.3,roles={'lead':dict(status='NOT_SPAWNED')})
    assert not defer_dynamic_role_entry(profile,row,10.0)


def test_manifest_records_route_and_bound_profile_identities(tmp_path):
    import hashlib
    recorder=episode(tmp_path)
    try:
        expected=hashlib.sha256(json.dumps(recorder.route,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert recorder.manifest['route_sha256']==expected
        assert set(recorder.manifest['assessment_profile_sha256'])==set(recorder.profiles)
        assert all(len(value)==64 for value in recorder.manifest['assessment_profile_sha256'].values())
    finally:
        recorder.stream.close()


def test_formal_traffic_uses_snapshot_and_excludes_walkers(tmp_path):
    recorder=episode(tmp_path)
    calls=[]
    def actors():
        calls.append(1)
        return [NS(id=2,type_id='vehicle.test'),NS(id=3,type_id='walker.pedestrian.test')]
    recorder.actor_provider=actors
    base=snapshot(1,0)
    front=snapshot(1,20).find(1)
    recorder.observe(NS(frame=1,timestamp=base.timestamp,
        find=lambda identity:base.find(1) if identity==1 else front if identity in (2,3) else None))
    recorder.actor_provider=lambda:[]
    recorder.observe(snapshot(2,.625))
    result=recorder.close()
    density=result['traffic_density']
    assert calls==[1]
    assert density['status']=='RECORDED'
    assert density['overall']['mean']['front_cone']==.5
    assert density['overall']['empty_ego_lane_fraction']==.5
    assert density['route_segments'][0]['from_m']==0
    rows=[json.loads(line) for line in (recorder.output/'episode_truth.jsonl').read_text().splitlines()]
    assert rows[0]['traffic_observation']['counts']['front_cone']==1
    assert '2' not in rows[0]['actors']  # Background truth stays outside task bindings.


def test_unconfigured_traffic_provider_is_not_empty_traffic_evidence(tmp_path):
    recorder=episode(tmp_path)
    recorder.observe(snapshot(1,0))
    density=recorder.close()['traffic_density']
    assert density['status']=='UNAVAILABLE'
    assert density['overall']['below_three_front_actor_fraction'] is None


@pytest.mark.parametrize('status',['SUCCESS','RUNNING','FAILURE'])
def test_episode_prepares_followup_only_from_independent_success(tmp_path,status):
    recorder=episode(tmp_path)
    profile=dict(recorder.profiles['c01_depart_45'],requires_task_success=['prior'])
    prior_result={'status':status,'evidence':[{'event':'SUCCESS','frame':1}] if status=='SUCCESS' else []}
    recorder.monitors['prior']=NS(oracle=NS(result=lambda:prior_result))
    try:
        recorder._prepare('c01_depart_45',profile,snapshot(2,.625),.625)
        if status=='SUCCESS':
            monitor=recorder.monitors['c01_depart_45']
            proof=monitor.collector.fixture['prerequisites']['prior']
            assert proof['completion_frame']==1
            assert proof['source']=='independent_task_oracle'
            monitor.close()
        else:
            assert 'prerequisite' in recorder.unavailable['c01_depart_45']
            assert 'c01_depart_45' not in recorder.monitors
    finally:
        recorder.stream.close()


@pytest.mark.parametrize('required_length,expected',[(200,'SUCCESS'),(5000,'SCENE_INVALID')])
def test_episode_destination_binds_actual_end_and_rejects_short_route(tmp_path,required_length,expected):
    recorder=episode(tmp_path)
    recorder.catalog=replace(recorder.catalog,route_length_m=required_length)
    profile=dict(recorder.profiles['c01_depart_45'])
    profile.pop('end_route_s_m',None)
    profile['steps']=[dict(kind='destination',max_distance_m=3,max_remaining_m=3)]
    recorder.profiles={'c01_depart_45':profile}
    for frame in range(1,321):
        recorder.observe(snapshot(frame,(frame-1)*.625))
    result=recorder.close()
    assert result['tasks']['c01_depart_45']['status']==expected
    if expected=='SUCCESS':
        fixture=json.loads((recorder.output/'tasks/c01_depart_45/fixture.json').read_text())
        assert fixture['steps']['0']['position_m']=={'x':200,'y':0,'z':0}


def test_segment_uses_explicit_origin_and_does_not_replay_earlier_tasks(tmp_path):
    recorder=episode(tmp_path,190)
    recorder.observe(snapshot(1,190))
    result=recorder.close()
    assert result['final_route_s_m']==190
    assert result['tasks']['c01_depart_45']['status']=='NOT_RUN'
    assert result['tasks']['c02_cruise_60']['status']=='NOT_RUN'
    assert result['assessed_tasks']==0


def test_replay_preserves_native_location_type_for_map_calls(tmp_path):
    class NativeLocation:
        def __init__(self,x,y,z):
            self.x,self.y,self.z=x,y,z
    class StrictMap(Map):
        def get_waypoint(self,location):
            assert isinstance(location,NativeLocation)
            return super().get_waypoint(location)
    recorder=episode(tmp_path)
    recorder.map=StrictMap()
    for frame in range(1,51):
        snap=snapshot(frame,(frame-1)*.625)
        snap.find(1).get_transform().location=NativeLocation((frame-1)*.625,0,0)
        recorder.observe(snap)
    result=recorder.close()
    assert result['tasks']['c01_depart_45']['status']=='SUCCESS'


def test_formal_episode_does_not_hide_unsupported_or_unreached_tasks(tmp_path):
    recorder=episode(tmp_path)
    for frame in range(1,51):
        recorder.observe(snapshot(frame,(frame-1)*.625))
    result=recorder.close()
    assert result['tasks']['c01_depart_45']['status']=='SUCCESS'
    assert result['tasks']['c04_change_left']['status']=='NOT_REACHED'
    assert result['tasks']['c02_cruise_60']['status']=='NOT_REACHED'
    assert result['tasks']['c10_keep_35']['status']=='NOT_REACHED'
    assert result['tasks']['c15_keep_to_goal']['status']=='NOT_REACHED'
    assert result['recorded_criteria_pass_rate']==pytest.approx(1/15)
    assert result['verified_completion_rate_lower_bound']==0
    assert result['tasks']['c01_depart_45']['instruction_status']=='UNVERIFIED'
    assert not result['benchmark_ready']
    assert recorder.close()==result
    assert len((tmp_path/'capture/episode_truth.jsonl').read_text().splitlines())==50


def test_episode_safety_can_fail_after_task_success(tmp_path):
    recorder=episode(tmp_path)
    for frame in range(1,51):
        recorder.observe(snapshot(frame,(frame-1)*.625))
    recorder.ledger.record('collision',49,other_actor_id=20)
    result=recorder.close()
    assert result['tasks']['c01_depart_45']['status']=='SUCCESS'
    assert result['episode_safety']['status']=='FAILURE'


def test_duplicate_frame_fails_and_teardown_is_available(tmp_path):
    recorder=episode(tmp_path)
    recorder.observe(snapshot(1,0))
    with pytest.raises(ConfigError,match='frame'):
        recorder.observe(snapshot(1,0))
    finish_episode(recorder)
    assert recorder.closed


def test_mutated_source_config_is_rejected_before_capture(tmp_path):
    config=tmp_path/'source.json'
    config.write_text('{}')
    with pytest.raises(ConfigError,match='configuration differs'):
        EpisodeAssessment('scene_1',[],Map(),None,tmp_path/'capture',config)
    assert not (tmp_path/'capture').exists()


def test_formal_scene2_journals_task_roles_and_missing_actor(tmp_path):
    ego=NS(id=1,bounding_box=NS(extent=NS(x=2,y=1),location=NS(x=0,y=0),rotation=NS(yaw=0)))
    ped=NS(id=2,type_id='walker.pedestrian.0001',
           attributes={'role_name':'scene2_crosswalk_pedestrian'},bounding_box=ego.bounding_box)
    world_map=Map()
    world_map.name='Town05_Opt'
    route=[dict(x=x,y=0,z=0,distance_m=x) for x in range(0,1201,5)]
    recorder=EpisodeAssessment('scene_2',route,world_map,ego,tmp_path/'capture',
                               CONFIG_ROOT/'scene_2_town05_runtime.json')
    recorder.actor_provider=lambda:[ped]
    base=snapshot(1,0)
    recorder.observe(NS(frame=1,timestamp=base.timestamp,
                        find=lambda identity:base.find(1) if identity in (1,2) else None))
    recorder.actor_provider=lambda:[]
    recorder.observe(snapshot(2,.625))
    result=recorder.close()
    rows=[json.loads(line) for line in (tmp_path/'capture/episode_truth.jsonl').read_text().splitlines()]
    assert rows[0]['roles']['scene2_crosswalk_pedestrian']['status']=='BOUND'
    assert '2' in rows[0]['actors']
    assert rows[0]['roles']['scene2_compound_slow_vehicle']['status']=='NOT_SPAWNED'
    assert rows[1]['roles']['scene2_crosswalk_pedestrian']['reason']=='bound_actor_missing'
    assert '2' not in rows[1]['actors']
    assert result['role_capture_errors']=={'scene2_crosswalk_pedestrian':'bound_actor_missing'}
    assert result['assessed_tasks']==0


def test_formal_role_task_uses_exact_source_crosswalk(tmp_path,monkeypatch):
    recorder=episode(tmp_path)
    profile=dict(recorder.profiles['c01_depart_45'])
    profile['steps']=[dict(kind='yield_pedestrian',target_role='ped',require_stop=False,
        min_speed_drop_kmh=5,stop_hold_s=.5,stopped_kmh=.5,clear_hold_s=.5)]
    recorder.catalog=replace(recorder.catalog,events=(dict(kind='crossing_pedestrian',
        ground_truth={'actor_roles':['ped']},crosswalk_polygon_index=2,anchor_progress_m=40),))
    calls=[]
    def fixture(world_map,route,index,anchor):
        calls.append((index,anchor))
        return dict(stop_line_route_s_m=35,conflict_polygon_xy=[[38,-3],[42,-3],[42,3],[38,3]])
    monkeypatch.setattr('benchmark.event_fixture.crosswalk_fixture',fixture)
    base=snapshot(1,0)
    snap=NS(frame=1,timestamp=base.timestamp,find=lambda identity:base.find(1) if identity in (1,2) else None)
    role={'ped':dict(status='BOUND',binding=asdict(ActorBinding(2,.3,.3,40)))}
    recorder._prepare('c01_depart_45',profile,snap,0,role)
    assert calls==[(2,40)]
    monitor=recorder.monitors['c01_depart_45']
    assert monitor.collector.bindings['ped'].actor_id==2
    assert monitor.collector.fixture['steps']['0']['stop_line_route_s_m']==35
    recorder.close()


def test_missing_role_at_activation_is_scene_invalid_not_unsupported(tmp_path):
    recorder=episode(tmp_path)
    profile=dict(recorder.profiles['c01_depart_45'])
    profile['steps']=[dict(kind='overtake',target_role='slow',rear_clearance_m=8,hold_s=1)]
    recorder._prepare('c01_depart_45',profile,snapshot(1,0),0,
                      {'slow':dict(status='INVALID',reason='actor_identity_changed')})
    result=recorder.close()
    assert result['tasks']['c01_depart_45']['status']=='SCENE_INVALID'
    assert 'actor_identity_changed' in result['tasks']['c01_depart_45']['reason']


def test_full_episode_diagnostic_preserves_original_results(tmp_path):
    from benchmark.episode_diagnostic import reassess
    recorder=episode(tmp_path)
    for frame in range(1,51):
        recorder.observe(snapshot(frame,(frame-1)*.625))
    recorder.close()
    original={p:p.read_bytes() for p in recorder.output.rglob('*') if p.is_file()}
    result=reassess(recorder.output,'c01_depart_45',tmp_path/'diagnostic',Map(),NS,60)
    assert result['diagnostic_result']['status']=='SUCCESS'
    assert result['scope']=='diagnostic_only_not_formal_acceptance'
    assert not result['benchmark_ready']
    assert result['diagnostic_timeout_s']==60
    assert all(p.read_bytes()==data for p,data in original.items())
    with pytest.raises(ConfigError,match='outside'):
        reassess(recorder.output,'c01_depart_45',recorder.output/'nested',Map(),NS)
    (recorder.output/'tasks/c01_depart_45/spec.json').write_text('{}')
    with pytest.raises(ConfigError,match='hash mismatch'):
        reassess(recorder.output,'c01_depart_45',tmp_path/'tampered',Map(),NS)


def test_map_geometry_is_captured_and_checked_before_replay(tmp_path,monkeypatch):
    import hashlib
    from benchmark.episode_diagnostic import reassess
    monkeypatch.setattr(Map,'to_opendrive',lambda self:'<OpenDRIVE>original</OpenDRIVE>',raising=False)
    recorder=episode(tmp_path)
    for frame in range(1,51):
        recorder.observe(snapshot(frame,(frame-1)*.625))
    summary=recorder.close()
    captured=recorder.output/'map.xodr'
    assert hashlib.sha256(captured.read_bytes()).hexdigest()==summary['map_geometry_sha256']
    result=reassess(recorder.output,'c01_depart_45',tmp_path/'matching',Map(),NS)
    assert result['original_map_geometry_hash_available']
    assert 'original_capture_has_no_map_geometry_hash' not in result['limitations']
    monkeypatch.setattr(Map,'to_opendrive',lambda self:'<OpenDRIVE>changed</OpenDRIVE>')
    with pytest.raises(ConfigError,match='geometry mismatch'):
        reassess(recorder.output,'c01_depart_45',tmp_path/'wrong_map',Map(),NS)
    assert not (tmp_path/'wrong_map').exists()
    monkeypatch.setattr(Map,'to_opendrive',lambda self:'<OpenDRIVE>original</OpenDRIVE>')
    captured.write_text('tampered')
    with pytest.raises(ConfigError,match='missing or changed'):
        reassess(recorder.output,'c01_depart_45',tmp_path/'wrong_file',Map(),NS)


def test_lane_diagnostic_resets_stability_on_error_and_gap():
    from benchmark.episode_diagnostic import lane_diagnostics
    spec={'activate_m':0,'max_frame_gap_s':.15,'steps':[dict(kind='lane_change',
        max_lateral_error_m=.35,max_heading_error_deg=5,hold_s=1)]}
    fixture={'steps':{'0':{'target_lane_key':'1:0:2'}}}
    rows=[dict(scenario_valid=True,sim_time_s=t,ego=dict(route_s_m=10,lane_key='1:0:2',
        in_junction=False,lateral_error_m=e,heading_error_deg=0))
        for t,e in [(0,.3),(.1,.3),(.2,.36),(.3,.3),(.4,.3),(.7,.3)]]
    result=lane_diagnostics(rows,spec,fixture)[0]
    assert result['target_lane_frames']==6
    assert result['longest_stable_s']==pytest.approx(.1)
