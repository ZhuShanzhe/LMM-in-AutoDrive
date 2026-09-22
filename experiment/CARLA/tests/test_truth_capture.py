import json
from types import SimpleNamespace as NS

import pytest

from benchmark.catalog import ConfigError
from benchmark.monitor import TaskMonitor
from benchmark.truth_capture import (ActorBinding, RouteProjector, SnapshotTruthCollector,
    polygons_intersect, prepare_lane_fixture, validate_polygon)


ROUTE=[dict(x=i,y=0,z=0,distance_m=i) for i in range(0,201,5)]
SPEC=dict(schema_version='task_oracle/1.0',task_id='capture_test',activate_m=0,
          timeout_s=10,max_frame_gap_s=1,
          steps=[dict(kind='speed',target_kmh=36,tolerance_kmh=1,hold_s=1,keep_lane=True)])


def transform(x=0,y=0,yaw=0):
    return NS(location=NS(x=x,y=y,z=0),rotation=NS(yaw=yaw))


def waypoint(y=0,lane=-1,yaw=0):
    return NS(road_id=1,section_id=0,lane_id=lane,is_junction=False,lane_type='Driving',
              lane_change='Both',transform=transform(y=y,yaw=yaw))


class Map:
    def get_waypoint(self, location):
        return waypoint()


class Snapshot:
    def __init__(self,t,actors):
        self.frame=int(t*20)
        self.timestamp=NS(elapsed_seconds=t)
        self.actors=actors

    def find(self,actor_id):
        return self.actors.get(actor_id)


def actor(x=0,y=0,v=10,yaw=0):
    return NS(get_transform=lambda:transform(x,y,yaw),get_velocity=lambda:NS(x=v,y=0,z=0))


def packets(t):
    return (dict(frame=int(t*20),complete=True,collisions=0,violations=0),
            dict(frame=int(t*20),complete=True,valid=True))


def collector(spec=None,roles=None,fixture=None):
    return SnapshotTruthCollector(spec or SPEC,ROUTE,Map(),ActorBinding(1,2,1,0),
                                  roles or {},fixture or {'steps':{}},'route_lap_0')


def test_snapshot_only_capture_drives_oracle_and_persists_hashes(tmp_path):
    with TaskMonitor(collector(),tmp_path/'capture') as monitor:
        assert monitor.observe(Snapshot(0,{1:actor()}),*packets(0))['status']=='RUNNING'
        feedback=monitor.observe(Snapshot(1,{1:actor(x=10)}),*packets(1))
        assert feedback['status']=='SUCCESS'
        assert 'ego' not in feedback and 'fixture' not in feedback
    result=json.loads((tmp_path/'capture/task_result.json').read_text())
    assert result['status']=='SUCCESS'
    assert result['evaluation_frames']==2
    assert len(result['observations_sha256'])==64
    log=[json.loads(l) for l in (tmp_path/'capture/task_truth.jsonl').read_text().splitlines()]
    assert log[1]['ego']['route_s_m']==10
    assert log[1]['frame']==20
    assert log[1]['ego']['speed_kmh']==36


@pytest.mark.parametrize('change',[dict(frame=99),dict(complete=False)])
def test_unfinalized_or_mixed_frame_safety_rejected(change):
    safety,validity=packets(0)
    safety.update(change)
    with pytest.raises(ConfigError,match='finalized'):
        collector().collect(Snapshot(0,{1:actor()}),safety,validity)


def test_missing_bound_actor_is_invalid_even_if_replacement_exists(tmp_path):
    with TaskMonitor(collector(),tmp_path/'capture') as monitor:
        result=monitor.observe(Snapshot(0,{999:actor()}),*packets(0))
        assert result['status']=='SCENE_INVALID'
    assert 'absent' in json.loads((tmp_path/'capture/task_result.json').read_text())['reason']


def test_abnormal_displacement_is_not_progress():
    capture=collector()
    capture.collect(Snapshot(0,{1:actor(v=0)}),*packets(0))
    with pytest.raises(ConfigError,match='displacement'):
        capture.collect(Snapshot(.05,{1:actor(x=20,v=0)}),*packets(.05))


def test_failed_observation_does_not_advance_collector_state():
    capture=collector()
    bad,_=packets(0)
    bad['complete']=False
    with pytest.raises(ConfigError):
        capture.collect(Snapshot(0,{1:actor()}),bad,packets(0)[1])
    assert capture.collect(Snapshot(0,{1:actor()}),*packets(0))['frame']==0


def test_duplicate_frame_and_large_gap_rejected():
    capture=collector()
    capture.collect(Snapshot(0,{1:actor()}),*packets(0))
    with pytest.raises(ConfigError,match='monotonic'):
        capture.collect(Snapshot(0,{1:actor()}),*packets(0))
    with pytest.raises(ConfigError,match='gap'):
        capture.collect(Snapshot(2,{1:actor(x=20)}),*packets(2))


def test_polygon_uses_footprint_not_just_center():
    zone=validate_polygon([[0,-1],[2,-1],[2,1],[0,1]])
    binding=ActorBinding(2,.5,.5,0)
    assert polygons_intersect(binding.footprint(transform(x=2.4)),zone)
    assert polygons_intersect(binding.footprint(transform(x=2.5)),zone)
    assert not polygons_intersect(binding.footprint(transform(x=2.6)),zone)


@pytest.mark.parametrize('vertices',[
    [[0,0],[1,0],[2,0]],[[0,0],[2,0],[1,.5],[2,2],[0,2]],
    [[0,0],[2,2],[0,2],[2,0]],[[0,0],[float('nan'),0],[0,2]]])
def test_invalid_conflict_polygon_rejected(vertices):
    with pytest.raises(ConfigError):
        validate_polygon(vertices)


def test_snapshot_generates_pedestrian_conflict_geometry():
    profile=dict(SPEC,steps=[dict(kind='yield_pedestrian',target_role='ped',stop_hold_s=.5,
                                stopped_kmh=.5,clear_hold_s=.5)])
    capture=collector(profile,{'ped':ActorBinding(2,.4,.4,20)},
        {'steps':{'0':dict(stop_line_route_s_m=15,conflict_polygon_xy=[[18,-2],[22,-2],[22,2],[18,2]])}})
    obs=capture.collect(Snapshot(0,{1:actor(v=0),2:actor(x=20,y=2.3,v=1)}),*packets(0))
    assert obs['actors']['ped']['in_conflict_zone']


def test_lane_entry_audit_checks_opposite_direction_and_legality():
    entry=waypoint()
    target=waypoint(y=-3.5,lane=-2)
    entry.get_left_lane=lambda:target
    world_map=NS(get_waypoint=lambda location:entry)
    fixture=prepare_lane_fixture(world_map,NS(),'LEFT')
    assert fixture['entry_lane_key']=='1:0:-1'
    assert fixture['target_lane_key']=='1:0:-2'
    target.transform.rotation.yaw=180
    with pytest.raises(ConfigError,match='same-direction'):
        prepare_lane_fixture(world_map,NS(),'LEFT')
    target.transform.rotation.yaw=0
    entry.lane_change='None'
    with pytest.raises(ConfigError,match='permitted'):
        prepare_lane_fixture(world_map,NS(),'LEFT')


def test_route_projection_is_continuous_not_nearest_sample():
    result=RouteProjector(ROUTE).project(NS(x=12.3,y=3,z=0),10)
    assert result['route_s_m']==pytest.approx(12.3)
    assert result['route_error_m']==pytest.approx(3)


def test_overlapping_laps_require_unambiguous_hint():
    route=[dict(x=x,y=0,z=0,distance_m=s) for x,s in [(0,0),(50,50),(0,100),(50,150)]]
    projector=RouteProjector(route)
    with pytest.raises(ConfigError,match='ambiguous'):
        projector.project(NS(x=10,y=0,z=0),50)
    assert projector.project(NS(x=10,y=0,z=0),10,window=15)['route_s_m']==10


def test_offroute_ego_is_behavior_failure_not_scene_invalid(tmp_path):
    with TaskMonitor(collector(),tmp_path/'capture') as monitor:
        result=monitor.observe(Snapshot(0,{1:actor(y=30)}),*packets(0))
        assert result['status']=='FAILURE'
        assert result['reason']=='ego_route_departure'


def test_incomplete_capture_does_not_succeed_and_directory_not_overwritten(tmp_path):
    with TaskMonitor(collector(),tmp_path/'capture') as monitor:
        monitor.observe(Snapshot(0,{1:actor()}),*packets(0))
    assert json.loads((tmp_path/'capture/task_result.json').read_text())['status']=='SCENE_INVALID'
    with pytest.raises(FileExistsError):
        TaskMonitor(collector(),tmp_path/'capture')


def test_static_binding_does_not_read_live_kinematics():
    box=NS(extent=NS(x=2,y=1),location=NS(x=.1,y=0),rotation=NS(yaw=0))
    live=NS(id=1,bounding_box=box,
            get_transform=lambda:(_ for _ in ()).throw(AssertionError('live transform read')))
    assert ActorBinding.from_actor(live,0).offset_x_m==.1


def test_front_bumper_crossing_line_fails_before_origin_crosses(tmp_path):
    profile=dict(SPEC,steps=[dict(kind='yield_pedestrian',target_role='ped',stop_hold_s=.5,
                                stopped_kmh=.5,clear_hold_s=.5)])
    capture=collector(profile,{'ped':ActorBinding(2,.4,.4,10)},
        {'steps':{'0':dict(stop_line_route_s_m=5,conflict_polygon_xy=[[8,-2],[12,-2],[12,2],[8,2]])}})
    with TaskMonitor(capture,tmp_path/'capture') as monitor:
        result=monitor.observe(Snapshot(0,{1:actor(x=4,v=0),2:actor(x=10,v=0)}),*packets(0))
        assert result['status']=='FAILURE'
        assert 'stop_line' in result['reason']


def test_capture_replays_through_public_cli_identically(tmp_path):
    from benchmark.__main__ import main
    output=tmp_path/'capture'
    with TaskMonitor(collector(),output) as monitor:
        monitor.observe(Snapshot(0,{1:actor()}),*packets(0))
        monitor.observe(Snapshot(1,{1:actor(x=10)}),*packets(1))
    replay=tmp_path/'replay.json'
    assert main(['evaluate','--spec',str(output/'spec.json'),'--observations',
                 str(output/'task_truth.jsonl'),'--output',str(replay)])==0
    online=json.loads((output/'task_result.json').read_text())
    offline=json.loads(replay.read_text())
    for key in ('status','evidence','observations_sha256','spec_sha256'):
        assert online[key]==offline[key]


def test_nonfinite_snapshot_time_is_recorded_as_invalid_not_crash(tmp_path):
    snapshot=Snapshot(0,{1:actor()})
    snapshot.timestamp.elapsed_seconds=float('nan')
    with TaskMonitor(collector(),tmp_path/'capture') as monitor:
        assert monitor.observe(snapshot,*packets(0))['status']=='SCENE_INVALID'
    data=json.loads((tmp_path/'capture/task_truth.jsonl').read_text())
    assert data['sim_time_s'] is None
