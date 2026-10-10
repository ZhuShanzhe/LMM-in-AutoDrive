import copy
from types import SimpleNamespace as NS

import pytest

from benchmark.catalog import ConfigError, load_catalog
from benchmark.task_oracle import TaskOracle, load_profile
from benchmark.truth_capture import SnapshotTruthCollector, ActorBinding


def cut_in_oracle():
    catalog = load_catalog('scene_3')
    return TaskOracle(load_profile(catalog, catalog.select('scene3_cut_in_decelerate')[0]))


def cut_in_row(t, *, merged=False, speed=20, target_speed=30, gap=25, seam=False):
    corridor = [dict(start_m=1250, end_m=1300, lane_keys=['34:0:-2']),
                dict(start_m=1300, end_m=1400, lane_keys=['35:0:-2'])]
    target_s = 1275 + gap
    return dict(schema_version='task_truth/1.0', source='simulator_truth',
        task_id='scene3_cut_in_decelerate', source_sha256=cut_in_oracle().spec['source_sha256'],
        frame=round(t*20), sim_time_s=t, scenario_valid=True,
        safety=dict(collisions=0, violations=0), fixture=dict(steps={'0':dict(lane_corridor=corridor)}),
        ego=dict(route_s_m=1270, front_route_s_m=1275, speed_kmh=speed,
                 lane_key='34:0:-2', road_key='34:0', in_junction=False,
                 route_corridor_id='main', lateral_error_m=0, heading_error_deg=0),
        actors={'scene3_cut_in_vehicle':dict(actor_id=2, alive=True,
            lane_key=('35:0:-2' if seam else '34:0:-2') if merged else '34:0:-1',
            route_s_m=target_s, rear_route_s_m=target_s, speed_kmh=target_speed,
            in_junction=False, route_corridor_id='main')})


def test_safe_already_slow_yield_requires_actual_merge_and_safe_gap():
    evaluator = cut_in_oracle()
    for i in range(30):
        result = evaluator.update(cut_in_row(i*.05, merged=i>=2, seam=True))
    assert result['status'] == 'SUCCESS'
    assert any(e['event']=='ALREADY_SLOW_SAFE_YIELD' for e in result['evidence'])
    assert any(e['event']=='CUT_IN_LANE_ENTRY' for e in result['evidence'])


@pytest.mark.parametrize('speed,target_speed,gap', [(20,30,2), (40,30,25), (30,20,8)])
def test_no_credit_for_close_gap_or_positive_closing_without_speed_response(speed,target_speed,gap):
    evaluator = cut_in_oracle()
    for i in range(30):
        result = evaluator.update(cut_in_row(i*.05, merged=i>=2, speed=speed,
                                 target_speed=target_speed, gap=gap, seam=gap>=25))
    assert result['status'] != 'SUCCESS'


def test_target_later_acceleration_cannot_create_already_safe_entry_credit():
    evaluator = cut_in_oracle()
    for i in range(30):
        result = evaluator.update(cut_in_row(i*.05, merged=i>=2, speed=40,
                                  target_speed=30 if i<4 else 50, seam=True))
    assert result['status'] != 'SUCCESS'


def test_road_number_change_alone_is_not_lane_departure():
    evaluator = cut_in_oracle()
    for i in range(30):
        row = cut_in_row(i*.05, merged=i>=2, speed=40 if i<4 else 30,
                         target_speed=35, seam=i>=4)
        if i<4:
            row['actors']['scene3_cut_in_vehicle']['route_s_m'] = 1295
            row['actors']['scene3_cut_in_vehicle']['rear_route_s_m'] = 1295
        result = evaluator.update(row)
    assert result['status']=='SUCCESS'


@pytest.mark.parametrize('field,value', [('lane_key','35:0:-3'), ('lane_key','99:0:-2'),
                                         ('route_s_m',1500), ('in_junction',True)])
def test_true_target_departure_or_unverified_topology_never_passes(field,value):
    evaluator = cut_in_oracle()
    evaluator.update(cut_in_row(0))
    evaluator.update(cut_in_row(.05, merged=True, seam=True))
    row = cut_in_row(.1, merged=True, seam=True)
    row['actors']['scene3_cut_in_vehicle'][field] = value
    assert evaluator.update(row)['status']=='SCENE_INVALID'


def test_no_merge_transition_still_invalid_even_when_already_slow():
    assert cut_in_oracle().update(cut_in_row(0, merged=True, seam=True))['status']=='SCENE_INVALID'


def deferred_collector(monkeypatch, *, lateral=0, heading=0):
    collector = SnapshotTruthCollector.__new__(SnapshotTruthCollector)
    collector.fixture = dict(steps={'1':dict(deferred_until_step_entry=True,corridor_end_m=1200)})
    collector.spec = dict(steps=[dict(kind='speed'), dict(kind='lane_change',direction='LEFT',
        max_lateral_error_m=.35,max_heading_error_deg=5)])
    collector.bindings = {'ego':ActorBinding(1,2,1,900)}
    collector.hints = {'ego':920}
    collector.projection_motion = {}
    collector.projector = NS(route=[dict(distance_m=0),dict(distance_m=1400)],
        project_actor=lambda *args,**kwargs:dict(route_s_m=920))
    center = NS(x=920,y=0,z=0)
    target = NS(road_id=10,section_id=0,lane_id=-1,is_junction=False,lane_type='Driving',
                transform=NS(location=NS(x=920,y=3.5,z=0),rotation=NS(yaw=0)))
    entry = NS(road_id=10,section_id=0,lane_id=-2,is_junction=False,lane_type='Driving',lane_change='Both',
        transform=NS(location=center,rotation=NS(yaw=0)),get_left_lane=lambda:target)
    collector.world_map = NS(get_waypoint=lambda location:entry)
    monkeypatch.setattr('benchmark.lane_continuity.trace_lane_change_keys',lambda *args,**kwargs:
                        dict(entry_lane_keys=['10:0:-2'],target_lane_keys=['10:0:-1']))
    pose = NS(location=NS(x=920,y=lateral,z=0),rotation=NS(yaw=heading))
    snapshot = NS(frame=50,timestamp=NS(elapsed_seconds=2.5),find=lambda identity:
                  NS(get_transform=lambda:pose) if identity==1 else None)
    return collector,snapshot


def test_deferred_lane_binds_actual_post_yield_snapshot(monkeypatch):
    collector,snapshot = deferred_collector(monkeypatch)
    collector.bind_step_fixture(snapshot,0)
    assert collector.fixture['steps']['1']['deferred_until_step_entry']
    collector.bind_step_fixture(snapshot,1)
    bound = copy.deepcopy(collector.fixture['steps']['1'])
    assert bound['binding_evidence']['route_s_m']==920
    assert bound['binding_evidence']['frame']==50
    assert bound['entry_lane_key']=='10:0:-2'
    collector.bind_step_fixture(snapshot,1)
    assert collector.fixture['steps']['1']==bound


@pytest.mark.parametrize('lateral,heading', [(.4,0),(0,6)])
def test_deferred_lane_rejects_already_departed_origin(monkeypatch,lateral,heading):
    collector,snapshot = deferred_collector(monkeypatch,lateral=lateral,heading=heading)
    with pytest.raises(ConfigError,match='outside centered origin'):
        collector.bind_step_fixture(snapshot,1)
