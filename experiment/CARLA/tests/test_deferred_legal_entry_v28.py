from types import SimpleNamespace as NS
import copy
import pytest
from benchmark.catalog import ConfigError
from benchmark.task_oracle import TaskOracle
from benchmark.truth_capture import SnapshotTruthCollector, ActorBinding


def test_junction_wait_keeps_unbound_fixture_then_binds_legal_origin(monkeypatch):
    collector=SnapshotTruthCollector.__new__(SnapshotTruthCollector)
    collector.fixture={'steps':{'1':dict(deferred_until_step_entry=True,corridor_end_m=1200)}}
    collector.spec={'steps':[{'kind':'speed'},dict(kind='lane_change',direction='LEFT',
        max_lateral_error_m=.35,max_heading_error_deg=5)]}
    collector.bindings={'ego':ActorBinding(1,2,1,900)}
    collector.hints={'ego':920}
    collector.projection_motion={}
    collector.projector=NS(route=[{'distance_m':0},{'distance_m':1400}],
        project_actor=lambda *args,**kwargs:{'route_s_m':920})
    center=NS(x=920,y=0,z=0)
    target=NS(road_id=10,section_id=0,lane_id=-1,is_junction=False,lane_type='Driving',
        transform=NS(location=NS(x=920,y=3.5,z=0),rotation=NS(yaw=0)))
    entry=NS(road_id=10,section_id=0,lane_id=-2,is_junction=True,lane_type='Driving',
        lane_change='Both',transform=NS(location=center,rotation=NS(yaw=0)),get_left_lane=lambda:target)
    collector.world_map=NS(get_waypoint=lambda location:entry)
    pose=NS(location=center,rotation=NS(yaw=0))
    snapshot=NS(frame=10,timestamp=NS(elapsed_seconds=.5),find=lambda actor_id:NS(get_transform=lambda:pose))
    monkeypatch.setattr('benchmark.lane_continuity.trace_lane_change_keys',lambda *args,**kwargs:
        dict(entry_lane_keys=['10:0:-2'],target_lane_keys=['10:0:-1']))
    before=copy.deepcopy(collector.fixture)
    collector.bind_step_fixture(snapshot,1)
    assert collector.fixture==before
    entry.is_junction=False
    collector.bind_step_fixture(snapshot,1)
    assert collector.fixture['steps']['1']['entry_lane_key']=='10:0:-2'
    assert 'deferred_until_step_entry' not in collector.fixture['steps']['1']


def evaluator():
    spec=dict(schema_version='task_oracle/1.0',task_id='test',activate_m=0,timeout_s=1,max_frame_gap_s=.15,violation_budget=0,
        steps=[dict(kind='speed',target_kmh=0,tolerance_kmh=1,hold_s=.05),
               dict(kind='lane_change',direction='LEFT',hold_s=.1,max_lateral_error_m=.35,
                    max_heading_error_deg=5)])
    oracle=TaskOracle(spec)
    oracle.update(row(0))
    assert oracle.update(row(1))['completed_steps']==1
    return oracle


def row(frame,junction=True,collisions=0):
    return dict(schema_version='task_truth/1.0',source='simulator_truth',frame=frame,
        sim_time_s=frame*.05,scenario_valid=True,
        ego=dict(speed_kmh=0,route_s_m=5,in_junction=junction),actors={},
        safety=dict(collisions=collisions,violations=0),
        fixture={'steps':{'1':dict(deferred_until_step_entry=True,corridor_end_m=100)}})


def test_pending_junction_never_counts_as_lane_change_success():
    oracle=evaluator()
    for frame in range(2,10):
        assert oracle.update(row(frame))['status']=='RUNNING'
    assert oracle.index==1 and not oracle.fixtures


def test_pending_still_has_deadline():
    oracle=evaluator()
    for frame in range(2,22):
        result=oracle.update(row(frame))
    assert result['status']=='TIMEOUT' and result['reason']=='task_deadline'


def test_pending_does_not_ignore_collision():
    oracle=evaluator()
    assert oracle.update(row(2,collisions=1))['status']=='FAILURE'


def test_unbound_fixture_outside_junction_is_invalid():
    oracle=evaluator()
    assert oracle.update(row(2,junction=False))['status']=='SCENE_INVALID'
