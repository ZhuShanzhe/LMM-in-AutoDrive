from types import SimpleNamespace as NS
import json

import pytest

from benchmark.catalog import ConfigError
from benchmark.compound_driver import CompoundDriver, lane_follow_extension


def fixture(tmp_path):
    driver=CompoundDriver.__new__(CompoundDriver)
    driver.command={'announce_at_m':800}
    driver.state='WAITING'
    driver.seen_conflict=False
    driver.clear_since=None
    driver.owns_plan=False
    driver.output=tmp_path/'driver.jsonl'
    driver._extend_pass_plan=lambda snapshot:None
    driver.crossing={'stop_line_route_s_m':850,
                     'conflict_polygon_xy':[[0,0],[10,0],[10,10],[0,10]]}
    position=NS(x=5,y=5,z=0)
    actor=NS(get_transform=lambda:NS(location=position,rotation=NS(yaw=0)))
    driver.pedestrian=NS(id=1)
    driver.lead=NS(id=2)
    driver.ego=NS(id=3)
    driver.ped_binding=NS(footprint=lambda pose:[(pose.location.x+dx,pose.location.y+dy)
                                  for dx,dy in [(-.2,-.2),(.2,-.2),(.2,.2),(-.2,.2)]])
    snapshot=NS(frame=1,timestamp=NS(elapsed_seconds=0),find=lambda identity:actor)
    return driver,snapshot,position


def test_yield_requires_seen_conflict_and_sustained_clearance(tmp_path):
    driver,snapshot,position=fixture(tmp_path)
    assert driver.update(snapshot,780,45)==45
    assert driver.update(snapshot,800,45)==30
    assert driver.update(snapshot,840,45)==0
    position.x=15
    snapshot.timestamp.elapsed_seconds=1
    assert driver.update(snapshot,840,45)==0
    snapshot.timestamp.elapsed_seconds=2.1
    assert driver.update(snapshot,840,45)==30
    assert driver.state=='APPROACH_LANE'
    assert not driver.owns_plan


def test_clear_before_any_crossing_does_not_skip_yield(tmp_path):
    driver,snapshot,position=fixture(tmp_path)
    position.x=15
    driver.update(snapshot,840,45)
    snapshot.timestamp.elapsed_seconds=5
    assert driver.update(snapshot,840,45)==0
    assert driver.state=='YIELD'


def test_missing_actor_fails_explicitly(tmp_path):
    driver,snapshot,_=fixture(tmp_path)
    snapshot.find=lambda identity:None
    with pytest.raises(ConfigError,match='missing'):
        driver.update(snapshot,800,45)


def route_fixture(tmp_path):
    driver=CompoundDriver.__new__(CompoundDriver)
    driver.output=tmp_path/'driver.jsonl'
    driver.state='PASS'
    driver.owns_plan=True
    driver.ego=NS(id=1)
    def waypoint(x,lane=-1):
        return NS(road_id=1,section_id=0,lane_id=lane,
            transform=NS(location=NS(x=x,y=0,z=0)),get_left_lane=lambda:None,get_right_lane=lambda:None)
    points=[waypoint(x) for x in range(0,201,20)]
    driver.route=[(w,'FOLLOW') for w in points]
    driver.distances=list(range(0,201,20))
    current=points[2]
    snapshot=NS(frame=1,timestamp=NS(elapsed_seconds=1),
        find=lambda identity:NS(get_transform=lambda:current.transform))
    driver.world=NS(get_map=lambda:NS(get_waypoint=lambda location:current))
    plans=[]
    def trace(a,b):
        assert a is current and b is points[5]
        return [(b,'FOLLOW')]
    driver.agent=NS(trace_route=trace,
                    set_global_plan=lambda plan,**kwargs:plans.append(plan))
    return driver,snapshot,current,plans


def test_route_resume_keeps_full_remaining_route(tmp_path):
    driver,snapshot,_,plans=route_fixture(tmp_path)
    assert driver._resume_route(snapshot,40)
    assert driver.state=='FINISHED' and not driver.owns_plan
    assert plans[0]==driver.route[5:]


def test_route_resume_does_not_cross_nonadjacent_lanes(tmp_path):
    driver,snapshot,current,plans=route_fixture(tmp_path)
    driver.route[2]=(NS(road_id=1,section_id=0,lane_id=-3),'FOLLOW')
    assert not driver._resume_route(snapshot,40)
    assert driver.state=='MERGE_WAIT' and driver.owns_plan
    assert not plans


def test_route_resume_rejects_missing_connector(tmp_path):
    driver,snapshot,_,plans=route_fixture(tmp_path)
    driver.agent.trace_route=lambda a,b:[]
    with pytest.raises(ConfigError,match='connector'):
        driver._resume_route(snapshot,40)
    assert not plans


def test_route_resume_waits_for_target_lane_gap(tmp_path,monkeypatch):
    driver,snapshot,current,plans=route_fixture(tmp_path)
    reference=NS(road_id=1,section_id=0,lane_id=-2)
    driver.route[2]=(reference,'FOLLOW')
    current.get_right_lane=lambda:reference
    driver.world.get_actors=lambda:NS(filter=lambda pattern:[])
    monkeypatch.setattr('benchmark.compound_driver.prepare_lane_fixture',lambda *args:{})
    monkeypatch.setattr('benchmark.compound_driver.gap_check',lambda *args:
                        {'clear':False,'reason':'target_lane_gap'})
    assert not driver._resume_route(snapshot,40)
    assert driver.state=='MERGE_WAIT' and not plans


def centering_fixture(tmp_path):
    driver,snapshot,wp,_=route_fixture(tmp_path)
    wp.transform.rotation=NS(yaw=0)
    wp.is_junction=False
    pose=NS(location=NS(x=40,y=0.2,z=0),rotation=NS(yaw=0))
    snapshot.find=lambda identity:NS(get_transform=lambda:pose)
    driver.pass_lane_keys={'1:0:-1'}
    driver.centered_since=None
    driver.last_center_sample=None
    return driver,snapshot,wp,pose


def test_passing_lane_requires_continuous_centering(tmp_path):
    driver,snapshot,_,pose=centering_fixture(tmp_path)
    for frame in range(32):
        snapshot.timestamp.elapsed_seconds=frame*.05
        stable=driver._pass_lane_stable(snapshot)
    assert stable
    pose.location.y=.36
    snapshot.timestamp.elapsed_seconds=1.6
    assert not driver._pass_lane_stable(snapshot)
    assert driver.centered_since is None


@pytest.mark.parametrize('fault',['wrong_lane','junction','heading','missing_actor','nan','gap','backward'])
def test_passing_lane_resets_on_invalid_sample(tmp_path,fault):
    driver,snapshot,wp,pose=centering_fixture(tmp_path)
    for frame in range(32):
        snapshot.timestamp.elapsed_seconds=frame*.05
        driver._pass_lane_stable(snapshot)
    snapshot.timestamp.elapsed_seconds=1.6
    if fault=='wrong_lane': wp.lane_id=-2
    elif fault=='junction': wp.is_junction=True
    elif fault=='heading': pose.rotation.yaw=6
    elif fault=='missing_actor': snapshot.find=lambda identity:None
    elif fault=='nan': pose.location.y=float('nan')
    elif fault=='gap': snapshot.timestamp.elapsed_seconds=2
    elif fault=='backward': snapshot.timestamp.elapsed_seconds=1
    assert not driver._pass_lane_stable(snapshot)


def test_merge_wait_rechecks_lane_stability(tmp_path):
    driver,snapshot,_=fixture(tmp_path)
    driver.state='MERGE_WAIT'
    driver._pass_lane_stable=lambda snapshot:False
    def forbidden(*args):
        pytest.fail('must not merge before reacquiring lane stability')
    driver._resume_route=forbidden
    assert driver.update(snapshot,1100,45)==30


def test_control_probe_records_actual_target_and_control(tmp_path):
    driver,snapshot,wp,pose=centering_fixture(tmp_path)
    snapshot.find=lambda identity:NS(get_transform=lambda:pose,
                                    get_velocity=lambda:NS(x=10,y=0,z=0))
    driver.agent.get_local_planner=lambda:NS(target_waypoint=wp)
    driver.record_control(snapshot,NS(steer=-.05,throttle=.4,brake=0))
    row=json.loads((tmp_path/'compound_control.jsonl').read_text())
    assert row['target']['lane_key']=='1:0:-1'
    assert row['target']['lateral_error_m']==pytest.approx(.2)
    assert row['speed_kmh']==36 and row['steer']==-.05
    assert row['simulator_truth_baseline'] is True


def test_control_probe_inactive_outside_owned_plan(tmp_path):
    driver,snapshot,_,_=centering_fixture(tmp_path)
    driver.owns_plan=False
    driver.record_control(snapshot,NS())
    assert not (tmp_path/'compound_control.jsonl').exists()


def test_lane_acquisition_does_not_accelerate_to_passing_speed(tmp_path):
    driver,snapshot,_=fixture(tmp_path)
    driver.owns_plan=True
    driver.state='LANE_ACQUIRE'
    actor=snapshot.find(2)
    actor.get_velocity=lambda:NS(x=5,y=0,z=0)
    driver._pass_lane_stable=lambda snapshot:False
    assert driver.update(snapshot,980,45)==18
    assert driver.state=='LANE_ACQUIRE'


def test_acceleration_requires_stable_passing_lane(tmp_path):
    driver,snapshot,_=fixture(tmp_path)
    driver.owns_plan=True
    driver.state='LANE_ACQUIRE'
    driver._pass_lane_stable=lambda snapshot:True
    driver.ego.bounding_box=NS(extent=NS(x=2))
    driver.lead.bounding_box=NS(extent=NS(x=2))
    assert driver.update(snapshot,980,45)==45
    assert driver.state=='PASS'


def extension_points():
    class Location:
        def __init__(self,x): self.x=x
        def distance(self,other): return abs(self.x-other.x)
    points=[NS(id=i,is_junction=False,lane_type='Driving',
               transform=NS(location=Location(i*2),rotation=NS(yaw=0))) for i in range(35)]
    for i,wp in enumerate(points):
        wp.next=lambda distance,i=i:points[i+1:i+2]
    return points


def test_lane_horizon_extension_is_bounded():
    points=extension_points()
    extension,reason=lane_follow_extension(points[0])
    assert extension==points[1:31]
    assert reason is None


@pytest.mark.parametrize('fault',['junction','branch','jump','opposite','loop','lane_type'])
def test_lane_horizon_rejects_unsafe_continuation(fault):
    points=extension_points()
    if fault=='junction': points[2].is_junction=True
    elif fault=='branch': points[1].next=lambda distance:[points[2],points[3]]
    elif fault=='jump': points[2].transform.location.x=20
    elif fault=='opposite': points[2].transform.rotation.yaw=180
    elif fault=='loop': points[1].next=lambda distance:[points[0]]
    elif fault=='lane_type': points[2].lane_type='Sidewalk'
    extension,reason=lane_follow_extension(points[0])
    assert extension==points[1:2] and reason is not None
