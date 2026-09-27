import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control.execution_wait import matching_execution_wait, traffic_wait_feedback
from control.map_signal_observer import connected_stop_distance, PassedStopLineMemory


def test_only_enforced_zero_speed_traffic_hold_pauses():
    traffic = dict(active=True, status='wait_for_observed_green', speed_cap_kmh=0.)
    decision = dict(action='stop', target_speed_kmh=0.)
    feedback = traffic_wait_feedback('a', 3., decision, traffic)
    assert matching_execution_wait(feedback, 'a', 3.)
    assert not matching_execution_wait(feedback, 'b', 3.)
    assert not matching_execution_wait(feedback, 'a', 3.1)
    for change in (dict(active=False), dict(speed_cap_kmh=10.), dict(status='observed_green_release')):
        assert not traffic_wait_feedback('a', 3., decision, {**traffic, **change})['safety_wait']
    assert not traffic_wait_feedback('a', 3., dict(action='keep_lane', target_speed_kmh=10.), traffic)['safety_wait']


class Waypoint:
    def __init__(self, x, road, lane=-1):
        self.road_id, self.section_id, self.lane_id, self.s = road, 0, lane, x
        self.transform = NS(location=NS(x=x, y=0., z=0.),
                            get_forward_vector=lambda: NS(x=1., y=0., z=0.))
        self.successors = []

    def next(self, distance):
        return self.successors


def test_stop_line_across_road_boundary_requires_connected_matching_lane():
    nodes = [Waypoint(x, 1 if x < 6 else 2) for x in range(0, 12, 2)]
    for a, b in zip(nodes, nodes[1:]):
        a.successors = [b]
    entry = dict(road_id=2, section_id=0, lane_id=-1,
                 position=[10., 0., 0.], forward=[1., 0., 0.])
    assert connected_stop_distance([entry], nodes[0], [2., 0., 0.]) == 8.
    assert connected_stop_distance([{**entry, 'lane_id': -2}], nodes[0], [2., 0., 0.]) is None
    assert connected_stop_distance([{**entry, 'forward': [-1., 0., 0.]}], nodes[0], [2., 0., 0.]) is None
    nodes[2].successors = []
    assert connected_stop_distance([entry], nodes[0], [2., 0., 0.]) is None


def test_passed_light_not_reacquired_inside_junction_but_next_approach_is_checked():
    memory=PassedStopLineMemory()
    entry=dict(position=[10.,0.,0.],forward=[1.,0.,0.])
    memory.record('a',[entry],[12.,0.,0.])
    assert memory.contains('a',[16.,0.,0.])
    assert not memory.contains('b',[16.,0.,0.])
    assert not memory.contains('a',[0.,0.,0.])
    memory.record('a',[entry],[12.,0.,0.])
    assert not memory.contains('a',[120.,0.,0.])


def test_prior_signal_survives_landmark_disappearance_across_road_boundary():
    import numpy as np
    from control.map_signal_observer import MapSignalObserver
    nodes=[Waypoint(x,1 if x<6 else 2) for x in range(0,12,2)]
    for a,b in zip(nodes,nodes[1:]):a.successors=[b]
    nodes[0].get_landmarks_of_type=lambda *args:[]
    observer=MapSignalObserver.__new__(MapSignalObserver)
    observer.last_frame=None;observer.last_observation=None
    observer.world_map=NS(get_waypoint=lambda _:nodes[0])
    observer.camera=NS(attributes={'fov':'100'})
    observer.state_classifier=object()
    observer._observe_static_heads=lambda *args:('GREEN',1.,[])
    observer.debug_directory=None
    observer.active_landmark=NS(id='a',distance=20.,waypoint=NS(road_id=99,section_id=0,lane_id=-1),
        transform=NS(location=NS(x=10.,y=0.,z=0.)))
    observer.stop_lines={'a':[dict(road_id=2,section_id=0,lane_id=-1,
        position=[10.,0.,0.],forward=[1.,0.,0.])]}
    ego=NS(get_transform=lambda:nodes[0].transform,get_location=lambda:nodes[0].transform.location,
           bounding_box=NS(extent=NS(x=2.)))
    result=observer.observe(ego,np.zeros((16,16,3)),frame=1,timestamp_s=1.,camera_inverse=np.eye(4))
    assert result['applicable'] and result['signal_id']=='a'
    assert result['state']=='GREEN' and result['stop_distance_m']==8.
