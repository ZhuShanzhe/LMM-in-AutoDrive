from dataclasses import dataclass
import math
from types import SimpleNamespace as NS
import pytest
from benchmark.lane_continuity import _coincident_connector_prefix
from benchmark.junction_alias import coincident_junction_alias_matches
from benchmark.task_oracle import TaskOracle


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.

    def distance(self,other):
        return math.dist((self.x,self.y,self.z),(other.x,other.y,other.z))


class Waypoint:
    def __init__(self,offset,branch=False):
        self.offset=offset;self.branch=branch
        self.road_id=2 if branch else 1
        self.section_id=0;self.lane_id=1
        x=offset+.25 if branch else offset
        self.is_junction=x>=.113
        self.transform=NS(location=Point(x,.05*x*x if branch else 0.),
                          rotation=NS(yaw=math.degrees(math.atan(.1*x)) if branch else 0.))

    def next(self,distance):
        if self.branch:return [Waypoint(self.offset+distance,True)]
        offset=self.offset+distance
        return [Waypoint(offset)]+([Waypoint(offset,True)] if offset>=.113 else [])

    def previous(self,distance):
        return [Waypoint(self.offset-distance,self.branch)]


def test_arc_sample_offset_does_not_hide_geometrically_shared_entry():
    ranges=_coincident_connector_prefix(Waypoint(0.),'1:0:1',100.,3.,
        position_limit=.2,heading_limit=10.)
    assert len(ranges)==1
    interval=ranges[0]
    assert interval['lane_key']=='2:0:1'
    assert 100.113 <= interval['start_m'] < 100.1132
    assert 101.7<interval['end_m']<101.9
    segment=dict(shared_entry_alias_source='static_topology_shared_entry_0.2m_10deg',
                 shared_entry_junction_lane_ranges=ranges)
    ego=dict(in_junction=True,lane_key='2:0:1',route_s_m=101.,
             route_error_m=.06,yaw_deg=.1,route_yaw_deg=0.)
    assert coincident_junction_alias_matches(ego,[segment],.35,5.)
    assert not coincident_junction_alias_matches(dict(ego,yaw_deg=6.),[segment],.35,5.)
    assert not coincident_junction_alias_matches(dict(ego,route_s_m=102.),[segment],.35,5.)


@pytest.mark.parametrize('length',[0.,-1.,15.1])
def test_unbounded_or_reversed_prefix_not_accepted(length):
    assert _coincident_connector_prefix(Waypoint(0.),'1:0:1',100.,length)==[]


def destination_observation(**changes):
    ego=dict(speed_kmh=30.,route_s_m=4949.,in_junction=True,lane_key='2:0:1',
        route_error_m=.067,yaw_deg=.15,route_yaw_deg=0.,lateral_error_m=.06,
        heading_error_deg=-6.3,position_m=dict(x=4949.,y=.067,z=0.))
    ego.update(changes)
    return dict(schema_version='task_truth/1.0',source='simulator_truth',frame=1,
        sim_time_s=0.,scenario_valid=True,ego=ego,actors={},safety=dict(collisions=0,violations=0),
        fixture=dict(steps={'0':dict(route_end_s_m=5000.,position_m=dict(x=5000.,y=0.,z=0.),
            lane_corridor=[dict(start_m=4600.,end_m=5000.,lane_keys=['1:0:1'],
                shared_entry_alias_source='static_topology_shared_entry_0.2m_10deg',
                shared_entry_junction_lane_ranges=[dict(lane_key='2:0:1',start_m=4947.8,end_m=4949.6)])])}))


def destination_oracle():
    return TaskOracle(dict(schema_version='task_oracle/1.0',task_id='destination',
        activate_m=4940.,timeout_s=120.,max_frame_gap_s=.15,violation_budget=0,
        steps=[dict(kind='destination',max_distance_m=3.,max_remaining_m=3.,keep_lane=True,
                    max_lateral_error_m=.35,max_heading_error_deg=5.)]))


def test_destination_uses_published_route_heading_for_a_proven_alias():
    oracle=destination_oracle()
    assert oracle.update(destination_observation())['status']=='RUNNING'
    row=destination_observation(route_s_m=4997.5,lane_key='1:0:1',heading_error_deg=.15,
                                position_m=dict(x=4997.5,y=.067,z=0.))
    row.update(frame=2,sim_time_s=.05)
    assert oracle.update(row)['status']=='SUCCESS'


@pytest.mark.parametrize('changes',[dict(yaw_deg=6.),dict(route_error_m=.351),
    dict(route_s_m=4949.7),dict(lane_key='unproven'),dict(in_junction=False)])
def test_destination_still_rejects_actual_departure_and_unproven_ids(changes):
    assert destination_oracle().update(destination_observation(**changes))['status']=='FAILURE'
