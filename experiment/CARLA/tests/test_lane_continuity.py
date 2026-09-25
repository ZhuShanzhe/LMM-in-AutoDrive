from types import SimpleNamespace as NS

import pytest

from benchmark.catalog import ConfigError
from benchmark.lane_continuity import (build_lane_corridor, trace_lane_corridor,
                                       trace_lane_change_keys)


class Location:
    def __init__(self,x,y=0,z=0): self.x,self.y,self.z=x,y,z
    def distance(self,other):
        return ((self.x-other.x)**2+(self.y-other.y)**2+(self.z-other.z)**2)**.5


def setup():
    points=[dict(x=i*5,y=0,z=0,distance_m=i*5,road_id=1 if i<2 else 2,
                 section_id=0,lane_id=-1) for i in range(4)]
    wps=[NS(road_id=p['road_id'],section_id=0,lane_id=-1,lane_type='Driving',
            transform=NS(location=Location(p['x']),rotation=NS(yaw=0))) for p in points]
    for i,w in enumerate(wps): w.next=lambda distance,i=i:wps[i+1:i+2]
    return points,wps,NS(get_waypoint=lambda loc:wps[int(loc.x/5)])


def test_lane_continuity_allows_connected_road_id_change():
    route,wps,world_map=setup()
    result=build_lane_corridor(world_map,route,2,Location)
    assert result[1]['lane_keys']==['1:0:-1','2:0:-1']
    assert result[-1]['end_m']==15


def test_bounded_corridor_does_not_validate_later_unrelated_turn():
    route,wps,world_map=setup()
    wps[2].next=lambda distance:[]
    result=build_lane_corridor(world_map,route,0,Location,10)
    assert result[-1]['end_m']==10
    with pytest.raises(ConfigError):
        build_lane_corridor(world_map,route,0,Location)


def test_unique_successor_resolves_spatially_overlapping_connector():
    route,wps,world_map=setup()
    other=NS(road_id=99,section_id=0,lane_id=-3,lane_type='Driving',
             transform=NS(location=Location(10),rotation=NS(yaw=0)))
    world_map.get_waypoint=lambda loc:other if loc.x==10 else wps[int(loc.x/5)]
    corridor=build_lane_corridor(world_map,route,0,Location)
    assert corridor[1]['lane_keys']==['1:0:-1','2:0:-1']
    assert all('99:0:-3' not in s['lane_keys'] for s in corridor)


def test_road_boundary_accepts_nearby_connected_successor():
    route,wps,world_map=setup()
    wrong=NS(road_id=1,section_id=0,lane_id=-1,lane_type='Driving',
             transform=NS(location=Location(9.7),rotation=NS(yaw=0)))
    wps[1].next=lambda distance:[wrong] if distance<=5 else [wps[2]]
    corridor=build_lane_corridor(world_map,route,0,Location)
    assert corridor[-1]['end_m']==15


def test_ambiguous_initial_anchor_is_not_overridden_by_route_label():
    route,wps,world_map=setup()
    wps[0].road_id=99
    with pytest.raises(ConfigError,match='route/map mismatch'):
        build_lane_corridor(world_map,route,0,Location)


def test_partial_corridor_records_unverified_future_without_accepting_branch():
    route,wps,world_map=setup()
    wps[1].next=lambda distance:[wps[2],wps[3]]
    result=trace_lane_corridor(world_map,route,0,Location)
    assert result['verified_end_m']==5
    assert result['requested_end_m']==15
    assert result['stop_reason']=='lane corridor has ambiguous forward topology'
    assert len(result['lane_corridor'])==1
    with pytest.raises(ConfigError):
        build_lane_corridor(world_map,route,0,Location)


@pytest.mark.parametrize('end',[0,-1,20,float('nan')])
def test_bad_corridor_end_is_rejected(end):
    route,wps,world_map=setup()
    with pytest.raises(ConfigError):
        build_lane_corridor(world_map,route,0,Location,end)


@pytest.mark.parametrize('fault',['branch','wrong_successor','off_map','reverse','sparse'])
def test_invalid_lane_corridors_are_rejected(fault):
    route,wps,world_map=setup()
    if fault=='branch': wps[1].next=lambda distance:[wps[2],wps[3]]
    elif fault=='wrong_successor': wps[1].next=lambda distance:[wps[3]]
    elif fault=='off_map': route[1]['y']=2
    elif fault=='reverse':
        successor=NS(road_id=2,section_id=0,lane_id=-1,lane_type='Driving',
                     transform=NS(location=Location(10),rotation=NS(yaw=180)))
        wps[1].next=lambda distance:[successor]
    elif fault=='sparse': route[-1]['distance_m']=100
    with pytest.raises(ConfigError): build_lane_corridor(world_map,route,0,Location)


def test_lane_change_target_identity_continues_after_junction():
    route = [dict(x=i*5, y=0, z=0, distance_m=i*5,
                  road_id=(1 if i == 0 else 99 if i == 1 else 2),
                  section_id=0, lane_id=6) for i in range(4)]
    entries = []
    for point in route:
        road = point['road_id']
        target = NS(road_id=road, section_id=0, lane_id=5,
                    lane_type='Driving', is_junction=False,
                    transform=NS(rotation=NS(yaw=0)))
        entry = NS(road_id=road, section_id=0, lane_id=6,
                   lane_type='Driving', is_junction=road == 99,
                   lane_change='Left', transform=NS(rotation=NS(yaw=0)),
                   get_left_lane=lambda target=target: target)
        entries.append(entry)
    world_map = NS(get_waypoint=lambda location: entries[int(location.x/5)])
    fixture = dict(entry_lane_key='1:0:6', target_lane_key='1:0:5')

    keys = trace_lane_change_keys(world_map, route, 0, 15, 'LEFT', Location, fixture)

    assert keys['entry_lane_keys'] == ['1:0:6', '2:0:6']
    assert keys['target_lane_keys'] == ['1:0:5', '2:0:5']
    route[2]['lane_id'] = 7
    with pytest.raises(ConfigError, match='route/map mismatch'):
        trace_lane_change_keys(world_map, route, 0, 15, 'LEFT', Location, fixture)


def test_lane_change_skips_overlapping_junction_connector():
    route, wps, world_map = setup()
    for wp in wps:
        wp.is_junction = False
        wp.lane_change = 'Left'
        wp.get_left_lane = lambda wp=wp: NS(
            road_id=wp.road_id, section_id=0, lane_id=-2,
            lane_type='Driving', is_junction=False,
            transform=NS(rotation=NS(yaw=0)))
    overlap = NS(road_id=99, section_id=0, lane_id=-1,
                 lane_type='Driving', is_junction=True)
    world_map.get_waypoint = lambda loc: overlap if loc.x == 5 else wps[int(loc.x/5)]
    fixture = dict(entry_lane_key='1:0:-1', target_lane_key='1:0:-2')
    keys = trace_lane_change_keys(world_map, route, 0, 15, 'LEFT', Location, fixture)
    assert keys['entry_lane_keys'] == ['1:0:-1', '2:0:-1']
    assert keys['target_lane_keys'] == ['1:0:-2', '2:0:-2']
