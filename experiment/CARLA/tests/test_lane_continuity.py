from types import SimpleNamespace as NS

import pytest

from benchmark.catalog import ConfigError
from benchmark.lane_continuity import (build_lane_corridor, trace_lane_corridor,
                                       trace_lane_change_keys, trace_speed_lane_corridor)


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


@pytest.mark.parametrize('disconnected',[False,True])
def test_lane_change_pair_tracks_original_lane_when_route_is_already_in_target_lane(disconnected):
    route=[];entries=[];targets=[]
    for i in range(3):
        road=1 if i==0 else 2
        entry=NS(road_id=road,section_id=0,lane_id=1,lane_type='Driving',
            is_junction=False,lane_change='Right',
            transform=NS(location=Location(i*5,3),rotation=NS(yaw=0)))
        target=NS(road_id=road,section_id=0,lane_id=2,lane_type='Driving',
            is_junction=False,lane_change='Left',
            transform=NS(location=Location(i*5),rotation=NS(yaw=0)))
        entry.get_right_lane=lambda target=target:target
        target.get_left_lane=lambda entry=entry:entry
        target.get_right_lane=lambda:None
        entries.append(entry);targets.append(target)
        route.append(dict(x=i*5,y=0,z=0,distance_m=i*5,road_id=road,section_id=0,lane_id=2))
    for i in range(3):
        entries[i].next=lambda distance,i=i:entries[i+1:i+2] if not disconnected else []
        targets[i].next=lambda distance,i=i:targets[i+1:i+2]
    world_map=NS(get_waypoint=lambda location:targets[int(location.x/5)])
    fixture=dict(entry_lane_key='1:0:1',target_lane_key='1:0:2')
    if disconnected:
        with pytest.raises(ConfigError,match='unique continuous successor'):
            trace_lane_change_keys(world_map,route,0,10,'RIGHT',Location,fixture,stop_before_junction=True)
    else:
        result=trace_lane_change_keys(world_map,route,0,10,'RIGHT',Location,fixture,stop_before_junction=True)
        assert result['entry_lane_keys']==['1:0:1','2:0:1']
        assert result['target_lane_keys']==['1:0:2','2:0:2']


def test_verified_twelve_meter_sample_gap_is_accepted():
    route=[dict(x=x,y=0,z=0,distance_m=x,road_id=1,section_id=0,lane_id=-1)
           for x in (0,12,24)]
    waypoints=[NS(road_id=1,section_id=0,lane_id=-1,lane_type='Driving',
                  transform=NS(location=Location(x),rotation=NS(yaw=0)))
               for x in (0,12,24)]
    for index,waypoint in enumerate(waypoints):
        waypoint.next=lambda distance,index=index:waypoints[index+1:index+2]
    world_map=NS(get_waypoint=lambda location:waypoints[int(location.x/12)])

    assert build_lane_corridor(world_map,route,0,Location)[-1]['end_m']==24
    route[2]['distance_m']=30
    with pytest.raises(ConfigError,match='samples too sparse'):
        build_lane_corridor(world_map,route,0,Location)


def test_bounded_corridor_does_not_validate_later_unrelated_turn():
    route,wps,world_map=setup()
    wps[2].next=lambda distance:[]
    result=build_lane_corridor(world_map,route,0,Location,10)
    assert result[-1]['end_m']==10
    with pytest.raises(ConfigError):
        build_lane_corridor(world_map,route,0,Location)


def test_non_sampled_endpoint_is_clipped_to_requested_proof_window():
    route,wps,world_map=setup()
    result=trace_speed_lane_corridor(world_map,route,0,Location,12.5)
    assert result['verified_end_m']==12.5
    assert result['requested_end_m']==12.5
    assert result['lane_corridor'][-1]['end_m']==12.5


def test_unique_successor_resolves_spatially_overlapping_connector():
    route,wps,world_map=setup()
    other=NS(road_id=99,section_id=0,lane_id=-3,lane_type='Driving',
             transform=NS(location=Location(10),rotation=NS(yaw=0)))
    world_map.get_waypoint=lambda loc:other if loc.x==10 else wps[int(loc.x/5)]
    corridor=build_lane_corridor(world_map,route,0,Location)
    assert corridor[1]['lane_keys']==['1:0:-1','2:0:-1']
    assert all('99:0:-3' not in s['lane_keys'] for s in corridor)


def test_branch_resolves_only_when_nearest_connector_is_wrong():
    route,wps,world_map=setup()
    other=NS(road_id=99,section_id=0,lane_id=-3,lane_type='Driving',
             transform=NS(location=Location(10),rotation=NS(yaw=0)))
    wps[1].next=lambda distance:[wps[2],other]
    world_map.get_waypoint=lambda loc:other if loc.x==10 else wps[int(loc.x/5)]
    result=trace_speed_lane_corridor(world_map,route,0,Location,15)
    assert result['stop_reason'] is None
    assert result['verified_end_m']==15


def test_road_boundary_accepts_nearby_connected_successor():
    route,wps,world_map=setup()
    wrong=NS(road_id=1,section_id=0,lane_id=-1,lane_type='Driving',
             transform=NS(location=Location(9.7),rotation=NS(yaw=0)))
    wps[1].next=lambda distance:[wrong] if distance<=5 else [wps[2]]
    corridor=build_lane_corridor(world_map,route,0,Location)
    assert corridor[-1]['end_m']==15


@pytest.mark.parametrize('fault',[None,'different_geometry','different_heading','not_junction'])
def test_boundary_extension_resolves_only_coincident_junction_connectors(fault):
    route,wps,world_map=setup()
    wps[2].is_junction=True
    other=NS(road_id=99,section_id=0,lane_id=-3,lane_type='Driving',is_junction=True,
             transform=NS(location=Location(10),rotation=NS(yaw=0)))
    boundary=NS(road_id=1,section_id=0,lane_id=-1,lane_type='Driving',
                transform=NS(location=Location(9.7),rotation=NS(yaw=0)))
    if fault=='different_geometry':other.transform.location=Location(11)
    if fault=='different_heading':other.transform.rotation.yaw=20
    if fault=='not_junction':other.is_junction=False
    wps[1].next=lambda distance:[boundary] if distance<=5 else [wps[2],other]
    result=trace_speed_lane_corridor(world_map,route,0,Location,15)
    assert result['verified_end_m']==(15 if fault is None else 5)
    assert result['stop_reason']==(None if fault is None else 'lane corridor has ambiguous forward topology')


@pytest.mark.parametrize('fault',[None,'offset','heading','unconnected','not_junction'])
def test_route_can_continue_through_proven_coincident_connector_alias(fault):
    route,wps,world_map=setup()
    wps[2].is_junction=True
    alias=NS(road_id=88,section_id=0,lane_id=-2,lane_type='Driving',is_junction=True,
             transform=NS(location=Location(10),rotation=NS(yaw=0)),
             next=lambda distance:[wps[3]])
    if fault=='offset':alias.transform.location=Location(11)
    if fault=='heading':alias.transform.rotation.yaw=20
    if fault=='not_junction':alias.is_junction=False
    wps[1].next=lambda distance:[wps[2]] if fault=='unconnected' else [wps[2],alias]
    dead=NS(road_id=77,section_id=0,lane_id=-3,lane_type='Driving',
            transform=NS(location=Location(15),rotation=NS(yaw=0)))
    wps[2].next=lambda distance:[dead]
    if fault is None:
        result=trace_speed_lane_corridor(world_map,route,0,Location,15)
        assert result['verified_end_m']==15
        assert result['stop_reason'] is None
    elif fault=='unconnected':
        with pytest.raises(ConfigError,match='route/map mismatch'):
            trace_speed_lane_corridor(world_map,route,0,Location,15)
    else:
        result=trace_speed_lane_corridor(world_map,route,0,Location,15)
        assert result['verified_end_m']==5
        assert result['stop_reason']=='lane corridor has ambiguous forward topology'


def test_ambiguous_initial_anchor_is_not_overridden_by_route_label():
    route,wps,world_map=setup()
    wps[0].road_id=99
    with pytest.raises(ConfigError,match='route/map mismatch'):
        build_lane_corridor(world_map,route,0,Location)


def test_curve_sampling_reanchors_only_after_lane_continuity_is_proven():
    route,wps,world_map=setup()
    dead=NS(road_id=77,section_id=0,lane_id=-3,lane_type='Driving',
            transform=NS(location=Location(10),rotation=NS(yaw=0)))
    drift=NS(road_id=1,section_id=0,lane_id=-1,lane_type='Driving',
             transform=NS(location=Location(4.7),rotation=NS(yaw=0)),
             next=lambda distance:[dead])
    wps[0].next=lambda distance:[drift]
    result=trace_speed_lane_corridor(world_map,route,0,Location,15)
    assert result['verified_end_m']==15
    assert result['stop_reason'] is None
    drift.road_id=99
    with pytest.raises(ConfigError,match='route/map mismatch'):
        trace_speed_lane_corridor(world_map,route,0,Location,15)


@pytest.mark.parametrize('fault',[None,'branch','diverge','heading','offset'])
def test_short_boundary_stub_requires_unique_geometric_convergence(fault):
    route,wps,world_map=setup()
    connected=NS(road_id=77,section_id=0,lane_id=-3,lane_type='Driving',
                 transform=NS(location=Location(9.7),rotation=NS(yaw=0)),
                 next=lambda distance:[wps[3]])
    wps[1].next=lambda distance:[connected]
    if fault=='branch':connected.next=lambda distance:[wps[3],wps[2]]
    if fault=='diverge':connected.next=lambda distance:[wps[1]]
    if fault=='heading':connected.transform.rotation.yaw=20
    if fault=='offset':connected.transform.location=Location(8)
    if fault is None:
        assert trace_speed_lane_corridor(world_map,route,0,Location,15)['verified_end_m']==15
    else:
        with pytest.raises(ConfigError,match='route/map mismatch'):
            trace_speed_lane_corridor(world_map,route,0,Location,15)


@pytest.mark.parametrize('connected',[True,False])
def test_boundary_initial_anchor_requires_verified_connected_predecessor(connected):
    route,wps,world_map=setup()
    other=NS(road_id=99,section_id=0,lane_id=-3,lane_type='Driving',
             transform=NS(location=Location(10),rotation=NS(yaw=0)))
    world_map.get_waypoint=lambda loc:other if loc.x==10 else wps[int(loc.x/5)]
    if not connected:wps[1].next=lambda distance:[other]
    if connected:
        result=build_lane_corridor(world_map,route,10,Location)
        assert result==[dict(start_m=10,end_m=15,lane_keys=['2:0:-1'])]
    else:
        with pytest.raises(ConfigError,match='route/map mismatch'):
            build_lane_corridor(world_map,route,10,Location)


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


def test_speed_corridor_keeps_proven_prefix_at_fork():
    route,wps,world_map=setup()
    wps[1].next=lambda distance:[wps[2],wps[3]]
    result=trace_speed_lane_corridor(world_map,route,0,Location,15)
    assert result['verified_end_m']==5
    assert result['stop_reason']=='lane corridor has ambiguous forward topology'


@pytest.mark.parametrize('fault',[None,'no_yaw','bad_heading','unconnected','duplicate'])
def test_destination_corridor_requires_unique_published_connected_successor(fault):
    route,wps,world_map=setup()
    for point in route:point['yaw']=0.
    wps[1].next=lambda distance:[wps[2],wps[3]]
    if fault=='no_yaw':route[2].pop('yaw')
    if fault=='bad_heading':route[2]['yaw']=90.
    if fault=='unconnected':wps[1].next=lambda distance:[wps[3]]
    if fault=='duplicate':wps[1].next=lambda distance:[wps[2],wps[2],wps[3]]
    if fault is None:
        assert build_lane_corridor(world_map,route,0,Location,follow_published_route=True)[-1]['end_m']==15
        with pytest.raises(ConfigError):build_lane_corridor(world_map,route,0,Location)
    else:
        with pytest.raises(ConfigError):
            build_lane_corridor(world_map,route,0,Location,follow_published_route=True)


def test_speed_corridor_rejects_real_route_map_mismatch():
    route,wps,world_map=setup()
    wps[2].road_id=99
    with pytest.raises(ConfigError,match='route/map mismatch'):
        trace_speed_lane_corridor(world_map,route,0,Location,15)


def test_unique_successor_can_cross_sampled_road_endpoint():
    route,wps,world_map=setup()
    boundary=NS(road_id=1,section_id=0,lane_id=-1,lane_type='Driving',
                transform=NS(location=Location(9.7),rotation=NS(yaw=0)))
    wps[1].next=lambda distance:[boundary] if distance<=5 else [wps[2]]
    result=trace_speed_lane_corridor(world_map,route,0,Location,15)
    assert result['verified_end_m']==15
    assert result['stop_reason'] is None


def test_unique_predecessor_resolves_short_road_boundary_overshoot():
    route,wps,world_map=setup()
    before=NS(road_id=2,section_id=0,lane_id=-1,lane_type='Driving',
              transform=NS(location=Location(9.8),rotation=NS(yaw=0)))
    wps[1].next=lambda distance:[before] if distance<5 else [
        NS(road_id=3,section_id=0,lane_id=-1,lane_type='Driving',
           transform=NS(location=Location(10.3),rotation=NS(yaw=0)))]
    result=trace_speed_lane_corridor(world_map,route,0,Location,10)
    assert result['verified_end_m']==10
    assert result['stop_reason'] is None


def test_endpoint_extension_does_not_choose_one_branch():
    route,wps,world_map=setup()
    boundary=NS(road_id=1,section_id=0,lane_id=-1,lane_type='Driving',
                transform=NS(location=Location(9.7),rotation=NS(yaw=0)))
    wps[1].next=lambda distance:[boundary] if distance<=5 else [wps[2],wps[3]]
    result=trace_speed_lane_corridor(world_map,route,0,Location,15)
    assert result['verified_end_m']==5
    assert result['stop_reason']=='lane corridor has ambiguous forward topology'


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
                   lane_change='Left',
                   transform=NS(location=Location(point['x']), rotation=NS(yaw=0)),
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

    keys = trace_lane_change_keys(world_map, route, 0, 15, 'LEFT', Location,
                                  fixture, stop_before_junction=True)
    assert keys == dict(entry_lane_keys=['1:0:6'], target_lane_keys=['1:0:5'],
                        lane_change_window_end_m=5)
    entries[1].is_junction = False
    route[1]['is_junction'] = True
    keys = trace_lane_change_keys(world_map, route, 0, 15, 'LEFT', Location,
                                  fixture, stop_before_junction=True)
    assert keys['lane_change_window_end_m'] == 5

    entries[0].is_junction = True
    with pytest.raises(ConfigError, match='invalid lane-change corridor'):
        trace_lane_change_keys(world_map, route, 0, 15, 'LEFT', Location,
                               fixture, stop_before_junction=True)


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


def test_lane_change_uses_connected_successor_at_road_boundary():
    route, wps, world_map = setup()
    for wp in wps:
        wp.is_junction = False
        wp.lane_change = 'Left'
        wp.get_left_lane = lambda wp=wp: NS(
            road_id=wp.road_id, section_id=0, lane_id=-2,
            lane_type='Driving', is_junction=False,
            transform=NS(rotation=NS(yaw=0)))
    wrong = NS(road_id=1, section_id=0, lane_id=-1, is_junction=False)
    world_map.get_waypoint = lambda loc: wrong if loc.x == 10 else wps[int(loc.x/5)]
    fixture = dict(entry_lane_key='1:0:-1', target_lane_key='1:0:-2')

    keys = trace_lane_change_keys(world_map, route, 0, 15, 'LEFT', Location, fixture)
    assert keys['entry_lane_keys'] == ['1:0:-1', '2:0:-1']
    wps[1].next = lambda distance: []
    with pytest.raises(ConfigError, match='route/map mismatch'):
        trace_lane_change_keys(world_map, route, 0, 15, 'LEFT', Location, fixture)
