from types import SimpleNamespace as NS
import pytest
from benchmark.catalog import ConfigError
from benchmark.overtake_geometry import build_overtake_geometry


def geometry(return_allowed=True):
    def waypoint(p):
        lane=-1 if p.y==0 else -2
        wp=NS(road_id=1,section_id=0,lane_id=lane,is_junction=False,lane_type='Driving',
            lane_change='Both' if lane==-1 or return_allowed else 'None',
            transform=NS(location=NS(x=p.x,y=p.y,z=0),rotation=NS(yaw=0)))
        wp.get_left_lane=lambda:waypoint(NS(x=p.x,y=-3.5))
        wp.get_right_lane=lambda:waypoint(NS(x=p.x,y=0))
        wp.next=lambda distance:[waypoint(NS(x=p.x+distance,y=p.y))]
        return wp
    route=[dict(x=x,y=0,z=0,yaw=0,distance_m=x,road_id=1,section_id=0,lane_id=-1)
           for x in range(0,401,5)]
    return NS(get_waypoint=waypoint),route


def test_bounded_corridor_preserves_source_and_checks_both_directions():
    world_map,route=geometry()
    result=build_overtake_geometry(world_map,route,NS,preferred_m=50)
    assert result['source_entry_m']==50
    assert result['remaining_after_requested_anchor_m']==280
    assert result['route'][-1]['distance_m']==280
    assert route[10]['distance_m']==50
    assert result['returning']['target_lane_key']==result['outgoing']['entry_lane_key']
    assert not result['execution_supported']


def test_illegal_return_and_distant_search_are_rejected():
    world_map,route=geometry(False)
    with pytest.raises(ConfigError,match='no legal'):
        build_overtake_geometry(world_map,route,NS,preferred_m=0)
    world_map,route=geometry()
    with pytest.raises(ConfigError,match='no legal'):
        build_overtake_geometry(world_map,route,NS,preferred_m=1000,max_anchor_shift_m=10)


def test_connected_road_identity_change_is_supported():
    world_map,route=geometry()
    original=world_map.get_waypoint
    def changed(p):
        wp=original(p)
        wp.road_id=1 if p.x<100 else 2
        wp.next=lambda distance:[changed(NS(x=p.x+distance,y=p.y))]
        wp.get_left_lane=lambda:changed(NS(x=p.x,y=-3.5))
        wp.get_right_lane=lambda:changed(NS(x=p.x,y=0))
        return wp
    world_map.get_waypoint=changed
    for p in route:
        p['road_id']=1 if p['x']<100 else 2
    result=build_overtake_geometry(world_map,route,NS,preferred_m=0)
    assert {p['outgoing']['entry_lane_key'] for p in result['lane_pairs']}=={'1:0:-1','2:0:-1'}


def test_nearby_disconnected_lane_is_not_a_valid_corridor():
    world_map,route=geometry()
    original=world_map.get_waypoint
    def disconnected(p):
        wp=original(p)
        wp.next=lambda distance:[]
        return wp
    world_map.get_waypoint=disconnected
    with pytest.raises(ConfigError,match='ambiguous forward topology'):
        build_overtake_geometry(world_map,route,NS,preferred_m=0)


def test_curved_forward_link_is_supported():
    import math
    from benchmark.overtake_geometry import check_forward_link
    def waypoint(angle):
        wp=NS(road_id=1,section_id=0,lane_id=-1,
            transform=NS(location=NS(x=50*math.sin(angle),y=50*(1-math.cos(angle)),z=0),
                         rotation=NS(yaw=math.degrees(angle))))
        wp.next=lambda distance:[waypoint(angle+distance/50)]
        return wp
    for angle in (0,.5,1):
        check_forward_link(waypoint(angle),waypoint(angle+.08))
