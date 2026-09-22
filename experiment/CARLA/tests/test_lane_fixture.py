from types import SimpleNamespace as NS

import pytest

from benchmark.catalog import ConfigError
from benchmark.lane_fixture import gap_check, select_lane_segment


def actor(x=0,y=0,v=10):
    pose=NS(location=NS(x=x,y=y,z=0),rotation=NS(yaw=0))
    return NS(get_transform=lambda:pose,get_velocity=lambda:NS(x=v,y=0,z=0))


def waypoint(y=0,lane=-1):
    return NS(road_id=1,section_id=0,lane_id=lane,is_junction=False,lane_type='Driving',
              lane_change='Both',transform=actor(y=y).get_transform())


def Snapshot(time,actors):
    return NS(find=actors.get)


def geometry():
    entry=waypoint()
    target=waypoint(y=-3.5,lane=-2)
    target.lane_width=3.5
    entry.get_left_lane=lambda:target
    return entry,target,NS(get_waypoint=lambda location:entry)


def vehicle(identity):
    return NS(id=identity,bounding_box=NS(extent=NS(x=2,y=1)))


FIXTURE=dict(direction='LEFT',legal=True,entry_lane_key='1:0:-1',target_lane_key='1:0:-2')


@pytest.mark.parametrize('x,y,speed,clear',[
    (12,-3.5,10,False),(-12,-3.5,10,False),(-30,-3.5,20,False),
    (40,-3.5,10,True),(5,3.5,10,True),(70,-3.5,0,True),
    (25,-3.5,0,False)])
def test_gap_rejects_nearby_or_fast_closing_traffic(x,y,speed,clear):
    _,_,world_map=geometry()
    snapshot=Snapshot(0,{1:actor(),2:actor(x=x,y=y,v=speed)})
    result=gap_check(snapshot,vehicle(1),[vehicle(2)],world_map,FIXTURE)
    assert result['clear'] is clear


def test_wrong_entry_not_treated_as_safe_gap():
    entry,_,world_map=geometry()
    entry.lane_id=-3
    result=gap_check(Snapshot(0,{1:actor()}),vehicle(1),[],world_map,FIXTURE)
    assert result['reason']=='entry_lane_changed'


def test_selects_and_rebases_without_mutating_source():
    _,_,world_map=geometry()
    route=[dict(x=x,y=0,z=0,yaw=0,distance_m=x,road_id=1,section_id=0,lane_id=-1)
           for x in range(0,201,5)]
    local,fixture,targets,start=select_lane_segment(world_map,route,'LEFT',NS,
                                                   preferred_m=50,length_m=100)
    assert start==50 and local[0]['distance_m']==0
    assert route[10]['distance_m']==50
    assert fixture==FIXTURE and len(targets)>=21


def test_illegal_or_short_corridor_rejected():
    entry,_,world_map=geometry()
    route=[dict(x=x,y=0,z=0,yaw=0,distance_m=x,road_id=1,section_id=0,lane_id=-1)
           for x in range(0,201,5)]
    entry.lane_change='None'
    with pytest.raises(ConfigError,match='corridor'):
        select_lane_segment(world_map,route,'LEFT',NS,preferred_m=0,length_m=100)
    entry.lane_change='Both'
    with pytest.raises(ConfigError,match='corridor'):
        select_lane_segment(world_map,route,'LEFT',NS,preferred_m=0,length_m=300)


@pytest.mark.parametrize('x,y,speed,clear',[
    (-30,3.5,20,False),(12,3.5,10,False),(50,3.5,10,True),
    (5,-3.5,10,True)])
def test_right_lane_gap_uses_right_neighbor(x,y,speed,clear):
    entry=waypoint()
    target=waypoint(y=3.5,lane=-2)
    target.lane_width=3.5
    entry.get_right_lane=lambda:target
    world_map=NS(get_waypoint=lambda location:entry)
    fixture=dict(FIXTURE,direction='RIGHT')
    result=gap_check(Snapshot(0,{1:actor(),2:actor(x=x,y=y,v=speed)}),
                     vehicle(1),[vehicle(2)],world_map,fixture)
    assert result['clear'] is clear


def test_right_lane_selection_preserves_direction():
    entry=waypoint()
    target=waypoint(y=3.5,lane=-2)
    entry.get_right_lane=lambda:target
    route=[dict(x=x,y=0,z=0,yaw=0,distance_m=x,road_id=1,section_id=0,lane_id=-1)
           for x in range(0,201,5)]
    _,fixture,targets,_=select_lane_segment(NS(get_waypoint=lambda location:entry),route,
                                           'RIGHT',NS,preferred_m=50,length_m=100)
    assert fixture['direction']=='RIGHT'
    assert all(p.y==3.5 for p in targets)
