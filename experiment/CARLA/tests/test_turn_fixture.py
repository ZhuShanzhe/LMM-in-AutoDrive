import pytest
from types import SimpleNamespace as NS

from benchmark.catalog import ConfigError
from benchmark.turn_fixture import turn_evidence, bind_route_turn, waypoint_route


def route():
    return [dict(x=x,y=y,z=0,distance_m=s,yaw=yaw,road_id=road,section_id=0,lane_id=-1,is_junction=j)
            for x,y,s,yaw,road,j in [(0,0,0,0,1,False),(5,0,5,-20,10,True),
                                   (8,-4,10,-60,10,True),(8,-9,15,-90,2,False)]]


def test_waypoint_route_preserves_canonical_distances_after_deduplication():
    def waypoint(x):
        return NS(transform=NS(location=NS(x=x,y=0,z=0),rotation=NS(yaw=0)),
                  road_id=1,section_id=0,lane_id=-1,is_junction=False)

    waypoints=[waypoint(0),waypoint(0),waypoint(10)]
    result=waypoint_route(waypoints,[0,2,12])
    assert [point['distance_m'] for point in result]==[0,12]
    with pytest.raises(ConfigError,match='must align'):
        waypoint_route(waypoints,[0,2])


def test_left_turn_requires_junction_and_distinct_roads():
    evidence=turn_evidence(route(),'LEFT')
    assert evidence['entry_road_key']=='1' and evidence['exit_road_key']=='2'
    assert evidence['heading_change_deg']==-90


def test_formal_turn_binds_next_junction():
    evidence=bind_route_turn(route(),'LEFT',0,20)
    assert evidence['junction_start_m']==5
    assert evidence['exit_road_key']=='2'


@pytest.mark.parametrize('direction,activation,end',[
    ('RIGHT',0,20),('LEFT',5,20),('LEFT',8,20),('LEFT',16,20),('LEFT',0,15)])
def test_formal_turn_rejects_wrong_or_late_binding(direction,activation,end):
    with pytest.raises(ConfigError):
        bind_route_turn(route(),direction,activation,end)


def test_formal_turn_does_not_skip_wrong_first_junction():
    points=route()
    points.extend([dict(p,x=p['x']+20,y=p['y']-15,distance_m=p['distance_m']+30,
                       yaw=-p['yaw']) for p in route()])
    with pytest.raises(ConfigError,match='next junction does not match'):
        bind_route_turn(points,'RIGHT',0,60)


@pytest.mark.parametrize('change',['wrong_direction','same_road','no_junction','shallow','jump'])
def test_incorrect_routes_cannot_claim_turn_success(change):
    points=route()
    if change=='same_road':
        points[-1]['road_id']=1
    if change=='no_junction':
        for p in points:
            p['is_junction']=False
    if change=='shallow':
        points[-1]['yaw']=-10
    if change=='jump':
        points[-1]['x']=100
    with pytest.raises(ConfigError):
        turn_evidence(points,'RIGHT' if change=='wrong_direction' else 'LEFT')
