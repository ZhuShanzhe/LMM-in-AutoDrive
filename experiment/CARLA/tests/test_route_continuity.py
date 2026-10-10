from types import SimpleNamespace as NS
import pytest
from scenarios.route_continuity import repair_route_continuity,RouteContinuityError


def wp(x,y=0.,yaw=0.,road=1,lane=1,z=0.):
    point=NS(transform=NS(location=NS(x=x,y=y,z=z),rotation=NS(yaw=yaw)),
        road_id=road,section_id=0,lane_id=lane)
    point.children=[];point.next=lambda sampling:point.children
    return point


def test_overshot_junction_is_replaced_with_real_successors():
    approach=wp(0.);entry=wp(2.);straight=wp(4.);overshoot=wp(6.)
    bend=wp(3.,1.,45.,road=2);target=wp(3.,3.,90.,road=2)
    approach.children=[entry];entry.children=[straight,bend];straight.children=[overshoot]
    bend.children=[target]
    source=[(p,'FOLLOW') for p in (approach,entry,straight,overshoot,target)]
    result,repairs=repair_route_continuity(source)
    assert [p for p,_ in result]==[approach,entry,bend,target]
    assert repairs[0]['removed_waypoints']==2
    assert repairs[0]['policy_truth_access'] is False
    assert [option for _,option in result[-2:]]==['RIGHT','RIGHT']


def test_repaired_left_turn_retains_navigation_enum():
    from enum import Enum
    class Option(Enum):
        FOLLOW=0
        LEFT=1
        RIGHT=2
    entry=wp(2.);straight=wp(4.);overshoot=wp(6.)
    bend=wp(3.,-1.,-45.,road=2);target=wp(3.,-3.,-90.,road=2)
    entry.children=[straight,bend];straight.children=[overshoot];bend.children=[target]
    result,_=repair_route_continuity([(p,Option.FOLLOW) for p in (entry,straight,overshoot,target)])
    assert result[-1][1] is Option.LEFT


def test_valid_tight_turn_and_ordinary_trace_are_not_rewritten():
    points=[wp(0.),wp(2.),wp(3.,1.,45.),wp(3.,3.,90.),wp(3.,5.,90.)]
    source=[(p,'FOLLOW') for p in points]
    result,repairs=repair_route_continuity(source)
    assert result==source and repairs==[]


def test_large_forward_gap_is_filled_by_directed_successors():
    points=[wp(float(x)) for x in range(0,19,2)]
    for a,b in zip(points,points[1:]):a.children=[b]
    result,repairs=repair_route_continuity([(points[0],'FOLLOW'),(points[-1],'FOLLOW')])
    assert all(b[0].transform.location.x-a[0].transform.location.x<=4. for a,b in zip(result,result[1:]))
    assert repairs


def test_unconnected_or_opposing_lane_is_rejected_not_interpolated():
    origin=wp(0.);other=wp(-12.,yaw=180.,road=2)
    with pytest.raises(RouteContinuityError):
        repair_route_continuity([(origin,'FOLLOW'),(other,'FOLLOW')])


def test_cyclic_map_search_is_bounded():
    origin=wp(0.);next_point=wp(2.);origin.children=[next_point];next_point.children=[origin]
    with pytest.raises(RouteContinuityError):
        repair_route_continuity([(origin,'FOLLOW'),(wp(-15.,road=3),'FOLLOW')])


def test_existing_legal_adjacent_lane_transition_is_preserved():
    origin=wp(0.,lane=2);neighbor=wp(0.,y=3.5,lane=1);target=wp(12.,y=3.5,lane=1)
    origin.lane_change='Right';origin.get_right_lane=lambda:neighbor
    source=[(origin,'CHANGE_RIGHT'),(target,'FOLLOW')]
    result,repairs=repair_route_continuity(source)
    assert result==source and repairs==[]


def test_lateral_spawn_stub_is_removed_before_vehicle_spawn():
    origin=wp(0.,lane=2);neighbor=wp(0.,y=3.5,lane=1)
    second=wp(-.7,y=3.5,lane=1);third=wp(1.3,y=3.5,lane=1)
    origin.get_left_lane=lambda:neighbor
    source=[(point,'FOLLOW') for point in (origin,second,third)]
    result,repairs=repair_route_continuity(source)
    assert result==source[1:]
    assert repairs[0]['source']=='map_adjacent_lane_spawn_anchor'


def test_solid_or_nonadjacent_lane_hop_is_rejected():
    origin=wp(0.,lane=2);target=wp(12.,y=3.5,lane=1)
    origin.lane_change='None';origin.get_right_lane=lambda:target
    with pytest.raises(RouteContinuityError):
        repair_route_continuity([(origin,'CHANGE_RIGHT'),(target,'FOLLOW')])


@pytest.mark.parametrize('sampling',[0.,-1.,float('nan')])
def test_invalid_sampling_is_rejected(sampling):
    with pytest.raises(ValueError):repair_route_continuity([],sampling)
