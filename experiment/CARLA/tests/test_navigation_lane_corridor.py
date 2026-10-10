import unittest
from types import SimpleNamespace as NS
from control.lane_corridor_navigation import navigation_reference,select_navigation_successor,select_navigation_branch


def waypoint(road,lane,x=20.,y=0.,z=0.,yaw=0.):
    return NS(road_id=road,section_id=0,lane_id=lane,lane_width=3.5,
        transform=NS(location=NS(x=x,y=y,z=z),rotation=NS(yaw=yaw)))


class NavigationLaneCorridorTests(unittest.TestCase):
    def test_public_navigation_selects_straight_or_turn_without_hardcoded_map(self):
        target=waypoint(100,-1)
        a=waypoint(201,-1,y=3.5)
        b=waypoint(202,-1,y=3.5,yaw=15.)
        for selected,reference in [(a,waypoint(201,-2)),(b,waypoint(202,-2,yaw=15.))]:
            with self.subTest(road=selected.road_id):
                self.assertIs(select_navigation_successor([a,b],reference,target),selected)

    def test_unknown_conflicting_or_ambiguous_navigation_rejected(self):
        target=waypoint(100,-1);reference=waypoint(201,-2)
        a=waypoint(201,-1,y=3.5);other=waypoint(202,-1)
        self.assertIsNone(select_navigation_successor([a,other],None,target))
        for fault in [waypoint(201,1),waypoint(201,-1,z=2.),waypoint(201,-1,yaw=90.),
                      waypoint(201,-1,x=100.),waypoint(201,-1,x=float('nan'))]:
            with self.subTest(fault=fault):
                self.assertIsNone(select_navigation_successor([fault,other],reference,target))
        self.assertIsNone(select_navigation_successor([a,waypoint(201,-1,y=3.5)],reference,target))

    def test_existing_unique_successor_unchanged(self):
        candidate=waypoint(10,-3)
        self.assertIs(select_navigation_successor([candidate],None,None),candidate)

    def test_lane_sign_change_between_roads_is_not_opposing_traffic(self):
        target=waypoint(100,-1)
        candidate=waypoint(201,1)
        reference=waypoint(201,2)
        self.assertIs(select_navigation_successor([candidate,waypoint(202,-1)],reference,target),candidate)

    def test_boundary_fork_uses_later_route_not_nearest_branch_guess(self):
        a=waypoint(201,1);b=waypoint(202,1)
        a.next=lambda distance:[waypoint(301,1,x=30.)]
        b.next=lambda distance:[waypoint(302,-1,x=25.,y=10.,yaw=90.)]
        self.assertIsNone(select_navigation_successor([a,b],waypoint(100,1),waypoint(100,1)))
        self.assertIs(select_navigation_branch([a,b],waypoint(301,2,x=30.),extension_m=10.),a)
        self.assertIs(select_navigation_branch([a,b],waypoint(302,-2,x=25.,y=10.,yaw=90.),extension_m=10.),b)

    def test_later_reference_must_be_unique_and_same_level(self):
        a=waypoint(201,1);b=waypoint(202,1)
        a.next=lambda distance:[waypoint(301,1,x=30.)]
        b.next=lambda distance:[waypoint(301,2,x=30.)]
        self.assertIsNone(select_navigation_branch([a,b],waypoint(301,1,x=30.),extension_m=10.))
        b.next=lambda distance:[waypoint(302,1,x=30.)]
        for invalid in [None,waypoint(301,1,x=30.,z=5.),waypoint(301,1,x=30.,yaw=90.)]:
            self.assertIsNone(select_navigation_branch([a,b],invalid,extension_m=10.))
        for invalid in [0.,True,float('nan'),float('inf')]:
            self.assertIsNone(select_navigation_branch([a,b],waypoint(301,1,x=30.),extension_m=invalid))

    def test_reference_interpolates_only_public_route(self):
        route=[dict(distance_m=float(i),x=float(i),y=0.,z=0.) for i in range(0,51,2)]
        controller=NS(route_manager=NS(route=route))
        seen=[]
        road_map=NS(get_waypoint=lambda location:seen.append(location) or waypoint(201,-2,x=location.x))
        result=navigation_reference(controller,road_map,NS(x=10.,y=0.,z=0.),10.25,20.)
        self.assertEqual(result.transform.location.x,30.25)
        self.assertEqual(seen[0].x,30.25)

    def test_no_route_invalid_route_and_outside_range_not_guessed(self):
        road_map=NS(get_waypoint=lambda _:self.fail('Invalid reference must not query map'))
        cases=[NS(),NS(route_manager=NS(route=[dict(distance_m=0.,x=0.,y=0.,z=0.),
            dict(distance_m=10.,x=10.,y=0.,z=0.)])),
            NS(route_manager=NS(route=[dict(distance_m=0.,x=float('nan'),y=0.,z=0.),
                dict(distance_m=2.,x=2.,y=0.,z=0.)]))]
        for controller in cases:
            self.assertIsNone(navigation_reference(controller,road_map,NS(x=0.,y=0.,z=0.),0.,1.))


if __name__=='__main__':unittest.main()
