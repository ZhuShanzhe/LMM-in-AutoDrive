import math
from types import SimpleNamespace as NS

import pytest
from control.generic_route_pid import GenericRoutePID


@pytest.mark.parametrize('yaw', [0., 35., 90., 180., -90.])
@pytest.mark.parametrize('translation', [(0., 0.), (300., -500.)])
def test_passed_adjacent_lane_stitch_is_rotation_and_location_invariant(yaw, translation):
    angle = math.radians(yaw)
    rotate = lambda p: (translation[0]+p[0]*math.cos(angle)-p[1]*math.sin(angle),
                        translation[1]+p[0]*math.sin(angle)+p[1]*math.cos(angle))
    ego = rotate((0., 0.))
    route = [rotate(p) for p in [(-2.5, 0.), (-.4, 3.5), (2., 3.5), (4., 3.5)]]
    controller = GenericRoutePID.__new__(GenericRoutePID)
    controller.ego = NS(get_location=lambda: NS(x=ego[0], y=ego[1], z=0.),
                        get_transform=lambda: NS(rotation=NS(yaw=yaw)),
                        bounding_box=NS(extent=NS(y=1.)))
    controller.route_context = None
    controller.route_plan = None
    controller.route_manager = NS(route=[dict(x=p[0], y=p[1], distance_m=i*2.)
                                         for i, p in enumerate(route)])
    controller.progress_m = lambda: 0.

    def mapped(location, **_):
        dx, dy = location.x-translation[0], location.y-translation[1]
        across = -dx*math.sin(angle)+dy*math.cos(angle)
        return NS(road_id=17, section_id=0, lane_id=-1 if across < 1.75 else -2,
                  lane_width=3.5, is_junction=False,
                  transform=NS(location=location, rotation=NS(yaw=yaw)))
    controller.world = NS(get_map=lambda: NS(get_waypoint=mapped))
    result = controller.radar_corridor_polyline(20.)
    assert result[0] == ego
    assert route[0] not in result[1:]
    assert result[-1] == route[-1]


@pytest.mark.parametrize('invalid', ['junction', 'opposite', 'different_road', 'off_road', 'non_adjacent'])
def test_topology_proof_rejects_unverified_corner_or_lane_jump(invalid):
    c = GenericRoutePID.__new__(GenericRoutePID)
    c.ego = NS(get_location=lambda: NS(x=0., y=0., z=1.))
    calls = []
    def mapped(location, **_):
        i = len(calls); calls.append(location)
        return NS(road_id=99 if invalid == 'different_road' and i == 2 else 17,
                  section_id=0, lane_id=-4 if invalid == 'non_adjacent' and i == 2 else -1,
                  lane_width=3.5, is_junction=invalid == 'junction',
                  transform=NS(location=NS(x=location.x+10 if invalid == 'off_road' else location.x,
                                           y=location.y),
                               rotation=NS(yaw=180. if invalid == 'opposite' else 0.)))
    c.world = NS(get_map=lambda: NS(get_waypoint=mapped))
    assert c._corridor_transition_width((-2., 0.), (1., 3.5), (1., 0.)) is None
