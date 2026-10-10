from types import SimpleNamespace as NS
import numpy as np
from control.generic_route_pid import GenericRoutePID
from lightweight_vla_adapter.src.risk_behavior import lidar_corridor_evidence


def controller(points, *, matched_road=True):
    c=GenericRoutePID.__new__(GenericRoutePID)
    c.ego=NS(get_location=lambda:NS(x=0.,y=0.,z=0.),
        get_transform=lambda:NS(rotation=NS(yaw=0.)),
        bounding_box=NS(extent=NS(x=2.4,y=.9)))
    c._sensor_road_map=lambda:NS(get_waypoint=lambda *a,**kw:NS(transform=NS(location=NS(z=0.))))
    c._corridor_transition_width=lambda *a:3.5 if matched_road else None
    c.progress_m=lambda:0.
    c.route_context=None;c.route_plan=None;c._completed_lane_navigation=False
    c.route_manager=NS(route=[dict(x=x,y=y,z=0.,distance_m=i) for i,(x,y) in enumerate(points)])
    return c


def test_two_reversed_passed_points_do_not_add_distance_to_sensor_corridor():
    c=controller([(-.8,0.),(-1.5,0.),(.7,0.),(2.7,0.),(30.,0.)])
    road=c.radar_corridor_polyline(60.,include_height=True)
    assert road==[(0.,0.,0.),(.7,0.,0.),(2.7,0.,0.),(30.,0.,0.)]
    cloud=np.array([[x,y,0.] for x in np.arange(4.4,30,.1) for y in (-.2,.2)])
    result=lidar_corridor_evidence(cloud,road,half_width_m=1.25,ego_front_m=2.4,
        sensor_frame=1,observed_at_s=1.)
    assert result['clear_distance_m']>=25


def test_road_mismatch_or_real_corner_is_retained():
    road=controller([(-.8,0.),(-1.5,0.),(.7,0.),(30.,0.)],matched_road=False).radar_corridor_polyline()
    assert road[1:3]==[(-.8,0.),(-1.5,0.)]
    road=controller([(-.8,2.),(-1.5,2.),(.7,2.),(30.,2.)]).radar_corridor_polyline()
    assert road[1:3]==[(-.8,2.),(-1.5,2.)]


def test_large_backtracking_prefix_is_not_silently_deleted():
    road=controller([(-4.,0.),(-5.,0.),(.7,0.),(30.,0.)]).radar_corridor_polyline()
    assert road[1:3]==[(-4.,0.),(-5.,0.)]


def test_static_connected_road_id_seam_is_not_a_corner():
    c=controller([])
    def waypoint(x, *, fork=False):
        w=NS(road_id=37 if x>=-.3 else 36,section_id=0,lane_id=-2,
            is_junction=False,lane_width=3.5,
            transform=NS(location=NS(x=x,y=0.,z=0.),rotation=NS(yaw=0.)))
        w.next=lambda distance:[waypoint(x+distance)]*(2 if fork else 1)
        return w
    c.world=NS()
    c._sensor_road_map=lambda:NS(get_waypoint=lambda location,**kw:waypoint(location.x))
    c._corridor_transition_width=GenericRoutePID._corridor_transition_width.__get__(c)
    assert c._corridor_transition_width((-.8,0.),(-1.5,0.),(1.,0.))==3.5
    c._sensor_road_map=lambda:NS(get_waypoint=lambda location,**kw:waypoint(location.x,fork=True))
    assert c._corridor_transition_width((-.8,0.),(-1.5,0.),(1.,0.)) is None
