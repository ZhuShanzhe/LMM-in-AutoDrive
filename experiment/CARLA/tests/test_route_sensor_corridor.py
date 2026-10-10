from types import SimpleNamespace as NS

import pytest
from control.generic_route_pid import GenericRoutePID


def corridor(ego,route,mode,yaw=None):
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller.ego=NS(get_location=lambda:NS(x=ego[0],y=ego[1]))
    if yaw is not None:
        controller.ego.get_transform=lambda:NS(rotation=NS(yaw=yaw))
        controller.ego.bounding_box=NS(extent=NS(y=1.))
    controller.progress_m=lambda:0.
    controller.route_context=None
    controller.route_plan=None
    controller.route_manager=NS(route=[dict(x=x,y=y,distance_m=i*2)
                                      for i,(x,y) in enumerate(route)])
    if mode=='waypoints':
        controller.route_context=NS(distances_m=[i*2 for i in range(len(route))])
        controller.route_plan=[(NS(transform=NS(location=NS(x=x,y=y))),None)
                               for x,y in route]
    return controller.radar_corridor_polyline(20.)


@pytest.mark.parametrize('mode',['points','waypoints'])
@pytest.mark.parametrize('yaw_sign',[1,-1])
def test_swept_corridor_does_not_fold_back_to_just_passed_cursor(mode,yaw_sign):
    route=[(yaw_sign*x,0) for x in (0,2,4,6)]
    result=corridor((yaw_sign*1.5,.1),route,mode)
    assert result==[(yaw_sign*1.5,.1)]+route[1:]


@pytest.mark.parametrize('mode',['points','waypoints'])
@pytest.mark.parametrize('ego,route',[
    ((-1,0),[(0,0),(2,0),(4,0)]),
    ((1.5,2),[(0,0),(2,0),(4,0)]),
    ((-1,0),[(0,0),(0,2),(0,4)]),
    ((1.5,0),[(0,0),(20,0),(40,0)]),
])
def test_corridor_retains_unpassed_corner_off_lane_and_sparse_geometry(mode,ego,route):
    assert corridor(ego,route,mode)[1:]==route


@pytest.mark.parametrize('mode',['points','waypoints'])
def test_heading_verified_corridor_trims_offset_passed_anchor(mode):
    ego=(30.966745,-132.333557)
    route=[(31.060387,-135.613373),(27.240799,-123.711243),(27.187325,-121.711960)]
    assert corridor(ego,route,mode,yaw=91.554)[1:]==route[1:]


@pytest.mark.parametrize('mode',['points','waypoints'])
def test_heading_corridor_trims_just_overrun_segment(mode):
    ego=(9.243639,-204.007629)
    route=[(11.280804,-204.000839),(9.280821,-203.992371),(7.280837,-203.983903)]
    assert corridor(ego,route,mode,yaw=180.)[1:]==route[2:]


@pytest.mark.parametrize('mode',['points','waypoints'])
def test_heading_corridor_trims_two_small_passed_anchors(mode):
    route=[(0.,0.),(2.,0.),(4.,0.),(6.,0.)]
    assert corridor((2.5,.1),route,mode,yaw=0.)[1:]==route[2:]


@pytest.mark.parametrize('mode',['points','waypoints'])
@pytest.mark.parametrize('ego,route,yaw',[
    ((1.5,2.),[(0.,0.),(2.,0.),(4.,0.)],0.),
    ((1.5,0.),[(0.,0.),(0.,2.),(0.,4.)],0.),
    ((1.5,0.),[(0.,0.),(-2.,0.),(-4.,0.)],0.),
])
def test_heading_trim_does_not_skip_off_lane_corner_or_reverse_route(mode,ego,route,yaw):
    assert corridor(ego,route,mode,yaw=yaw)[1:]==route
