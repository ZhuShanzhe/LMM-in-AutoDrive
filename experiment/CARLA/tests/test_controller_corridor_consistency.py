from types import SimpleNamespace as NS
import pytest
from control.generic_route_pid import GenericRoutePID


@pytest.mark.parametrize('sign',[1.,-1.])
def test_observation_uses_controller_logical_lane_translation(sign):
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller.ego=NS(get_location=lambda:NS(x=0.,y=0.))
    controller.progress_m=lambda:0.
    controller.route_context=NS(distances_m=[0.,2.,4.,6.])
    controller.route_plan=[(NS(transform=NS(location=NS(x=sign*i*2.,y=3.5))),None) for i in range(4)]
    controller.logical_lane_adapter=object()
    controller._waypoint_target=lambda offset,**snapshot:dict(x=sign*offset,y=0.)
    assert controller.radar_corridor_polyline(20.)==[(0.,0.),(sign*2.,0.),(sign*4.,0.),(sign*6.,0.)]


def test_unknown_controller_translation_is_not_replaced_by_a_raw_other_lane():
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller.ego=NS(get_location=lambda:NS(x=0.,y=0.))
    controller.progress_m=lambda:0.
    controller.route_context=NS(distances_m=[0.,2.,4.])
    controller.route_plan=[(NS(transform=NS(location=NS(x=i*2.,y=3.5))),None) for i in range(3)]
    controller.logical_lane_adapter=object()
    controller._waypoint_target=lambda offset,**snapshot:None
    controller._route_target=lambda:None
    assert controller.radar_corridor_polyline(20.)==[(0.,0.)]
