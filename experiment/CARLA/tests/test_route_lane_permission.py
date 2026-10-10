from types import SimpleNamespace as NS
import carla
import pytest
from control.generic_route_pid import GenericRoutePID


@pytest.mark.parametrize('direction,permission,allowed',[
    ('left',carla.LaneChange.Left,True),('left',carla.LaneChange.Right,False),
    ('right',carla.LaneChange.Left,False),('right',carla.LaneChange.Right,True),
    ('left',carla.LaneChange.Both,True),('right',carla.LaneChange.Both,True),
    ('left',carla.LaneChange.NONE,False),('right',carla.LaneChange.NONE,False)])
def test_route_controller_uses_carla_permission_enums(direction,permission,allowed):
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller._action='lane_change_'+direction
    controller.world=NS(get_map=lambda:NS(get_waypoint=lambda *a,**kw:NS(lane_change=permission)))
    controller.ego=NS(get_location=lambda:None)
    assert controller._lane_change_legal()==allowed


def test_unknown_map_permission_is_not_authority():
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller._action='lane_change_left'
    controller.world=NS(get_map=lambda:NS(get_waypoint=lambda *a,**kw:None))
    controller.ego=NS(get_location=lambda:None)
    assert not controller._lane_change_legal()


@pytest.mark.parametrize('same_command,same_lane,allowed',[
    (True,True,True),(False,True,False),(True,False,False)])
def test_centering_existing_target_does_not_initiate_another_change(same_command,same_lane,allowed):
    controller=GenericRoutePID.__new__(GenericRoutePID)
    controller._action='lane_change_left';controller._command_id='merge_1'
    controller._pid=NS(_lane_change_command_id='merge_1' if same_command else 'old_merge',
        _lane_change_target_lane_id=-1 if same_lane else -2)
    controller.world=NS(get_map=lambda:NS(get_waypoint=lambda *a,**kw:NS(
        lane_change=carla.LaneChange.NONE,lane_id=-1)))
    controller.ego=NS(get_location=lambda:None)
    assert controller._lane_change_legal()==allowed
