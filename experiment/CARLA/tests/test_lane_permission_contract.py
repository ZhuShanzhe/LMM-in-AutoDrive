import carla
import pytest
from control.pid_controller import EgoPIDController
from experiment.CARLA.tests.test_control_safety_regression import FakeVehicle,FakeMap,FakeWaypoint,transform


@pytest.mark.parametrize('permission,left,right',[
    (carla.LaneChange.Left,True,False),
    (carla.LaneChange.Right,False,True),
    (carla.LaneChange.Both,True,True),
    (carla.LaneChange.NONE,False,False),
])
def test_lane_permission_uses_carla_enums_not_reversed_literals(permission,left,right):
    waypoint=FakeWaypoint(transform())
    waypoint.lane_change=permission
    state=EgoPIDController(FakeVehicle(),FakeMap(waypoint)).get_execution_state()
    assert state['left_lane_change_allowed'] is left
    assert state['right_lane_change_allowed'] is right
