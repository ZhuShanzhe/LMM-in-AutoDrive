import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from continuous.route_manager import RouteManager


def waypoint(yaw,junction):
    return NS(is_junction=junction,transform=NS(rotation=NS(yaw=yaw)))


def test_straight_uses_exit_not_connector_entry_heading():
    manager=RouteManager(NS(get_map=lambda:None))
    current=waypoint(0,False)
    deceptive_turn=waypoint(0,True)
    straight=waypoint(5,True)
    manager._trace_junction_exit=lambda candidate:waypoint(90 if candidate is deceptive_turn else 0,False)
    assert manager._choose_straight(current,[deceptive_turn,straight]) is straight


def test_unresolved_connector_is_not_selected_as_straight():
    manager=RouteManager(NS(get_map=lambda:None))
    manager._trace_junction_exit=lambda candidate:None
    assert manager._choose_straight(waypoint(0,False),[waypoint(0,True)]) is None
