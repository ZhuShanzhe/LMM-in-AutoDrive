import copy

import pytest

from control.scheduled_scene_bridge_policy import ScheduledSceneBridgePolicy


def _intent(direction):
    return {
        "intent": {
            "steps": [
                {"action": "KEEP_LANE", "parameters": {}},
                {"action": "TURN", "parameters": {"direction": direction}},
                {"action": "TURN", "parameters": {"direction": "LEFT"}},
            ]
        }
    }


@pytest.mark.parametrize("direction", ["LEFT", "RIGHT"])
def test_route_target_is_bound_only_to_next_scheduled_turn(direction):
    intent = _intent(direction)
    original = copy.deepcopy(intent)

    result = ScheduledSceneBridgePolicy._bind_planned_turn_target(
        intent,
        {"x": 12.0, "y": -3.0, "z": 0.5, "yaw": 90.0},
    )

    assert intent == original
    assert result["intent"]["steps"][1]["parameters"]["target_location"] == {
        "x": 12.0,
        "y": -3.0,
        "z": 0.5,
        "yaw": 90.0,
    }
    assert "target_location" not in result["intent"]["steps"][2]["parameters"]


def test_missing_or_invalid_route_target_keeps_turn_unresolved():
    intent = _intent("RIGHT")

    for target in (None, {}, {"x": "not-a-number", "y": 2.0}):
        result = ScheduledSceneBridgePolicy._bind_planned_turn_target(intent, target)
        assert "target_location" not in result["intent"]["steps"][1]["parameters"]


def test_explicit_turn_target_is_not_overwritten():
    intent = _intent("RIGHT")
    explicit = {"x": 1.0, "y": 2.0, "z": 0.0}
    intent["intent"]["steps"][1]["parameters"]["target_location"] = explicit

    result = ScheduledSceneBridgePolicy._bind_planned_turn_target(
        intent, {"x": 12.0, "y": -3.0}
    )

    assert result["intent"]["steps"][1]["parameters"]["target_location"] == explicit
