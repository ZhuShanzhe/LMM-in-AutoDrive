import pytest

from control.generic_instruction_fsm import GenericInstructionFSM
from scene2_runtime_interface import _scene2_step_contract

class Service:
    def __init__(self, action, parameters=None):
        self.action = action
        self.parameters = parameters or {}
        self.calls = []

    def parse_text(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return dict(
            intent=dict(steps=[dict(step_id="s1", action=self.action, parameters=self.parameters, trigger=dict(type="IMMEDIATE"))]),
            parse_result=dict(status="VALID", confidence=0.8, source="structured_command_parser"),
        )

def _record(text, expected, actual, source):
    return dict(text=text, expected=expected, actual=actual, source=source)

def test_speed_unit_and_target_field_in_text_mode():
    parsed = GenericInstructionFSM().parse(
        dict(text="Keep the current lane at 18 kilometres per hour.")
    )
    record = _record("Keep the current lane at 18 kilometres per hour.", "KEEP_LANE@18kmh", parsed.parsed_intent, "rule_parser")
    assert parsed.parsed_intent == "KEEP_LANE"
    assert parsed.target_speed_kmh == 18.0
    assert record["source"] == "rule_parser"

def test_speed_number_variant_mps_is_converted_to_kmh():
    fsm = GenericInstructionFSM(
        parser=Service("ADJUST_SPEED", dict(change="DECREASE", target_speed_mps=12.5))
    )
    parsed = fsm.parse(dict(id="n1", text="Ease off to the requested pace."))
    assert parsed.parsed_intent == "DECELERATE"
    assert parsed.target_speed_kmh == 45.0

def test_configured_speed_step_keeps_mps_target_field():
    action, parameters, _, completion = _scene2_step_contract("SET_SPEED:12.50mps")
    assert action == "SET_SPEED"
    assert parameters["target_speed_mps"] == 12.5
    assert completion == {"type": "TARGET_SPEED_REACHED"}

def test_negation_does_not_trigger_positive_stop():
    service = Service("KEEP_LANE")
    parsed = GenericInstructionFSM(parser=service).parse(dict(id="neg1", text="Do not stop."))
    assert parsed.parsed_intent == "KEEP_LANE"
    assert parsed.target_speed_kmh is None

def test_return_to_original_lane_contract_points_right_and_waits():
    action, parameters, on_blocked, completion = _scene2_step_contract(
        "CHANGE_LANE:RETURN_WHEN_SAFE"
    )
    assert action == "CHANGE_LANE"
    assert parameters["direction"] == "RIGHT"
    assert on_blocked == "WAIT_FOR_SAFE"
    assert completion == {"type": "LANE_CHANGE_COMPLETED"}

def test_wait_condition_is_encoded_as_gate():
    action, parameters, on_blocked, _ = _scene2_step_contract("CHECK:PATH_CLEAR")
    assert action == "CHECK"
    assert parameters["condition"] == "PATH_CLEAR"
    assert on_blocked == "WAIT_FOR_SAFE"

@pytest.mark.parametrize("text", [
    "Turn right at the junction.",
    "Please make a right turn.",
])
def test_right_turn_synonyms_share_one_expected_semantic(text):
    fsm = GenericInstructionFSM(parser=Service("TURN", dict(direction="RIGHT")))
    parsed = fsm.parse(dict(id="turn-" + text[:8], text=text))
    assert parsed.parsed_intent == "TURN_RIGHT"

def test_missing_direction_is_not_silently_resolved():
    fsm = GenericInstructionFSM(parser=Service("CHANGE_LANE"))
    parsed = fsm.parse(dict(id="amb1", text="Switch to another lane."))
    assert parsed.requested_lane_direction is None

@pytest.mark.xfail(
    reason="规则解析器把复合顺序/指代折叠为首个关键词；交 朱善哲 定位",
    strict=False,
)
def test_compound_order_reference_is_not_collapsed():
    parsed = GenericInstructionFSM().parse(dict(text="先右转，然后直行通过路口，再次右转。"))
    assert parsed.semantic_goal == ("TURN_RIGHT", "PROCEED_STRAIGHT", "TURN_RIGHT")