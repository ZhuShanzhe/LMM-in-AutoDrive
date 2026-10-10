import pytest
from control.generic_instruction_fsm import GenericInstructionFSM, ParsedInstruction
from control.cruise_goal_contract import qualitative_instruction_bound


@pytest.mark.parametrize('text',['Slow down for the road hazard.','Avoid the vehicle cutting in.'])
def test_rule_derived_non_numeric_slowdown_is_temporary(text):
    fsm=GenericInstructionFSM()
    document=fsm.driving_intent(dict(id='hazard',text=text),
        parsed=ParsedInstruction(parsed_intent='DECELERATE',target_speed_kmh=18.))
    assert qualitative_instruction_bound(document)==18.
    assert document['parse_result']['model_prediction'] is False


@pytest.mark.parametrize('text',['Slow down to 20 km/h.','Reduce speed to 20.'])
def test_explicit_numeric_target_is_not_marked_as_a_qualitative_adapter(text):
    fsm=GenericInstructionFSM()
    document=fsm.driving_intent(dict(id='speed',text=text),
        parsed=ParsedInstruction(parsed_intent='DECELERATE',target_speed_kmh=20.))
    assert qualitative_instruction_bound(document) is None
