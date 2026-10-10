import pytest
from control.generic_instruction_fsm import GenericInstructionFSM, ParsedInstruction


def test_unquantified_slowdown_is_not_frozen_at_a_red_light_speed():
    fsm=GenericInstructionFSM(default_speed_kmh=50.)
    command=dict(id='warning',text='Slow down.')
    parsed=ParsedInstruction(parsed_intent='DECELERATE',source_text='Slow down.')
    first=fsm.resolve_relative_speed(command,parsed,ego_speed_kmh=1.,cruise_speed_kmh=50.)
    assert first.target_speed_kmh==30.
    later=fsm.resolve_relative_speed(command,parsed,ego_speed_kmh=20.,cruise_speed_kmh=30.)
    assert later.target_speed_kmh==30.
    document=fsm.driving_intent(command,first)
    assert document['intent']['steps'][0]['parameters']['target_speed_mps']==pytest.approx(30./3.6)
    assert document['intent']['steps'][0]['completion']['type']=='TARGET_SPEED_REACHED'
    assert fsm.driving_intent(command,later)==document


def test_explicit_delta_and_absolute_target_are_not_reinterpreted():
    fsm=GenericInstructionFSM()
    parsed=ParsedInstruction(parsed_intent='DECELERATE',speed_delta_kmh=5.,source_text='Reduce by 5 km/h.')
    resolved=fsm.resolve_relative_speed(dict(id='numeric'),parsed,ego_speed_kmh=16.,cruise_speed_kmh=50.)
    assert resolved.target_speed_kmh==11.
    absolute=ParsedInstruction(parsed_intent='SET_SPEED',target_speed_kmh=40.)
    assert fsm.resolve_relative_speed({},absolute,ego_speed_kmh=0.,cruise_speed_kmh=50.) is absolute


@pytest.mark.parametrize('text',['Do not stop, then turn right.','Slow down and stop before the truck.'])
def test_compound_instruction_cannot_be_reduced_to_a_single_speed_plan(text):
    assert GenericInstructionFSM().driving_intent(dict(text=text),
        ParsedInstruction(parsed_intent='DECELERATE',target_speed_kmh=20.)) is None
