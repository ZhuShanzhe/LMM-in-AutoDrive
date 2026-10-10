from copy import deepcopy
from types import SimpleNamespace
import pytest
from control.completed_maneuver_context import CompletedManeuverContext
from control.generic_instruction_fsm import GenericInstructionFSM


def prior(direction='RIGHT'):
    return SimpleNamespace(document=dict(request_id='previous',intent=dict(steps=[dict(step_id='s1',
        action='TURN',parameters=dict(direction=direction))])),state=dict(plan_status='COMPLETED',
        step_states=[dict(step_id='s1',action='TURN',status='COMPLETED',terminal_frame_id='f100',
            reason_codes=['observed_physical_completion'])]))


@pytest.mark.parametrize('side,direction',[('右','RIGHT'),('左','LEFT')])
def test_completed_turn_context_does_not_repeat_a_new_turn(side,direction):
    raw=f'完成{side}转后，保持当前车道，以30公里每小时行驶。'
    command=dict(id='next',voice_text=raw,parser_text_en=f'After the {direction.lower()} turn, keep the current lane at 30 km/h.')
    resolver=CompletedManeuverContext();resolved=resolver.resolve(command,prior(direction))
    parsed=GenericInstructionFSM.__new__(GenericInstructionFSM)._parse_text_rules(resolved['text'])
    assert parsed.parsed_intent=='KEEP_LANE' and parsed.target_speed_kmh==30.
    assert resolved['voice_text']==raw and resolved['original_source_text']==raw
    assert resolved['parser_text_en']=='keep the current lane at 30 km/h.'
    assert resolver.resolve(command,None)==resolved
    assert resolver.resolve(dict(command,id='new-request'),None)==dict(command,id='new-request')


@pytest.mark.parametrize('fault',['missing','blocked','unobserved','wrong_direction','same_request','wrong_step'])
def test_unproven_history_never_removes_a_required_turn(fault):
    command=dict(id='next',text='After the right turn, keep the current lane.')
    previous=prior()
    if fault=='missing':previous=None
    elif fault=='blocked':previous.state['plan_status']='BLOCKED'
    elif fault=='unobserved':previous.state['step_states'][0]['reason_codes']=[]
    elif fault=='wrong_direction':previous=prior('LEFT')
    elif fault=='same_request':command['id']='previous'
    elif fault=='wrong_step':previous.state['step_states'][0]['step_id']='other'
    assert CompletedManeuverContext().resolve(command,previous)==command


def test_new_turn_after_confirmed_history_is_preserved():
    command=dict(id='next',text='完成右转后，再左转。')
    resolved=CompletedManeuverContext().resolve(command,prior())
    assert resolved['text']=='左转。'
    assert GenericInstructionFSM.__new__(GenericInstructionFSM)._parse_text_rules(resolved['text']).parsed_intent=='TURN_LEFT'


def test_conflicting_translation_cannot_erase_a_turn():
    command=dict(id='next',text='完成右转后，保持当前车道。',parser_text_en='After the left turn, keep the current lane.')
    assert CompletedManeuverContext().resolve(command,prior())==command
