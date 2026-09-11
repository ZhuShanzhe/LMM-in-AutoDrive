import pytest
from control.generic_instruction_fsm import GenericInstructionFSM, ParsedInstruction


class Service:
    def __init__(self,action,parameters=None):
        self.action=action;self.parameters=parameters or {};self.calls=[]

    def parse_text(self,text,**kwargs):
        self.calls.append((text,kwargs))
        return dict(intent=dict(steps=[dict(action=self.action,parameters=self.parameters,trigger=dict(type='IMMEDIATE'))]),
            parse_result=dict(status='VALID',confidence=.9))


def test_multi_step_cached_document_is_retained_without_reparsing():
    service=Service('ADJUST_SPEED',dict(target_speed_mps=4.))
    original=service.parse_text
    def parse(text,**kwargs):
        result=original(text,**kwargs)
        result['intent']['steps'][0]['step_id']='step_1'
        result['intent']['steps'].append(dict(step_id='step_2',action='STOP',parameters={},depends_on=['step_1'],trigger=dict(type='AFTER_STEP',step_id='step_1')))
        return result
    service.parse_text=parse;fsm=GenericInstructionFSM(parser=service)
    command=dict(text='Drive at 15 kilometers per hour, then stop.')
    fsm.parse(command);document=fsm.driving_intent(command)
    assert len(document['intent']['steps'])==2 and len(service.calls)==1
    document['intent']['steps'].clear()
    assert len(fsm.driving_intent(command)['intent']['steps'])==2


@pytest.mark.parametrize('text,expected',[
    ('Stop the vehicle.','STOP'),('Please stop the car now.','STOP'),
    ('Emergency brake now.','EMERGENCY_BRAKE'),('Please emergency brake.','EMERGENCY_BRAKE')])
def test_plain_english_stop_without_id(text,expected):
    r=GenericInstructionFSM().parse(dict(text=text))
    assert r.parsed_intent==expected and r.target_speed_kmh==0.


@pytest.mark.parametrize('direction',['LEFT','RIGHT'])
def test_full_driving_intent_and_direction_without_id(direction):
    service=Service('CHANGE_LANE',dict(direction=direction))
    r=GenericInstructionFSM(parser=service).parse(dict(text=f'Change to the {direction.lower()} lane.'))
    assert r.parsed_intent=='CHANGE_LANE_'+direction and r.requested_lane_direction==direction.lower()
    assert service.calls[0][1]['source_language']=='en-US'


def test_model_stop_is_not_lost_with_sibling_intent():
    r=GenericInstructionFSM(parser=Service('STOP')).parse(dict(id='x',text='Bring us to a halt.'))
    assert r.parsed_intent=='STOP' and r.target_speed_kmh==0.


@pytest.mark.parametrize('unit',['kilometers per hour','kilometres per hour','KM/H'])
def test_neutral_english_cruise_preserves_explicit_speed_without_model_override(unit):
    service=Service('STOP');fsm=GenericInstructionFSM(parser=service)
    r=fsm.parse(dict(text=f'Keep the current lane at 20 {unit}.'))
    assert r.parsed_intent=='KEEP_LANE' and r.target_speed_kmh==20.
    assert not service.calls


def test_target_speed_mps_is_converted_and_decrease_is_not_accelerate():
    fsm=GenericInstructionFSM(parser=Service('ADJUST_SPEED',dict(change='DECREASE',target_speed_mps=5.)))
    r=fsm.parse(dict(text='Ease off to the requested pace.'))
    assert r.parsed_intent=='DECELERATE' and r.target_speed_kmh==18.


def test_negated_stop_does_not_use_positive_stop_shortcut():
    service=Service('KEEP_LANE')
    r=GenericInstructionFSM(parser=service).parse(dict(text='Do not stop.'))
    assert r.parsed_intent=='KEEP_LANE' and len(service.calls)==1


def test_missing_direction_does_not_silently_turn_left():
    fsm=GenericInstructionFSM(parser=Service('CHANGE_LANE'))
    assert fsm.parse(dict(text='Move to another lane.')).requested_lane_direction is None


def test_model_invalid_speed_is_not_consumed():
    fsm=GenericInstructionFSM(parser=Service('SET_SPEED',dict(target_speed_mps=float('nan'))))
    assert fsm.parse(dict(text='Use that speed.')).target_speed_kmh is None
