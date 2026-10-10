from copy import deepcopy

import pytest

from control.generic_instruction_fsm import GenericInstructionFSM


def steps():
    return [dict(step_id='s1', action='TURN', parameters={'direction': 'RIGHT'},
                 depends_on=[], trigger={'type': 'IMMEDIATE'}),
            dict(step_id='s2', action='PROCEED', parameters={'condition': 'STRAIGHT_THROUGH_JUNCTION'},
                 depends_on=['s1'], trigger={'type': 'AFTER_STEP', 'step_id': 's1'}),
            dict(step_id='s3', action='TURN', parameters={'direction': 'RIGHT'},
                 depends_on=['s2'], trigger={'type': 'AFTER_STEP', 'step_id': 's2'})]


class Parser:
    def __init__(self, sequence=None, status='VALID'):
        self.sequence = steps() if sequence is None else sequence
        self.status = status
        self.calls = 0

    def parse_text(self, text, **kwargs):
        self.calls += 1
        return dict(intent={'steps': deepcopy(self.sequence)},
                    parse_result=dict(status=self.status, confidence=.9, source='test_parser'))


@pytest.mark.parametrize('field', ['text', 'voice_text', 'normalized_text', 'parser_text_en'])
def test_ordered_plan_survives_rule_keywords_and_caching(field):
    parser = Parser()
    fsm = GenericInstructionFSM(parser=parser)
    command = {field: '先右转，然后直行通过路口，再次右转。'}
    parsed = fsm.parse(command)
    plan = fsm.driving_intent(command)
    assert parsed.semantic_goal == ('TURN_RIGHT', 'PROCEED_STRAIGHT', 'TURN_RIGHT')
    assert plan['intent']['steps'] == steps()
    assert plan['input']['language'] == 'zh-CN'
    fsm.parse(command)
    assert parser.calls == 1


@pytest.mark.parametrize('fault', ['single', 'truncated', 'duplicate', 'dependency', 'future', 'direction', 'status'])
def test_bad_compound_cannot_leak_cached_plan(fault):
    sequence = steps()
    if fault == 'single': sequence = sequence[:1]
    if fault == 'truncated': sequence = sequence[:2]
    if fault == 'duplicate': sequence[1]['step_id'] = 's1'
    if fault == 'dependency': sequence[1]['depends_on'] = []
    if fault == 'future': sequence[0]['depends_on'] = ['s3']
    if fault == 'direction': sequence[0]['parameters'] = {}
    fsm = GenericInstructionFSM(parser=Parser(sequence, 'NEEDS_CLARIFICATION' if fault == 'status' else 'VALID'))
    command = dict(text='Turn right, then go straight, then turn right.')
    parsed = fsm.parse(command)
    assert parsed.parse_status == 'NEEDS_CLARIFICATION'
    assert fsm.driving_intent(command) is None
    issued, blocked = fsm.enforce_parse_status(
        dict(action='accelerate', target_speed_kmh=40., target_acceleration_mps2=2.), parsed)
    assert blocked and issued['action'] == 'stop'
    assert issued['target_speed_kmh'] == 0.
    assert 'target_acceleration_mps2' not in issued


@pytest.mark.parametrize('emergency', [False, True])
def test_clarification_preserves_output_contract_and_removes_motion(emergency):
    fsm = GenericInstructionFSM(parser=Parser([], 'NEEDS_CLARIFICATION'))
    parsed = fsm.parse(dict(text='Ambiguous driving instruction.'))
    original = dict(schema_version='decision/1.0', request_id='request-1', frame=123,
        action='accelerate', target_speed_kmh=40., emergency=emergency,
        blocked_reason_codes=['existing_constraint'], target_lane='left',
        target_location=[1., 2.], target_acceleration_mps2=2.,
        longitudinal_sequence_schema='sequence/1.0', sequence_valid_until_s=10.,
        risk_motion_permission={'max_speed_kmh': 6.})
    issued, blocked = fsm.enforce_parse_status(original, parsed)
    assert blocked and issued['emergency'] is emergency
    assert issued['action'] == ('emergency_brake' if emergency else 'stop')
    assert issued['target_speed_kmh'] == 0. and not issued['allow_positive_acceleration']
    assert issued['decision_status'] == 'BLOCKED'
    assert issued['schema_version'] == original['schema_version']
    assert issued['request_id'] == 'request-1' and issued['frame'] == 123
    assert issued['target_lane'] is None and issued['target_location'] is None
    assert issued['blocked_reason_codes'] == ['existing_constraint', 'instruction_requires_clarification']
    for key in ('target_acceleration_mps2', 'longitudinal_sequence_schema',
                'sequence_valid_until_s', 'risk_motion_permission'):
        assert key not in issued
    assert original['action'] == 'accelerate'


def test_translated_input_preserves_original_and_uses_english():
    class EnglishParser(Parser):
        def parse_text(self, text, **kwargs):
            assert text == 'Turn right, then go straight, then turn right.'
            assert kwargs['source_text'].startswith('先右转')
            return super().parse_text(text, **kwargs)
    command = dict(text='先右转，然后直行通过路口，再次右转。',
                   parser_text_en='Turn right, then go straight, then turn right.')
    fsm = GenericInstructionFSM(parser=EnglishParser())
    assert fsm.parse(command).parse_status == 'VALID'
    assert fsm.driving_intent(command)['input']['normalized_text'] == command['parser_text_en']


def test_plan_executor_advances_only_on_observed_speed_then_stop():
    from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
    from lightweight_vla_adapter.tests.fixtures import integration_documents
    sequence = [dict(step_id='s1', action='ADJUST_SPEED', parameters={'target_speed_mps': 4.},
                     depends_on=[], trigger={'type': 'IMMEDIATE'}, completion={'type': 'TARGET_SPEED_REACHED'},
                     preconditions=[], on_blocked='WAIT_FOR_SAFE'),
                dict(step_id='s2', action='STOP', parameters={}, depends_on=['s1'],
                     trigger={'type': 'AFTER_STEP', 'step_id': 's1'}, completion={'type': 'VEHICLE_STOPPED'},
                     preconditions=[], on_blocked='WAIT_FOR_SAFE')]
    fsm = GenericInstructionFSM(parser=Parser(sequence))
    command = dict(text='Accelerate to the target speed, then stop.')
    fsm.parse(command)
    plan = fsm.driving_intent(command)
    base, world, _, risk = integration_documents(parser_action='ADJUST_SPEED')
    base['intent']['steps'] = plan['intent']['steps']
    runtime = DrivingPlanRuntime()
    for index, speed in enumerate([1., 4., 4., 4., 4., 4., 4., 4.]):
        frame = f'f{index}'
        world['frame_id'] = risk['frame_id'] = frame
        world['ego']['speed_mps'] = speed
        active = runtime.prepare(base, frame_id=frame, timestamp_s=index*.1, speed_mps=speed)
        decision = runtime.advance(world, risk)
        if index == 0: assert active['step_id'] == 's1'
    assert runtime.state['step_states'][0]['status'] == 'COMPLETED'
    assert decision['source_step_id'] == 's2' and decision['action'] == 'stop'
