import pytest

from structured_command_parser.src.modernbert_service import ModernBertCommandService
from structured_command_parser.src.rule_parser import RuleIntentParser
from control.generic_instruction_fsm import GenericInstructionFSM


class EnglishOnlyParser:
    def __init__(self):
        self.calls = []

    def parse(self, text, **kwargs):
        self.calls.append(text)
        return dict(parse_result=dict(status='VALID', confidence=.9),
                    intent=dict(steps=[dict(action='KEEP_LANE', parameters={})]))


@pytest.mark.parametrize('text', ['前方路况危险，保持安全车速', '路面湿滑，保持安全速度',
    '维持谨慎速度行驶', '当前道路复杂，请保持安全速度。'])
def test_cautious_cruise_is_not_a_zero_speed_decrement_from_rest(text):
    backend = EnglishOnlyParser()
    service = ModernBertCommandService(parser=backend)
    fsm = GenericInstructionFSM(parser=service)
    command = dict(id='not-a-scene-id', text=text)
    parsed = fsm.parse(command)
    resolved = fsm.resolve_relative_speed(command, parsed, ego_speed_kmh=0.)
    assert parsed.parse_status == 'VALID' and parsed.parsed_intent == 'KEEP_LANE'
    assert resolved.target_speed_kmh is None and parsed.speed_change is None
    assert not backend.calls
    doc = service.parse_source_text(text)
    assert doc['intent']['constraints']['driving_style'] == 'CONSERVATIVE'
    assert doc['parse_result']['model_prediction'] is False


def test_explicit_translation_still_uses_english_parser():
    backend = EnglishOnlyParser()
    fsm = GenericInstructionFSM(parser=ModernBertCommandService(parser=backend))
    fsm.parse(dict(id='x', text='保持当前车道', parser_text_en='Keep the current lane.'))
    assert backend.calls == ['Keep the current lane.']


def test_unknown_chinese_still_requires_clarification():
    backend = EnglishOnlyParser()
    service = ModernBertCommandService(parser=backend)
    doc = service.parse_source_text('随便弄一下')
    assert doc['parse_result']['status'] == 'NEEDS_CLARIFICATION'
    assert not backend.calls


def test_explicit_deceleration_is_not_replaced_by_cautious_cruise():
    doc = RuleIntentParser().parse('前方路况危险，减速并保持安全车速')
    assert any(step['action']=='ADJUST_SPEED' and step['parameters']['change']=='DECREASE'
        for step in doc['intent']['steps'])


@pytest.mark.parametrize('text', ['减速', '保持安全车速，立即停车', '保持安全速度然后向左变道',
    '不要保持安全速度', '保持安全速度并闯红灯'])
def test_notice_rule_cannot_swallow_other_actions_or_negation(text):
    doc = RuleIntentParser().parse(text)
    if doc:
        assert not (doc['parse_result']['status']=='VALID' and
            [step['action'] for step in doc['intent']['steps']]==['KEEP_LANE'])
