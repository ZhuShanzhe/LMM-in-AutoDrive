from control.generic_instruction_fsm import GenericInstructionFSM
from scene2_runtime_interface import build_scheduled_driving_intent

class Service:
    def __init__(self, status="VALID", steps=None):
        self.status = status
        self.steps = steps if steps is not None else [
            dict(step_id="s1", action="SET_SPEED", parameters={"target_speed_mps": 5.0}, trigger=dict(type="IMMEDIATE")),
        ]
        self.calls = []

    def parse_text(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return dict(
            intent=dict(steps=self.steps),
            parse_result=dict(status=self.status, confidence=0.8, source="structured_command_parser"),
        )

def _configured_command():
    return dict(
        id="s2_t05_cmd_01",
        spoken_text="保持当前车道，将车速调整至45公里每小时，在前方路口右转，然后保持车道继续行驶。",
        category="NAVIGATION",
        urgency="NORMAL",
        steps=["KEEP_LANE", "SET_SPEED:12.50mps", "TURN:RIGHT", "KEEP_LANE"],
    )

def test_configured_plan_is_explicitly_marked():
    plan = build_scheduled_driving_intent(_configured_command(), 0, 0.0, 0.0)
    result = plan["parse_result"]
    assert result["source"] == "competition_schedule"
    assert result["source_kind"] == "CONFIGURED_PLAN"
    assert result["model_prediction"] is False

def test_configured_plan_remains_executable_isolation():
    plan = build_scheduled_driving_intent(_configured_command(), 0, 0.0, 0.0)
    assert plan["parse_result"]["status"] == "VALID"
    assert len(plan["intent"]["steps"]) == 4

def test_text_mode_actually_invokes_parser_and_records_source():
    service = Service()
    fsm = GenericInstructionFSM(parser=service)
    command = dict(id="text-1", text="Ease off to the requested pace.")
    parsed = fsm.parse(command)
    assert service.calls and service.calls[0][0] == command["text"]
    assert parsed.parse_status == "VALID"
    assert parsed.parse_source == "structured_command_parser"

def test_clarification_is_passed_through_without_preset_answer():
    fsm = GenericInstructionFSM(parser=Service(status="NEEDS_CLARIFICATION"))
    parsed = fsm.parse(dict(id="text-2", text="Do the thing at the place."))
    assert parsed.parse_status == "NEEDS_CLARIFICATION"
    assert parsed.parsed_intent == "KEEP_LANE"

def test_supplied_configured_plan_is_not_a_model_prediction():
    plan = build_scheduled_driving_intent(_configured_command(), 0, 0.0, 0.0)
    fsm = GenericInstructionFSM()
    document = fsm.driving_intent(
        dict(id="s2_t05_cmd_01", text="ignored", driving_intent=plan)
    )
    assert document["parse_result"]["model_prediction"] is False
    assert len(document["intent"]["steps"]) == 4

def test_text_multi_step_document_is_marked_as_model_prediction():
    steps = [
        dict(step_id="s1", action="SET_SPEED", parameters={"target_speed_mps": 5.0}, trigger=dict(type="IMMEDIATE")),
        dict(step_id="s2", action="STOP", parameters={}, depends_on=["s1"], trigger=dict(type="AFTER_STEP", step_id="s1")),
    ]
    fsm = GenericInstructionFSM(parser=Service(steps=steps))
    command = dict(id="text-3", text="Ease off to the requested pace, then stop.")
    fsm.parse(command)
    document = fsm.driving_intent(command)
    assert document["parse_result"]["model_prediction"] is True
    assert document["parse_result"]["source"] == "structured_command_parser"