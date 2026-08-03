import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scene_understanding.src.control_plan_executor import (
    advance_control_plan,
    validate_control_plan_state,
    validate_step_feedback,
)
from scene_understanding.tests.test_control_decision import (
    driving_intent,
    load_example,
)


def three_step_intent() -> dict:
    intent = driving_intent(
        "ADJUST_SPEED", {"change": "DECREASE"}, on_blocked="WAIT_FOR_SAFE"
    )
    intent["intent"]["steps"].extend(
        [
            {
                "step_id": "step_2",
                "action": "CHANGE_LANE",
                "target": None,
                "parameters": {"direction": "LEFT", "lane_count": 1},
                "depends_on": ["step_1"],
                "preconditions": [],
                "on_blocked": "WAIT_FOR_SAFE",
            },
            {
                "step_id": "step_3",
                "action": "OVERTAKE",
                "target": None,
                "parameters": {},
                "depends_on": ["step_2"],
                "preconditions": [],
                "on_blocked": "SAFE_STOP",
            },
        ]
    )
    return intent


def alignment_for(intent: dict, frame_id: str, *, first_success=True) -> dict:
    alignments = []
    for index, step in enumerate(intent["intent"]["steps"]):
        required = index == 0 and step.get("target") is not None
        success = first_success if required else None
        entity = None
        if success is True:
            entity = {
                "entity_type": "actor",
                "entity_id": "carla_actor_25",
                "category": "pedestrian",
                "distance_m": 20.0,
                "relative_position": "front_left",
                "lane_relation": "crossing_ego_path",
                "risk_level": "low",
            }
        alignments.append(
            {
                "step_id": step["step_id"],
                "action": step["action"],
                "target": step.get("target"),
                "alignment_required": required,
                "alignment_success": success,
                "candidate_count": int(success is True),
                "matched_entity": entity,
                "reason_code": (
                    "matched_world_object"
                    if success is True
                    else "no_matching_entity"
                    if required
                    else "target_not_required"
                ),
            }
        )
    required_count = sum(item["alignment_required"] for item in alignments)
    matched_count = sum(item["alignment_success"] is True for item in alignments)
    return {
        "schema_version": "1.0.0",
        "request_id": intent["request_id"],
        "world_state_frame_id": frame_id,
        "parse_status": intent["parse_result"]["status"],
        "alignment_status": (
            "NOT_REQUIRED"
            if required_count == 0
            else "COMPLETE"
            if matched_count == required_count
            else "FAILED"
        ),
        "target_count": required_count,
        "matched_target_count": matched_count,
        "step_alignments": alignments,
    }


def feedback(intent: dict, step_id: str, outcome: str, frame_id="carla_feedback"):
    return {
        "schema_version": "1.0.0",
        "request_id": intent["request_id"],
        "frame_id": frame_id,
        "step_id": step_id,
        "outcome": outcome,
        "reason_codes": [f"test_{outcome.lower()}"],
    }


class ControlPlanExecutorTests(unittest.TestCase):
    def setUp(self):
        self.world = load_example("world_state.example.json")
        self.risk = load_example("risk_assessment.example.json")
        self.risk["risk_level"] = "low"
        self.risk["reason_codes"] = ["distance_above_25m"]
        self.risk["recommended_action"] = "monitor"

    def run_plan(self, intent, *, state=None, event=None, alignment=None):
        return advance_control_plan(
            intent,
            self.world,
            alignment or alignment_for(intent, self.world["frame_id"]),
            self.risk,
            prior_state=state,
            feedback=event,
        )

    def test_advances_three_steps_only_on_explicit_feedback(self):
        intent = three_step_intent()
        state, decision = self.run_plan(intent)
        self.assertEqual(state["revision"], 0)
        self.assertEqual(state["active_step_id"], "step_1")
        self.assertEqual(decision["action"], "decelerate")

        state, decision = self.run_plan(
            intent,
            state=state,
            event=feedback(intent, "step_1", "COMPLETED"),
        )
        self.assertEqual(state["active_step_id"], "step_2")
        self.assertEqual(decision["action"], "lane_change_left")

        state, decision = self.run_plan(
            intent,
            state=state,
            event=feedback(intent, "step_2", "COMPLETED"),
        )
        self.assertEqual(state["active_step_id"], "step_3")
        self.assertEqual(decision["action"], "accelerate")

        state, decision = self.run_plan(
            intent,
            state=state,
            event=feedback(intent, "step_3", "COMPLETED"),
        )
        self.assertEqual(state["revision"], 3)
        self.assertEqual(state["plan_status"], "COMPLETED")
        self.assertIsNone(state["active_step_id"])
        self.assertEqual(decision["action"], "keep_lane")
        self.assertEqual(decision["reason"], "plan_completed")
        self.assertEqual(validate_control_plan_state(state), [])

    def test_waiting_step_recovers_when_alignment_becomes_available(self):
        intent = driving_intent(
            "ADJUST_SPEED",
            {"change": "DECREASE"},
            target={"type": "PEDESTRIAN", "relation": "AHEAD_CROSSING"},
            on_blocked="WAIT_FOR_SAFE",
        )
        missing = alignment_for(intent, self.world["frame_id"], first_success=False)
        state, decision = self.run_plan(intent, alignment=missing)
        self.assertEqual(state["plan_status"], "ACTIVE")
        self.assertEqual(state["step_states"][0]["status"], "WAITING")
        self.assertEqual(decision["decision_status"], "BLOCKED")

        available = alignment_for(intent, self.world["frame_id"], first_success=True)
        state, decision = self.run_plan(intent, state=state, alignment=available)
        self.assertEqual(state["step_states"][0]["status"], "ACTIVE")
        self.assertEqual(decision["decision_status"], "READY")

    def test_safe_stop_policy_terminates_blocked_plan(self):
        intent = driving_intent(
            "ADJUST_SPEED",
            {"change": "DECREASE"},
            target={"type": "PEDESTRIAN", "relation": "AHEAD_CROSSING"},
            on_blocked="SAFE_STOP",
        )
        missing = alignment_for(intent, self.world["frame_id"], first_success=False)
        state, decision = self.run_plan(intent, alignment=missing)
        self.assertEqual(state["plan_status"], "BLOCKED")
        self.assertEqual(state["step_states"][0]["status"], "BLOCKED")
        self.assertEqual(decision["action"], "stop")

    def test_skip_policy_activates_dependent_next_step(self):
        intent = driving_intent(
            "ADJUST_SPEED",
            {"change": "DECREASE"},
            target={"type": "PEDESTRIAN", "relation": "AHEAD_CROSSING"},
            on_blocked="SKIP_STEP",
        )
        intent["intent"]["steps"].append(
            {
                "step_id": "step_2",
                "action": "KEEP_LANE",
                "target": None,
                "parameters": {},
                "depends_on": ["step_1"],
                "preconditions": [],
                "on_blocked": "SAFE_STOP",
            }
        )
        missing = alignment_for(intent, self.world["frame_id"], first_success=False)
        state, decision = self.run_plan(intent, alignment=missing)
        self.assertEqual(state["step_states"][0]["status"], "SKIPPED")
        self.assertEqual(state["active_step_id"], "step_2")
        self.assertEqual(decision["action"], "keep_lane")

    def test_failed_feedback_stops_plan(self):
        intent = three_step_intent()
        state, _ = self.run_plan(intent)
        state, decision = self.run_plan(
            intent,
            state=state,
            event=feedback(intent, "step_1", "FAILED"),
        )
        self.assertEqual(state["plan_status"], "FAILED")
        self.assertEqual(decision["action"], "stop")
        self.assertEqual(decision["decision_status"], "SAFE_FALLBACK")

    def test_feedback_must_match_active_step(self):
        intent = three_step_intent()
        state, _ = self.run_plan(intent)
        with self.assertRaisesRegex(ValueError, "does not match the active step"):
            self.run_plan(
                intent,
                state=state,
                event=feedback(intent, "step_2", "COMPLETED"),
            )

    def test_feedback_validator_rejects_duplicate_reasons(self):
        intent = three_step_intent()
        event = feedback(intent, "step_1", "CONTINUE")
        event["reason_codes"] = ["same", "same"]
        self.assertIn("reason_codes: values must be unique", validate_step_feedback(event))

    def test_cli_initializes_state_and_writes_decision(self):
        intent = three_step_intent()
        alignment = alignment_for(intent, self.world["frame_id"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in {
                "intent.json": intent,
                "world.json": self.world,
                "alignment.json": alignment,
                "risk.json": self.risk,
            }.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            state_path = root / "state.json"
            decision_path = root / "decision.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scene_understanding.scripts.advance_control_plan",
                    "--driving-intent", str(root / "intent.json"),
                    "--world-state", str(root / "world.json"),
                    "--semantic-alignment", str(root / "alignment.json"),
                    "--risk-assessment", str(root / "risk.json"),
                    "--state-output", str(state_path),
                    "--decision-output", str(decision_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(state_path.read_text())["active_step_id"], "step_1")
            self.assertEqual(json.loads(decision_path.read_text())["action"], "decelerate")


if __name__ == "__main__":
    unittest.main()
