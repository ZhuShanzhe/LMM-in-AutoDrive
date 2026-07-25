"""Contract tests for the parser-to-CARLA JSON bridge."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scene_understanding.scripts.run_carla_decision_bridge import (
    main as bridge_main,
    pending_feedback,
    process_current_frame,
)


ROOT = Path(__file__).resolve().parents[2]
CARLA_ROOT = ROOT / "experiment" / "CARLA"
if str(CARLA_ROOT) not in sys.path:
    sys.path.insert(0, str(CARLA_ROOT))

from control.decision_provider import JsonFileDecisionPolicy


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class CarlaDecisionBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intent = read_json(
            ROOT / "structured_command_parser" / "examples" / "basic_set_speed.json"
        )
        self.world_state = read_json(
            ROOT / "scene_understanding" / "schemas" / "examples" / "world_state.example.json"
        )

    def test_parser_example_compiles_keep_lane_and_set_speed(self) -> None:
        safe_world = deepcopy(self.world_state)
        safe_world["objects"] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intent_path = root / "intent.json"
            world_path = root / "world.json"
            decision_path = root / "decision.json"
            plan_path = root / "plan.json"
            intent_path.write_text(json.dumps(self.intent), encoding="utf-8")
            world_path.write_text(json.dumps(safe_world), encoding="utf-8")

            frame_id, result = process_current_frame(
                driving_intent_path=intent_path,
                world_state_path=world_path,
                decision_output_path=decision_path,
                plan_state_output_path=plan_path,
            )

            decision = read_json(decision_path)
            policy = JsonFileDecisionPolicy(str(decision_path), max_age_frames=0)
            accepted = policy.decide({"simulation_frame": safe_world["simulation_frame"]})

        self.assertEqual(frame_id, safe_world["frame_id"])
        self.assertEqual(self.intent["schema_version"], "1.2.0")
        self.assertEqual(decision["frame_id"], safe_world["frame_id"])
        self.assertEqual(accepted["decision_age_frames"], 0)
        self.assertEqual(policy.telemetry()["status"], "accepted")
        self.assertEqual(result["plan_state"]["step_states"][0]["status"], "SKIPPED")
        self.assertEqual(result["plan_state"]["active_step_id"], "step_2")
        self.assertEqual(result["decision"]["target_speed_kmh"], 60.0012)

    def test_bridge_accepts_utf8_bom_input_documents(self) -> None:
        safe_world = deepcopy(self.world_state)
        safe_world["objects"] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intent_path = root / "intent.json"
            world_path = root / "world.json"
            decision_path = root / "decision.json"
            plan_path = root / "plan.json"
            intent_path.write_text(json.dumps(self.intent), encoding="utf-8-sig")
            world_path.write_text(json.dumps(safe_world), encoding="utf-8-sig")

            _, result = process_current_frame(
                driving_intent_path=intent_path,
                world_state_path=world_path,
                decision_output_path=decision_path,
                plan_state_output_path=plan_path,
            )

        self.assertEqual(result["decision"]["target_speed_kmh"], 60.0012)

    def test_completed_feedback_advances_to_set_speed_step(self) -> None:
        sequential_intent = deepcopy(self.intent)
        sequential_intent["intent"]["steps"][0].update(
            {
                "action": "ADJUST_SPEED",
                "parameters": {"change": "DECREASE"},
                "completion": {"type": "ACTION_REACHED"},
            }
        )
        sequential_intent["intent"]["steps"][1]["depends_on"] = ["step_1"]
        safe_world = deepcopy(self.world_state)
        safe_world["objects"] = []
        first_state, _ = process_current_frame_for_test(sequential_intent, safe_world)
        next_world = deepcopy(safe_world)
        next_world["frame_id"] = "carla_000124"
        next_world["simulation_frame"] = 124
        feedback = {
            "schema_version": "1.0.0",
            "request_id": sequential_intent["request_id"],
            "frame_id": self.world_state["frame_id"],
            "step_id": "step_1",
            "outcome": "COMPLETED",
            "reason_codes": ["lane_held"],
        }
        state, decision = process_current_frame_for_test(
            sequential_intent,
            next_world,
            prior_plan_state=first_state,
            feedback=feedback,
        )

        self.assertEqual(state["active_step_id"], "step_2")
        self.assertEqual(decision["source_step_id"], "step_2")
        self.assertEqual(decision["target_speed_kmh"], 60.0012)

    def test_completed_feedback_is_ignored_after_bridge_restart(self) -> None:
        sequential_intent = deepcopy(self.intent)
        sequential_intent["intent"]["steps"][0].update(
            {
                "action": "ADJUST_SPEED",
                "parameters": {"change": "DECREASE"},
                "completion": {"type": "ACTION_REACHED"},
            }
        )
        sequential_intent["intent"]["steps"][1]["depends_on"] = ["step_1"]
        safe_world = deepcopy(self.world_state)
        safe_world["objects"] = []
        first_state, _ = process_current_frame_for_test(sequential_intent, safe_world)
        feedback = {
            "schema_version": "1.0.0",
            "request_id": sequential_intent["request_id"],
            "frame_id": safe_world["frame_id"],
            "step_id": "step_1",
            "outcome": "COMPLETED",
            "reason_codes": ["lane_held"],
        }
        next_world = deepcopy(safe_world)
        next_world["frame_id"] = "carla_000124"
        next_world["simulation_frame"] = 124
        completed_state, _ = process_current_frame_for_test(
            sequential_intent,
            next_world,
            prior_plan_state=first_state,
            feedback=feedback,
        )

        self.assertEqual(completed_state["active_step_id"], "step_2")
        self.assertIsNone(pending_feedback(feedback, completed_state))

    def test_service_restart_ignores_persisted_terminal_feedback(self) -> None:
        sequential_intent = deepcopy(self.intent)
        sequential_intent["intent"]["steps"][0].update(
            {
                "action": "ADJUST_SPEED",
                "parameters": {"change": "DECREASE"},
                "completion": {"type": "ACTION_REACHED"},
            }
        )
        sequential_intent["intent"]["steps"][1]["depends_on"] = ["step_1"]
        safe_world = deepcopy(self.world_state)
        safe_world["objects"] = []
        feedback = {
            "schema_version": "1.0.0",
            "request_id": sequential_intent["request_id"],
            "frame_id": safe_world["frame_id"],
            "step_id": "step_1",
            "outcome": "COMPLETED",
            "reason_codes": ["lane_held"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intent_path = root / "intent.json"
            world_path = root / "world.json"
            feedback_path = root / "feedback.json"
            decision_path = root / "decision.json"
            plan_path = root / "plan.json"
            intent_path.write_text(json.dumps(sequential_intent), encoding="utf-8")
            world_path.write_text(json.dumps(safe_world), encoding="utf-8")
            feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
            process_current_frame(
                driving_intent_path=intent_path,
                world_state_path=world_path,
                decision_output_path=decision_path,
                plan_state_output_path=plan_path,
            )

            next_world = deepcopy(safe_world)
            next_world["frame_id"] = "carla_000124"
            next_world["simulation_frame"] = 124
            world_path.write_text(json.dumps(next_world), encoding="utf-8")
            first_restart = bridge_main(
                [
                    "--driving-intent", str(intent_path),
                    "--world-state", str(world_path),
                    "--decision-output", str(decision_path),
                    "--plan-state-output", str(plan_path),
                    "--step-feedback", str(feedback_path),
                    "--once",
                ]
            )
            self.assertEqual(first_restart, 0)
            self.assertEqual(read_json(plan_path)["active_step_id"], "step_2")

            newest_world = deepcopy(next_world)
            newest_world["frame_id"] = "carla_000125"
            newest_world["simulation_frame"] = 125
            world_path.write_text(json.dumps(newest_world), encoding="utf-8")
            second_restart = bridge_main(
                [
                    "--driving-intent", str(intent_path),
                    "--world-state", str(world_path),
                    "--decision-output", str(decision_path),
                    "--plan-state-output", str(plan_path),
                    "--step-feedback", str(feedback_path),
                    "--once",
                ]
            )

        self.assertEqual(second_restart, 0)


def process_current_frame_for_test(
    intent: dict,
    world_state: dict,
    *,
    prior_plan_state: dict | None = None,
    feedback: dict | None = None,
) -> tuple[dict, dict]:
    from scene_understanding.scripts.run_carla_decision_bridge import build_decision

    plan_state, decision, _, _ = build_decision(
        intent,
        world_state,
        prior_plan_state=prior_plan_state,
        feedback=feedback,
    )
    return plan_state, decision


if __name__ == "__main__":
    unittest.main()
