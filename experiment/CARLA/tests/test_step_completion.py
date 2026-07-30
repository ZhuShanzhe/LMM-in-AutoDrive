from __future__ import annotations

import unittest

from control.step_completion import StepCompletionEvaluator


def intent(completion, *, action="KEEP_LANE", parameters=None):
    return {
        "request_id": "request-1",
        "intent": {
            "steps": [
                {
                    "step_id": "step_1",
                    "action": action,
                    "parameters": parameters or {},
                    "completion": {"type": completion},
                }
            ]
        },
    }


def state():
    return {"plan_status": "ACTIVE", "active_step_id": "step_1"}


def world(speed_mps=0.0, junction=False):
    return {
        "frame_id": "frame-1",
        "ego": {"speed_mps": speed_mps, "is_junction": junction},
        "environment": {"is_intersection": junction},
        "objects": [],
    }


def alignment(success=None):
    return {
        "step_alignments": [
            {
                "step_id": "step_1",
                "alignment_success": success,
                "matched_entity": None,
            }
        ]
    }


class StepCompletionTest(unittest.TestCase):
    def test_target_speed_requires_stable_frames(self):
        evaluator = StepCompletionEvaluator(stable_frames_required=3)
        command = intent(
            "TARGET_SPEED_REACHED",
            action="SET_SPEED",
            parameters={"target_speed_mps": 10.0},
        )
        self.assertIsNone(
            evaluator.evaluate(command, state(), world(10.0), alignment())
        )
        self.assertIsNone(
            evaluator.evaluate(command, state(), world(10.0), alignment())
        )
        self.assertEqual(
            evaluator.evaluate(command, state(), world(10.0), alignment())[
                "outcome"
            ],
            "COMPLETED",
        )

    def test_deceleration_completes_after_crossing_target_threshold(self):
        evaluator = StepCompletionEvaluator(stable_frames_required=2)
        command = intent(
            "TARGET_SPEED_REACHED",
            action="SET_SPEED",
            parameters={"target_speed_mps": 10.0},
        )

        self.assertIsNone(
            evaluator.evaluate(command, state(), world(14.0), alignment())
        )
        self.assertIsNone(
            evaluator.evaluate(command, state(), world(10.4), alignment())
        )
        self.assertEqual(
            evaluator.evaluate(command, state(), world(9.8), alignment())[
                "outcome"
            ],
            "COMPLETED",
        )

    def test_target_cleared_requires_prior_match(self):
        evaluator = StepCompletionEvaluator()
        command = intent("TARGET_CLEARED", action="YIELD")
        self.assertIsNone(
            evaluator.evaluate(command, state(), world(), alignment(False))
        )
        self.assertIsNone(
            evaluator.evaluate(command, state(), world(), alignment(True))
        )
        self.assertEqual(
            evaluator.evaluate(command, state(), world(), alignment(False))[
                "reason_codes"
            ],
            ["aligned_target_cleared"],
        )

    def test_junction_exit_requires_entry(self):
        evaluator = StepCompletionEvaluator()
        command = intent("JUNCTION_EXITED", action="TURN")
        self.assertIsNone(
            evaluator.evaluate(command, state(), world(junction=False), alignment())
        )
        self.assertIsNone(
            evaluator.evaluate(command, state(), world(junction=True), alignment())
        )
        self.assertIsNotNone(
            evaluator.evaluate(command, state(), world(junction=False), alignment())
        )

    def test_lane_change_uses_controller_completion_phase(self):
        evaluator = StepCompletionEvaluator()
        command = intent("LANE_CHANGE_COMPLETED", action="CHANGE_LANE")
        self.assertIsNone(
            evaluator.evaluate(
                command,
                state(),
                world(),
                alignment(),
                lateral_diagnostics={"phase": "RECENTER"},
            )
        )
        self.assertIsNotNone(
            evaluator.evaluate(
                command,
                state(),
                world(),
                alignment(),
                lateral_diagnostics={"phase": "COMPLETE"},
            )
        )


if __name__ == "__main__":
    unittest.main()
