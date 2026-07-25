"""Bridge parser output and CARLA WorldState into guarded control decisions.

The bridge is intentionally process-separated from CARLA.  It consumes the
schema-valid WorldState emitted by ``run_control_experiment.py`` and atomically
writes a ControlDecision that the existing JSON-file policy can read.  The
parser can replace the DrivingIntent file independently without importing the
CARLA runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from scene_understanding.src.control_plan_executor import advance_control_plan
from scene_understanding.src.driving_intent_alignment import align_driving_intent
from scene_understanding.src.risk_interface import assess_scene_risk


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def pending_feedback(
    candidate: dict[str, Any], prior_plan_state: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Return feedback that still applies to the persisted active step.

    A feedback snapshot is deliberately retained by the execution side so a
    bridge restart must recognise an already-terminal step instead of trying
    to apply that old event to the next active step.
    """

    if prior_plan_state is None:
        return candidate
    if candidate.get("request_id") != prior_plan_state.get("request_id"):
        raise ValueError("StepFeedback request_id does not match ControlPlanState")

    step_id = candidate.get("step_id")
    matching_steps = [
        step
        for step in prior_plan_state.get("step_states", [])
        if step.get("step_id") == step_id
    ]
    if len(matching_steps) != 1:
        raise ValueError("StepFeedback step_id is not present in ControlPlanState")
    step = matching_steps[0]
    if step.get("status") in {
        "COMPLETED",
        "SKIPPED",
        "BLOCKED",
        "FAILED",
        "CANCELLED",
    }:
        return None
    if step_id != prior_plan_state.get("active_step_id"):
        raise ValueError("StepFeedback step_id does not match the active step")
    return candidate


def build_decision(
    driving_intent: dict[str, Any],
    world_state: dict[str, Any],
    *,
    prior_plan_state: dict[str, Any] | None = None,
    feedback: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return plan state, decision, alignment, and risk for one CARLA frame."""

    alignment = align_driving_intent(driving_intent, world_state)
    risk = assess_scene_risk(world_state)
    plan_state, decision = advance_control_plan(
        driving_intent,
        world_state,
        alignment,
        risk,
        prior_state=prior_plan_state,
        feedback=feedback,
    )
    return plan_state, decision, alignment, risk


def process_current_frame(
    *,
    driving_intent_path: Path,
    world_state_path: Path,
    decision_output_path: Path,
    plan_state_output_path: Path,
    prior_plan_state: dict[str, Any] | None = None,
    feedback: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Process the current atomic snapshots and persist a controller decision."""

    driving_intent = _read_json(driving_intent_path)
    world_state = _read_json(world_state_path)
    plan_state, decision, alignment, risk = build_decision(
        driving_intent,
        world_state,
        prior_plan_state=prior_plan_state,
        feedback=feedback,
    )
    _write_json_atomic(plan_state_output_path, plan_state)
    _write_json_atomic(decision_output_path, decision)
    return str(world_state["frame_id"]), {
        "plan_state": plan_state,
        "decision": decision,
        "alignment": alignment,
        "risk": risk,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driving-intent", type=Path, required=True)
    parser.add_argument("--world-state", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    parser.add_argument("--plan-state-output", type=Path, required=True)
    parser.add_argument(
        "--step-feedback",
        type=Path,
        default=None,
        help="Optional latest StepFeedback JSON written by the execution layer",
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=float,
        default=10.0,
        help="Snapshot polling interval for service mode",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process the current WorldState once and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_interval_ms <= 0:
        raise SystemExit("--poll-interval-ms must be positive")

    prior_plan_state = (
        _read_json(args.plan_state_output) if args.plan_state_output.exists() else None
    )
    previous_frame_id: str | None = None
    consumed_feedback_signature: str | None = None
    while True:
        try:
            current_world_state = _read_json(args.world_state)
            current_frame_id = str(current_world_state["frame_id"])
            if current_frame_id == previous_frame_id:
                if args.once:
                    return 0
                time.sleep(args.poll_interval_ms / 1000.0)
                continue
            feedback = None
            feedback_signature = None
            if args.step_feedback and args.step_feedback.exists():
                candidate = _read_json(args.step_feedback)
                feedback_signature = json.dumps(
                    candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                if feedback_signature != consumed_feedback_signature:
                    feedback = pending_feedback(candidate, prior_plan_state)
            frame_id, result = process_current_frame(
                driving_intent_path=args.driving_intent,
                world_state_path=args.world_state,
                decision_output_path=args.decision_output,
                plan_state_output_path=args.plan_state_output,
                prior_plan_state=prior_plan_state,
                feedback=feedback,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            if args.once:
                print(f"ERROR: {error}")
                return 1
            time.sleep(args.poll_interval_ms / 1000.0)
            continue

        previous_frame_id = frame_id
        prior_plan_state = result["plan_state"]
        if feedback_signature is not None:
            consumed_feedback_signature = feedback_signature
        print(
            f"frame={frame_id} action={result['decision']['action']} "
            f"status={result['decision']['decision_status']}"
        )
        if args.once:
            return 0
        time.sleep(args.poll_interval_ms / 1000.0)


if __name__ == "__main__":
    raise SystemExit(main())
