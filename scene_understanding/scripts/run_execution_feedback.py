"""Emit conservative StepFeedback from current execution observations."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from scene_understanding.src.execution_feedback import evaluate_execution_feedback


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def process_current_frame(
    *,
    driving_intent_path: Path,
    world_state_path: Path,
    control_decision_path: Path,
    plan_state_path: Path,
    tracker_path: Path,
    feedback_output_path: Path,
    speed_tolerance_mps: float,
    stop_speed_threshold_mps: float,
    target_tolerance_m: float,
    pedestrian_clearance_lateral_m: float,
    minimum_speed_reduction_mps: float,
    overtake_rear_clearance_m: float,
    required_stable_frames: int,
) -> tuple[str, dict[str, Any] | None]:
    world_state = _read_json(world_state_path)
    tracker = _read_json(tracker_path) if tracker_path.exists() else None
    updated_tracker, feedback = evaluate_execution_feedback(
        _read_json(driving_intent_path),
        _read_json(plan_state_path),
        _read_json(control_decision_path),
        world_state,
        tracker=tracker,
        speed_tolerance_mps=speed_tolerance_mps,
        stop_speed_threshold_mps=stop_speed_threshold_mps,
        target_tolerance_m=target_tolerance_m,
        pedestrian_clearance_lateral_m=pedestrian_clearance_lateral_m,
        minimum_speed_reduction_mps=minimum_speed_reduction_mps,
        overtake_rear_clearance_m=overtake_rear_clearance_m,
        required_stable_frames=required_stable_frames,
    )
    if updated_tracker is not None:
        _write_json_atomic(tracker_path, updated_tracker)
    if feedback is not None:
        _write_json_atomic(feedback_output_path, feedback)
    return str(world_state["frame_id"]), feedback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driving-intent", type=Path, required=True)
    parser.add_argument("--world-state", type=Path, required=True)
    parser.add_argument("--control-decision", type=Path, required=True)
    parser.add_argument("--plan-state", type=Path, required=True)
    parser.add_argument("--tracker-output", type=Path, required=True)
    parser.add_argument("--feedback-output", type=Path, required=True)
    parser.add_argument("--speed-tolerance-mps", type=float, default=0.5)
    parser.add_argument("--stop-speed-threshold-mps", type=float, default=0.2)
    parser.add_argument("--target-tolerance-m", type=float, default=3.0)
    parser.add_argument("--pedestrian-clearance-lateral-m", type=float, default=2.5)
    parser.add_argument("--minimum-speed-reduction-mps", type=float, default=3.0)
    parser.add_argument("--overtake-rear-clearance-m", type=float, default=8.0)
    parser.add_argument("--required-stable-frames", type=int, default=5)
    parser.add_argument("--poll-interval-ms", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_interval_ms <= 0:
        raise SystemExit("--poll-interval-ms must be positive")
    previous_frame_id: str | None = None
    while True:
        try:
            frame_id, feedback = process_current_frame(
                driving_intent_path=args.driving_intent,
                world_state_path=args.world_state,
                control_decision_path=args.control_decision,
                plan_state_path=args.plan_state,
                tracker_path=args.tracker_output,
                feedback_output_path=args.feedback_output,
                speed_tolerance_mps=args.speed_tolerance_mps,
                stop_speed_threshold_mps=args.stop_speed_threshold_mps,
                target_tolerance_m=args.target_tolerance_m,
                pedestrian_clearance_lateral_m=args.pedestrian_clearance_lateral_m,
                minimum_speed_reduction_mps=args.minimum_speed_reduction_mps,
                overtake_rear_clearance_m=args.overtake_rear_clearance_m,
                required_stable_frames=args.required_stable_frames,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            if args.once:
                print(f"ERROR: {error}")
                return 1
            time.sleep(args.poll_interval_ms / 1000.0)
            continue
        if frame_id != previous_frame_id:
            print(f"frame={frame_id} feedback={None if feedback is None else feedback['outcome']}")
            previous_frame_id = frame_id
        if args.once:
            return 0
        time.sleep(args.poll_interval_ms / 1000.0)


if __name__ == "__main__":
    raise SystemExit(main())
