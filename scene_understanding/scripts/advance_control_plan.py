"""Advance a multi-step DrivingIntent using persisted state and JSON feedback."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from scene_understanding.src.control_plan_executor import advance_control_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driving-intent", required=True, type=Path)
    parser.add_argument("--world-state", required=True, type=Path)
    parser.add_argument("--semantic-alignment", required=True, type=Path)
    parser.add_argument("--risk-assessment", required=True, type=Path)
    parser.add_argument(
        "--state",
        type=Path,
        help="prior ControlPlanState JSON; omit only for plan initialization",
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        help="StepFeedback JSON for the active step; requires --state",
    )
    parser.add_argument("--state-output", required=True, type=Path)
    parser.add_argument("--decision-output", required=True, type=Path)
    return parser


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.feedback is not None and args.state is None:
        print("ERROR: --feedback requires --state", file=sys.stderr)
        return 2
    if args.state_output.resolve() == args.decision_output.resolve():
        print("ERROR: state and decision outputs must be different files", file=sys.stderr)
        return 2
    try:
        state, decision = advance_control_plan(
            _read_json(args.driving_intent),
            _read_json(args.world_state),
            _read_json(args.semantic_alignment),
            _read_json(args.risk_assessment),
            prior_state=_read_json(args.state) if args.state is not None else None,
            feedback=_read_json(args.feedback) if args.feedback is not None else None,
        )
        _write_json_atomic(args.state_output, state)
        _write_json_atomic(args.decision_output, decision)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Wrote control-plan state to {args.state_output}")
    print(f"Wrote control decision to {args.decision_output}")
    print(f"Plan status: {state['plan_status']}")
    print(f"Active step: {state['active_step_id']}")
    print(f"Action: {decision['action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
