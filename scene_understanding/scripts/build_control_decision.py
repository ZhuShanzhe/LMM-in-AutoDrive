"""Build one safety-gated ControlDecision from four integration JSON files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from scene_understanding.src.control_decision import build_control_decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--driving-intent", required=True, type=Path, help="DrivingIntent JSON"
    )
    parser.add_argument(
        "--world-state", required=True, type=Path, help="WorldState JSON"
    )
    parser.add_argument(
        "--semantic-alignment",
        required=True,
        type=Path,
        help="SemanticAlignment JSON",
    )
    parser.add_argument(
        "--risk-assessment", required=True, type=Path, help="RiskAssessment JSON"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="ControlDecision JSON output"
    )
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
    try:
        result = build_control_decision(
            _read_json(args.driving_intent),
            _read_json(args.world_state),
            _read_json(args.semantic_alignment),
            _read_json(args.risk_assessment),
        )
        _write_json_atomic(args.output, result)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Wrote control decision to {args.output}")
    print(f"Decision status: {result['decision_status']}")
    print(f"Action: {result['action']}")
    print(f"Reason: {result['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
