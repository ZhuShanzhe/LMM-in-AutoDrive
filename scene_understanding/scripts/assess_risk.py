"""Assess one WorldState JSON file and write a risk-assessment JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scene_understanding.src.risk_interface import assess_scene_risk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-state",
        required=True,
        type=Path,
        help="WorldState JSON input path",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="risk-assessment JSON output path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        world_state = json.loads(args.world_state.read_text(encoding="utf-8"))
        result = assess_scene_risk(world_state)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Wrote risk assessment to {args.output}")
    print(f"Risk level: {result['risk_level']}")
    print(f"Recommended action: {result['recommended_action']}")
    for direction in ("left", "right"):
        judgment = result["lane_change"][direction]
        state = "safe" if judgment["is_safe"] else "blocked"
        reasons = ", ".join(judgment["reason_codes"]) or "none"
        print(f"Lane change {direction}: {state} ({reasons})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
