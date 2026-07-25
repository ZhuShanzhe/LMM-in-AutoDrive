"""Align a DrivingIntent JSON file with a WorldState JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scene_understanding.src.driving_intent_alignment import align_driving_intent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--driving-intent",
        required=True,
        type=Path,
        help="Supported DrivingIntent JSON file",
    )
    parser.add_argument(
        "--world-state",
        required=True,
        type=Path,
        help="WorldState JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="semantic-alignment JSON output path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        driving_intent = json.loads(args.driving_intent.read_text(encoding="utf-8"))
        world_state = json.loads(args.world_state.read_text(encoding="utf-8"))
        result = align_driving_intent(driving_intent, world_state)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Wrote semantic alignment to {args.output}")
    print(
        "Alignment status: "
        f"{result['alignment_status']} "
        f"({result['matched_target_count']}/{result['target_count']} targets matched)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
