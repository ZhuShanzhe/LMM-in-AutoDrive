#!/usr/bin/env python3
"""Evaluate Scene 3 scheduled Chinese commands without stopping on one parser error."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from structured_command_parser.src.rule_parser import RuleIntentParser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(
    schedule: list[dict[str, Any]],
    parser: Any,
) -> list[dict[str, Any]]:
    rows = []
    for source in schedule:
        error = None
        try:
            intent = parser.parse(
                source["text"],
                modality="VOICE",
                request_id=source["command_id"],
            )
        except Exception as exc:  # Preserve per-command failures as evidence.
            intent = None
            error = {"type": type(exc).__name__, "message": str(exc)}
        steps = [] if intent is None else intent.get("intent", {}).get("steps", [])
        parse_result = {} if intent is None else intent.get("parse_result", {})
        rows.append(
            {
                "command_id": source["command_id"],
                "text": source["text"],
                "semantic_goal": source.get("semantic_goal", []),
                "parse_status": parse_result.get("status"),
                "error": error,
                "method": parse_result.get("method"),
                "latency_ms": parse_result.get("latency_ms"),
                "steps": [
                    {
                        "action": step.get("action"),
                        "parameters": step.get("parameters"),
                        "preconditions": step.get("preconditions"),
                        "on_blocked": step.get("on_blocked"),
                    }
                    for step in steps
                ],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = evaluate(read_jsonl(args.schedule), RuleIntentParser())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    valid = sum(row["parse_status"] == "VALID" for row in rows)
    errors = sum(row["error"] is not None for row in rows)
    print(f"commands={len(rows)} valid={valid} errors={errors} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
