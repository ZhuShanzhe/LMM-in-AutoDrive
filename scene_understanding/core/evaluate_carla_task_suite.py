#!/usr/bin/env python3
"""Summarize CARLA task-completion metrics without mixing detector metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    runs = len(records)
    completed = sum(record.get("task_completed") is True for record in records)
    collision_free = sum(record.get("collision_free") is True for record in records)
    violation_free = sum(record.get("violation_free") is True for record in records)
    completion_rate = completed / runs if runs else 0.0
    return {
        "runs": runs,
        "task_completed": completed,
        "task_completion_rate": round(completion_rate, 6),
        "collision_free_rate": round(collision_free / runs, 6) if runs else 0.0,
        "violation_free_rate": round(violation_free / runs, 6) if runs else 0.0,
        "competition_threshold": 0.9,
        "meets_90_percent_task_completion": runs > 0 and completion_rate >= 0.9,
        "scenarios": [
            {
                "scenario": record.get("scenario"),
                "task_completed": record.get("task_completed") is True,
                "scenario_reason": record.get("scenario_reason"),
                "collision_free": record.get("collision_free") is True,
                "violation_free": record.get("violation_free") is True,
            }
            for record in records
        ],
        "note": (
            "Task completion is a system-level CARLA metric. It is not object "
            "detection recall, precision, or mAP."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.metrics
    ]
    result = summarize(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
