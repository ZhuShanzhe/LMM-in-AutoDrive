#!/usr/bin/env python3
"""Summarize Scene 3 text-to-VLA closed-loop JSONL decisions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="vla_control_decisions.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path (relative paths use the current directory)",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def summarize(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    proposal_actions: Counter[str] = Counter()
    final_actions: Counter[str] = Counter()
    command_counts: Counter[str] = Counter()
    risk_levels: Counter[str] = Counter()
    overrides: Counter[str] = Counter()
    blocked_reasons: Counter[str] = Counter()
    per_command: dict[str, dict[str, Counter[str] | int]] = defaultdict(
        lambda: {
            "decisions": 0,
            "model_applied": 0,
            "proposal_actions": Counter(),
            "final_actions": Counter(),
            "overrides": Counter(),
        }
    )
    latencies: list[float] = []
    model_applied = 0

    for record in records:
        command_id = str(record.get("command_id", "unknown"))
        proposal_action = str(
            record.get("vla_proposal", {}).get("action", "unknown")
        )
        decision = record.get("control_decision", {})
        final_action = str(decision.get("action", "unknown"))
        applied = bool(record.get("model_output_applied", False))
        override = record.get("liveness_override") or "none"
        risk_level = str(record.get("risk_assessment", {}).get("risk_level", "unknown"))

        command_counts[command_id] += 1
        proposal_actions[proposal_action] += 1
        final_actions[final_action] += 1
        risk_levels[risk_level] += 1
        overrides[str(override)] += 1
        blocked_reasons.update(str(item) for item in decision.get("blocked_reason_codes", []))
        model_applied += int(applied)
        latencies.append(float(record.get("full_decision_latency_ms", 0.0)))

        bucket = per_command[command_id]
        bucket["decisions"] = int(bucket["decisions"]) + 1
        bucket["model_applied"] = int(bucket["model_applied"]) + int(applied)
        bucket["proposal_actions"][proposal_action] += 1  # type: ignore[index]
        bucket["final_actions"][final_action] += 1  # type: ignore[index]
        bucket["overrides"][str(override)] += 1  # type: ignore[index]

    command_summary: dict[str, Any] = {}
    for command_id, bucket in sorted(per_command.items()):
        command_summary[command_id] = {
            "decisions": bucket["decisions"],
            "model_applied": bucket["model_applied"],
            "proposal_actions": counter_dict(bucket["proposal_actions"]),  # type: ignore[arg-type]
            "final_actions": counter_dict(bucket["final_actions"]),  # type: ignore[arg-type]
            "overrides": counter_dict(bucket["overrides"]),  # type: ignore[arg-type]
        }

    return {
        "input": str(path),
        "decision_count": len(records),
        "model_applied_count": model_applied,
        "model_applied_rate": round(model_applied / len(records), 6) if records else 0.0,
        "command_counts": counter_dict(command_counts),
        "proposal_action_counts": counter_dict(proposal_actions),
        "final_action_counts": counter_dict(final_actions),
        "risk_level_counts": counter_dict(risk_levels),
        "liveness_override_counts": counter_dict(overrides),
        "blocked_reason_counts": counter_dict(blocked_reasons),
        "full_decision_latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "median": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "per_command": command_summary,
    }


def main() -> int:
    args = parse_args()
    result = summarize(args.input)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
