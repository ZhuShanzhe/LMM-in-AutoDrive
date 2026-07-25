from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any

from structured_command_parser.src.modernbert_parser import (
    ModernBertEnglishIntentParser,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _observed(document: dict[str, Any]) -> dict[str, Any]:
    steps = document["intent"]["steps"]
    return {
        "status": document["parse_result"]["status"],
        "actions": [step["action"] for step in steps],
        "directions": [
            step["parameters"]["direction"]
            for step in steps
            if "direction" in step["parameters"]
        ],
        "predicates": sorted(
            {
                condition["predicate"]
                for step in steps
                for condition in step.get("goal_conditions", [])
            }
        ),
        "suppressed_actions": [
            item["action"]
            for item in document["intent"]["suppressed_intents"]
        ],
        "entity_count": len(document["intent"]["entities"]),
    }


def _score(row: dict[str, Any], observed: dict[str, Any]) -> dict[str, bool]:
    expected = row["expected"]
    expected_entity_count = sum(
        span["role"] == "ENTITY" for span in row.get("spans", [])
    )
    return {
        "status": observed["status"] == expected["status"],
        "actions": observed["actions"] == expected["actions"],
        "directions": sorted(observed["directions"])
        == sorted(expected["directions"]),
        "predicates": observed["predicates"]
        == sorted(expected["predicates"]),
        "suppression": observed["suppressed_actions"]
        == expected["suppressed_actions"],
        "entities": (
            observed["entity_count"] >= 1
            if expected_entity_count >= 1
            else observed["entity_count"] == 0
        ),
    }


def _aggregate(items: list[dict[str, bool]]) -> dict[str, float]:
    if not items:
        return {}
    fields = list(items[0])
    metrics = {
        f"{field}_accuracy": sum(item[field] for item in items) / len(items)
        for field in fields
    }
    metrics["exact_graph_match"] = (
        sum(all(item.values()) for item in items) / len(items)
    )
    metrics["rows"] = len(items)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = _read_jsonl(args.data)
    intent_parser = ModernBertEnglishIntentParser(
        str(args.model),
        device=args.device,
    )
    intent_parser.warmup()
    scores = []
    by_phenomenon: dict[str, list[dict[str, bool]]] = defaultdict(list)
    failures = []
    latencies = []
    for row in rows:
        document = intent_parser.parse(
            row["text_en"],
            request_id=row["sample_id"],
        )
        observed = _observed(document)
        score = _score(row, observed)
        scores.append(score)
        by_phenomenon[row["phenomenon"]].append(score)
        latencies.append(document["parse_result"]["latency_ms"])
        if not all(score.values()) and len(failures) < 100:
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "phenomenon": row["phenomenon"],
                    "text_en": row["text_en"],
                    "expected": row["expected"],
                    "observed": observed,
                    "score": score,
                }
            )

    ordered_latency = sorted(latencies)
    p95_index = max(0, round(0.95 * len(ordered_latency)) - 1)
    report = {
        "schema": "compositional-pipeline-evaluation-v1",
        "model": str(args.model),
        "data": str(args.data),
        "overall": _aggregate(scores),
        "by_phenomenon": {
            key: _aggregate(value)
            for key, value in sorted(by_phenomenon.items())
        },
        "latency_ms": {
            "mean": mean(latencies),
            "p95": ordered_latency[p95_index],
            "max": max(latencies),
        },
        "failure_examples": failures,
        "metric_scope": (
            "Synthetic compositional contract accuracy; not human-gold "
            "closed-loop driving accuracy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
