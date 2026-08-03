#!/usr/bin/env python3
"""Produce a reproducible lexical evidence report for every mining batch."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = MODULE_ROOT / "data" / "corpus" / "knowledge_mining"

FAMILIES = {
    "KEEP_LANE": r"\b(?:keep|stay|remain|maintain)(?:\s+\w+){0,3}\s+lane\b|\bcontinue driving on (?:your|the) current lane\b",
    "SET_SPEED": r"\b(?:set|maintain|hold|adjust|settle at|aim to maintain)(?:\s+\w+){0,5}\s+-?\d+(?:\.\d+)?\s*(?:km/h|kph|mph|m/s)\b",
    "INCREASE_SPEED": r"\b(?:accelerate|speed up|increase (?:the )?speed|go faster|pick up the pace|step on the gas|push the accelerator)\b",
    "DECREASE_SPEED": r"\b(?:decelerate|slow down|reduce (?:the )?speed|cut down on speed|ease off the accelerator|hit the brakes|lower your speed)\b",
    "STOP": r"\b(?:stop|halt|standstill|cease (?:all |forward )?movement|full stop|hold position)\b",
    "CHANGE_LANE": r"\b(?:change|switch|shift|move|merge|transition)(?:\s+\w+){0,5}\s+lane\b|\blane change\b",
    "TURN": r"\bturn(?:ing)? (?:left|right)\b|\b(?:make|take) (?:a |the )?(?:left|right)\b",
    "YIELD": r"\b(?:yield to|give way to|let (?:the )?\w+ pass|allow (?:the )?\w+ to pass)\b",
    "PULL_OVER": r"\b(?:pull over|pull up)\b",
    "PARK": r"\b(?:park|parking)\b",
    "OVERTAKE": r"\b(?:overtake|pass (?:the |that |a )?(?:car|vehicle|truck|van|bus)|get past)\b",
    "AVOID": r"\b(?:avoid|go around|drive around|steer clear of|maneuver around|dodge)\b",
    "RESUME": r"\b(?:resume|continue (?:driving|forward|straight))\b",
    "DELIBERATE_COLLISION": r"\b(?:crash(?:\s+\w+){0,3}\s+into|collide with|smash into|run into|ram|bump into)\b",
    "NEGATION": r"\b(?:do not|don't|never|without)\b",
    "ORDER": r"\b(?:then|after|before|once|until|followed by|subsequently)\b",
    "CONDITION": r"\b(?:if|when|once|until|as soon as)\b",
    "DISTANCE": r"\b\d+(?:\.\d+)?\s*(?:m|meter|meters|metre|metres)\b",
    "SPEED_UNIT": r"\b-?\d+(?:\.\d+)?\s*(?:km/h|kph|mph|m/s)\b",
    "FOLLOW": r"\b(?:follow|trail|stay behind)\b",
    "APPROACH": r"\b(?:approach|move closer|drive closer|get closer)\b",
    "NAVIGATE_TO": r"\b(?:go to|drive to|head to|take me to|get to|towards?|toward)\b",
    "REVERSE": r"\b(?:reverse|back up|back into)\b",
    "U_TURN": r"\b(?:u[- ]?turn|turn around)\b",
    "MERGE": r"\bmerge\b",
    "PROCEED": r"\b(?:proceed|go forward|continue straight|drive through)\b",
    "PASS_BY": r"\b(?:drive past|go past|pass by)\b",
    "ENTER_EXIT_AREA": r"\b(?:enter|exit|leave the)\b",
    "WAIT": r"\b(?:wait|hold position|remain in place)\b",
}
COMPILED = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in FAMILIES.items()}


def read_batch(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    aggregate: Counter[str] = Counter()
    batch_reports: list[dict[str, Any]] = []
    unmatched_command_ids: list[str] = []
    for path in sorted((args.root / "gpt_inputs").glob("batch_*.input.jsonl")):
        rows = read_batch(path)
        family_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        for row in rows:
            source_counts[str(row["source"])] += 1
            matched = [
                name for name, pattern in COMPILED.items() if pattern.search(row["text_en"])
            ]
            family_counts.update(matched)
            aggregate.update(matched)
            if (
                row["mining_scope"] == "COMMAND_TERMINOLOGY_AND_PARSE_RULES"
                and not matched
            ):
                unmatched_command_ids.append(str(row["sample_id"]))
        batch_reports.append(
            {
                "batch_id": path.name.removesuffix(".input.jsonl"),
                "rows": len(rows),
                "source_counts": dict(sorted(source_counts.items())),
                "family_counts": dict(sorted(family_counts.items())),
            }
        )

    report = {
        "schema": "english-knowledge-batch-analysis-v1",
        "batch_count": len(batch_reports),
        "total_rows": sum(report["rows"] for report in batch_reports),
        "aggregate_family_counts": dict(sorted(aggregate.items())),
        "unmatched_command_count": len(unmatched_command_ids),
        "unmatched_command_sample_ids": unmatched_command_ids[:100],
        "batches": batch_reports,
        "limitations": [
            "Counts are lexical evidence indicators and may overlap.",
            "Lexical matches are not final labels.",
            "Unmatched commands require terminology or pattern review.",
        ],
    }
    output = args.root / "manifests" / "batch_evidence_analysis.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "batches"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
