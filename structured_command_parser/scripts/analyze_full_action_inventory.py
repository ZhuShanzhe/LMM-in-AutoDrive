#!/usr/bin/env python3
"""Inventory action language in the complete Talk2Car and SimLingo corpus."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = MODULE_ROOT / "data" / "corpus" / "processed"
DEFAULT_OUTPUT = (
    MODULE_ROOT
    / "data"
    / "corpus"
    / "knowledge_mining"
    / "manifests"
    / "full_corpus_action_inventory.json"
)

FAMILIES = {
    "KEEP_LANE": r"\b(?:keep|stay|remain|maintain)(?:\s+\w+){0,4}\s+(?:in |on )?(?:the )?(?:current |same )?lane\b",
    "SET_SPEED": r"\b(?:set|maintain|hold|keep|adjust|settle at|drive at|aim(?:ing)? for)(?:\s+\w+){0,4}\s+-?\d+(?:\.\d+)?\s*(?:km/h|kph|mph|m/s)\b",
    "ADJUST_SPEED_INCREASE": r"\b(?:accelerat\w*|speed up|go faster|drive faster|increase (?:the )?speed|step on the gas|pick up the pace|push the accelerator)\b",
    "ADJUST_SPEED_DECREASE": r"\b(?:decelerat\w*|slow down|go slower|drive more slowly|reduce (?:the )?speed|cut down on speed|ease (?:up on|off) the (?:gas|accelerator)|apply the brakes|hit the brakes)\b",
    "STOP": r"\b(?:stop|halt|standstill|cease (?:all |forward )?movement|hold (?:your )?position|remain stopped)\b",
    "WAIT": r"\b(?:wait|hold position|remain in place|stay put)\b",
    "FOLLOW": r"\b(?:follow|trail|stay behind|keep behind|get (?:right )?behind|same direction as|catch up(?: to)?)\b",
    "APPROACH": r"\b(?:approach|move closer|drive closer|get closer|come closer|advance towards?)\b",
    "NAVIGATE_TO": r"\b(?:go to|drive to|head to|take me to|get to|navigate to|where .+ is|towards? the destination)\b",
    "CHANGE_LANE": r"\b(?:change|switch|shift|move|transition|navigate|head)(?:\s+\w+){0,5}\s+lane\b|\blane change\b|\bmove over\b",
    "MERGE": r"\bmerge(?:\s+\w+){0,4}\b",
    "TURN": r"\bturn(?:ing)? (?:to the )?(?:left|right)\b|\bmake (?:a )?(?:left|right)\b|\btake (?:a |the )?(?:left|right)\b",
    "U_TURN": r"\b(?:u[- ]?turn|turn around)\b",
    "PROCEED": r"\b(?:proceed|advance|continue (?:driving|forward|straight|on)|go (?:ahead|forward|straight)|drive through|go through|cross (?:the )?(?:road|intersection|junction))\b",
    "YIELD": r"\b(?:yield|give way|let (?:the )?.{0,30} pass)\b",
    "PULL_OVER": r"\b(?:pull over|pull up)\b",
    "PARK": r"\b(?:park|parking)\b",
    "OVERTAKE": r"\b(?:overtake|get past|pass (?:the |that |a )?(?:car|vehicle|truck|van|bus|taxi|suv))\b",
    "PASS_BY": r"\b(?:drive past|go past|pass by|pass (?:the |that |a )?(?:person|pedestrian|man|woman|cyclist|cone|sign|building|object))\b",
    "AVOID": r"\b(?:avoid|dodge|swerve|go around|drive around|maneuver around|make space for)\b",
    "REVERSE": r"\b(?:reverse|back up|back into)\b",
    "RESUME": r"\b(?:resume|return to (?:the |your )?(?:route|lane)|continue to follow)\b",
    "ENTER_OR_EXIT_AREA": r"\b(?:enter|exit|take the exit|leave the)\b",
    "PICKUP_DROPOFF_PURPOSE": r"\b(?:pick up|pickup|drop off|dropoff)\b",
    "UNSAFE_COLLISION_OR_VIOLATION": r"\b(?:crash|collide|collision|ram|run into|smash into|red light violation|go through (?:a |the )?red light)\b",
    "CANCEL": r"\b(?:cancel|never mind|disregard (?:the )?(?:previous|last))\b",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = (
        args.processed / "talk2car_all.jsonl",
        args.processed / "simlingo_dreamer_unique.jsonl",
        args.processed / "simlingo_commentary_unique.jsonl",
    )
    counts: Counter[tuple[str, str]] = Counter()
    source_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = {name: [] for name in FAMILIES}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                source = str(row["source"])
                text = str(row["text_en"]).strip()
                source_counts[source] += 1
                for family, pattern in FAMILIES.items():
                    if not re.search(pattern, text, re.I):
                        continue
                    counts[("ALL", family)] += 1
                    counts[(source, family)] += 1
                    if len(examples[family]) < 8:
                        examples[family].append(
                            {"sample_id": str(row["sample_id"]), "text_en": text}
                        )

    report = {
        "schema": "full-corpus-action-inventory-v1",
        "method": "overlapping_case_insensitive_lexical_patterns",
        "status": "AI_CURATED_PENDING_HUMAN_REVIEW",
        "source_rows": dict(sorted(source_counts.items())),
        "total_rows": sum(source_counts.values()),
        "families": [
            {
                "family": family,
                "matches": counts[("ALL", family)],
                "source_matches": {
                    source: counts[(source, family)] for source in sorted(source_counts)
                },
                "examples": examples[family],
            }
            for family in FAMILIES
        ],
        "warnings": [
            "Counts overlap because one command can contain multiple actions.",
            "Pattern matches are inventory evidence, not parser accuracy labels.",
            "SimLingo Commentary is context vocabulary, not passenger-command gold.",
            "Unsafe collision/violation requests must remain UNSUPPORTED.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "total_rows": report["total_rows"],
                "families": len(report["families"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
