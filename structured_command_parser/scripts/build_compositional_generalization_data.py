from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


COLORS = ("red", "blue", "white", "black", "silver", "yellow")
VEHICLES = ("car", "truck", "van", "bus", "sedan", "SUV")
STATES = ("", "parked ", "moving ", "slow ", "large ", "small ")
POSITIONS = ("", " ahead", " on the left", " on the right")
LANDMARKS = ("bus stop", "traffic light", "traffic sign")
ORDINALS = ("first", "second", "third")

RELATION_PHRASES = (
    "before",
    "behind",
    "after",
    "next to",
    "near",
    "between",
    "when you see",
    "once you see",
)

TRAIN_TEMPLATES = (
    {
        "template": "Slow down and stop before {entity}.",
        "actions": ["ADJUST_SPEED", "STOP"],
        "predicates": ["BEFORE"],
        "phenomenon": "multi_action",
    },
    {
        "template": "Follow {entity}.",
        "actions": ["FOLLOW"],
        "predicates": [],
        "phenomenon": "reference",
    },
    {
        "template": "Park behind {entity}.",
        "actions": ["PARK"],
        "predicates": ["BEHIND"],
        "phenomenon": "relative_relation",
    },
    {
        "template": "Overtake {entity}, then resume the original lane.",
        "actions": ["OVERTAKE", "RESUME"],
        "predicates": [],
        "phenomenon": "multi_action",
    },
    {
        "template": "When you see the {landmark}, pull over and stop on the right.",
        "actions": ["PULL_OVER", "STOP"],
        "directions": ["RIGHT"],
        "predicates": ["VISIBLE"],
        "phenomenon": "conditional",
    },
    {
        "template": "Turn {direction} after the {ordinal} junction.",
        "actions": ["TURN"],
        "directions_from": "direction",
        "predicates": ["AFTER"],
        "phenomenon": "ordinal_condition",
    },
    {
        "template": "Follow {entity}, but keep a safe distance.",
        "actions": ["FOLLOW"],
        "predicates": ["SAFE_DISTANCE"],
        "phenomenon": "constraint",
    },
    {
        "template": "Do not turn {negative_direction}; instead continue straight.",
        "actions": ["PROCEED"],
        "directions": ["STRAIGHT"],
        "suppressed_actions": ["TURN"],
        "phenomenon": "counterfactual_negation",
    },
)

TEST_TEMPLATES = (
    {
        "template": "Find an opportunity to merge into the {direction} lane.",
        "actions": ["CHANGE_LANE"],
        "directions_from": "direction",
        "predicates": [],
        "phenomenon": "heldout_paraphrase",
    },
    {
        "template": "Wait for a safe gap, then move into the {direction} lane.",
        "actions": ["CHANGE_LANE"],
        "directions_from": "direction",
        "predicates": [],
        "phenomenon": "heldout_paraphrase",
    },
    {
        "template": "When it is safe, switch to the {direction}-hand lane.",
        "actions": ["CHANGE_LANE"],
        "directions_from": "direction",
        "predicates": [],
        "phenomenon": "heldout_paraphrase",
    },
    {
        "template": "Use the {direction}-hand lane when possible.",
        "actions": ["CHANGE_LANE"],
        "directions_from": "direction",
        "predicates": [],
        "phenomenon": "heldout_paraphrase",
    },
    {
        "template": "Get past {entity} and move back into the original lane.",
        "actions": ["OVERTAKE", "RESUME"],
        "predicates": [],
        "phenomenon": "heldout_action_composition",
    },
    {
        "template": "Pass {entity}, then return to the previous lane.",
        "actions": ["OVERTAKE", "RESUME"],
        "predicates": [],
        "phenomenon": "heldout_action_composition",
    },
    {
        "template": "Overtake {entity} and get back to the original lane.",
        "actions": ["OVERTAKE", "RESUME"],
        "predicates": [],
        "phenomenon": "heldout_action_composition",
    },
    {
        "template": "Once you see the {landmark}, pull over to the right and stop.",
        "actions": ["PULL_OVER", "STOP"],
        "directions": ["RIGHT"],
        "predicates": ["VISIBLE"],
        "phenomenon": "heldout_condition",
    },
    {
        "template": "After you spot the {landmark}, pull over on the right and stop.",
        "actions": ["PULL_OVER", "STOP"],
        "directions": ["RIGHT"],
        "predicates": ["VISIBLE"],
        "phenomenon": "heldout_condition",
    },
    {
        "template": "When the {landmark} comes into view, pull over on the right and stop.",
        "actions": ["PULL_OVER", "STOP"],
        "directions": ["RIGHT"],
        "predicates": ["VISIBLE"],
        "phenomenon": "heldout_condition",
    },
    {
        "template": "Upon seeing the {landmark}, pull over to the right and stop.",
        "actions": ["PULL_OVER", "STOP"],
        "directions": ["RIGHT"],
        "predicates": ["VISIBLE"],
        "phenomenon": "heldout_condition",
    },
    {
        "template": "Make a {direction} turn after passing the {ordinal} junction.",
        "actions": ["TURN"],
        "directions_from": "direction",
        "predicates": ["AFTER"],
        "phenomenon": "heldout_ordinal",
    },
    {
        "template": "After passing the {ordinal} junction, turn {direction}.",
        "actions": ["TURN"],
        "directions_from": "direction",
        "predicates": ["AFTER"],
        "phenomenon": "heldout_ordinal",
    },
    {
        "template": "Turn {direction} once you have passed the {ordinal} junction.",
        "actions": ["TURN"],
        "directions_from": "direction",
        "predicates": ["AFTER"],
        "phenomenon": "heldout_ordinal",
    },
    {
        "template": "Trail the one ahead, but do not get too close.",
        "actions": ["FOLLOW"],
        "predicates": ["SAFE_DISTANCE"],
        "phenomenon": "ellipsis_constraint",
    },
    {
        "template": "Stay behind {entity}, without getting too close.",
        "actions": ["FOLLOW"],
        "predicates": ["BEHIND", "SAFE_DISTANCE"],
        "phenomenon": "ellipsis_constraint",
    },
    {
        "template": "Keep up with {entity}, while maintaining a safe gap.",
        "actions": ["FOLLOW"],
        "predicates": ["SAFE_DISTANCE"],
        "phenomenon": "ellipsis_constraint",
    },
    {
        "template": "Do not turn {negative_direction}; carry on straight.",
        "actions": ["PROCEED"],
        "directions": ["STRAIGHT"],
        "suppressed_actions": ["TURN"],
        "phenomenon": "heldout_counterfactual",
    },
    {
        "template": "Don't turn {negative_direction}; keep going straight.",
        "actions": ["PROCEED"],
        "directions": ["STRAIGHT"],
        "suppressed_actions": ["TURN"],
        "phenomenon": "heldout_counterfactual",
    },
    {
        "template": "Skip the {negative_direction} turn and proceed straight.",
        "actions": ["PROCEED"],
        "directions": ["STRAIGHT"],
        "suppressed_actions": ["TURN"],
        "phenomenon": "heldout_counterfactual",
    },
    {
        "template": "Take the write lane when it is safe.",
        "actions": ["CHANGE_LANE"],
        "directions": ["RIGHT"],
        "predicates": [],
        "phenomenon": "asr_noise",
    },
    {
        "template": "Move into the write lane when clear.",
        "actions": ["CHANGE_LANE"],
        "directions": ["RIGHT"],
        "predicates": [],
        "phenomenon": "asr_noise",
    },
    {
        "template": "Make a write turn at the next junction.",
        "actions": ["TURN"],
        "directions": ["RIGHT"],
        "predicates": [],
        "phenomenon": "asr_noise",
    },
)


def _spans(text: str, entities: list[str]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    cursor = 0
    for entity in entities:
        start = text.casefold().find(entity.casefold(), cursor)
        if start < 0:
            continue
        end = start + len(entity)
        spans.append(
            {
                "role": "ENTITY",
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )
        cursor = end
    for phrase in RELATION_PHRASES:
        cursor = 0
        while True:
            start = text.casefold().find(phrase, cursor)
            if start < 0:
                break
            end = start + len(phrase)
            spans.append(
                {
                    "role": "RELATION",
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
            cursor = end
    return sorted(spans, key=lambda item: (item["start"], item["end"]))


def _row(template: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    text = template["template"].format(**values)
    entities = [
        values[key]
        for key in ("entity", "landmark")
        if key in values
    ]
    if "junction" in text.casefold():
        if "{ordinal}" in template["template"]:
            entities.append(f"the {values['ordinal']} junction")
        elif "next junction" in text.casefold():
            entities.append("the next junction")
    if "one ahead" in text.casefold():
        entities.append("the one ahead")
    directions = list(template.get("directions", []))
    if template.get("directions_from"):
        directions.append(values[template["directions_from"]].upper())
    return {
        "sample_id": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
        "text_en": text,
        "spans": _spans(text, entities),
        "expected": {
            "status": "VALID",
            "actions": template["actions"],
            "directions": directions,
            "predicates": template.get("predicates", []),
            "suppressed_actions": template.get("suppressed_actions", []),
        },
        "phenomenon": template["phenomenon"],
    }


def _combinations(
    templates: tuple[dict[str, Any], ...],
    *,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: dict[str, dict[str, Any]] = {}
    for _ in range(repetitions):
        for template in templates:
            values = {
                "entity": (
                    f"the {rng.choice(STATES)}{rng.choice(COLORS)} "
                    f"{rng.choice(VEHICLES)}{rng.choice(POSITIONS)}"
                ),
                "landmark": rng.choice(LANDMARKS),
                "ordinal": rng.choice(ORDINALS),
                "direction": rng.choice(("left", "right")),
                "negative_direction": rng.choice(("left", "right")),
            }
            row = _row(template, values)
            rows[row["sample_id"]] = row
    return sorted(rows.values(), key=lambda item: item["sample_id"])


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def _balanced_test(
    rows: list[dict[str, Any]],
    *,
    limit_per_phenomenon: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["phenomenon"], []).append(row)
    rng = random.Random(seed)
    selected = []
    for phenomenon in sorted(grouped):
        candidates = grouped[phenomenon]
        rng.shuffle(candidates)
        selected.extend(candidates[:limit_per_phenomenon])
    return sorted(selected, key=lambda item: item["sample_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "compositional_generalization",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    train = _combinations(TRAIN_TEMPLATES, repetitions=1200, seed=args.seed)
    validation = _combinations(
        TRAIN_TEMPLATES, repetitions=240, seed=args.seed + 1
    )
    test = _balanced_test(
        _combinations(TEST_TEMPLATES, repetitions=720, seed=args.seed + 2),
        limit_per_phenomenon=64,
        seed=args.seed + 3,
    )
    _write(args.output / "train.jsonl", train)
    _write(args.output / "validation.jsonl", validation)
    _write(args.output / "test.jsonl", test)
    manifest = {
        "schema": "compositional-driving-language-v1",
        "seed": args.seed,
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "split_policy": {
            "train_validation": "seen semantic skeletons with disjoint surfaces",
            "test": "held-out paraphrases, action combinations, conditions, counterfactuals, and ASR noise",
            "random_row_split": False,
            "test_limit_per_phenomenon": 64,
        },
        "limitations": [
            "Synthetic labels test compositional contract behavior, not road safety.",
            "Chinese ASR ambiguity is covered by rule tests because the online ModernBERT parser consumes normalized English.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
