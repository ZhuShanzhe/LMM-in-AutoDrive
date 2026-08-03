from __future__ import annotations

import argparse
import json
from pathlib import Path

from structured_command_parser.scripts.evaluate_parser import (
    matches_expected,
    summarize_result,
)
from structured_command_parser.src.rule_parser import RuleIntentParser


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    MODULE_ROOT / "tests" / "fixtures" / "golden_commands.jsonl",
    MODULE_ROOT / "data" / "processed" / "chinese_diverse_dev.jsonl",
]
DEFAULT_OUTPUT = MODULE_ROOT / "configs" / "semantic_prototypes.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        dest="candidates",
        help="Add only candidates whose proposed label is confirmed by the rule parser.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows: list[dict] = []
    seen: set[str] = set()
    for path in args.inputs or DEFAULT_INPUTS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            text = sample["text"].strip()
            if text in seen:
                continue
            seen.add(text)
            rows.append(
                {
                    "prototype_id": sample["sample_id"],
                    "text": text,
                    "expected": sample["expected"],
                    "source": "reviewed",
                    "weight": 1.0,
                }
            )

    accepted_candidates = 0
    rejected_candidates = 0
    rule_parser = RuleIntentParser()
    for path in args.candidates or []:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            text = (sample.get("text") or sample.get("text_zh") or "").strip()
            expected = sample.get("expected") or sample.get("proposed_expected")
            intent = rule_parser.parse(text) if text and isinstance(expected, dict) else None
            if intent is None or not matches_expected(summarize_result(intent), expected):
                rejected_candidates += 1
                continue
            if text in seen:
                continue
            seen.add(text)
            accepted_candidates += 1
            rows.append(
                {
                    "prototype_id": sample["sample_id"],
                    "text": text,
                    "expected": expected,
                    "source": sample.get("source", "rule_confirmed_candidate"),
                    "weight": 0.7,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"prototypes: {len(rows)}")
    print(f"accepted_candidates: {accepted_candidates}")
    print(f"rejected_candidates: {rejected_candidates}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
