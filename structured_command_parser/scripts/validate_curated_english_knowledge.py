#!/usr/bin/env python3
"""Validate curated English terminology and parsing rules against source evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = MODULE_ROOT / "data" / "corpus" / "knowledge_mining"
DEFAULT_TERMS = MODULE_ROOT / "configs" / "english_terminology.json"
DEFAULT_RULES = MODULE_ROOT / "configs" / "english_parsing_rules.json"

ACTIONS = {
    "KEEP_LANE",
    "SET_SPEED",
    "ADJUST_SPEED",
    "STOP",
    "WAIT",
    "FOLLOW",
    "APPROACH",
    "NAVIGATE_TO",
    "CHANGE_LANE",
    "MERGE",
    "TURN",
    "U_TURN",
    "PROCEED",
    "YIELD",
    "PULL_OVER",
    "PARK",
    "OVERTAKE",
    "PASS_BY",
    "AVOID",
    "REVERSE",
    "ENTER_AREA",
    "EXIT_AREA",
    "EMERGENCY_BRAKE",
    "RESUME",
    "CANCEL",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def check_evidence(
    owner: str, source_ids: Any, known_ids: set[str], errors: list[str]
) -> None:
    if not isinstance(source_ids, list) or not source_ids:
        errors.append(f"{owner}: source_sample_ids must be a non-empty array")
        return
    unknown = sorted(set(map(str, source_ids)) - known_ids)
    if unknown:
        errors.append(f"{owner}: unknown source IDs: {unknown[:10]}")


def find_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if "regex" in key.casefold() or "code" == key.casefold():
                found.append(f"{path}.{key}")
            found.extend(find_forbidden_keys(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_forbidden_keys(child, f"{path}[{index}]"))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--terminology", type=Path, default=DEFAULT_TERMS)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    args = parser.parse_args()

    terms_doc = json.loads(args.terminology.read_text(encoding="utf-8"))
    rules_doc = json.loads(args.rules.read_text(encoding="utf-8"))
    rows = list(read_jsonl(args.corpus / "representative_samples.jsonl"))
    known_ids = {str(row["sample_id"]) for row in rows}
    errors: list[str] = []

    allowed_statuses = {
        "AI_CURATED_PENDING_HUMAN_REVIEW",
        "REVIEWED_APPROVED_WITH_CORRECTIONS",
    }
    if terms_doc.get("status") not in allowed_statuses:
        errors.append("terminology has an invalid review status")
    if rules_doc.get("status") not in allowed_statuses:
        errors.append("rules have an invalid review status")
    for owner, document in (("terminology", terms_doc), ("rules", rules_doc)):
        if document.get("status") == "REVIEWED_APPROVED_WITH_CORRECTIONS":
            review = document.get("review")
            if not isinstance(review, dict):
                errors.append(f"{owner}: approved status requires review metadata")
                continue
            for key in ("reviewed_at", "reviewer", "review_type", "decision"):
                if not review.get(key):
                    errors.append(f"{owner}: review metadata missing {key}")

    term_ids: set[str] = set()
    action_counts: Counter[str] = Counter()
    for index, term in enumerate(terms_doc.get("terms", [])):
        owner = f"terms[{index}]"
        term_id = str(term.get("term_id") or "")
        if not term_id:
            errors.append(f"{owner}: missing term_id")
        elif term_id in term_ids:
            errors.append(f"{owner}: duplicate term_id {term_id}")
        term_ids.add(term_id)
        action = term.get("canonical_action")
        if action is not None and action not in ACTIONS:
            errors.append(f"{owner}: invalid canonical_action {action}")
        if action:
            action_counts[action] += 1
        if not isinstance(term.get("expressions_en"), list) or not term["expressions_en"]:
            errors.append(f"{owner}: expressions_en must be non-empty")
        check_evidence(owner, term.get("source_sample_ids"), known_ids, errors)
        if int(term.get("evidence_count", 0)) < len(term.get("source_sample_ids") or []):
            errors.append(f"{owner}: evidence_count is smaller than cited evidence")

    gap_ids: set[str] = set()
    for index, gap in enumerate(terms_doc.get("ontology_gaps", [])):
        owner = f"ontology_gaps[{index}]"
        gap_id = str(gap.get("concept_id") or "")
        if not gap_id:
            errors.append(f"{owner}: missing concept_id")
        elif gap_id in gap_ids:
            errors.append(f"{owner}: duplicate concept_id {gap_id}")
        gap_ids.add(gap_id)
        check_evidence(owner, gap.get("source_sample_ids"), known_ids, errors)

    rule_ids: set[str] = set()
    priorities: set[int] = set()
    rule_types: Counter[str] = Counter()
    for index, rule in enumerate(rules_doc.get("rules", [])):
        owner = f"rules[{index}]"
        rule_id = str(rule.get("rule_id") or "")
        if not rule_id:
            errors.append(f"{owner}: missing rule_id")
        elif rule_id in rule_ids:
            errors.append(f"{owner}: duplicate rule_id {rule_id}")
        rule_ids.add(rule_id)
        priority = rule.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool):
            errors.append(f"{owner}: priority must be an integer")
        elif priority in priorities:
            errors.append(f"{owner}: duplicate priority {priority}")
        else:
            priorities.add(priority)
        rule_types[str(rule.get("rule_type") or "MISSING")] += 1
        check_evidence(owner, rule.get("source_sample_ids"), known_ids, errors)
        if not isinstance(rule.get("positive_patterns"), list) or not rule["positive_patterns"]:
            errors.append(f"{owner}: positive_patterns must be non-empty")

    forbidden = find_forbidden_keys({"terminology": terms_doc, "rules": rules_doc})
    if forbidden:
        errors.append(f"executable regex/code fields are forbidden: {forbidden}")

    batch_files = sorted((args.corpus / "gpt_inputs").glob("batch_*.input.jsonl"))
    report = {
        "schema": "curated-english-knowledge-validation-v1",
        "status": "PASS" if not errors else "FAIL",
        "source_samples": len(rows),
        "source_batches": len(batch_files),
        "terminology_entries": len(terms_doc.get("terms", [])),
        "ontology_gaps": len(terms_doc.get("ontology_gaps", [])),
        "parsing_rules": len(rules_doc.get("rules", [])),
        "action_term_counts": dict(sorted(action_counts.items())),
        "rule_type_counts": dict(sorted(rule_types.items())),
        "review_status": {
            "terminology": terms_doc.get("status"),
            "rules": rules_doc.get("status"),
        },
        "errors": errors,
    }
    report_path = args.corpus / "manifests" / "curated_knowledge_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
