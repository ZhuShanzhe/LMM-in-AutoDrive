#!/usr/bin/env python3
"""Validate GPT terminology/rule mining outputs without approving their content."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = MODULE_ROOT / "data" / "corpus" / "knowledge_mining"

ACTIONS = {
    "KEEP_LANE",
    "SET_SPEED",
    "ADJUST_SPEED",
    "STOP",
    "CHANGE_LANE",
    "TURN",
    "YIELD",
    "PULL_OVER",
    "OVERTAKE",
    "AVOID",
    "EMERGENCY_BRAKE",
    "RESUME",
    "CANCEL",
}
RULE_TYPES = {
    "ACTION",
    "SLOT",
    "ORDER",
    "NEGATION",
    "AMBIGUITY",
    "UNSUPPORTED",
    "NORMALIZATION",
}
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def load_batch_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line)["sample_id"]))
    return ids


def confidence_error(value: Any) -> bool:
    return not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1


def validate_output(
    document: Any, allowed_ids: set[str], expected_batch_id: str
) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["top level must be an object"]
    for field in ("batch_id", "terminology", "rules", "unresolved"):
        if field not in document:
            errors.append(f"missing top-level field: {field}")
    if document.get("batch_id") != expected_batch_id:
        errors.append(
            f"batch_id mismatch: {document.get('batch_id')} != {expected_batch_id}"
        )
    if CJK_RE.search(json.dumps(document, ensure_ascii=False)):
        errors.append("Chinese text found in an English-only mining output")

    terminology = document.get("terminology")
    if not isinstance(terminology, list):
        errors.append("terminology must be an array")
        terminology = []
    for index, term in enumerate(terminology):
        prefix = f"terminology[{index}]"
        if not isinstance(term, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in (
            "term_id",
            "concept",
            "canonical_action",
            "expressions_en",
            "definition_en",
            "required_slots",
            "optional_slots",
            "confusable_with",
            "negative_patterns",
            "source_sample_ids",
            "confidence",
        ):
            if field not in term:
                errors.append(f"{prefix} missing field: {field}")
        action = term.get("canonical_action")
        if action is not None and action not in ACTIONS:
            errors.append(f"{prefix} invalid canonical_action: {action}")
        if not isinstance(term.get("expressions_en"), list) or not term.get("expressions_en"):
            errors.append(f"{prefix} expressions_en must be a non-empty array")
        source_ids = term.get("source_sample_ids")
        if not isinstance(source_ids, list) or not source_ids:
            errors.append(f"{prefix} source_sample_ids must be a non-empty array")
        else:
            unknown = sorted(set(map(str, source_ids)) - allowed_ids)
            if unknown:
                errors.append(f"{prefix} cites IDs outside its input batch: {unknown[:5]}")
        if confidence_error(term.get("confidence")):
            errors.append(f"{prefix} confidence must be between 0 and 1")

    rules = document.get("rules")
    if not isinstance(rules, list):
        errors.append("rules must be an array")
        rules = []
    for index, rule in enumerate(rules):
        prefix = f"rules[{index}]"
        if not isinstance(rule, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in (
            "rule_id",
            "rule_type",
            "priority",
            "description",
            "positive_patterns",
            "negative_patterns",
            "output_constraints",
            "source_sample_ids",
            "confidence",
        ):
            if field not in rule:
                errors.append(f"{prefix} missing field: {field}")
        if rule.get("rule_type") not in RULE_TYPES:
            errors.append(f"{prefix} invalid rule_type: {rule.get('rule_type')}")
        if not isinstance(rule.get("priority"), int) or isinstance(rule.get("priority"), bool):
            errors.append(f"{prefix} priority must be an integer")
        source_ids = rule.get("source_sample_ids")
        if not isinstance(source_ids, list) or not source_ids:
            errors.append(f"{prefix} source_sample_ids must be a non-empty array")
        else:
            unknown = sorted(set(map(str, source_ids)) - allowed_ids)
            if unknown:
                errors.append(f"{prefix} cites IDs outside its input batch: {unknown[:5]}")
        if confidence_error(rule.get("confidence")):
            errors.append(f"{prefix} confidence must be between 0 and 1")

        constraints = rule.get("output_constraints")
        if isinstance(constraints, dict):
            action = constraints.get("action")
            if action is not None and action not in ACTIONS:
                errors.append(f"{prefix} invalid output action: {action}")

    unresolved = document.get("unresolved")
    if not isinstance(unresolved, list):
        errors.append("unresolved must be an array")
    else:
        for index, item in enumerate(unresolved):
            prefix = f"unresolved[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            source_ids = item.get("sample_ids")
            if not isinstance(source_ids, list) or not source_ids:
                errors.append(f"{prefix} sample_ids must be a non-empty array")
            else:
                unknown = sorted(set(map(str, source_ids)) - allowed_ids)
                if unknown:
                    errors.append(f"{prefix} cites IDs outside its input batch: {unknown[:5]}")
            if not isinstance(item.get("issue"), str) or not item.get("issue", "").strip():
                errors.append(f"{prefix} issue must be a non-empty string")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    raw_root = args.root / "gpt_outputs" / "raw"
    validated_root = args.root / "gpt_outputs" / "validated"
    error_root = args.root / "gpt_outputs" / "errors"
    validated_root.mkdir(parents=True, exist_ok=True)
    error_root.mkdir(parents=True, exist_ok=True)

    summary = {"checked": 0, "valid": 0, "invalid": 0, "files": []}
    for output_path in sorted(raw_root.glob("batch_*.output.json")):
        batch_name = output_path.name.removesuffix(".output.json")
        input_path = args.root / "gpt_inputs" / f"{batch_name}.input.jsonl"
        file_result: dict[str, Any] = {"file": output_path.name, "errors": []}
        if not input_path.is_file():
            file_result["errors"].append(f"missing matching input: {input_path.name}")
            document = None
            allowed_ids: set[str] = set()
        else:
            allowed_ids = load_batch_ids(input_path)
            try:
                document = json.loads(output_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                document = None
                file_result["errors"].append(f"invalid JSON: {exc}")
        if document is not None:
            file_result["errors"].extend(
                validate_output(document, allowed_ids, batch_name)
            )

        summary["checked"] += 1
        if file_result["errors"]:
            summary["invalid"] += 1
            (error_root / f"{batch_name}.validation.json").write_text(
                json.dumps(file_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            summary["valid"] += 1
            (validated_root / f"{batch_name}.validated.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        summary["files"].append(file_result)

    (args.root / "manifests" / "gpt_output_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if summary["invalid"] == 0 else 1)


if __name__ == "__main__":
    main()
