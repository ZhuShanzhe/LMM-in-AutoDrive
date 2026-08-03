from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from structured_command_parser.src.english_parser import QwenEnglishIntentParser
from structured_command_parser.src.intent_boundaries import classify_english_braking
from structured_command_parser.src.modernbert_labels import (
    ACTION_LABELS,
    CATEGORY_LABELS,
    DIRECTION_LABELS,
    STATUS_LABELS,
    URGENCY_LABELS,
    label_schema,
)


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = MODULE_ROOT / "data" / "corpus" / "processed"
DEFAULT_OUTPUT = MODULE_ROOT / "data" / "processed" / "english_pseudolabels"
SOURCE_FILES = (
    "talk2car_all.jsonl",
    "simlingo_dreamer_unique.jsonl",
    "simlingo_commentary_unique.jsonl",
)

SPARSE_AUGMENTATION = {
    "EMERGENCY_BRAKE": tuple(
        f"{command} {suffix}".strip() + "."
        for command in (
            "Emergency brake",
            "Apply the emergency brake",
            "Slam on the brakes",
            "Brake hard",
            "Apply full braking",
        )
        for suffix in (
            "now",
            "immediately",
            "at once",
            "as quickly as possible",
            "because there is collision danger ahead",
            "because a pedestrian stepped into the road",
            "because the lane is suddenly blocked",
            "because an obstacle appeared ahead",
            "before we collide with the vehicle ahead",
            "due to the immediate hazard",
            "this is an emergency",
            "without delay",
        )
    ),
    "CANCEL": tuple(
        f"{command}{suffix}."
        for command in (
            "Cancel",
            "Abort",
            "Revoke",
            "Withdraw",
            "Do not execute",
            "Disregard",
            "Ignore",
            "Forget",
        )
        for suffix in (
            " the previous command",
            " my last instruction",
            " the last maneuver",
            " that driving request",
            " what I just asked",
            " the current instruction",
        )
    ),
}


def compact_text(value: str) -> str:
    return " ".join(value.strip().split())


def normalized_key(text: str) -> str:
    return compact_text(text).casefold()


def split_hash(text: str) -> str:
    return hashlib.sha1(normalized_key(text).encode("utf-8")).hexdigest()


def commands_from_expected(expected: dict[str, Any]) -> list[dict[str, Any]]:
    actions = expected.get("actions") or expected.get("actions_unordered") or []
    directions = list(expected.get("directions") or [])
    speeds = list(expected.get("target_speed_mps") or [])
    commands: list[dict[str, Any]] = []
    for action in actions:
        command: dict[str, Any] = {"action": action}
        if action in {"CHANGE_LANE", "MERGE", "TURN"} and directions:
            command["direction"] = directions.pop(0)
        if action == "SET_SPEED" and speeds:
            command["target_speed_mps"] = speeds.pop(0)
        commands.append(command)
    return commands


def teacher_payload(row: dict[str, Any], text: str) -> tuple[dict[str, Any], str]:
    source = row.get("source", "")
    proposed = row.get("proposed_expected")
    use_source_label = bool(
        source == "SimLingo-Dreamer"
        and isinstance(proposed, dict)
        and (proposed.get("actions") or proposed.get("status") == "UNSUPPORTED")
    )
    payload: dict[str, Any] = {"commands": []}
    origin = "CURATED_RULE_HEURISTIC"
    if use_source_label:
        payload.update(
            {
                "commands": commands_from_expected(proposed),
                "status": proposed.get("status", "VALID"),
                "category": proposed.get("category", "BASIC_CONTROL"),
                "urgency": proposed.get("urgency", "NORMAL"),
            }
        )
        origin = "SOURCE_MODE_AND_RULE"
    if not payload["commands"]:
        braking = classify_english_braking(text)
        if braking is not None:
            command: dict[str, Any] = {"action": braking.action}
            if braking.action == "ADJUST_SPEED":
                command["change"] = "DECREASE"
            payload.update(
                {
                    "commands": [command],
                    "status": "VALID",
                    "category": (
                        "EMERGENCY_RESPONSE"
                        if braking.action == "EMERGENCY_BRAKE"
                        else "BASIC_CONTROL"
                    ),
                    "urgency": braking.urgency,
                }
            )
    normalized = QwenEnglishIntentParser._normalize_payload(payload, text)
    commands = list(normalized.get("commands") or [])
    status = normalized.get("status", "VALID" if commands else "NEEDS_CLARIFICATION")
    if status == "VALID" and not commands:
        status = "NEEDS_CLARIFICATION"
    normalized["status"] = status
    if not commands and status == "NEEDS_CLARIFICATION":
        origin = "UNRESOLVED_TEXT"
    return normalized, origin


def expected_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    commands = list(payload.get("commands") or [])
    actions = [
        command.get("action")
        for command in commands
        if command.get("action") in ACTION_LABELS
    ]
    directions = [
        command.get("direction")
        for command in commands
        if command.get("direction") in DIRECTION_LABELS
    ]
    changes = [
        command.get("change")
        for command in commands
        if command.get("change") in {"INCREASE", "DECREASE"}
    ]
    expected: dict[str, Any] = {
        "status": (
            payload.get("status")
            if payload.get("status") in STATUS_LABELS
            else "NEEDS_CLARIFICATION"
        ),
        "category": (
            payload.get("category")
            if payload.get("category") in CATEGORY_LABELS
            else "BASIC_CONTROL"
        ),
        "urgency": (
            payload.get("urgency")
            if payload.get("urgency") in URGENCY_LABELS
            else "NORMAL"
        ),
        "actions": actions,
        "directions": directions,
        "change": changes[0] if changes else "NONE",
        "commands": commands,
    }
    if expected["status"] != "VALID":
        expected["actions"] = []
        expected["directions"] = []
        expected["change"] = "NONE"
        expected["commands"] = []
    return expected


def quality_weight(source: str, origin: str) -> float:
    if origin == "SOURCE_MODE_AND_RULE":
        return 1.0
    if origin == "UNRESOLVED_TEXT":
        return 0.35
    if source == "Talk2Car":
        return 0.9
    if source == "SimLingo-Commentary":
        return 0.5
    return 0.8


def iter_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL: {path}:{line_number}: {error}") from error


def pseudo_label(row: dict[str, Any]) -> dict[str, Any]:
    text = compact_text(str(row.get("text_en") or ""))
    if not text:
        raise ValueError(f"Missing text_en for {row.get('sample_id')}")
    payload, origin = teacher_payload(row, text)
    source = str(row.get("source") or "UNKNOWN")
    metadata = row.get("metadata") or {}
    return {
        "sample_id": row["sample_id"],
        "source": source,
        "source_split": row.get("source_split", ""),
        "text_en": text,
        "expected": expected_from_payload(payload),
        "pseudo_label": {
            "origin": origin,
            "weight": quality_weight(source, origin),
            "human_verified": False,
            "teacher": "reviewed_terminology_rules_v1.1.0",
        },
        "metadata": {
            key: metadata[key]
            for key in ("mode", "scenario_name", "occurrences", "allowed", "safe_to_execute")
            if key in metadata
        },
        "split_key": split_hash(text),
    }


def assign_splits(rows: list[dict[str, Any]]) -> Counter[str]:
    rows.sort(key=lambda row: (row["split_key"], row["sample_id"]))
    targets = {
        "train": round(len(rows) * 0.70),
        "validation": round(len(rows) * 0.20),
    }
    targets["test"] = len(rows) - targets["train"] - targets["validation"]
    counts: Counter[str] = Counter()
    index = 0
    while index < len(rows):
        end = index + 1
        key = rows[index]["split_key"]
        while end < len(rows) and rows[end]["split_key"] == key:
            end += 1
        group_size = end - index
        if counts["train"] < targets["train"]:
            split = "train"
        elif counts["validation"] < targets["validation"]:
            split = "validation"
        else:
            split = "test"
        for row in rows[index:end]:
            row["split"] = split
        counts[split] += group_size
        index = end
    return counts


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def sparse_augmentation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action, texts in SPARSE_AUGMENTATION.items():
        for index, text in enumerate(texts, start=1):
            payload = QwenEnglishIntentParser._normalize_payload(
                {"commands": [{"action": action}]}, text
            )
            rows.append(
                {
                    "sample_id": f"sparse-{action.casefold()}-{index:03d}",
                    "source": "CURATED_SPARSE_AUGMENTATION",
                    "source_split": "train_only",
                    "text_en": text,
                    "expected": expected_from_payload(payload),
                    "pseudo_label": {
                        "origin": "POLICY_CURATED_SPARSE_AUGMENTATION",
                        "weight": 1.5,
                        "human_verified": False,
                        "teacher": "reviewed_braking_cancel_policy_v1.1.0",
                    },
                    "metadata": {"augmentation_action": action},
                    "split_key": split_hash(text),
                    "split": "train",
                }
            )
    return rows


def distribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    urgency_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for row in rows:
        expected = row["expected"]
        source_counts[row["source"]] += 1
        origin_counts[row["pseudo_label"]["origin"]] += 1
        status_counts[expected["status"]] += 1
        category_counts[expected["category"]] += 1
        urgency_counts[expected["urgency"]] += 1
        action_counts.update(expected["actions"])
    return {
        "sources": dict(sorted(source_counts.items())),
        "origins": dict(sorted(origin_counts.items())),
        "statuses": dict(sorted(status_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "urgencies": dict(sorted(urgency_counts.items())),
        "actions": dict(sorted(action_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full English pseudo labels")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_paths = [args.corpus / name for name in SOURCE_FILES]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing corpus files: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)

    rows = [pseudo_label(row) for row in iter_rows(source_paths)]
    if len(rows) != 662700:
        raise ValueError(f"Expected 662700 full-corpus rows, got {len(rows)}")
    split_counts = assign_splits(rows)

    write_jsonl(args.output / "all.jsonl", rows)
    for split in ("train", "validation", "test"):
        write_jsonl(
            args.output / f"{split}.jsonl",
            (row for row in rows if row["split"] == split),
        )
    augmentation = sparse_augmentation_rows()
    write_jsonl(args.output / "train_sparse_augmentation.jsonl", augmentation)

    manifest = {
        "schema": "english-driving-command-pseudolabels-v1",
        "full_corpus_rows": len(rows),
        "split_policy": {
            "method": "SHA1_GROUPED_NORMALIZED_TEXT",
            "requested_ratio": {"train": 0.7, "validation": 0.2, "test": 0.1},
            "counts": dict(split_counts),
            "actual_ratio": {
                key: round(split_counts[key] / len(rows), 8)
                for key in ("train", "validation", "test")
            },
            "same_normalized_text_cross_split": False,
        },
        "label_schema": label_schema(),
        "train_only_sparse_augmentation_rows": len(augmentation),
        "train_only_sparse_augmentation_distribution": distribution(augmentation),
        "distribution": distribution(rows),
        "limitations": [
            "All labels are pseudo labels and are not human gold annotations.",
            "SimLingo Commentary is assigned lower training weight because it is context evidence.",
            "Test metrics measure agreement with the deterministic teacher, not real-world accuracy.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "label_schema.json").write_text(
        json.dumps(label_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
