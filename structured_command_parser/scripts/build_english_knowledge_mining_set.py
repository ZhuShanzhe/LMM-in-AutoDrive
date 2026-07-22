#!/usr/bin/env python3
"""Build deterministic, diverse English batches for terminology/rule mining."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


MODULE_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = MODULE_ROOT / "data" / "corpus"
PROCESSED_ROOT = CORPUS_ROOT / "processed"
MINING_ROOT = CORPUS_ROOT / "knowledge_mining"

NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
SPACE_RE = re.compile(r"\s+")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pattern_fingerprint(text: str) -> str:
    text = NUMBER_RE.sub("<num>", text.casefold())
    text = re.sub(r"[^a-z<>]+", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def length_bucket(text: str) -> str:
    words = len(text.split())
    if words <= 6:
        return "SHORT"
    if words <= 14:
        return "MEDIUM"
    return "LONG"


def frequency_bucket(row: dict[str, Any]) -> str:
    occurrences = int((row.get("metadata") or {}).get("occurrences") or 1)
    if occurrences == 1:
        return "SINGLETON"
    if occurrences <= 20:
        return "REPEATED"
    return "FREQUENT"


def talk2car_group(row: dict[str, Any]) -> tuple[str, ...]:
    expected = row.get("proposed_expected") or {}
    actions = expected.get("actions") or ["UNMAPPED"]
    directions = expected.get("directions") or ["NO_DIRECTION"]
    return (
        str(expected.get("status") or "UNMAPPED"),
        "+".join(sorted(map(str, actions))),
        "+".join(sorted(map(str, directions))),
        length_bucket(row["text_en"]),
    )


def dreamer_group(row: dict[str, Any]) -> tuple[str, ...]:
    expected = row.get("proposed_expected") or {}
    actions = expected.get("actions") or ["UNMAPPED"]
    metadata = row.get("metadata") or {}
    return (
        str(metadata.get("mode") or "UNMAPPED"),
        str(expected.get("status") or "UNMAPPED"),
        "+".join(sorted(map(str, actions))),
        length_bucket(row["text_en"]),
        frequency_bucket(row),
    )


def commentary_group(row: dict[str, Any]) -> tuple[str, ...]:
    metadata = row.get("metadata") or {}
    return (
        str(metadata.get("scenario_name") or "UNMAPPED"),
        length_bucket(row["text_en"]),
        frequency_bucket(row),
    )


def select_diverse(
    path: Path,
    limit: int,
    group_fn: Callable[[dict[str, Any]], tuple[str, ...]],
    *,
    primary_index: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    seen_patterns: set[str] = set()
    for row in read_jsonl(path):
        fingerprint = pattern_fingerprint(row["text_en"])
        if not fingerprint or fingerprint in seen_patterns:
            continue
        seen_patterns.add(fingerprint)
        groups[group_fn(row)].append(row)

    queues: dict[tuple[str, ...], deque[dict[str, Any]]] = {}
    for key, rows in groups.items():
        rows.sort(key=lambda row: stable_hash(row["sample_id"]))
        queues[key] = deque(rows)

    selected: list[dict[str, Any]] = []
    keys = sorted(queues, key=lambda key: stable_hash("|".join(key)))
    primary_values = {key[primary_index] for key in keys}
    primary_cap = math.ceil(limit / max(len(primary_values), 1))
    primary_counts: Counter[str] = Counter()

    while len(selected) < limit:
        progressed = False
        for key in keys:
            primary = key[primary_index]
            queue = queues[key]
            if queue and primary_counts[primary] < primary_cap and len(selected) < limit:
                selected.append(queue.popleft())
                primary_counts[primary] += 1
                progressed = True
        if not progressed:
            break

    # Some primary groups may have fewer patterns than their quota. Fill the
    # remainder only after every available primary group received a fair pass.
    while len(selected) < limit:
        progressed = False
        for key in keys:
            queue = queues[key]
            if queue and len(selected) < limit:
                selected.append(queue.popleft())
                progressed = True
        if not progressed:
            break
    return selected


def mining_view(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    expected = row.get("proposed_expected")
    source = row["source"]
    scope = (
        "CONTEXT_TERMINOLOGY_ONLY"
        if source == "SimLingo-Commentary"
        else "COMMAND_TERMINOLOGY_AND_PARSE_RULES"
    )
    weak_hints: dict[str, Any] = {}
    if expected:
        weak_hints["proposed_expected"] = expected
    for key in ("mode", "allowed", "safe_to_execute", "scenario_name", "cause_object"):
        if metadata.get(key) is not None:
            weak_hints[key] = metadata[key]
    return {
        "sample_id": row["sample_id"],
        "source": source,
        "source_split": row.get("source_split"),
        "mining_scope": scope,
        "text_en": row["text_en"],
        "weak_hints": weak_hints,
        "provenance": {
            "source_ref": row.get("source_ref"),
            "occurrences": metadata.get("occurrences", 1),
        },
    }


def reset_generated_directories() -> None:
    gpt_inputs = MINING_ROOT / "gpt_inputs"
    if gpt_inputs.exists():
        shutil.rmtree(gpt_inputs)
    for path in (
        MINING_ROOT / "gpt_inputs",
        MINING_ROOT / "gpt_outputs" / "raw",
        MINING_ROOT / "gpt_outputs" / "validated",
        MINING_ROOT / "gpt_outputs" / "errors",
        MINING_ROOT / "gpt_outputs" / "merged",
        MINING_ROOT / "artifacts" / "candidates",
        MINING_ROOT / "artifacts" / "approved",
        MINING_ROOT / "manifests",
    ):
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--talk2car", type=int, default=500)
    parser.add_argument("--dreamer", type=int, default=300)
    parser.add_argument("--commentary", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    selected = {
        "Talk2Car": select_diverse(
            PROCESSED_ROOT / "talk2car_all.jsonl",
            args.talk2car,
            talk2car_group,
            primary_index=1,
        ),
        "SimLingo-Dreamer": select_diverse(
            PROCESSED_ROOT / "simlingo_dreamer_unique.jsonl",
            args.dreamer,
            dreamer_group,
            primary_index=0,
        ),
        "SimLingo-Commentary": select_diverse(
            PROCESSED_ROOT / "simlingo_commentary_unique.jsonl",
            args.commentary,
            commentary_group,
            primary_index=0,
        ),
    }
    rows = [mining_view(row) for source_rows in selected.values() for row in source_rows]
    rows.sort(key=lambda row: stable_hash(row["sample_id"]))

    reset_generated_directories()
    write_jsonl(MINING_ROOT / "representative_samples.jsonl", rows)
    batch_files: list[str] = []
    for start in range(0, len(rows), args.batch_size):
        batch_number = start // args.batch_size + 1
        batch_path = MINING_ROOT / "gpt_inputs" / f"batch_{batch_number:03d}.input.jsonl"
        write_jsonl(batch_path, rows[start : start + args.batch_size])
        batch_files.append(batch_path.name)

    action_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    for row in rows:
        scope_counts[row["mining_scope"]] += 1
        hints = row["weak_hints"]
        expected = hints.get("proposed_expected") or {}
        for action in expected.get("actions") or ["UNMAPPED"]:
            action_counts[str(action)] += 1
        if hints.get("mode"):
            mode_counts[str(hints["mode"])] += 1

    summary = {
        "schema": "english-knowledge-mining-selection-v1",
        "total_samples": len(rows),
        "batch_size": args.batch_size,
        "batch_count": len(batch_files),
        "source_counts": {source: len(values) for source, values in selected.items()},
        "scope_counts": dict(sorted(scope_counts.items())),
        "weak_action_hint_counts": dict(sorted(action_counts.items())),
        "dreamer_mode_counts": dict(sorted(mode_counts.items())),
        "batch_files": batch_files,
        "warnings": [
            "weak_hints are noisy source metadata or heuristic labels, not ground truth",
            "SimLingo Commentary is context terminology only, not passenger commands",
            "Chinese translation is intentionally deferred to a later phase",
        ],
    }
    (MINING_ROOT / "manifests" / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
