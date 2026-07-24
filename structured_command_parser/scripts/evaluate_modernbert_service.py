from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from time import perf_counter
from typing import Any

from structured_command_parser import ModernBertCommandService


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST = MODULE_ROOT / "data" / "processed" / "english_pseudolabels" / "test.jsonl"
DEFAULT_MODEL = Path("/root/autodl-tmp/models/modernbert-drive-command-base")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_rows(
    rows: list[dict[str, Any]], limit: int | None, stratified: bool
) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    if not stratified:
        return rows[:limit]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        actions = row["expected"].get("actions") or []
        key = "+".join(actions) if actions else row["expected"].get("status", "NO_ACTION")
        groups[key].append(row)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    index = 0
    while len(selected) < limit and keys:
        next_keys: list[str] = []
        for key in keys:
            if index < len(groups[key]) and len(selected) < limit:
                selected.append(groups[key][index])
                next_keys.append(key)
        keys = next_keys
        index += 1
    return selected


def contract(document: dict[str, Any]) -> dict[str, Any]:
    steps = document["intent"]["steps"]
    directions = [
        step["parameters"]["direction"]
        for step in steps
        if step["parameters"].get("direction")
    ]
    changes = [
        step["parameters"]["change"]
        for step in steps
        if step["parameters"].get("change")
    ]
    return {
        "status": document["parse_result"]["status"],
        "category": document["intent"]["category"],
        "urgency": document["intent"]["urgency"],
        "actions": [step["action"] for step in steps],
        "directions": directions,
        "change": changes[0] if changes else "NONE",
    }


def evaluate(service: ModernBertCommandService, rows: list[dict[str, Any]]) -> dict[str, Any]:
    warmup_started = perf_counter()
    service.warmup()
    warmup_seconds = perf_counter() - warmup_started
    exact = Counter()
    action_tp = action_fp = action_fn = 0
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        document = service.parse_text(row["text_en"], request_id=row["sample_id"])
        actual = contract(document)
        expected = row["expected"]
        for key in ("status", "category", "urgency", "actions", "directions", "change"):
            exact[key] += int(actual[key] == expected[key])
        predicted_actions = set(actual["actions"])
        gold_actions = set(expected["actions"])
        action_tp += len(predicted_actions & gold_actions)
        action_fp += len(predicted_actions - gold_actions)
        action_fn += len(gold_actions - predicted_actions)
        latencies.append(document["parse_result"]["latency_ms"])
        if actual["actions"] != expected["actions"] and len(failures) < 100:
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "text_en": row["text_en"],
                    "expected": expected,
                    "actual": actual,
                }
            )
    precision = action_tp / max(1, action_tp + action_fp)
    recall = action_tp / max(1, action_tp + action_fn)
    total = len(rows)
    return {
        "schema": "modernbert-service-evaluation-v1",
        "samples": total,
        "action_sequence_exact_match": exact["actions"] / max(1, total),
        "action_micro_precision": precision,
        "action_micro_recall": recall,
        "action_micro_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "status_accuracy": exact["status"] / max(1, total),
        "category_accuracy": exact["category"] / max(1, total),
        "urgency_accuracy": exact["urgency"] / max(1, total),
        "direction_exact_match": exact["directions"] / max(1, total),
        "change_accuracy": exact["change"] / max(1, total),
        "warmup_seconds": warmup_seconds,
        "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "latency_p95_ms": sorted(latencies)[round(0.95 * (len(latencies) - 1))]
        if latencies
        else 0.0,
        "failure_examples": failures,
        "metric_scope": "PSEUDO_LABEL_TEACHER_AGREEMENT_NOT_HUMAN_GOLD_ACCURACY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ModernBERT service boundary")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows = select_rows(load_rows(args.dataset), args.limit, args.stratified)
    report = evaluate(
        ModernBertCommandService(str(args.model), device=args.device),
        rows,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
