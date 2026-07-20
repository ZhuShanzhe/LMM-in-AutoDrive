from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from structured_command_parser.scripts.evaluate_parser import (
    load_jsonl,
    matches_expected,
    percentile,
    summarize_result,
)
from structured_command_parser.src.english_parser import QwenEnglishIntentParser
from structured_command_parser.src.schema_tools import schema_errors, semantic_errors


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "simlingo_candidates_en.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the English intent parser")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    command_parser = QwenEnglishIntentParser(args.model)
    valid_count = 0
    exact_count = 0
    latencies: list[float] = []
    failures: list[str] = []
    report_rows: list[dict[str, Any]] = []
    for sample in samples:
        intent = command_parser.parse(
            sample["text_en"], request_id=sample["sample_id"]
        )
        errors = schema_errors(intent) + semantic_errors(intent)
        valid_count += int(not errors)
        actual = summarize_result(intent)
        exact = matches_expected(actual, sample["expected"])
        exact_count += int(exact)
        latencies.append(intent["parse_result"]["latency_ms"])
        if errors:
            failures.append(f"{sample['sample_id']}: {'; '.join(errors)}")
        if not exact:
            failures.append(
                f"{sample['sample_id']}: expected={sample['expected']} actual={actual}"
            )
        report_rows.append(
            {
                "sample_id": sample["sample_id"],
                "text_en": sample["text_en"],
                "expected": sample["expected"],
                "actual": actual,
                "exact_match": exact,
                "document": intent,
            }
        )

    total = len(samples)
    print(f"samples: {total}")
    print(f"json_valid_rate: {valid_count / total:.2%}")
    print(f"exact_match_rate: {exact_count / total:.2%}")
    print(f"latency_mean_ms: {statistics.fmean(latencies):.3f}")
    print(f"latency_p50_ms: {percentile(latencies, 0.50):.3f}")
    print(f"latency_p95_ms: {percentile(latencies, 0.95):.3f}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as file:
            for row in report_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"report: {args.report}")
    if failures:
        print("failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
