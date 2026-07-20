from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from typing import Any

from structured_command_parser import ChineseEnglishCommandPipeline
from structured_command_parser.scripts.evaluate_parser import (
    load_jsonl,
    matches_expected,
    percentile,
    summarize_result,
)
from structured_command_parser.src.schema_tools import schema_errors, semantic_errors


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "golden_commands.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Chinese-English pipeline")
    parser.add_argument("--translator-model", required=True)
    parser.add_argument("--parser-model", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    pipeline = ChineseEnglishCommandPipeline(
        args.translator_model, args.parser_model
    )

    valid_count = 0
    exact_count = 0
    term_pass_count = 0
    translation_latencies: list[float] = []
    parsing_latencies: list[float] = []
    total_latencies: list[float] = []
    failures: list[str] = []
    report_rows: list[dict[str, Any]] = []
    slice_totals: Counter[str] = Counter()
    slice_matches: Counter[str] = Counter()
    for sample in samples:
        result = pipeline.parse(
            sample["text"], modality="VOICE", request_id=sample["sample_id"]
        )
        intent = result["driving_intent"]
        errors = schema_errors(intent) + semantic_errors(intent)
        if not errors:
            valid_count += 1
        else:
            failures.append(f"{sample['sample_id']}: {'; '.join(errors)}")
        actual = summarize_result(intent)
        exact = matches_expected(actual, sample["expected"])
        exact_count += int(exact)
        slice_name = sample.get("metadata", {}).get("slice", "unspecified")
        slice_totals[slice_name] += 1
        slice_matches[slice_name] += int(exact)
        term_pass = result["translation"]["term_constraints_passed"]
        term_pass_count += int(term_pass)
        translation_latencies.append(result["translation"]["latency_ms"])
        parsing_latencies.append(intent["parse_result"]["latency_ms"])
        total_latencies.append(result["total_latency_ms"])
        if not exact:
            failures.append(
                f"{sample['sample_id']}: expected={sample['expected']} actual={actual}"
            )
        report_rows.append(
            {
                "sample_id": sample["sample_id"],
                "source_text": sample["text"],
                "translated_text": result["translation"]["translated_text"],
                "term_constraints_passed": term_pass,
                "expected": sample["expected"],
                "actual": actual,
                "exact_match": exact,
                "pipeline_result": result,
            }
        )

    total = len(samples)
    print(f"samples: {total}")
    print(f"json_valid_rate: {valid_count / total:.2%}")
    print(f"intent_exact_match_rate: {exact_count / total:.2%}")
    print(f"term_constraint_pass_rate: {term_pass_count / total:.2%}")
    print(f"translation_latency_mean_ms: {statistics.fmean(translation_latencies):.3f}")
    print(f"translation_latency_p95_ms: {percentile(translation_latencies, 0.95):.3f}")
    print(f"parsing_latency_mean_ms: {statistics.fmean(parsing_latencies):.3f}")
    print(f"parsing_latency_p95_ms: {percentile(parsing_latencies, 0.95):.3f}")
    print(f"total_latency_mean_ms: {statistics.fmean(total_latencies):.3f}")
    print(f"total_latency_p95_ms: {percentile(total_latencies, 0.95):.3f}")
    print("slice_contract_match_rates:")
    for slice_name in sorted(slice_totals):
        matched = slice_matches[slice_name]
        slice_total = slice_totals[slice_name]
        print(f"  {slice_name}: {matched}/{slice_total} ({matched / slice_total:.2%})")
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
