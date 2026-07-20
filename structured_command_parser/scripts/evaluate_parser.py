from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from structured_command_parser import HybridCommandParser
from structured_command_parser.src.schema_tools import schema_errors, semantic_errors


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "golden_commands.jsonl"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
    return samples


def summarize_result(document: dict[str, Any]) -> dict[str, Any]:
    steps = document["intent"]["steps"]
    return {
        "status": document["parse_result"]["status"],
        "category": document["intent"]["category"],
        "actions": [step["action"] for step in steps],
        "directions": [
            step["parameters"]["direction"]
            for step in steps
            if "direction" in step["parameters"]
        ],
        "target_speed_mps": [
            round(step["parameters"]["target_speed_mps"], 3)
            for step in steps
            if "target_speed_mps" in step["parameters"]
        ],
        "warnings": document["parse_result"]["warnings"],
    }


def matches_expected(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if key == "actions_unordered":
            if sorted(actual.get("actions", [])) != sorted(value):
                return False
        elif key == "target_speed_mps":
            actual_values = actual.get(key, [])
            if len(actual_values) != len(value) or any(
                abs(actual_value - expected_value) > 0.05
                for actual_value, expected_value in zip(
                    actual_values, value, strict=True
                )
            ):
                return False
        elif actual.get(key, []) != value:
            return False
    return True


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return ordered[index]


def main() -> int:
    argument_parser = argparse.ArgumentParser(description="Evaluate command parsing")
    argument_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    argument_parser.add_argument("--model", help="Local Qwen model directory")
    argument_parser.add_argument("--report", type=Path, help="Write per-sample JSONL results")
    argument_parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate LLM samples too; requires --model",
    )
    args = argument_parser.parse_args()
    if args.all and not args.model:
        argument_parser.error("--all requires --model")

    samples = load_jsonl(args.dataset)
    if not args.all:
        samples = [sample for sample in samples if sample["parser_path"] == "RULE"]

    parser = HybridCommandParser(model_path=args.model)
    valid_count = 0
    exact_count = 0
    latencies: list[float] = []
    failures: list[str] = []
    report_rows: list[dict[str, Any]] = []
    for sample in samples:
        document = parser.parse(sample["text"], request_id=sample["sample_id"])
        errors = schema_errors(document) + semantic_errors(document)
        if not errors:
            valid_count += 1
        else:
            failures.append(f"{sample['sample_id']}: {'; '.join(errors)}")
        actual = summarize_result(document)
        report_rows.append(
            {
                "sample_id": sample["sample_id"],
                "text": sample["text"],
                "expected": sample["expected"],
                "actual": actual,
                "document": document,
            }
        )
        if matches_expected(actual, sample["expected"]):
            exact_count += 1
        else:
            failures.append(
                f"{sample['sample_id']}: expected={sample['expected']} actual={actual}"
            )
        latencies.append(document["parse_result"]["latency_ms"])

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
