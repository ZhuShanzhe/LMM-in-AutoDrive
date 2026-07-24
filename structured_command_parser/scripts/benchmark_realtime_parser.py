from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from structured_command_parser.scripts.evaluate_parser import (
    matches_expected,
    summarize_result,
)
from structured_command_parser.src.pipeline import ChineseEnglishCommandPipeline


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = MODULE_ROOT / "tests" / "fixtures" / "golden_commands.jsonl"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    argument_parser.add_argument("--repeat", type=int, default=100)
    argument_parser.add_argument("--budget-ms", type=float, default=50.0)
    argument_parser.add_argument("--semantic-model")
    argument_parser.add_argument("--semantic-threshold", type=float, default=0.58)
    argument_parser.add_argument("--semantic-top-k", type=int, default=7)
    argument_parser.add_argument(
        "--semantic-device", choices=["cpu", "cuda", "auto"], default="cpu"
    )
    argument_parser.add_argument("--semantic-cpu-threads", type=int, default=1)
    argument_parser.add_argument("--min-match-rate", type=float, default=1.0)
    argument_parser.add_argument("--show-failures", type=int, default=0)
    args = argument_parser.parse_args()

    samples = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pipeline = ChineseEnglishCommandPipeline(
        "unused-model",
        "unused-model",
        allow_llm_fallback=False,
        semantic_model_path=args.semantic_model,
        semantic_similarity_threshold=args.semantic_threshold,
        semantic_top_k=args.semantic_top_k,
        semantic_device=args.semantic_device,
        semantic_cpu_threads=args.semantic_cpu_threads,
    )
    pipeline.warmup()
    latencies: list[float] = []
    correct = 0
    total = 0
    paths: dict[str, int] = {}
    failures: list[dict[str, object]] = []
    for _ in range(args.repeat):
        for sample in samples:
            started = perf_counter()
            result = pipeline.parse(sample["text"], request_id=sample["sample_id"])
            latency_ms = (perf_counter() - started) * 1000
            latencies.append(latency_ms)
            total += 1
            actual = summarize_result(result["driving_intent"])
            matched = matches_expected(actual, sample["expected"])
            correct += int(matched)
            if not matched and len(failures) < args.show_failures:
                failures.append(
                    {
                        "sample_id": sample["sample_id"],
                        "text": sample["text"],
                        "expected": sample["expected"],
                        "actual": actual,
                        "execution_path": result["execution_path"],
                        "confidence": result["driving_intent"]["parse_result"][
                            "confidence"
                        ],
                    }
                )
            path = result["execution_path"]
            paths[path] = paths.get(path, 0) + 1

    over_budget = sum(value > args.budget_ms for value in latencies)
    print(f"samples_per_round: {len(samples)}")
    print(f"repeat: {args.repeat}")
    print(f"requests: {total}")
    print(f"contract_match_rate: {correct / total:.2%}")
    print(f"mean_latency_ms: {mean(latencies):.3f}")
    print(f"p95_latency_ms: {percentile(latencies, 0.95):.3f}")
    print(f"p99_latency_ms: {percentile(latencies, 0.99):.3f}")
    print(f"max_latency_ms: {max(latencies):.3f}")
    print(f"over_budget: {over_budget}/{total}")
    print(f"execution_paths: {json.dumps(paths, ensure_ascii=False, sort_keys=True)}")
    for failure in failures:
        print("failure: " + json.dumps(failure, ensure_ascii=False, sort_keys=True))
    match_rate = correct / total
    raise SystemExit(1 if match_rate < args.min_match_rate or over_budget else 0)


if __name__ == "__main__":
    main()
