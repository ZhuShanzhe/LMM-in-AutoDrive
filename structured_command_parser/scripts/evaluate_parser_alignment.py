from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from scene_understanding.src.driving_intent_alignment import (
    align_driving_intent,
)
from structured_command_parser.src.modernbert_parser import (
    ModernBertEnglishIntentParser,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORLD_STATE = (
    ROOT
    / "scene_understanding"
    / "schemas"
    / "examples"
    / "world_state.example.json"
)


def _red_truck(base: dict[str, Any], object_id: str) -> dict[str, Any]:
    obj = copy.deepcopy(base)
    obj["object_id"] = object_id
    obj["subtype"] = "vehicle.test.truck"
    obj["semantic_matches"] = [
        {
            "camera_name": "front",
            "visual_object_id": f"visual-{object_id}",
            "bbox_2d": [0.1, 0.1, 0.3, 0.4],
            "description": "vehicle; red; truck; front",
            "confidence": 0.98,
        }
    ]
    return obj


def _run_case(
    parser: ModernBertEnglishIntentParser,
    world_state: dict[str, Any],
    *,
    case_id: str,
    text: str,
    expected_parse_status: str,
    expected_alignment_status: str,
    source_text: str | None = None,
    source_language: str | None = None,
) -> dict[str, Any]:
    document = parser.parse(
        text,
        request_id=case_id,
        source_text=source_text,
        source_language=source_language,
        modality="VOICE" if source_text is not None else "TEXT",
    )
    started = perf_counter()
    alignment = align_driving_intent(document, world_state)
    alignment_latency_ms = (perf_counter() - started) * 1000
    passed = (
        document["parse_result"]["status"] == expected_parse_status
        and alignment["alignment_status"] == expected_alignment_status
    )
    return {
        "case_id": case_id,
        "passed": passed,
        "text": text,
        "parse_status": document["parse_result"]["status"],
        "alignment_status": alignment["alignment_status"],
        "parser_latency_ms": document["parse_result"]["latency_ms"],
        "alignment_latency_ms": round(alignment_latency_ms, 3),
        "intent": document,
        "alignment": alignment,
    }


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model", type=Path, required=True)
    argument_parser.add_argument(
        "--world-state",
        type=Path,
        default=DEFAULT_WORLD_STATE,
    )
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument("--device", default="cuda")
    args = argument_parser.parse_args()

    base_state = json.loads(args.world_state.read_text(encoding="utf-8"))
    base_object = base_state["objects"][0]
    unique_state = copy.deepcopy(base_state)
    unique_state["objects"] = [_red_truck(base_object, "red_truck")]
    ambiguous_state = copy.deepcopy(base_state)
    ambiguous_state["objects"] = [
        _red_truck(base_object, "red_truck_1"),
        _red_truck(base_object, "red_truck_2"),
    ]

    parser = ModernBertEnglishIntentParser(
        str(args.model),
        device=args.device,
    )
    parser.warmup()
    cases = [
        _run_case(
            parser,
            unique_state,
            case_id="unique-red-truck",
            text="Slow down and stop before the red truck.",
            expected_parse_status="VALID",
            expected_alignment_status="COMPLETE",
        ),
        _run_case(
            parser,
            ambiguous_state,
            case_id="ambiguous-red-truck",
            text="Stop before the red truck.",
            expected_parse_status="VALID",
            expected_alignment_status="FAILED",
        ),
        _run_case(
            parser,
            base_state,
            case_id="second-junction",
            text="Turn right after the second junction.",
            expected_parse_status="VALID",
            expected_alignment_status="FAILED",
        ),
        _run_case(
            parser,
            base_state,
            case_id="asr-homophone-clarification",
            text="Turn right at the junction.",
            source_text="\u524d\u65b9\u8def\u53e3\u53c8\u8f6c",
            source_language="zh-CN",
            expected_parse_status="NEEDS_CLARIFICATION",
            expected_alignment_status="SKIPPED",
        ),
        _run_case(
            parser,
            base_state,
            case_id="unresolved-anaphora",
            text="Follow it, but keep a safe distance.",
            expected_parse_status="NEEDS_CLARIFICATION",
            expected_alignment_status="SKIPPED",
        ),
    ]
    parser_latencies = [float(item["parser_latency_ms"]) for item in cases]
    alignment_latencies = [
        float(item["alignment_latency_ms"]) for item in cases
    ]
    report = {
        "schema": "parser-scene-alignment-evaluation-v1",
        "model": str(args.model),
        "passed": sum(item["passed"] for item in cases),
        "total": len(cases),
        "all_passed": all(item["passed"] for item in cases),
        "latency_ms": {
            "parser_mean": mean(parser_latencies),
            "parser_max": max(parser_latencies),
            "alignment_mean": mean(alignment_latencies),
            "alignment_max": max(alignment_latencies),
        },
        "cases": cases,
        "limitations": [
            "World states are deterministic fixtures, not live CARLA frames.",
            "A second junction requires a route-level entity graph that the current WorldState does not expose.",
            "Alignment output is not a vehicle-control command.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("passed", "total", "all_passed", "latency_ms")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
