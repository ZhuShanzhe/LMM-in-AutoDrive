"""Adapt an ASR-plus-translation result into the parser ingress contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from structured_command_parser.scripts.run_english_parser_service import (
    write_json_atomic,
)


def _non_empty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ASR result {field} must be a non-empty string")
    return " ".join(value.split())


def _latency_ms(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"ASR result {field} must be a non-negative number")
    return round(float(value) * 1000.0, 3)


def build_ingress_message(
    asr_result: Mapping[str, Any], *, request_id: str | None = None
) -> dict[str, Any]:
    """Return the English parser message while retaining voice provenance."""

    resolved_request_id = request_id or asr_result.get("request_id")
    if not isinstance(resolved_request_id, str) or not resolved_request_id.strip():
        raise ValueError("request_id is required for an ASR result")

    message = {
        "request_id": resolved_request_id.strip(),
        "text": _non_empty_text(asr_result.get("english_translation"), "english_translation"),
        "language": "en-US",
        "modality": "VOICE",
        "source_text": _non_empty_text(asr_result.get("chinese_text"), "chinese_text"),
        "source_language": "zh-CN",
    }
    for source_field, output_field in (
        ("asr_processing_time_seconds", "asr_latency_ms"),
        ("translation_time_seconds", "translation_latency_ms"),
        ("total_time_seconds", "voice_pipeline_latency_ms"),
    ):
        latency = _latency_ms(asr_result.get(source_field), source_field)
        if latency is not None:
            message[output_field] = latency
    return message


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        message = build_ingress_message(
            _read_json(args.asr_result), request_id=args.request_id
        )
        write_json_atomic(args.output, message)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}")
        return 1
    print(f"request_id={message['request_id']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
