"""Continuously convert translated English command snapshots to DrivingIntent.

This is an ingress boundary, not an ASR implementation and not a CARLA control
process.  An upstream ASR/translation component atomically writes one English
message, and this service keeps one warmed ModernBERT instance in memory before
atomically replacing the corresponding DrivingIntent document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from structured_command_parser import ModernBertCommandService


ENGLISH_LANGUAGES = {"en", "en-US", "en-GB"}


def read_message(path: Path) -> dict[str, Any]:
    """Read the translator snapshot while accepting UTF-8 BOM files."""

    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if value.get("language") not in ENGLISH_LANGUAGES:
        raise ValueError("message.language must identify English text")
    if value.get("modality", "VOICE") not in {"VOICE", "TEXT"}:
        raise ValueError("message.modality must be VOICE or TEXT")
    if not isinstance(value.get("request_id"), str) or not value["request_id"].strip():
        raise ValueError("message.request_id must be a non-empty string")
    if not isinstance(value.get("text"), str) or not value["text"].strip():
        raise ValueError("message.text must be a non-empty string")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Replace one JSON snapshot without exposing a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def message_signature(message: dict[str, Any]) -> str:
    """Identify an upstream command independently of JSON whitespace/order."""

    canonical = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def process_message(
    service: ModernBertCommandService,
    message: dict[str, Any],
    *,
    output_path: Path,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Parse one validated upstream message and persist its two output records."""

    intent = service.handle_message(message)
    write_json_atomic(output_path, intent)
    receipt = {
        "request_id": intent["request_id"],
        "input_language": message["language"],
        "input_modality": message.get("modality", "VOICE"),
        "parse_status": intent["parse_result"]["status"],
        "parser_latency_ms": intent["parse_result"]["latency_ms"],
        "method": intent["parse_result"]["method"],
    }
    if receipt_path is not None:
        write_json_atomic(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="English translator JSON")
    parser.add_argument("--driving-intent-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument(
        "--model",
        default=os.environ.get("MODERNBERT_MODEL_PATH"),
        help="Fine-tuned model directory; defaults to MODERNBERT_MODEL_PATH",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-input-chars", type=int, default=512)
    parser.add_argument("--poll-interval-ms", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model:
        raise SystemExit("--model or MODERNBERT_MODEL_PATH is required")
    if args.max_input_chars <= 0:
        raise SystemExit("--max-input-chars must be positive")
    if args.poll_interval_ms <= 0:
        raise SystemExit("--poll-interval-ms must be positive")

    service = ModernBertCommandService(
        str(args.model), device=args.device, max_input_chars=args.max_input_chars
    )
    service.warmup()
    previous_signature: str | None = None
    while True:
        try:
            message = read_message(args.input)
            signature = message_signature(message)
            if signature != previous_signature:
                receipt = process_message(
                    service,
                    message,
                    output_path=args.driving_intent_output,
                    receipt_path=args.receipt_output,
                )
                previous_signature = signature
                print(
                    f"request_id={receipt['request_id']} "
                    f"status={receipt['parse_status']} "
                    f"latency_ms={receipt['parser_latency_ms']}"
                )
            if args.once:
                return 0
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            if args.once:
                print(f"ERROR: {error}")
                return 1
        time.sleep(args.poll_interval_ms / 1000.0)


if __name__ == "__main__":
    raise SystemExit(main())
