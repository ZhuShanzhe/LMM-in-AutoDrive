"""Reapply the current deterministic adapter to existing auditable VLM results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scene_understanding.core.normalize_scene_output import normalize_scene_output
from scene_understanding.core.validate_scene_output import parse_json_text, validate_scene_output


def reprocess_record(record: dict[str, Any]) -> dict[str, Any]:
    """Reprocess one inference record without rerunning the model."""

    raw_parsed = record.get("raw_parsed_output")
    if raw_parsed is None and isinstance(record.get("raw_output"), str):
        raw_parsed = parse_json_text(record["raw_output"])

    updated = dict(record)
    previous_status = record.get("status")
    updated["previous_status"] = previous_status
    if not isinstance(raw_parsed, dict):
        updated["status"] = "invalid"
        updated["parsed_output"] = None
        updated["normalization_actions"] = []
        updated["validation_errors"] = ["raw parsed output is unavailable"]
        return updated

    raw_errors = validate_scene_output(
        raw_parsed,
        expected_frame_id=record.get("frame_id"),
        expected_source=record.get("source"),
        expected_camera_name=record.get("camera_name"),
    )
    processed_size = record.get("processed_image_size") or {}
    parsed_output, actions = normalize_scene_output(
        raw_parsed,
        processed_width=processed_size.get("width"),
        processed_height=processed_size.get("height"),
    )
    errors = validate_scene_output(
        parsed_output,
        expected_frame_id=record.get("frame_id"),
        expected_source=record.get("source"),
        expected_camera_name=record.get("camera_name"),
    )
    updated["raw_validation_errors"] = raw_errors
    updated["parsed_output"] = parsed_output
    updated["normalization_actions"] = actions
    updated["validation_errors"] = errors
    updated["status"] = "valid" if not errors else "invalid"
    return updated


def reprocess_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Write reprocessed JSONL and return total and valid record counts."""

    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    valid = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "x", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON: {exc}") from exc
            updated = reprocess_record(record)
            destination.write(json.dumps(updated, ensure_ascii=False) + "\n")
            total += 1
            valid += int(updated["status"] == "valid")
    return total, valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    total, valid = reprocess_file(args.input, args.output)
    print(f"Reprocessed {total} frames; schema-valid outputs: {valid}/{total}")
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
