#!/usr/bin/env python3
"""Audit the downloaded and processed driving-language corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def count_jsonl(path: Path, required: set[str]) -> tuple[int, list[str]]:
    count = 0
    errors: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                continue
            missing = sorted(required - row.keys())
            if missing:
                errors.append(f"{path}:{line_number}: missing {missing}")
    return count, errors


def audit(root: Path) -> dict[str, Any]:
    manifests = root / "manifests"
    download = json.loads((manifests / "simlingo_download.json").read_text(encoding="utf-8"))
    summary = json.loads((manifests / "corpus_summary.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    archive_bytes = 0
    for item in download["files"]:
        path = root / "raw" / "simlingo" / item["directory"] / item["filename"]
        if not path.is_file():
            errors.append(f"missing archive: {path}")
            continue
        actual_size = path.stat().st_size
        archive_bytes += actual_size
        if actual_size != item["size"]:
            errors.append(
                f"archive size mismatch: {path.name}: {actual_size} != {item['size']}"
            )

    jsonl_specs = {
        "talk2car": (
            root / "processed" / "talk2car_all.jsonl",
            summary["talk2car_rows"],
            {"sample_id", "source", "text_en", "review_status"},
        ),
        "dreamer": (
            root / "processed" / "simlingo_dreamer_unique.jsonl",
            summary["dreamer_rows"],
            {"sample_id", "source", "text_en", "review_status"},
        ),
        "commentary": (
            root / "processed" / "simlingo_commentary_unique.jsonl",
            summary["commentary_rows"],
            {"sample_id", "source", "text_en", "use"},
        ),
        "deferred_bilingual_seed": (
            root / "deferred_translation" / "legacy_bilingual_seed_627.jsonl",
            summary["deferred_bilingual_seed_rows"],
            set(),
        ),
    }
    counts: dict[str, int] = {}
    for name, (path, expected, required) in jsonl_specs.items():
        actual, row_errors = count_jsonl(path, required)
        counts[name] = actual
        errors.extend(row_errors[:20])
        if actual != expected:
            errors.append(f"{name} row mismatch: {actual} != {expected}")

    mining_root = root / "knowledge_mining"
    selection = json.loads(
        (mining_root / "manifests" / "selection_summary.json").read_text(
            encoding="utf-8"
        )
    )
    representative_path = mining_root / "representative_samples.jsonl"
    representative_count, representative_errors = count_jsonl(
        representative_path,
        {"sample_id", "source", "mining_scope", "text_en", "weak_hints", "provenance"},
    )
    errors.extend(representative_errors[:20])
    if representative_count != selection["total_samples"]:
        errors.append(
            "representative row mismatch: "
            f"{representative_count} != {selection['total_samples']}"
        )

    batch_count = 0
    batch_ids: set[str] = set()
    duplicate_batch_ids: set[str] = set()
    for filename in selection["batch_files"]:
        path = mining_root / "gpt_inputs" / filename
        actual, batch_errors = count_jsonl(
            path, {"sample_id", "source", "mining_scope", "text_en"}
        )
        batch_count += actual
        errors.extend(batch_errors[:20])
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sample_id = json.loads(line)["sample_id"]
                if sample_id in batch_ids:
                    duplicate_batch_ids.add(sample_id)
                batch_ids.add(sample_id)
    if batch_count != representative_count:
        errors.append(f"GPT batch row mismatch: {batch_count} != {representative_count}")
    if duplicate_batch_ids:
        errors.append(f"duplicate GPT batch sample IDs: {len(duplicate_batch_ids)}")

    result = {
        "schema": "full-driving-language-corpus-audit-v2",
        "status": "PASS" if not errors else "FAIL",
        "archive_files": len(download["files"]),
        "archive_bytes": archive_bytes,
        "jsonl_rows": counts,
        "knowledge_mining": {
            "representative_samples": representative_count,
            "gpt_batches": len(selection["batch_files"]),
            "gpt_batch_rows": batch_count,
        },
        "errors": errors,
    }
    (manifests / "corpus_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "corpus",
    )
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
