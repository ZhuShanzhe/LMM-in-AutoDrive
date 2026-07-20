from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = MODULE_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
DEFAULT_SIMLINGO = PROCESSED_ROOT / "simlingo_candidates_zh.jsonl"
DEFAULT_TALK2CAR = PROCESSED_ROOT / "talk2car_review_queue_zh.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rows(rows: list[dict[str, Any]], source: str) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_english: set[str] = set()
    for index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        if not sample_id or sample_id in seen_ids:
            errors.append(f"{source}:{index}: missing or duplicate sample_id")
        seen_ids.add(sample_id)
        english = str(row.get("text_en") or "").strip()
        chinese = str(row.get("text_zh") or "").strip()
        if not english:
            errors.append(f"{sample_id}: empty text_en")
        if not chinese:
            errors.append(f"{sample_id}: empty text_zh")
        elif not re.search(r"[\u4e00-\u9fff]", chinese):
            errors.append(f"{sample_id}: text_zh contains no Chinese characters")
        if english.casefold() in seen_english:
            errors.append(f"{sample_id}: duplicate text_en")
        seen_english.add(english.casefold())
        if row.get("review_status") != "REQUIRES_HUMAN_REVIEW":
            errors.append(f"{sample_id}: machine data must require human review")
    return errors


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "source",
        "mode_or_object",
        "text_en",
        "text_zh",
        "expected_json",
        "review_result",
        "reviewer",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metadata = row.get("metadata", {})
            expected = row.get("expected") or row.get("proposed_expected")
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "source": row["source"],
                    "mode_or_object": metadata.get("mode") or metadata.get("object_type"),
                    "text_en": row["text_en"],
                    "text_zh": row["text_zh"],
                    "expected_json": json.dumps(expected, ensure_ascii=False),
                    "review_result": "PENDING",
                    "reviewer": "",
                    "notes": "",
                }
            )


def print_examples(rows: list[dict[str, Any]], key: str) -> None:
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row.get("metadata", {}).get(key, "UNKNOWN"))
        if len(examples[group]) < 2:
            examples[group].append(row)
    for group, group_rows in sorted(examples.items()):
        print(f"[{group}]")
        for row in group_rows:
            print(f"  EN: {row['text_en']}")
            print(f"  ZH: {row['text_zh']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate translated command data")
    parser.add_argument("--simlingo", type=Path, default=DEFAULT_SIMLINGO)
    parser.add_argument("--talk2car", type=Path, default=DEFAULT_TALK2CAR)
    parser.add_argument("--output", type=Path, default=PROCESSED_ROOT)
    args = parser.parse_args()

    simlingo = load_jsonl(args.simlingo)
    talk2car = load_jsonl(args.talk2car)
    errors = validate_rows(simlingo, "simlingo") + validate_rows(talk2car, "talk2car")
    simlingo_modes = Counter(row["metadata"]["mode"] for row in simlingo)
    talk2car_objects = Counter(row["metadata"]["object_type"] for row in talk2car)
    translation_models = Counter(
        row.get("translation_status", "UNKNOWN") for row in simlingo + talk2car
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_review_csv(args.output / "simlingo_review.csv", simlingo)
    write_review_csv(args.output / "talk2car_review.csv", talk2car)
    manifest = {
        "schema": "external-command-candidates-v1",
        "created_on": "2026-07-20",
        "status": "REQUIRES_HUMAN_REVIEW",
        "simlingo": {
            "count": len(simlingo),
            "mode_counts": dict(sorted(simlingo_modes.items())),
            "sha256": sha256(args.simlingo),
            "label_quality": "SOURCE_MODE_MAPPED",
        },
        "talk2car": {
            "count": len(talk2car),
            "object_counts": dict(talk2car_objects.most_common()),
            "sha256": sha256(args.talk2car),
            "label_quality": "HEURISTIC_PROPOSAL_ONLY",
        },
        "translation": {
            "statuses": dict(translation_models),
            "quality": "MACHINE_TRANSLATION_REQUIRES_HUMAN_REVIEW",
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nSimLingo translation examples:")
    print_examples(simlingo, "mode")
    print("\nTalk2Car translation examples:")
    print_examples(talk2car[:10], "object_type")
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
