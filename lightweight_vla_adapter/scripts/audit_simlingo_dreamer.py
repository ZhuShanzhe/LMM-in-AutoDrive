from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SimLingo Dreamer labels")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    archives = sorted(Path(args.input_dir).glob("dreamer_*.tar.gz"))
    if not archives:
        raise FileNotFoundError("no dreamer_*.tar.gz archives found")
    mode_counts: Counter[str] = Counter()
    safety_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    instruction_count = 0
    frame_count = 0
    invalid_count = 0
    for archive_path in archives:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json.gz"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    invalid_count += 1
                    continue
                try:
                    payload = json.loads(gzip.decompress(extracted.read()))
                except (OSError, json.JSONDecodeError):
                    invalid_count += 1
                    continue
                if not isinstance(payload, dict):
                    invalid_count += 1
                    continue
                frame_count += 1
                parts = member.name.split("/")
                if len(parts) >= 4:
                    scenario_counts[parts[-4]] += 1
                for mode, alternatives in payload.items():
                    if not isinstance(alternatives, list):
                        invalid_count += 1
                        continue
                    mode_counts[str(mode)] += len(alternatives)
                    for alternative in alternatives:
                        if not isinstance(alternative, dict):
                            invalid_count += 1
                            continue
                        safety_counts[str(alternative.get("safe_to_execute"))] += 1
                        instructions = alternative.get("dreamer_instruction") or []
                        instruction_count += sum(
                            isinstance(value, str) and bool(value.strip())
                            for value in instructions
                        )
    result = {
        "schema_version": "1.0.0",
        "archives": len(archives),
        "frames": frame_count,
        "alternatives": sum(mode_counts.values()),
        "instructions": instruction_count,
        "invalid_records": invalid_count,
        "mode_counts": dict(mode_counts.most_common()),
        "safety_counts": dict(safety_counts.most_common()),
        "scenario_counts": dict(scenario_counts.most_common()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
