"""Merge prepared VLA datasets without copying their large sensor assets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PATH_FIELDS = ("tensor_path", "intent_tensor_path", "image_tensor_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (root / path).resolve())


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    seen: set[str] = set()
    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as writer:
        for dataset_index, dataset in enumerate(args.dataset):
            root = dataset.resolve()
            with (root / "manifest.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    original_id = str(row.get("sample_id", counters["samples"]))
                    sample_id = f"d{dataset_index}_{original_id}"
                    if sample_id in seen:
                        raise ValueError(f"duplicate sample id: {sample_id}")
                    seen.add(sample_id)
                    row["sample_id"] = sample_id
                    for field in PATH_FIELDS:
                        if row.get(field):
                            row[field] = resolve_path(root, str(row[field]))
                    if row.get("image_paths"):
                        row["image_paths"] = [
                            resolve_path(root, str(path)) for path in row["image_paths"]
                        ]
                    row.setdefault("camera_view_mask", [True, True, True, True])
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counters["samples"] += 1
                    counters[f"dataset:{dataset_index}"] += 1
                    counters[f"split:{row.get('split', 'implicit')}"] += 1
                    counters[f"action:{row['label']['action']}"] += 1
                    counters[f"risk:{row['risk_level']}"] += 1
    inventory = {
        "schema_version": "universal_vla_merged_dataset/1.0",
        "datasets": [str(path) for path in args.dataset],
        "counts": dict(sorted(counters.items())),
        "assets_copied": False,
    }
    (output / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
