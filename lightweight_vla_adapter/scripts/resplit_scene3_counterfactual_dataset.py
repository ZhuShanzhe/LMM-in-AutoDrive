"""Atomically apply the builder's group-stratified split to an existing dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.scripts.build_scene3_counterfactual_dataset import (
    assign_grouped_stratified_splits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    manifest = dataset / "manifest.jsonl"
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    with manifest.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    split_report = assign_grouped_stratified_splits(rows, spec)
    backup = dataset / "manifest.pre_stratified.jsonl"
    if not backup.exists():
        shutil.copy2(manifest, backup)
    temporary = dataset / "manifest.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(manifest)
    report_path = dataset / "dataset_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["splits"] = dict(Counter(row["split"] for row in rows))
    report["split_strategy"] = split_report
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(split_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
