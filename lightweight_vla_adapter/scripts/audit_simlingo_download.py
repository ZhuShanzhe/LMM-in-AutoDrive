from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile


DEFAULT_REPO_ID = "RenzKa/simlingo"
DEFAULT_TARGET = Path("/root/autodl-tmp/datasets/vla_student/simlingo_hf")
PATTERNS = (
    "README.md",
    "LICENSE",
    "buckets_paths.pkl",
    "data_*.tar.gz",
    "dreamer_*.tar.gz",
    "commentary_*.tar.gz",
    "drivelm_*.tar.gz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the selected official SimLingo files against Hub metadata."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def is_selected(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in PATTERNS)


def main() -> None:
    args = parse_args()
    entries = HfApi().list_repo_tree(
        args.repo_id,
        repo_type="dataset",
        recursive=True,
        expand=True,
    )
    expected = {
        item.path: int(item.size or 0)
        for item in entries
        if isinstance(item, RepoFile) and is_selected(item.path)
    }

    complete = {}
    missing = []
    size_mismatches = []
    for path, expected_size in sorted(expected.items()):
        local_path = args.target / path
        if not local_path.is_file():
            missing.append(path)
            continue
        actual_size = local_path.stat().st_size
        complete[path] = actual_size
        if expected_size and actual_size != expected_size:
            size_mismatches.append(
                {
                    "path": path,
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                }
            )

    partial = {
        item.name: item.stat().st_size
        for item in (
            args.target / ".cache" / "huggingface" / "download"
        ).glob("*.incomplete")
    }
    result = {
        "repo_id": args.repo_id,
        "expected_files": len(expected),
        "expected_bytes": sum(expected.values()),
        "complete_files": len(complete),
        "complete_bytes": sum(complete.values()),
        "partial_files": len(partial),
        "partial_bytes": sum(partial.values()),
        "missing_files": len(missing),
        "size_mismatches": size_mismatches,
        "missing_paths": missing,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")

    is_complete = (
        len(expected) == 182
        and not missing
        and not partial
        and not size_mismatches
    )
    if args.require_complete and not is_complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
