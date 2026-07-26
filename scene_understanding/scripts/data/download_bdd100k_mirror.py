#!/usr/bin/env python3
"""Download the task-relevant BDD100K mirror as portable Parquet shards."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

EXPECTED_SHARDS = 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/scene_understanding/bdd100k/source_parquet"
        ),
    )
    parser.add_argument(
        "--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_ENDPOINT"] = args.endpoint
    result = snapshot_download(
        repo_id="lance-format/BDD100K-enriched",
        repo_type="dataset",
        revision="refs/convert/parquet",
        allow_patterns=["default/train/*.parquet"],
        local_dir=args.output,
        endpoint=args.endpoint,
    )
    shards = sorted(args.output.glob("default/train/*.parquet"))
    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"incomplete BDD100K mirror: found {len(shards)}/{EXPECTED_SHARDS} "
            f"Parquet shards under {args.output / 'default/train'}"
        )
    print(f"{result}: verified {len(shards)} Parquet shards")


if __name__ == "__main__":
    main()
