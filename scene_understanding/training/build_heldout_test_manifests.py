"""Build deterministic test manifests disjoint from the specialized YOLO validation set."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_heldout(
    source: Path,
    excluded_images: set[str],
    *,
    limit: int,
    seed: int,
) -> list[dict]:
    candidates = [
        record
        for record in read_jsonl(source)
        if str(Path(record["image_path"]).resolve()) not in excluded_images
    ]
    if len(candidates) < limit:
        raise ValueError(
            f"{source} has only {len(candidates)} held-out records; need {limit}"
        )
    selected = random.Random(seed).sample(candidates, limit)
    selected.sort(key=lambda item: str(item.get("frame_id") or item.get("image_id")))
    return selected


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--bdd-source", type=Path, required=True)
    parser.add_argument("--nuscenes-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    validation = read_jsonl(args.validation_audit)
    excluded = {
        str(Path(record["image_path"]).resolve()) for record in validation
    }
    bdd = select_heldout(
        args.bdd_source, excluded, limit=args.limit, seed=args.seed
    )
    nuscenes = select_heldout(
        args.nuscenes_source,
        excluded,
        limit=args.limit,
        seed=args.seed + 1,
    )
    bdd_path = args.output / "bdd100k_test.jsonl"
    nuscenes_path = args.output / "nuscenes_cam_front_test.jsonl"
    write_jsonl(bdd_path, bdd)
    write_jsonl(nuscenes_path, nuscenes)
    selected_paths = {
        str(Path(record["image_path"]).resolve()) for record in [*bdd, *nuscenes]
    }
    overlap = selected_paths & excluded
    inventory = {
        "seed": args.seed,
        "selection": "uniform random without replacement after image-level exclusion",
        "excluded_validation_images": len(excluded),
        "bdd100k_test_records": len(bdd),
        "nuscenes_test_records": len(nuscenes),
        "validation_test_image_overlap": len(overlap),
        "manifests": {
            "bdd100k": str(bdd_path),
            "nuscenes_cam_front": str(nuscenes_path),
        },
    }
    (args.output / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
