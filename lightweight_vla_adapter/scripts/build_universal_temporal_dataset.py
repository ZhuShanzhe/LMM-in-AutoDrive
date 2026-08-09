#!/usr/bin/env python3
"""Build leakage-safe temporal features for the universal three-scene VLA."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.scripts.train_scene3_multimodal import (
    Scene3Dataset,
    model_kwargs,
)


PATH_FIELDS = ("tensor_path", "intent_tensor_path", "image_tensor_path")
CANONICAL_VARIANT_PRIORITY = {
    "observed_command": 0,
    "environment_pair_observed": 1,
    "scene2_baseline": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge explicit-split manifests and precompute four historical "
            "sensor features for temporal risk training."
        )
    )
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument(
        "--extra-dataset", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--temporal-block-size", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_manifest(root: Path) -> list[dict]:
    manifest = root / "manifest.jsonl"
    with manifest.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relative_asset_path(value: str, source_root: Path, output_root: Path) -> str:
    path = Path(value)
    absolute = path if path.is_absolute() else source_root / path
    return os.path.relpath(absolute.resolve(), output_root.resolve())


def rebase_row(row: dict, source_root: Path, output_root: Path) -> dict:
    rebased = copy.deepcopy(row)
    for field in PATH_FIELDS:
        value = rebased.get(field)
        if value:
            rebased[field] = relative_asset_path(
                str(value), source_root, output_root
            )
    if rebased.get("image_paths"):
        rebased["image_paths"] = [
            relative_asset_path(str(value), source_root, output_root)
            for value in rebased["image_paths"]
        ]
    return rebased


def merge_rows(dataset_roots: list[Path], output_root: Path) -> list[dict]:
    merged: dict[str, dict] = {}
    for root in dataset_roots:
        for row in read_manifest(root):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"sample without sample_id in {root}")
            merged[sample_id] = rebase_row(row, root, output_root)
    rows = list(merged.values())
    if not rows:
        raise ValueError("no samples were merged")
    return rows


def repair_cross_split_groups(rows: list[dict], block_size: int) -> dict:
    """Atomically re-split leaked CARLA trajectories into time blocks."""

    if block_size < 8:
        raise ValueError("--temporal-block-size must be at least 8")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        source = str(row.get("source_dataset", "CARLA"))
        group = str(
            row.get("split_group")
            or row.get("trajectory_id")
            or row.get("counterfactual_set_id")
            or row["sample_id"]
        )
        grouped[(source, group)].append(row)

    repaired = 0
    repaired_rows = 0
    for (source, group), members in grouped.items():
        splits = {str(row["split"]) for row in members}
        if len(splits) <= 1:
            continue
        frames = sorted({frame_number(row) for row in members})
        blocks = [
            frames[index : index + block_size]
            for index in range(0, len(frames), block_size)
        ]
        if len(blocks) < 3:
            # Tiny event groups cannot support three independent temporal
            # partitions. Keep the whole atomic group in training; validation
            # and test remain supplied by longer trajectories.
            for row in members:
                row["split"] = "train"
                row["split_group"] = f"{group}:temporal_block_000"
            repaired += 1
            repaired_rows += len(members)
            continue
        seed = int.from_bytes(
            hashlib.sha256(f"{source}|{group}".encode()).digest()[:4],
            "big",
        )
        validation_index = seed % len(blocks)
        test_index = (validation_index + max(1, len(blocks) // 2)) % len(blocks)
        if test_index == validation_index:
            test_index = (validation_index + 1) % len(blocks)
        frame_to_block = {
            frame: block_index
            for block_index, block in enumerate(blocks)
            for frame in block
        }
        for row in members:
            block_index = frame_to_block[frame_number(row)]
            split = "train"
            if block_index == validation_index:
                split = "validation"
            elif block_index == test_index:
                split = "test"
            row["split"] = split
            row["split_group"] = f"{group}:temporal_block_{block_index:03d}"
        repaired += 1
        repaired_rows += len(members)
    return {
        "repaired_cross_split_groups": repaired,
        "repaired_rows": repaired_rows,
        "temporal_block_size": block_size,
    }


def validate_splits(rows: list[dict]) -> dict[str, object]:
    split_counts = Counter(str(row.get("split", "")) for row in rows)
    invalid = set(split_counts) - {"train", "validation", "test"}
    if invalid:
        raise ValueError(f"all rows need explicit splits; invalid={invalid}")
    if any(split_counts[name] == 0 for name in ("train", "validation", "test")):
        raise ValueError(f"train/validation/test must be non-empty: {split_counts}")

    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        source = str(row.get("source_dataset", "CARLA"))
        group = str(
            row.get("split_group")
            or row.get("trajectory_id")
            or row.get("counterfactual_set_id")
            or row["sample_id"]
        )
        group_splits[(source, group)].add(str(row["split"]))
    leaked = {
        f"{source}|{group}": sorted(splits)
        for (source, group), splits in group_splits.items()
        if len(splits) > 1
    }
    if leaked:
        preview = dict(list(leaked.items())[:20])
        raise ValueError(f"episode groups cross data splits: {preview}")
    return {
        "split_counts": dict(split_counts),
        "split_risk_counts": {
            split: dict(
                Counter(
                    row["risk_level"]
                    for row in rows
                    if row["split"] == split
                )
            )
            for split in ("train", "validation", "test")
        },
        "split_action_counts": {
            split: dict(
                Counter(
                    row["label"]["action"]
                    for row in rows
                    if row["split"] == split
                )
            )
            for split in ("train", "validation", "test")
        },
        "episode_groups": len(group_splits),
    }


def sequence_key(row: dict) -> tuple[str, str, str]:
    if row.get("source_frame") is None:
        # Public-image samples are intentionally isolated. Repeating their own
        # feature supplies useful static negatives without inventing temporal
        # adjacency between unrelated images.
        return (
            str(row["split"]),
            str(row.get("source_dataset", "static")),
            f"static:{row['sample_id']}",
        )
    return (
        str(row["split"]),
        str(row.get("source_dataset", "CARLA")),
        str(
            row.get("split_group")
            or row.get("trajectory_id")
            or row.get("counterfactual_set_id")
            or row["sample_id"]
        ),
    )


def frame_number(row: dict) -> int:
    value = row.get("source_frame")
    if value is None:
        return 0
    return int(value)


def canonical_rows(
    rows: list[dict],
) -> tuple[list[dict], dict[tuple[tuple[str, str, str], int], int], dict]:
    candidates: dict[tuple[tuple[str, str, str], int], list[dict]] = defaultdict(list)
    for row in rows:
        candidates[(sequence_key(row), frame_number(row))].append(row)

    canonical = []
    canonical_index = {}
    frames_by_sequence: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for key in sorted(candidates, key=lambda item: (item[0], item[1])):
        choices = candidates[key]
        selected = min(
            choices,
            key=lambda row: (
                CANONICAL_VARIANT_PRIORITY.get(
                    str(row.get("variant_type", "")), 10
                ),
                str(row["sample_id"]),
            ),
        )
        canonical_index[key] = len(canonical)
        canonical.append(selected)
        frames_by_sequence[key[0]].append(key[1])
    for sequence in frames_by_sequence:
        frames_by_sequence[sequence].sort()
    return canonical, canonical_index, frames_by_sequence


def load_compatible_checkpoint(model: torch.nn.Module, checkpoint: Path) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return {
        "compatible_parameters": len(compatible),
        "initialized_parameters": len(current),
        "skipped_checkpoint_parameters": len(state) - len(compatible),
    }


def extract_features(
    rows: list[dict],
    *,
    dataset_root: Path,
    config: dict,
    checkpoint: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    model = build_model(config)
    initialization = load_compatible_checkpoint(model, checkpoint)
    model.to(device).eval()
    loader = DataLoader(
        Scene3Dataset(
            dataset_root,
            rows,
            augment=False,
            intent_max_length=int(config.get("intent_max_length", 32)),
            bev_input_size=tuple(config.get("bev_input_size", (64, 64))),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    features = []
    with torch.inference_mode():
        for batch in loader:
            output = model(**model_kwargs(batch, device, config))
            if output.risk_input_features is None:
                raise RuntimeError("config must enable use_temporal_risk")
            features.append(output.risk_input_features.detach().cpu().half())
    return torch.cat(features, dim=0), initialization


def make_histories(
    rows: list[dict],
    canonical_features: torch.Tensor,
    canonical_index: dict,
    frames_by_sequence: dict,
    history_size: int,
) -> torch.Tensor:
    feature_size = int(canonical_features.shape[-1])
    histories = torch.empty(
        (len(rows), history_size, feature_size), dtype=torch.float16
    )
    for row_index, row in enumerate(rows):
        sequence = sequence_key(row)
        current_frame = frame_number(row)
        frames = frames_by_sequence[sequence]
        position = bisect_left(frames, current_frame)
        selected = frames[max(0, position - history_size) : position]
        if not selected:
            selected = [frames[min(position, len(frames) - 1)]]
        while len(selected) < history_size:
            selected.insert(0, selected[0])
        feature_indices = [
            canonical_index[(sequence, frame)] for frame in selected
        ]
        histories[row_index] = canonical_features[feature_indices]
        row["temporal_history_index"] = row_index
    return histories


def main() -> None:
    args = parse_args()
    if args.history_size < 1:
        raise ValueError("--history-size must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output.iterdir()]
    if existing:
        raise FileExistsError(
            f"output directory must be empty: {output} contains {len(existing)} items"
        )

    roots = [args.base_dataset.resolve()] + [
        path.resolve() for path in args.extra_dataset
    ]
    rows = merge_rows(roots, output)
    repair_report = repair_cross_split_groups(
        rows, args.temporal_block_size
    )
    split_report = validate_splits(rows)
    canonical, canonical_index, frames_by_sequence = canonical_rows(rows)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not bool(config.get("use_temporal_risk", False)):
        raise ValueError("config must set use_temporal_risk=true")
    device = torch.device(args.device)
    features, initialization = extract_features(
        canonical,
        dataset_root=output,
        config=config,
        checkpoint=args.initialize_from,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    histories = make_histories(
        rows,
        features,
        canonical_index,
        frames_by_sequence,
        args.history_size,
    )
    torch.save(histories, output / "temporal_history_features.pt")
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "universal_temporal_dataset/1.0",
        "rows": len(rows),
        "canonical_frames": len(canonical),
        "history_size": args.history_size,
        "history_feature_size": int(histories.shape[-1]),
        "risk_counts": dict(Counter(row["risk_level"] for row in rows)),
        "action_counts": dict(
            Counter(row["label"]["action"] for row in rows)
        ),
        **split_report,
        **repair_report,
        "initialization": initialization,
        "inputs": [os.path.relpath(root, output) for root in roots],
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
