from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a multimodal driving JSONL dataset"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_rows(root: Path) -> list[dict]:
    with (root / "manifest.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def inferred_group(row: dict) -> str:
    explicit = row.get("split_group") or row.get("trajectory_id")
    if explicit:
        return str(explicit)
    sample_id = str(row.get("sample_id", "unknown"))
    source = sample_id.split("_", 1)[0]
    route_bucket = int(float(row.get("route_s_m", 0.0)) // 100.0)
    return f"{source}:route_{route_bucket:03d}"


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    rows = load_rows(root)
    actions = Counter(str(row["label"]["action"]) for row in rows)
    risks = Counter(str(row.get("risk_level", "unknown")) for row in rows)
    commands = Counter(str(row.get("command_id", "unknown")) for row in rows)
    groups = Counter(inferred_group(row) for row in rows)
    action_by_text: dict[str, Counter] = defaultdict(Counter)
    risk_by_text: dict[str, Counter] = defaultdict(Counter)
    speed_by_action: dict[str, list[float]] = defaultdict(list)
    route_buckets = Counter()
    missing_images = 0
    missing_tensors = 0
    missing_image_tensors = 0
    environment_vectors = Counter()
    splits = Counter()
    split_actions: dict[str, Counter] = defaultdict(Counter)
    split_risks: dict[str, Counter] = defaultdict(Counter)
    group_splits: dict[str, set[str]] = defaultdict(set)
    set_splits: dict[str, set[str]] = defaultdict(set)
    environment_pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        text = str(row.get("source_text", ""))
        action = str(row["label"]["action"])
        risk = str(row.get("risk_level", "unknown"))
        split = str(row.get("split", "implicit"))
        splits[split] += 1
        split_actions[split][action] += 1
        split_risks[split][risk] += 1
        group_splits[inferred_group(row)].add(split)
        counterfactual_set = str(row.get("counterfactual_set_id", ""))
        if counterfactual_set:
            set_splits[counterfactual_set].add(split)
        variant = str(row.get("variant_type", ""))
        if variant.startswith("environment_pair_"):
            environment_pairs[counterfactual_set][variant] = row
        action_by_text[text][action] += 1
        risk_by_text[text][risk] += 1
        speed_by_action[action].append(float(row["label"]["target_speed_kmh"]))
        route_buckets[int(float(row.get("route_s_m", 0.0)) // 500.0)] += 1
        missing_images += sum(
            not (root / image_path).resolve().exists()
            for image_path in row.get("image_paths", [])
        )
        if row.get("image_tensor_path") and not (
            root / str(row["image_tensor_path"])
        ).exists():
            missing_image_tensors += 1
        tensor_path = root / str(row.get("tensor_path", ""))
        if not tensor_path.exists():
            missing_tensors += 1
            continue
        saved = torch.load(tensor_path, map_location="cpu", weights_only=True)
        environment = saved.get("environment_features")
        if environment is not None:
            key = tuple(round(float(value), 4) for value in environment.flatten())
            environment_vectors[key] += 1
    text_action_diversity = {
        text: dict(counts)
        for text, counts in action_by_text.items()
        if len(counts) > 1
    }
    text_risk_diversity = {
        text: dict(counts)
        for text, counts in risk_by_text.items()
        if len(counts) > 1
    }
    speed_stats = {
        action: {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.fmean(values), 4),
            "unique": sorted(set(values)),
        }
        for action, values in sorted(speed_by_action.items())
    }
    invalid_environment_pairs = []
    for pair_id, pair in environment_pairs.items():
        observed = pair.get("environment_pair_observed")
        counterfactual = pair.get("environment_pair_counterfactual")
        valid = bool(
            observed
            and counterfactual
            and observed["image_paths"] == counterfactual["image_paths"]
            and observed["source_text"] == counterfactual["source_text"]
            and observed["split"] == counterfactual["split"]
            and observed["control_speed_cap_kmh"]
            != counterfactual["control_speed_cap_kmh"]
        )
        if not valid:
            invalid_environment_pairs.append(pair_id)
    leaking_groups = {
        group: sorted(values)
        for group, values in group_splits.items()
        if len(values) > 1
    }
    leaking_sets = {
        set_id: sorted(values)
        for set_id, values in set_splits.items()
        if len(values) > 1
    }
    report = {
        "schema_version": "multimodal_dataset_audit/1.0",
        "dataset": str(root),
        "samples": len(rows),
        "actions": dict(sorted(actions.items())),
        "risks": dict(sorted(risks.items())),
        "commands": dict(sorted(commands.items())),
        "speed_by_action": speed_stats,
        "inferred_split_groups": len(groups),
        "largest_group_samples": max(groups.values(), default=0),
        "splits": dict(sorted(splits.items())),
        "actions_by_split": {
            split: dict(sorted(values.items()))
            for split, values in sorted(split_actions.items())
        },
        "risks_by_split": {
            split: dict(sorted(values.items()))
            for split, values in sorted(split_risks.items())
        },
        "split_group_leakage_count": len(leaking_groups),
        "counterfactual_set_leakage_count": len(leaking_sets),
        "environment_pairs": len(environment_pairs),
        "invalid_environment_pair_count": len(invalid_environment_pairs),
        "route_500m_buckets": {
            str(index): count for index, count in sorted(route_buckets.items())
        },
        "unique_environment_vectors": len(environment_vectors),
        "same_text_multiple_actions": text_action_diversity,
        "same_text_multiple_risks": text_risk_diversity,
        "missing_image_files": missing_images,
        "missing_tensor_files": missing_tensors,
        "missing_image_tensor_files": missing_image_tensors,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
