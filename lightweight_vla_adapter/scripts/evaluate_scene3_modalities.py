from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.scripts.train_scene3_multimodal import (
    Scene3Dataset,
    load_rows,
    model_kwargs,
    split_rows,
)
from lightweight_vla_adapter.src.contracts import ACTION_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VLA modality ablations")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def apply_ablation(inputs: dict[str, torch.Tensor], mode: str) -> None:
    if mode == "no_images":
        inputs["camera_images"].zero_()
    elif mode == "no_text":
        inputs["intent_tokens"].zero_()
    elif mode == "no_vehicle_state":
        inputs["ego_features"].zero_()
    elif mode == "no_environment":
        inputs["environment_features"].zero_()
    elif mode.startswith("no_view_"):
        view = int(mode.rsplit("_", 1)[1])
        inputs["camera_view_mask"][:, view] = False
    elif mode == "shuffled_images":
        inputs["camera_images"] = inputs["camera_images"].roll(1, dims=0)
    elif mode == "shuffled_text":
        inputs["intent_tokens"] = inputs["intent_tokens"].roll(1, dims=0)
        inputs["intent_mask"] = inputs["intent_mask"].roll(1, dims=0)
    elif mode == "shuffled_vehicle_state":
        inputs["ego_features"] = inputs["ego_features"].roll(1, dims=0)
    elif mode == "shuffled_environment":
        inputs["environment_features"] = inputs["environment_features"].roll(
            1, dims=0
        )


def run(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> dict:
    action_probs = []
    risk_probs = []
    speeds = []
    action_labels = []
    risk_labels = []
    speed_labels = []
    per_action = defaultdict(lambda: [0, 0])
    environment_pairs = defaultdict(dict)
    with torch.inference_mode():
        for batch in loader:
            inputs = model_kwargs(batch, device)
            apply_ablation(inputs, mode)
            output = model(**inputs)
            action_probability = F.softmax(output.action_logits, dim=-1)
            risk_probability = F.softmax(output.visual_risk_logits, dim=-1)
            action_probs.append(action_probability.cpu())
            risk_probs.append(risk_probability.cpu())
            speeds.append(output.target_speed_kmh.cpu())
            action_labels.append(batch["action_label"])
            risk_labels.append(batch["risk_label"])
            speed_labels.append(batch["speed_label"])
            predicted = action_probability.argmax(-1).cpu()
            for label, prediction in zip(batch["action_label"], predicted):
                name = ACTION_LABELS[int(label)]
                per_action[name][1] += 1
                per_action[name][0] += int(label == prediction)
            for set_id, variant, predicted_speed, target_speed in zip(
                batch["counterfactual_set_id"],
                batch["variant_type"],
                output.target_speed_kmh.detach().cpu().tolist(),
                batch["speed_label"].tolist(),
            ):
                if str(variant).startswith("environment_pair_"):
                    environment_pairs[str(set_id)][str(variant)] = (
                        float(predicted_speed), float(target_speed)
                    )
    action_probs = torch.cat(action_probs)
    risk_probs = torch.cat(risk_probs)
    speeds = torch.cat(speeds)
    action_labels = torch.cat(action_labels)
    risk_labels = torch.cat(risk_labels)
    speed_labels = torch.cat(speed_labels)
    pair_count = 0
    pair_order_correct = 0
    pair_delta_error = 0.0
    for pair in environment_pairs.values():
        observed = pair.get("environment_pair_observed")
        counterfactual = pair.get("environment_pair_counterfactual")
        if observed is None or counterfactual is None:
            continue
        pair_count += 1
        predicted_delta = counterfactual[0] - observed[0]
        target_delta = counterfactual[1] - observed[1]
        pair_order_correct += int(predicted_delta * target_delta > 0.0)
        pair_delta_error += abs(predicted_delta - target_delta)
    return {
        "mode": mode,
        "samples": int(action_labels.numel()),
        "action_accuracy": float(
            (action_probs.argmax(-1) == action_labels).float().mean()
        ),
        "visual_risk_accuracy": float(
            (risk_probs.argmax(-1) == risk_labels).float().mean()
        ),
        "speed_mae_kmh": float((speeds - speed_labels).abs().mean()),
        "environment_pair_count": pair_count,
        "environment_pair_order_accuracy": pair_order_correct / max(1, pair_count),
        "environment_pair_delta_mae_kmh": pair_delta_error / max(1, pair_count),
        "per_action_accuracy": {
            name: {"correct": values[0], "samples": values[1],
                   "accuracy": values[0] / values[1]}
            for name, values in sorted(per_action.items())
        },
        "_action_probs": action_probs,
        "_risk_probs": risk_probs,
        "_speeds": speeds,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = load_rows(args.dataset)
    _, validation_rows, test_rows = split_rows(rows)
    evaluation_rows = test_rows or validation_rows
    dataset = Scene3Dataset(
        args.dataset,
        evaluation_rows,
        augment=False,
        intent_max_length=int(config.get("intent_max_length", 32)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, persistent_workers=True)
    model = build_model(config)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model.to(device).eval()
    modes = [
        "baseline", "no_images", "no_text", "no_vehicle_state",
        "no_environment", "no_view_0", "no_view_1", "no_view_2", "no_view_3",
        "shuffled_images", "shuffled_text", "shuffled_vehicle_state",
        "shuffled_environment",
    ]
    results = [run(model, loader, device, mode) for mode in modes]
    baseline = results[0]
    baseline_action_probs = baseline["_action_probs"]
    baseline_risk_probs = baseline["_risk_probs"]
    baseline_speeds = baseline["_speeds"]
    for result in results:
        result["mean_action_probability_l1_vs_baseline"] = float(
            (result["_action_probs"] - baseline_action_probs).abs().mean()
        )
        result["mean_risk_probability_l1_vs_baseline"] = float(
            (result["_risk_probs"] - baseline_risk_probs).abs().mean()
        )
        result["mean_speed_delta_kmh_vs_baseline"] = float(
            (result["_speeds"] - baseline_speeds).abs().mean()
        )
        for private in ("_action_probs", "_risk_probs", "_speeds"):
            del result[private]
    report = {
        "schema_version": "scene3_modality_ablation/1.0",
        "checkpoint": str(args.checkpoint),
        "evaluation_split": "test" if test_rows else "validation",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
