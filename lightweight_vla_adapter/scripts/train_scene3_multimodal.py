from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.io import read_image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.contracts import ACTION_LABELS


RISK_LABELS = {"low": 0, "medium": 1, "high": 2}
LANE_LABELS = {None: 0, "left": 1, "right": 2}


class Scene3Dataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: list[dict],
        *,
        augment: bool,
        intent_max_length: int = 32,
        bev_input_size: tuple[int, int] = (64, 64),
    ) -> None:
        self.root = root
        self.rows = rows
        self.augment = augment
        self.intent_max_length = int(intent_max_length)
        self.bev_input_size = tuple(int(value) for value in bev_input_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        saved = torch.load(
            self.root / row["tensor_path"], map_location="cpu", weights_only=True
        )
        if row.get("intent_tensor_path"):
            intent = torch.load(
                self.root / row["intent_tensor_path"],
                map_location="cpu",
                weights_only=True,
            )
            saved.update(intent)
        if row.get("image_tensor_path"):
            images = torch.load(
                self.root / row["image_tensor_path"],
                map_location="cpu",
                weights_only=True,
            ).float().div_(255.0)
        else:
            images = torch.stack(
                [
                    read_image(str(self.root / path))[:3]
                    for path in row["image_paths"]
                ]
            ).float().div_(255.0)
            images = F.interpolate(
                images,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
        if self.augment:
            brightness = random.uniform(0.70, 1.25)
            contrast = random.uniform(0.75, 1.25)
            mean = images.mean(dim=(-2, -1), keepdim=True)
            images = ((images - mean) * contrast + mean) * brightness
            images = images.clamp_(0.0, 1.0)
            if random.random() < 0.20:
                images = F.avg_pool3d(
                    images.unsqueeze(0), kernel_size=(1, 3, 3),
                    stride=1, padding=(0, 1, 1)
                ).squeeze(0)
        tokens = saved["intent_tokens"][: self.intent_max_length]
        mask = saved["intent_mask"][: self.intent_max_length]
        padded_tokens = torch.zeros(
            self.intent_max_length, tokens.shape[-1], dtype=tokens.dtype
        )
        padded_mask = torch.zeros(self.intent_max_length, dtype=torch.bool)
        padded_tokens[: tokens.shape[0]] = tokens
        padded_mask[: mask.shape[0]] = mask
        saved["intent_tokens"] = padded_tokens
        saved["intent_mask"] = padded_mask
        for name in ("camera_bev", "lidar_bev"):
            tensor = saved[name].float()
            if tensor.shape[-2:] != self.bev_input_size:
                tensor = F.interpolate(
                    tensor.unsqueeze(0),
                    size=self.bev_input_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            saved[name] = tensor
        configured_mask = row.get(
            "camera_view_mask", saved.get("camera_view_mask")
        )
        if configured_mask is None:
            view_mask = torch.ones(images.shape[0], dtype=torch.bool)
        else:
            view_mask = torch.as_tensor(configured_mask, dtype=torch.bool)
        if view_mask.numel() != images.shape[0]:
            raise ValueError(
                f"camera_view_mask has {view_mask.numel()} entries for "
                f"{images.shape[0]} images"
            )
        # Front-only dropout teaches the same checkpoint to serve Scene 1,
        # while partial side-view dropout makes transient sensor loss safe.
        if self.augment and images.shape[0] > 1:
            if random.random() < 0.25:
                view_mask[1:] = False
            else:
                for view_index in range(1, images.shape[0]):
                    if random.random() < 0.10:
                        view_mask[view_index] = False
        if not bool(view_mask.any()):
            view_mask[0] = True
        action = ACTION_LABELS.index(row["label"]["action"])
        return {
            **saved,
            "camera_images": images,
            "camera_view_mask": view_mask,
            "action_label": torch.tensor(action),
            "speed_label": torch.tensor(row["label"]["target_speed_kmh"]),
            "lane_label": torch.tensor(LANE_LABELS[row["label"]["target_lane"]]),
            "risk_label": torch.tensor(RISK_LABELS[row["risk_level"]]),
            "speed_cap_label": torch.tensor(
                float(row.get("control_speed_cap_kmh", 100.0))
            ),
            "variant_type": str(row.get("variant_type", "observed_command")),
            "counterfactual_set_id": str(
                row.get("counterfactual_set_id", row.get("sample_id", index))
            ),
            "source_dataset": str(row.get("source_dataset", "CARLA")),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the raw-multiview Scene 3 VLA adapter"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--balance-power",
        type=float,
        default=0.75,
        help="Inverse joint action/risk frequency exponent used by the sampler.",
    )
    parser.add_argument(
        "--high-recall-margin",
        type=float,
        default=0.0,
        help=(
            "Clamp penalty pushing P(high) above this margin for every "
            "high-risk sample (directly targets high-risk recall)."
        ),
    )
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=None,
        help="Epoch at which the raw-camera backbone is unfrozen.",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rows(root: Path) -> list[dict]:
    with (root / "manifest.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    explicit = {str(row.get("split", "")) for row in rows}
    if explicit - {"", "train", "validation", "test"}:
        raise ValueError(f"invalid explicit dataset splits: {sorted(explicit)}")
    if explicit - {""}:
        if "" in explicit:
            raise ValueError("dataset mixes explicit and implicit splits")
        result = tuple(
            [row for row in rows if row["split"] == name]
            for name in ("train", "validation", "test")
        )
        if any(not subset for subset in result):
            raise ValueError("train, validation, and test splits must be non-empty")
        return result
    return (
        [row for index, row in enumerate(rows) if index % 5 != 0],
        [row for index, row in enumerate(rows) if index % 5 == 0],
        [],
    )


def model_kwargs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    config: dict | None = None,
):
    floating = (
        "camera_bev", "lidar_bev", "ego_features", "candidate_features",
        "intent_tokens", "camera_images", "environment_features"
    )
    result = {name: batch[name].to(device) for name in floating}
    result.update(
        candidate_mask=batch["candidate_mask"].to(device),
        intent_mask=batch["intent_mask"].to(device),
        camera_view_mask=batch["camera_view_mask"].to(device),
    )
    if config is not None and not bool(
        config.get("use_candidate_entities", True)
    ):
        result["candidate_features"].zero_()
        result["candidate_mask"].zero_()
    allowed_structured_sources = (
        set(config.get("structured_sensor_sources", []))
        if config is not None
        else set()
    )
    if config is not None and "structured_sensor_sources" in config:
        blocked = torch.tensor(
            [
                str(source) not in allowed_structured_sources
                for source in batch["source_dataset"]
            ],
            dtype=torch.bool,
            device=device,
        )
        result["camera_bev"][blocked] = 0
        result["lidar_bev"][blocked] = 0
    return result


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict | None = None,
) -> dict:
    model.eval()
    totals = Counter()
    action_totals = Counter()
    action_correct = Counter()
    risk_totals = Counter()
    risk_correct = Counter()
    speed_error = 0.0
    environment_pair_predictions: dict[str, dict[str, tuple[float, float]]] = {}
    with torch.inference_mode():
        for batch in loader:
            output = model(**model_kwargs(batch, device, config))
            action = batch["action_label"].to(device)
            risk = batch["risk_label"].to(device)
            totals["count"] += action.numel()
            totals["action_correct"] += int(
                (output.action_logits.argmax(-1) == action).sum()
            )
            totals["risk_correct"] += int(
                (output.visual_risk_logits.argmax(-1) == risk).sum()
            )
            predicted_risk = output.visual_risk_logits.argmax(-1)
            for label, prediction in zip(
                risk.detach().cpu(), predicted_risk.detach().cpu()
            ):
                name = next(name for name, value in RISK_LABELS.items() if value == int(label))
                risk_totals[name] += 1
                risk_correct[name] += int(label == prediction)
            predicted = output.action_logits.argmax(-1)
            for label, prediction, variant in zip(
                action.detach().cpu(),
                predicted.detach().cpu(),
                batch["variant_type"],
            ):
                name = ACTION_LABELS[int(label)]
                action_totals[name] += 1
                action_correct[name] += int(label == prediction)
                if variant == "visual_counterfactual":
                    totals["visual_counterfactual_count"] += 1
                    totals["visual_counterfactual_correct"] += int(
                        label == prediction
                    )
            speed_error += float(
                (output.target_speed_kmh - batch["speed_label"].to(device))
                .abs().sum()
            )
            for set_id, variant, predicted_speed, target_speed in zip(
                batch["counterfactual_set_id"],
                batch["variant_type"],
                output.target_speed_kmh.detach().cpu().tolist(),
                batch["speed_label"].tolist(),
            ):
                if str(variant).startswith("environment_pair_"):
                    environment_pair_predictions.setdefault(str(set_id), {})[
                        str(variant)
                    ] = (float(predicted_speed), float(target_speed))
            totals["speed_cap_violations"] += int(
                (
                    output.target_speed_kmh
                    > batch["speed_cap_label"].to(device) + 1e-4
                ).sum()
            )
    count = max(1, totals["count"])
    per_action = {
        name: {
            "correct": action_correct[name],
            "samples": action_totals[name],
            "accuracy": action_correct[name] / action_totals[name],
        }
        for name in sorted(action_totals)
    }
    per_risk = {
        name: {
            "correct": risk_correct[name],
            "samples": risk_totals[name],
            "accuracy": risk_correct[name] / risk_totals[name],
        }
        for name in sorted(risk_totals)
    }
    pair_count = 0
    pair_order_correct = 0
    pair_delta_error = 0.0
    for pair in environment_pair_predictions.values():
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
        "samples": totals["count"],
        "action_accuracy": totals["action_correct"] / count,
        "macro_action_accuracy": statistics.fmean(
            item["accuracy"] for item in per_action.values()
        ),
        "per_action_accuracy": per_action,
        "visual_risk_accuracy": totals["risk_correct"] / count,
        "visual_risk_macro_accuracy": statistics.fmean(
            item["accuracy"] for item in per_risk.values()
        ),
        "per_risk_accuracy": per_risk,
        "speed_mae_kmh": speed_error / count,
        "visual_counterfactual_accuracy": (
            totals["visual_counterfactual_correct"]
            / max(1, totals["visual_counterfactual_count"])
        ),
        "speed_cap_violation_rate": totals["speed_cap_violations"] / count,
        "environment_pair_count": pair_count,
        "environment_pair_order_accuracy": pair_order_correct / max(1, pair_count),
        "environment_pair_delta_mae_kmh": pair_delta_error / max(1, pair_count),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bev_input_size = tuple(config.get("bev_input_size", (64, 64)))
    rows = load_rows(args.dataset)
    if len(rows) < 50:
        raise ValueError("at least 50 synchronized training samples are required")
    train_rows, validation_rows, test_rows = split_rows(rows)
    model = build_model(config)
    initialization = None
    if args.initialize_from is not None:
        state = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and current[key].shape == value.shape
        }
        incompatible = model.load_state_dict(compatible, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(str(incompatible.unexpected_keys))
        initialization = {
            "checkpoint": str(args.initialize_from),
            "compatible_parameters": len(compatible),
            "skipped_parameters": sorted(set(state) - set(compatible)),
        }
    else:
        model.raw_camera_encoder.load_imagenet_initialization()
    model.raw_camera_encoder.freeze_backbone(True)
    model.to(device)
    counts = Counter(row["label"]["action"] for row in train_rows)
    risk_counts = Counter(row["risk_level"] for row in train_rows)
    joint_counts = Counter(
        (row["label"]["action"], row["risk_level"])
        for row in train_rows
    )
    if not 0.0 <= args.balance_power <= 1.0:
        raise ValueError("--balance-power must be in [0, 1]")
    sample_weights = [
        float(row.get("sampling_weight", 1.0))
        / (
            joint_counts[(row["label"]["action"], row["risk_level"])]
            ** args.balance_power
        )
        for row in train_rows
    ]
    sampler = WeightedRandomSampler(sample_weights, len(train_rows), replacement=True)
    train_loader = DataLoader(
        Scene3Dataset(
            args.dataset,
            train_rows,
            augment=True,
            intent_max_length=int(config.get("intent_max_length", 32)),
            bev_input_size=bev_input_size,
        ),
        batch_size=args.batch_size, sampler=sampler, num_workers=4,
        pin_memory=True, persistent_workers=True,
    )
    validation_loader = DataLoader(
        Scene3Dataset(
            args.dataset,
            validation_rows,
            augment=False,
            intent_max_length=int(config.get("intent_max_length", 32)),
            bev_input_size=bev_input_size,
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    test_loader = (
        DataLoader(
            Scene3Dataset(
                args.dataset,
                test_rows,
                augment=False,
                intent_max_length=int(config.get("intent_max_length", 32)),
                bev_input_size=bev_input_size,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )
        if test_rows
        else None
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=0.02,
    )
    action_loss_weights = torch.tensor(
        [
            math.sqrt(max(counts.values()) / max(1, counts[action]))
            for action in ACTION_LABELS
        ],
        dtype=torch.float32,
        device=device,
    ).clamp_(max=10.0)
    action_loss_weights.div_(action_loss_weights.mean())
    risk_loss_weights = torch.tensor(
        [
            max(risk_counts.values()) / max(1, risk_counts[risk])
            for risk in RISK_LABELS
        ],
        dtype=torch.float32,
        device=device,
    ).clamp_(max=30.0)
    risk_loss_weights.div_(risk_loss_weights.mean())
    best_score = -float("inf")
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unfreeze_epoch = (
        args.unfreeze_epoch
        if args.unfreeze_epoch is not None
        else max(3, args.epochs // 3)
    )
    for epoch in range(1, args.epochs + 1):
        if epoch == unfreeze_epoch:
            model.raw_camera_encoder.freeze_backbone(False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate * 0.15, weight_decay=0.02
            )
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            inputs = model_kwargs(batch, device, config)
            if random.random() < 0.10:
                inputs["candidate_features"] = torch.zeros_like(
                    inputs["candidate_features"]
                )
                inputs["candidate_mask"] = torch.zeros_like(
                    inputs["candidate_mask"]
                )
            output = model(**inputs)
            action = batch["action_label"].to(device)
            speed = batch["speed_label"].to(device)
            lane = batch["lane_label"].to(device)
            risk = batch["risk_label"].to(device)
            loss = (
                F.cross_entropy(
                    output.action_logits,
                    action,
                    weight=action_loss_weights,
                    label_smoothing=0.02,
                )
                + 0.035 * F.smooth_l1_loss(output.target_speed_kmh, speed)
                + 0.25 * F.cross_entropy(output.target_lane_logits, lane)
                + 0.40
                * F.cross_entropy(
                    output.visual_risk_logits, risk, weight=risk_loss_weights
                )
            )
            if args.high_recall_margin > 0.0:
                high_mask = risk == 2
                if bool(high_mask.any()):
                    high_probs = F.softmax(
                        output.visual_risk_logits, dim=-1
                    )[high_mask, 2]
                    loss = loss + 0.5 * torch.clamp(
                        args.high_recall_margin - high_probs, min=0.0
                    ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        metrics = evaluate(model, validation_loader, device, config)
        metrics.update(epoch=epoch, train_loss=loss_sum / max(1, len(train_loader)))
        emergency_accuracy = metrics["per_action_accuracy"].get(
            "emergency_brake", {"accuracy": 0.0}
        )["accuracy"]
        high_risk_accuracy = metrics["per_risk_accuracy"].get(
            "high", {"accuracy": 0.0}
        )["accuracy"]
        metrics["selection_score"] = (
            0.20 * metrics["macro_action_accuracy"]
            + 0.10 * metrics["visual_counterfactual_accuracy"]
            + 0.10 * metrics["visual_risk_macro_accuracy"]
            + 0.10 * metrics["environment_pair_order_accuracy"]
            + 0.25 * emergency_accuracy
            + 0.25 * high_risk_accuracy
            - 0.005 * metrics["speed_mae_kmh"]
        )
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        torch.save(model.state_dict(), args.output_dir / "model_last.pt")
        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            torch.save(model.state_dict(), args.output_dir / "model.pt")
    if test_loader:
        best_state = torch.load(
            args.output_dir / "model.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(best_state)
        test_metrics = evaluate(model, test_loader, device, config)
    else:
        test_metrics = None
    report = {
        "schema_version": "scene3_multimodal_training/1.0",
        "dataset": str(args.dataset),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "test_samples": len(test_rows),
        "action_counts": dict(counts),
        "risk_counts": dict(risk_counts),
        "joint_action_risk_counts": {
            f"{action}|{risk}": count
            for (action, risk), count in sorted(joint_counts.items())
        },
        "balance_power": args.balance_power,
        "high_recall_margin": args.high_recall_margin,
        "unfreeze_epoch": unfreeze_epoch,
        "action_loss_weights": {
            action: float(weight)
            for action, weight in zip(ACTION_LABELS, action_loss_weights.cpu())
        },
        "risk_loss_weights": {
            risk: float(risk_loss_weights[index])
            for risk, index in RISK_LABELS.items()
        },
        "best_selection_score": best_score,
        "test_metrics": test_metrics,
        "initialization": initialization,
        "history": history,
        "seed": args.seed,
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
