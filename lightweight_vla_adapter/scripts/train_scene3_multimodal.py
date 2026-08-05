from __future__ import annotations

import argparse
import json
import random
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
    ) -> None:
        self.root = root
        self.rows = rows
        self.augment = augment
        self.intent_max_length = int(intent_max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        saved = torch.load(
            self.root / row["tensor_path"], map_location="cpu", weights_only=True
        )
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
        action = ACTION_LABELS.index(row["label"]["action"])
        return {
            **saved,
            "camera_images": images,
            "camera_view_mask": torch.ones(4, dtype=torch.bool),
            "action_label": torch.tensor(action),
            "speed_label": torch.tensor(row["label"]["target_speed_kmh"]),
            "lane_label": torch.tensor(LANE_LABELS[row["label"]["target_lane"]]),
            "risk_label": torch.tensor(RISK_LABELS[row["risk_level"]]),
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
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rows(root: Path) -> list[dict]:
    with (root / "manifest.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_kwargs(batch: dict[str, torch.Tensor], device: torch.device):
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
    return result


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    totals = Counter()
    speed_error = 0.0
    with torch.inference_mode():
        for batch in loader:
            output = model(**model_kwargs(batch, device))
            action = batch["action_label"].to(device)
            risk = batch["risk_label"].to(device)
            totals["count"] += action.numel()
            totals["action_correct"] += int(
                (output.action_logits.argmax(-1) == action).sum()
            )
            totals["risk_correct"] += int(
                (output.visual_risk_logits.argmax(-1) == risk).sum()
            )
            speed_error += float(
                (output.target_speed_kmh - batch["speed_label"].to(device))
                .abs().sum()
            )
    count = max(1, totals["count"])
    return {
        "samples": totals["count"],
        "action_accuracy": totals["action_correct"] / count,
        "visual_risk_accuracy": totals["risk_correct"] / count,
        "speed_mae_kmh": speed_error / count,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = load_rows(args.dataset)
    if len(rows) < 50:
        raise ValueError("at least 50 synchronized training samples are required")
    train_rows = [row for index, row in enumerate(rows) if index % 5 != 0]
    validation_rows = [row for index, row in enumerate(rows) if index % 5 == 0]
    model = build_model(config)
    if args.initialize_from is not None:
        state = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(str(incompatible.unexpected_keys))
    model.raw_camera_encoder.load_imagenet_initialization()
    model.raw_camera_encoder.freeze_backbone(True)
    model.to(device)
    counts = Counter(row["label"]["action"] for row in train_rows)
    sample_weights = [1.0 / counts[row["label"]["action"]] for row in train_rows]
    sampler = WeightedRandomSampler(sample_weights, len(train_rows), replacement=True)
    train_loader = DataLoader(
        Scene3Dataset(
            args.dataset,
            train_rows,
            augment=True,
            intent_max_length=int(config.get("intent_max_length", 32)),
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
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate, weight_decay=0.02,
    )
    best_score = -float("inf")
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        if epoch == max(3, args.epochs // 3):
            model.raw_camera_encoder.freeze_backbone(False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate * 0.15, weight_decay=0.02
            )
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            inputs = model_kwargs(batch, device)
            if random.random() < 0.50:
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
                F.cross_entropy(output.action_logits, action)
                + 0.035 * F.smooth_l1_loss(output.target_speed_kmh, speed)
                + 0.25 * F.cross_entropy(output.target_lane_logits, lane)
                + 0.40 * F.cross_entropy(output.visual_risk_logits, risk)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        metrics = evaluate(model, validation_loader, device)
        metrics.update(epoch=epoch, train_loss=loss_sum / max(1, len(train_loader)))
        metrics["selection_score"] = (
            metrics["action_accuracy"]
            + 0.15 * metrics["visual_risk_accuracy"]
            - 0.005 * metrics["speed_mae_kmh"]
        )
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        torch.save(model.state_dict(), args.output_dir / "model_last.pt")
        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            torch.save(model.state_dict(), args.output_dir / "model.pt")
    report = {
        "schema_version": "scene3_multimodal_training/1.0",
        "dataset": str(args.dataset),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "action_counts": dict(counts),
        "best_selection_score": best_score,
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
