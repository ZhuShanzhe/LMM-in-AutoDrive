"""Add rainy-night event-positive samples from the scene-3 counterfactual
capture to the fine-tuning dataset (frames outside the validation set)."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch
from torchvision.io import read_image
from torch.nn import functional as F


SOURCE = Path(
    "/root/autodl-tmp/LMM-in-AutoDrive/experiment/CARLA/outputs/"
    "scene3_cf_rainy_night_seed101_20260805"
)
STAGE4_MANIFEST = Path(
    "/root/autodl-tmp/datasets/training/"
    "universal_three_scene_v6_finetune_stage4/manifest.jsonl"
)
STAGE8_MANIFEST = Path(
    "/root/autodl-tmp/datasets/training/"
    "universal_three_scene_v6_finetune_stage8/manifest.jsonl"
)
OUTPUT = Path(
    "/root/autodl-tmp/datasets/training/"
    "universal_three_scene_v6_finetune_rainy_extra"
)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ("images", "tensors", "intents"):
        (OUTPUT / name).mkdir(exist_ok=True)

    # Event ACTIVE windows.
    active_windows: list[tuple[int, int]] = []
    active_start: dict[str, int] = {}
    with (SOURCE / "event_timeline.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame = int(row["simulation_frame"])
            if row["state"] == "ACTIVE":
                active_start[row["event_id"]] = frame
            elif row["state"] == "RESOLVED":
                start = active_start.pop(row["event_id"], frame)
                active_windows.append((start, frame))

    # Existing frames and validation high frames to exclude.
    existing_frames = set()
    for row in load_manifest(STAGE4_MANIFEST):
        frame = row.get("source_frame")
        if frame is not None:
            existing_frames.add(int(frame))
    validation_high_frames = set()
    for row in load_manifest(STAGE8_MANIFEST):
        if (
            row.get("split") == "validation"
            and row.get("risk_level") == "high"
            and row.get("source_frame") is not None
        ):
            validation_high_frames.add(int(row["source_frame"]))

    # Vehicle state lookup.
    vehicle_state: dict[int, dict] = {}
    with (SOURCE / "vehicle_state.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            vehicle_state[int(row["simulation_frame"])] = row

    # Rainy-night environment features (consistent with the live capture).
    env_template = None
    for row in load_manifest(STAGE4_MANIFEST):
        if str(row.get("weather_profile", "")).startswith(
            "official-rainy-night"
        ):
            tensors = torch.load(
                STAGE4_MANIFEST.parent / row["tensor_path"],
                map_location="cpu",
                weights_only=True,
            )
            env_template = tensors["environment_features"]
            break
    if env_template is None:
        env_template = torch.tensor(
            [1.0, 1.0, 1.0, 0.2, 0.5, -0.3, 0.9, 0.5, 0.02, 1.0, 0.1, 0.8, 0.32, 0.32]
        )

    view_names = ("front", "left", "right", "rear")
    frames = sorted(
        int(path.stem)
        for path in (SOURCE / "rgb" / "front_rgb").glob("*.png")
    )
    rows = []
    samples = 0
    for frame in frames:
        if frame in existing_frames or frame in validation_high_frames:
            continue

        def in_window(margin: int) -> bool:
            return any(start - margin <= frame <= end + margin for start, end in active_windows)

        if in_window(10):
            risk = "high"
            intent_name = "yield_10"
        elif in_window(60):
            risk = "medium"
            intent_name = "decel_20"
        else:
            continue

        images = []
        ok = True
        for view in view_names:
            path = SOURCE / "rgb" / f"{view}_rgb" / f"{frame}.png"
            if not path.exists():
                ok = False
                break
            image = read_image(str(path))[:3]
            image = F.interpolate(
                image.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            images.append(image)
        if not ok:
            continue
        image_tensor = torch.stack(images).to(torch.uint8)

        state = vehicle_state.get(frame, {})
        velocity = state.get("velocity_mps", {})
        acceleration = state.get("acceleration_mps2", {})
        angular = state.get("angular_velocity_deg_s", {})
        control = state.get("control", {})
        speed_mps = math.sqrt(
            float(velocity.get("x", 0.0)) ** 2
            + float(velocity.get("y", 0.0)) ** 2
            + float(velocity.get("z", 0.0)) ** 2
        )
        ego_features = torch.tensor(
            [
                speed_mps,
                math.sqrt(
                    float(acceleration.get("x", 0.0)) ** 2
                    + float(acceleration.get("y", 0.0)) ** 2
                    + float(acceleration.get("z", 0.0)) ** 2
                ),
                math.radians(float(angular.get("z", 0.0))),
                float(control.get("steer", 0.0)),
                float(control.get("throttle", 0.0)),
                float(control.get("brake", 0.0)),
                32.0 / 3.6,
                0.0,
            ]
        )
        sample_id = f"rainy_extra_{frame}_{risk}"
        torch.save(image_tensor, OUTPUT / "images" / f"{sample_id}.pt")
        torch.save(
            {
                "camera_bev": torch.zeros(8, 64, 64),
                "lidar_bev": torch.zeros(4, 64, 64),
                "ego_features": ego_features,
                "candidate_features": torch.zeros(32, 12),
                "candidate_mask": torch.zeros(32, dtype=torch.bool),
                "environment_features": env_template,
                "camera_view_mask": torch.tensor([True, True, True, True]),
            },
            OUTPUT / "tensors" / f"{sample_id}.pt",
        )
        split = "validation" if samples % 10 == 0 else "train"
        rows.append(
            {
                "schema_version": "scene3_multimodal_training_sample/2.0",
                "sample_id": sample_id,
                "source_dataset": "CARLA_scene3_rainy_counterfactual_extra",
                "source_frame": frame,
                "route_s_m": float(state.get("route_progress_m", 0.0)),
                "split_group": "rainy_extra:seed101",
                "split": split,
                "counterfactual_set_id": f"rainy_extra:{frame}",
                "variant_type": f"rainy_extra_{risk}",
                "command_id": intent_name,
                "source_text": (
                    "Slow down and yield to the road user ahead."
                    if risk == "high"
                    else "Slow down smoothly to 20.0 kilometers per hour."
                ),
                "normalized_text": (
                    "Slow down and yield to the road user ahead."
                    if risk == "high"
                    else "Slow down smoothly to 20.0 kilometers per hour."
                ),
                "camera_order": list(view_names),
                "camera_view_mask": [True, True, True, True],
                "image_paths": [f"images/{sample_id}.pt"] * 4,
                "image_tensor_path": f"images/{sample_id}.pt",
                "tensor_path": f"tensors/{sample_id}.pt",
                "intent_tensor_path": f"intents/{intent_name}.pt",
                "label": {
                    "action": "decelerate",
                    "target_speed_kmh": 20.0 if risk == "high" else 20.0,
                    "target_lane": None,
                },
                "risk_level": risk,
                "risk_reason_codes": [],
                "weather_profile": "official-rainy-night",
                "control_speed_cap_kmh": 32.0,
                "capture_quality": "scene3_rainy_counterfactual_extra",
                "sampling_weight": 250.0 if risk == "high" else 30.0,
            }
        )
        samples += 1

    with (OUTPUT / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("added samples:", samples)
    print("high:", sum(1 for r in rows if r["risk_level"] == "high"))
    print("medium:", sum(1 for r in rows if r["risk_level"] == "medium"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
