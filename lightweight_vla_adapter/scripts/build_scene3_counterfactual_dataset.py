from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torchvision.io import read_image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CARLA_DIR = REPO_ROOT / "experiment" / "CARLA"
if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

from lightweight_vla_adapter.scripts.scene3_command_profiles import (
    COMMAND_PROFILES,
    active_text_command,
)
from structured_command_parser.src.modernbert_service import ModernBertCommandService


CAMERA_ORDER = ("front", "left", "right", "rear")
CAMERA_DIRS = {
    "front": "front_rgb",
    "left": "left_rgb",
    "right": "right_rgb",
    "rear": "rear_rgb",
}
WEATHER_OVERRIDES = {
    "official-rainy-night": {},
    "rainy-daylight": {
        "cloudiness": 95.0,
        "precipitation": 55.0,
        "precipitation_deposits": 80.0,
        "wind_intensity": 25.0,
        "sun_altitude_angle": 18.0,
        "fog_density": 8.0,
        "fog_distance": 250.0,
        "wetness": 100.0,
    },
    "clear-daylight": {
        "cloudiness": 15.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 5.0,
        "sun_altitude_angle": 65.0,
        "fog_density": 0.0,
        "fog_distance": 1000.0,
        "wetness": 0.0,
    },
}
TEMPLATES = {
    "visual_safe": (
        "根据当前道路情况安全行驶",
        "Drive safely according to the current road situation.",
    ),
    "keep_lane": ("保持当前车道", "Keep the current lane."),
    "accelerate": ("在安全范围内逐步提速", "Accelerate when it is safe."),
    "decelerate": ("平稳减速", "Slow down smoothly."),
    "stop": ("在安全位置停车", "Stop the vehicle safely."),
    "emergency_brake": ("立即紧急制动", "Brake immediately."),
    "lane_change_left": ("确认安全后向左变道", "Move to the left lane when safe."),
    "lane_change_right": ("确认安全后向右变道", "Move to the right lane when safe."),
    "turn_left": ("前方路口左转", "Turn left at the next junction."),
    "turn_right": ("前方路口右转", "Turn right at the next junction."),
}
BASIC_VARIANTS = (
    "keep_lane",
    "accelerate",
    "decelerate",
    "stop",
    "lane_change_left",
    "lane_change_right",
    "turn_left",
    "turn_right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build grouped Scene 3 multimodal and counterfactual samples"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--parser-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rows_by_frame(path: Path) -> dict[int, dict]:
    return {int(row["simulation_frame"]): row for row in read_jsonl(path)}


def nearest_with_delta(
    rows: dict[int, dict], frame: int, tolerance: int
) -> tuple[dict | None, int | None]:
    for offset in range(tolerance + 1):
        candidates = (frame,) if offset == 0 else (frame - offset, frame + offset)
        for candidate in candidates:
            if candidate in rows:
                return rows[candidate], candidate - frame
    return None, None


def stable_unit(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def stable_index(value: str, size: int) -> int:
    return int(stable_unit(value) * size) % size


def split_for_group(group: str, config: dict) -> str:
    ratios = config["split_ratios"]
    value = stable_unit(f"{config['split_seed']}:{group}")
    train_end = float(ratios["train"])
    validation_end = train_end + float(ratios["validation"])
    if value < train_end:
        return "train"
    if value < validation_end:
        return "validation"
    return "test"


def assign_grouped_stratified_splits(
    rows: list[dict[str, Any]], config: dict
) -> dict[str, Any]:
    """Assign whole route groups while reserving safety-critical evaluation."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["split_group"]), []).append(row)
    split_names = ("train", "validation", "test")
    assignments: dict[str, str] = {}

    def critical_score(group: str) -> tuple[int, int, float]:
        group_rows = grouped[group]
        high = sum(row["risk_level"] == "high" for row in group_rows)
        emergency = sum(
            row["label"]["action"] == "emergency_brake" for row in group_rows
        )
        return high, emergency, stable_unit(
            f"critical:{config['split_seed']}:{group}"
        )

    critical_groups = sorted(grouped, key=critical_score, reverse=True)
    if len(critical_groups) < 3 or critical_score(critical_groups[2])[0] == 0:
        raise ValueError("at least three route groups with high-risk samples are required")
    # Give the strongest held-out evidence to validation/test, while retaining
    # one independent critical group for learning.
    for split, group in zip(
        ("validation", "test", "train"), critical_groups[:3]
    ):
        assignments[group] = split

    total = len(rows)
    ratios = config["split_ratios"]
    targets = {name: total * float(ratios[name]) for name in split_names}
    assigned_samples = Counter(
        assignments[group] for group in assignments for _ in grouped[group]
    )
    remaining = sorted(
        (group for group in grouped if group not in assignments),
        key=lambda group: stable_unit(
            f"remaining:{config['split_seed']}:{group}"
        ),
    )
    for group in remaining:
        split = max(
            split_names,
            key=lambda name: (
                (targets[name] - assigned_samples[name]) / max(1.0, targets[name]),
                -split_names.index(name),
            ),
        )
        assignments[group] = split
        assigned_samples[split] += len(grouped[group])
    for row in rows:
        row["split"] = assignments[str(row["split_group"])]

    actions_by_split = {
        split: Counter(
            row["label"]["action"] for row in rows if row["split"] == split
        )
        for split in split_names
    }
    risks_by_split = {
        split: Counter(
            row["risk_level"] for row in rows if row["split"] == split
        )
        for split in split_names
    }
    required_actions = {row["label"]["action"] for row in rows}
    for split in split_names:
        missing_actions = required_actions - set(actions_by_split[split])
        if missing_actions or risks_by_split[split]["high"] == 0:
            raise ValueError(
                f"stratified split {split} lacks actions={sorted(missing_actions)} "
                f"or high-risk samples"
            )
    return {
        "strategy": "500m_route_group_stratified_with_critical_reservation",
        "groups_by_split": dict(Counter(assignments.values())),
        "samples_by_split": dict(assigned_samples),
        "actions_by_split": {
            split: dict(values) for split, values in actions_by_split.items()
        },
        "risks_by_split": {
            split: dict(values) for split, values in risks_by_split.items()
        },
    }


def encode_intents(
    templates: dict[str, tuple[str, str]], parser_model: Path, device: str
) -> dict[str, dict[str, torch.Tensor]]:
    service = ModernBertCommandService(str(parser_model), device=device)
    service.parser.load()
    parser = service.parser
    result = {}
    for name, (_, text_en) in templates.items():
        encoded = parser.tokenizer(
            text_en,
            return_tensors="pt",
            truncation=True,
            max_length=parser.max_length,
        )
        encoded = {key: value.to(parser.device) for key, value in encoded.items()}
        with torch.inference_mode():
            tokens = parser.model.backbone(**encoded).last_hidden_state
        result[name] = {
            "intent_tokens": tokens[0].detach().float().cpu(),
            "intent_mask": encoded["attention_mask"][0].detach().bool().cpu(),
        }
    return result


def environment_features(
    runtime: dict, profile: str, control_speed_cap_kmh: float
) -> torch.Tensor:
    if profile not in WEATHER_OVERRIDES:
        raise ValueError(f"unknown weather profile: {profile}")
    weather = dict(runtime["weather"])
    weather.update(WEATHER_OVERRIDES[profile])
    road_limit = float(
        runtime.get("surface_and_visibility", {}).get(
            "wet_speed_limit_kmh", control_speed_cap_kmh
        )
    )
    if profile == "clear-daylight":
        road_limit = max(road_limit, control_speed_cap_kmh)
    return torch.tensor(
        [
            float(weather["cloudiness"]) / 100.0,
            float(weather["precipitation"]) / 100.0,
            float(weather["precipitation_deposits"]) / 100.0,
            float(weather["wind_intensity"]) / 100.0,
            float(weather.get("sun_azimuth_angle", 280.0)) / 360.0,
            max(-1.0, min(1.0, float(weather["sun_altitude_angle"]) / 90.0)),
            float(weather["fog_density"]) / 100.0,
            float(weather["fog_distance"]) / 1000.0,
            0.02,
            float(weather["wetness"]) / 100.0,
            0.10,
            0.80,
            road_limit / 100.0,
            float(control_speed_cap_kmh) / 100.0,
        ],
        dtype=torch.float32,
    )


def ego_features(
    vehicle: dict, truth: dict, control_speed_cap_kmh: float
) -> torch.Tensor:
    acceleration = vehicle.get("acceleration_mps2", {})
    angular = vehicle.get("angular_velocity_deg_s", {})
    control = vehicle.get("control", {})
    lane = truth.get("ego", {}).get("lane", {})
    acceleration_xy = math.hypot(
        float(acceleration.get("x", 0.0)),
        float(acceleration.get("y", 0.0)),
    )
    return torch.tensor(
        [
            float(vehicle.get("speed_kmh", 0.0)) / 3.6,
            acceleration_xy,
            math.radians(float(angular.get("z", 0.0))),
            float(control.get("steer", 0.0)),
            float(control.get("throttle", 0.0)),
            float(control.get("brake", 0.0)),
            float(control_speed_cap_kmh) / 3.6,
            float(bool(lane.get("is_junction", False))),
        ],
        dtype=torch.float32,
    )


def visual_risk(truth: dict) -> tuple[str, list[str]]:
    ego_speed_mps = float(truth.get("ego", {}).get("speed_kmh", 0.0)) / 3.6
    high = False
    medium = False
    reasons: list[str] = []
    for role, actor in truth.get("actors", {}).items():
        relation = actor.get("relation_to_ego", {})
        longitudinal = float(relation.get("longitudinal_m", math.inf))
        lateral = abs(float(relation.get("lateral_m", math.inf)))
        distance = float(relation.get("euclidean_distance_m", math.inf))
        ttc_value = relation.get("time_to_collision_s")
        ttc = float(ttc_value) if isinstance(ttc_value, (int, float)) else math.inf
        if longitudinal > 0.0 and lateral < 4.5:
            emergency_distance = 7.5 + 0.8 * ego_speed_mps
            caution_distance = max(16.0, emergency_distance + 1.5 * ego_speed_mps)
            if distance <= emergency_distance or ttc < 1.5:
                high = True
                reasons.append(f"imminent:{role}")
            elif distance < caution_distance or ttc < 3.0:
                medium = True
                reasons.append(f"caution:{role}")
    for event in truth.get("active_events", []):
        event_id = str(event.get("event_id"))
        phase = str(event.get("runtime_phase", ""))
        if event_id == "scene3_temporary_pedestrian" and phase in {
            "BRAKING", "CROSSING", "ACTIVE"
        }:
            high = True
            reasons.append(f"event:{event_id}:{phase}")
        elif event_id in {
            "scene3_cut_in",
            "scene3_cone_taper",
            "scene3_work_zone",
            "scene3_temporary_pedestrian",
            "scene3_blocked_lane",
        }:
            medium = True
            reasons.append(f"event:{event_id}:{phase}")
    return ("high" if high else "medium" if medium else "low"), sorted(set(reasons))


def base_label(command_id: str, risk: str, cap: float) -> dict[str, Any]:
    profile = COMMAND_PROFILES[command_id]
    action = str(profile["action"])
    if risk == "high" and action not in {"stop", "emergency_brake"}:
        action = "emergency_brake"
    elif risk == "medium" and action == "accelerate":
        action = "decelerate"
    speed = min(float(profile["target_speed_kmh"]), cap)
    if action == "emergency_brake":
        speed = 0.0
    elif action == "decelerate":
        speed = min(speed, 18.0)
    return {
        "action": action,
        "target_speed_kmh": speed,
        "target_lane": (
            action.removeprefix("lane_change_")
            if action.startswith("lane_change_")
            else None
        ),
    }


def label_for_variant(name: str, risk: str, cap: float) -> dict[str, Any]:
    action = name
    speed = cap
    lane = None
    if name == "visual_safe":
        action = {
            "low": "keep_lane",
            "medium": "decelerate",
            "high": "emergency_brake",
        }[risk]
    elif risk == "high" and action not in {"stop", "emergency_brake"}:
        action = "emergency_brake"
    elif risk == "medium" and action == "accelerate":
        action = "decelerate"
    if action == "accelerate":
        speed = cap
    elif action == "decelerate":
        speed = min(18.0 if risk != "low" else 25.0, cap)
    elif action in {"stop", "emergency_brake"}:
        speed = 0.0
    elif action.startswith("lane_change_"):
        speed = min(25.0, cap)
        lane = action.removeprefix("lane_change_")
    elif action.startswith("turn_"):
        speed = min(20.0, cap)
    return {"action": action, "target_speed_kmh": speed, "target_lane": lane}


def main() -> None:
    args = parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    runtime_path = REPO_ROOT / spec["runtime_config"]
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    commands = runtime["voice_input"]["commands"]
    output = args.output_dir.resolve()
    if (output / "manifest.jsonl").exists():
        raise FileExistsError(f"dataset already exists: {output}")
    (output / "tensors").mkdir(parents=True, exist_ok=True)
    (output / "intents").mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(parents=True, exist_ok=True)

    command_templates = {
        command_id: (
            next(
                (
                    str(item["text"])
                    for item in commands
                    if str(item["id"]) == command_id
                ),
                "继续沿当前车道安全行驶",
            ),
            str(profile["text_en"]),
        )
        for command_id, profile in COMMAND_PROFILES.items()
    }
    all_templates = {**command_templates, **TEMPLATES}
    intents = encode_intents(all_templates, args.parser_model, args.device)
    intent_paths = {}
    for name, tensors in intents.items():
        path = output / "intents" / f"{name}.pt"
        torch.save(tensors, path)
        intent_paths[name] = path

    manifest_rows = []
    skipped = Counter()
    truth_frame_deltas = Counter()
    group_size = float(spec.get("group_size_m", 500.0))
    for capture_spec in spec["captures"]:
        trajectory_id = str(capture_spec["trajectory_id"])
        capture = (REPO_ROOT / capture_spec["capture_dir"]).resolve()
        summary_path = capture / "scene_summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else None
        )
        capture_complete = bool(
            summary is not None and summary.get("complete_scene_success", False)
        )
        if not capture_complete and not capture_spec.get("allow_incomplete", False):
            raise ValueError(
                f"capture is not a strict complete scene: {capture_spec['capture_dir']}"
            )
        capture_quality = (
            "strict_complete" if capture_complete else "labeled_hard_negative"
        )
        cap = float(capture_spec["control_speed_cap_kmh"])
        environment = environment_features(
            runtime, str(capture_spec["weather_profile"]), cap
        )
        truth_rows = rows_by_frame(capture / "frame_ground_truth.jsonl")
        vehicle_rows = rows_by_frame(capture / "vehicle_state.jsonl")
        front_images = sorted((capture / "rgb" / "front_rgb").glob("*.png"))
        for front_image in front_images:
            frame = int(front_image.stem)
            images = [
                capture / "rgb" / CAMERA_DIRS[name] / front_image.name
                for name in CAMERA_ORDER
            ]
            if not all(path.is_file() for path in images):
                skipped["missing_view"] += 1
                continue
            # Sensors tick at 1 Hz while the recorder samples every 20 world
            # frames. Their phases may differ by up to half a sampling period.
            # Keep the actual offset for audit rather than claiming exact sync.
            truth, truth_frame_delta = nearest_with_delta(
                truth_rows,
                frame,
                int(spec.get("truth_alignment_tolerance_frames", 12)),
            )
            vehicle, vehicle_frame_delta = nearest_with_delta(
                vehicle_rows, frame, 2
            )
            if truth is None or vehicle is None:
                skipped["missing_state"] += 1
                continue
            truth_frame_deltas[int(truth_frame_delta)] += 1
            route_s_m = float(truth["route_s_m"])
            command = active_text_command(commands, route_s_m)
            command_id = str(command["id"])
            risk, risk_reasons = visual_risk(truth)
            group = (
                f"{trajectory_id}:route_"
                f"{int(route_s_m // group_size):03d}"
            )
            split = split_for_group(group, spec)
            state_relative = Path("tensors") / trajectory_id / f"{frame:08d}.pt"
            state_path = output / state_relative
            state_path.parent.mkdir(parents=True, exist_ok=True)
            def sensor_state(speed_cap_kmh: float, environment_tensor: torch.Tensor):
                return {
                    # Structured BEV is disabled in the v3 configuration; use
                    # one-cell placeholders rather than duplicating hundreds
                    # of megabytes of zeros across source frames.
                    "camera_bev": torch.zeros(8, 1, 1),
                    "lidar_bev": torch.zeros(4, 1, 1),
                    "ego_features": ego_features(vehicle, truth, speed_cap_kmh),
                    "candidate_features": torch.zeros(32, 12),
                    "candidate_mask": torch.zeros(32, dtype=torch.bool),
                    "environment_features": environment_tensor,
                }

            torch.save(sensor_state(cap, environment), state_path)
            alternate_caps = [value for value in (25.0, 30.0, 40.0) if value != cap]
            alternate_cap = alternate_caps[
                stable_index(
                    f"environment:{trajectory_id}:{frame}", len(alternate_caps)
                )
            ]
            alternate_state_relative = (
                Path("tensors") / trajectory_id / f"{frame:08d}_environment_cf.pt"
            )
            torch.save(
                sensor_state(
                    alternate_cap,
                    environment_features(
                        runtime, str(capture_spec["weather_profile"]), alternate_cap
                    ),
                ),
                output / alternate_state_relative,
            )
            relative_images = [
                Path(os.path.relpath(path, output)).as_posix() for path in images
            ]
            image_tensor_relative = (
                Path("images") / trajectory_id / f"{frame:08d}.pt"
            )
            image_tensor_path = output / image_tensor_relative
            image_tensor_path.parent.mkdir(parents=True, exist_ok=True)
            decoded_images = torch.stack(
                [read_image(str(path))[:3] for path in images]
            ).float()
            resized_images = F.interpolate(
                decoded_images,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).round().clamp_(0.0, 255.0).to(torch.uint8)
            torch.save(resized_images, image_tensor_path)
            counterfactual_set_id = f"{trajectory_id}:{frame:08d}"

            def append_variant(
                variant: str,
                label: dict[str, Any],
                *,
                intent_name: str,
                source_text: str,
                normalized_text: str,
                tensor_relative: Path = state_relative,
                label_speed_cap_kmh: float = cap,
            ) -> None:
                manifest_rows.append(
                    {
                        "schema_version": "scene3_multimodal_training_sample/2.0",
                        "sample_id": f"{trajectory_id}_{frame:08d}_{variant}",
                        "trajectory_id": trajectory_id,
                        "source_frame": frame,
                        "truth_frame_delta": int(truth_frame_delta),
                        "vehicle_frame_delta": int(vehicle_frame_delta),
                        "route_s_m": round(route_s_m, 3),
                        "split_group": group,
                        "split": split,
                        "counterfactual_set_id": counterfactual_set_id,
                        "variant_type": variant,
                        "command_id": intent_name,
                        "source_text": source_text,
                        "normalized_text": normalized_text,
                        "camera_order": list(CAMERA_ORDER),
                        "image_paths": relative_images,
                        "image_tensor_path": image_tensor_relative.as_posix(),
                        "tensor_path": tensor_relative.as_posix(),
                        "intent_tensor_path": Path(
                            os.path.relpath(intent_paths[intent_name], output)
                        ).as_posix(),
                        "label": label,
                        "risk_level": risk,
                        "risk_reason_codes": risk_reasons,
                        "weather_profile": capture_spec["weather_profile"],
                        "control_speed_cap_kmh": label_speed_cap_kmh,
                        "capture_quality": capture_quality,
                    }
                )

            source_text, normalized = command_templates[command_id]
            append_variant(
                "observed_command",
                base_label(command_id, risk, cap),
                intent_name=command_id,
                source_text=source_text,
                normalized_text=normalized,
            )
            visual_text, visual_en = TEMPLATES["visual_safe"]
            append_variant(
                "visual_counterfactual",
                label_for_variant("visual_safe", risk, cap),
                intent_name="visual_safe",
                source_text=visual_text,
                normalized_text=visual_en,
            )
            basic = BASIC_VARIANTS[
                stable_index(counterfactual_set_id, len(BASIC_VARIANTS))
            ]
            basic_text, basic_en = TEMPLATES[basic]
            append_variant(
                "instruction_counterfactual",
                label_for_variant(basic, risk, cap),
                intent_name=basic,
                source_text=basic_text,
                normalized_text=basic_en,
            )
            if risk == "low":
                accelerate_text, accelerate_en = TEMPLATES["accelerate"]
                append_variant(
                    "environment_pair_observed",
                    label_for_variant("accelerate", risk, cap),
                    intent_name="accelerate",
                    source_text=accelerate_text,
                    normalized_text=accelerate_en,
                )
                append_variant(
                    "environment_pair_counterfactual",
                    label_for_variant("accelerate", risk, alternate_cap),
                    intent_name="accelerate",
                    source_text=accelerate_text,
                    normalized_text=accelerate_en,
                    tensor_relative=alternate_state_relative,
                    label_speed_cap_kmh=alternate_cap,
                )
            if risk == "high":
                emergency_text, emergency_en = TEMPLATES["emergency_brake"]
                append_variant(
                    "high_risk_emergency",
                    label_for_variant("emergency_brake", risk, cap),
                    intent_name="emergency_brake",
                    source_text=emergency_text,
                    normalized_text=emergency_en,
                )

    split_report = assign_grouped_stratified_splits(manifest_rows, spec)
    manifest_rows.sort(
        key=lambda row: (
            row["trajectory_id"], row["source_frame"], row["variant_type"]
        )
    )
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "scene3_counterfactual_dataset/1.0",
        "spec": str(spec_path),
        "samples": len(manifest_rows),
        "source_frames": len(
            {row["counterfactual_set_id"] for row in manifest_rows}
        ),
        "trajectories": dict(Counter(row["trajectory_id"] for row in manifest_rows)),
        "splits": dict(Counter(row["split"] for row in manifest_rows)),
        "actions": dict(Counter(row["label"]["action"] for row in manifest_rows)),
        "risks": dict(Counter(row["risk_level"] for row in manifest_rows)),
        "variants": dict(Counter(row["variant_type"] for row in manifest_rows)),
        "weather_profiles": dict(
            Counter(row["weather_profile"] for row in manifest_rows)
        ),
        "capture_quality": dict(
            Counter(row["capture_quality"] for row in manifest_rows)
        ),
        "skipped": dict(skipped),
        "truth_frame_deltas": {
            str(delta): count for delta, count in sorted(truth_frame_deltas.items())
        },
        "image_paths_are_relative": True,
        "split_is_grouped": True,
        "split_strategy": split_report,
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
