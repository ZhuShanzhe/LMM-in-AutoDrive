from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CARLA_DIR = REPO_ROOT / "experiment" / "CARLA"
if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

from scene3_vla_controller import COMMAND_PROFILES, active_text_command
from structured_command_parser.src.modernbert_service import ModernBertCommandService


CAMERA_ORDER = ("front", "left", "right", "rear")
CAPTURE_CAMERA_DIRS = {
    "front": "front_rgb",
    "left": "left_rgb",
    "right": "right_rgb",
    "rear": "rear_rgb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a complete four-view Scene 3 capture to VLA samples"
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--parser-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_jsonl_by_frame(path: Path, wanted: set[int]) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            frame = int(row["simulation_frame"])
            if frame in wanted:
                rows[frame] = row
    return rows


def encode_intents(
    commands: list[dict], parser_model: Path, device: str
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    service = ModernBertCommandService(str(parser_model), device=device)
    service.parser.load()
    parser = service.parser
    result = {}
    command_ids = {"scene3_cruise", *(str(item["id"]) for item in commands)}
    for command_id in sorted(command_ids):
        text = str(COMMAND_PROFILES[command_id]["text_en"])
        encoded = parser.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=parser.max_length,
        )
        encoded = {key: value.to(parser.device) for key, value in encoded.items()}
        with torch.inference_mode():
            tokens = parser.model.backbone(**encoded).last_hidden_state
        result[command_id] = (
            tokens[0].detach().float().cpu(),
            encoded["attention_mask"][0].detach().bool().cpu(),
        )
    return result


def risk_label(truth: dict) -> str:
    active = truth.get("active_events") or []
    event_ids = {str(item.get("event_id")) for item in active}
    phases = {str(item.get("runtime_phase")) for item in active}
    if "scene3_temporary_pedestrian" in event_ids and phases & {
        "CROSSING", "BRAKING", "ACTIVE"
    }:
        return "high"
    if event_ids & {
        "scene3_cut_in",
        "scene3_cone_taper",
        "scene3_work_zone",
        "scene3_temporary_pedestrian",
        "scene3_blocked_lane",
    }:
        return "medium"
    return "low"


def ego_features(vehicle: dict, truth: dict) -> torch.Tensor:
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
            32.0 / 3.6,
            float(bool(lane.get("is_junction", False))),
        ],
        dtype=torch.float32,
    )


def main() -> None:
    args = parse_args()
    capture = args.capture_dir.resolve()
    output = args.output_dir.resolve()
    tensor_dir = output / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    front_images = sorted((capture / "rgb" / "front_rgb").glob("*.png"))
    frames = {int(path.stem) for path in front_images}
    if not frames:
        raise ValueError("capture contains no front_rgb PNG frames")
    truth_rows = read_jsonl_by_frame(capture / "frame_ground_truth.jsonl", frames)
    vehicle_rows = read_jsonl_by_frame(capture / "vehicle_state.jsonl", frames)
    runtime = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    commands = runtime["voice_input"]["commands"]
    intents = encode_intents(commands, args.parser_model, args.device)
    environment = torch.tensor(
        [0.80, 0.80, 0.85, 0.35, 0.0, -0.18, 0.35, 0.02, 0.05, 0.90, 0.0, 0.0],
        dtype=torch.float32,
    )
    written = 0
    skipped = []
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for image_path in front_images:
            frame = int(image_path.stem)
            truth = truth_rows.get(frame)
            vehicle = vehicle_rows.get(frame)
            if truth is None or vehicle is None:
                skipped.append(frame)
                continue
            route_s_m = float(truth["route_s_m"])
            command = active_text_command(commands, route_s_m)
            command_id = str(command["id"])
            profile = COMMAND_PROFILES[command_id]
            tokens, mask = intents[command_id]
            tensor_relative = Path("tensors") / f"frame_{frame:08d}.pt"
            torch.save(
                {
                    "camera_bev": torch.zeros(8, 64, 64),
                    "lidar_bev": torch.zeros(4, 64, 64),
                    "ego_features": ego_features(vehicle, truth),
                    "candidate_features": torch.zeros(32, 12),
                    "candidate_mask": torch.zeros(32, dtype=torch.bool),
                    "intent_tokens": tokens,
                    "intent_mask": mask,
                    "environment_features": environment,
                },
                output / tensor_relative,
            )
            image_paths = [
                (Path("..") / "rgb" / CAPTURE_CAMERA_DIRS[name] / image_path.name)
                .as_posix()
                for name in CAMERA_ORDER
            ]
            row = {
                "schema_version": "scene3_multimodal_training_sample/1.0",
                "sample_id": f"historic_{frame:08d}",
                "frame": frame,
                "route_s_m": round(route_s_m, 3),
                "command_id": command_id,
                "source_text": command.get("text", "继续沿当前车道安全行驶"),
                "camera_order": list(CAMERA_ORDER),
                "image_paths": image_paths,
                "tensor_path": tensor_relative.as_posix(),
                "label": {
                    "action": profile["action"],
                    "target_speed_kmh": float(profile["target_speed_kmh"]),
                    "target_lane": (
                        profile["action"].removeprefix("lane_change_")
                        if str(profile["action"]).startswith("lane_change_")
                        else None
                    ),
                },
                "risk_level": risk_label(truth),
                "risk_reason_codes": [
                    str(item.get("event_id"))
                    for item in truth.get("active_events", [])
                ],
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    report = {
        "schema_version": "scene3_capture_conversion/1.0",
        "capture_dir": str(capture),
        "samples_written": written,
        "frames_skipped": skipped,
        "camera_order": list(CAMERA_ORDER),
        "image_paths_are_relative": True,
    }
    (output / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
