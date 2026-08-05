#!/usr/bin/env python3
"""Audit a Scene-3 VLA checkpoint against the 15 Scene-2 commands.

This is an adaptation diagnostic, not a closed-loop competition score.  It
uses one exact-frame four-view Scene-2 capture and the recorded vehicle/weather
state while varying the scheduled raw Chinese command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.contracts import SensorTensorBatch
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
from structured_command_parser.src.modernbert_service import ModernBertCommandService


EXPECTED_FIRST_ACTIONS = {
    "s2_t05_cmd_01": {"decelerate"},
    "s2_t05_cmd_02": {"keep_lane", "accelerate"},
    "s2_t05_cmd_03": {"decelerate", "stop"},
    "s2_t05_cmd_04": {"lane_change_right", "stop"},
    "s2_t05_cmd_05": {"lane_change_right", "decelerate"},
    "s2_t05_cmd_06": {"stop", "keep_lane"},
    "s2_t05_cmd_07": {"decelerate"},
    "s2_t05_cmd_08": {"lane_change_right", "keep_lane"},
    "s2_t05_cmd_09": {"keep_lane", "lane_change_left"},
    "s2_t05_cmd_10": {"keep_lane", "accelerate"},
    "s2_t05_cmd_11": {"decelerate"},
    "s2_t05_cmd_12": {"lane_change_left", "stop"},
    "s2_t05_cmd_13": {"keep_lane"},
    "s2_t05_cmd_14": {"decelerate"},
    "s2_t05_cmd_15": {"keep_lane", "stop"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene2-config",
        type=Path,
        default=Path("experiment/CARLA/configs/scene_2_town05_runtime.json"),
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path(
            "experiment/CARLA/outputs/"
            "scene2_town05_variant1_multimodal_20260805"
        ),
    )
    parser.add_argument("--frame", type=int, default=46906)
    parser.add_argument(
        "--vla-config",
        type=Path,
        default=Path("lightweight_vla_adapter/configs/scene3_multimodal_v3.json"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parser-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_frame(capture_dir: Path, frame: int, size: tuple[int, int]) -> torch.Tensor:
    views = []
    for name in ("front_rgb", "left_rgb", "right_rgb", "rear_rgb"):
        path = capture_dir / "rgb" / name / f"{frame:08d}.png"
        image = read_image(str(path))[:3]
        views.append(resize(image, list(size), antialias=True))
    return torch.stack(views).unsqueeze(0)


def load_world_row(path: Path, frame: int) -> dict:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["simulation_frame"]) == frame:
                return row
    raise ValueError(f"frame {frame} missing from {path}")


def environment_tensor(config: dict, world_row: dict) -> torch.Tensor:
    weather = config["weather"]
    values = [
        float(weather["cloudiness"]) / 100.0,
        float(weather["precipitation"]) / 100.0,
        float(weather["precipitation_deposits"]) / 100.0,
        float(weather["wind_intensity"]) / 100.0,
        float(weather["sun_azimuth_angle"]) / 360.0,
        float(weather["sun_altitude_angle"]) / 90.0,
        float(weather["fog_density"]) / 100.0,
        float(weather["fog_distance"]) / 1000.0,
        float(weather["fog_falloff"]) / 10.0,
        float(weather["wetness"]) / 100.0,
        0.0,
        0.0,
        0.50,
        float(config["route"]["target_speed_kmh"]) / 100.0,
    ]
    assert len(values) == 14
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def main() -> None:
    args = parse_args()
    scene2 = json.loads(args.scene2_config.read_text(encoding="utf-8"))
    model_config = json.loads(args.vla_config.read_text(encoding="utf-8"))
    world_row = load_world_row(args.capture_dir / "world_state.jsonl", args.frame)
    camera_images = load_frame(
        args.capture_dir,
        args.frame,
        (int(model_config["camera_input_height"]), int(model_config["camera_input_width"])),
    )
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.precision
    ]
    pipeline = LightweightVLAPipeline.from_checkpoint(
        build_model(model_config),
        str(args.checkpoint),
        model_name=model_config["model_name"],
        device=args.device,
        dtype=dtype,
        strict_checkpoint=not bool(model_config.get("allow_legacy_checkpoint", False)),
    )
    parser_service = ModernBertCommandService(str(args.parser_model), device=args.device)
    parser_service.warmup()
    parser = parser_service.parser
    parser.load()

    rows = []
    for command in scene2["commands"]:
        text = str(command["text"])
        parsed = parser_service.parse_text(
            text,
            request_id=f"scene2-audit-{command['id']}",
            modality="TEXT",
            source_text=text,
            source_language="zh-CN",
        )
        encoded = parser.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=parser.max_length,
        )
        encoded = {name: tensor.to(parser.device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            tokens = parser.model.backbone(**encoded).last_hidden_state.detach().float().cpu()
        speed_mps = float(world_row["ego"]["speed_kmh"]) / 3.6
        ego_features = torch.tensor(
            [[speed_mps, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0 / 3.6, 0.0]],
            dtype=torch.float32,
        )
        batch = SensorTensorBatch(
            camera_bev=torch.zeros((1, 8, 8, 8)),
            lidar_bev=torch.zeros((1, 4, 8, 8)),
            ego_features=ego_features,
            candidate_features=torch.zeros((1, 32, 12)),
            candidate_mask=torch.zeros((1, 32), dtype=torch.bool),
            intent_tokens=tokens,
            intent_mask=encoded["attention_mask"].detach().bool().cpu(),
            camera_images=camera_images,
            camera_view_mask=torch.ones((1, 4), dtype=torch.bool),
            environment_features=environment_tensor(scene2, world_row),
        )
        risk = {
            "risk_level": "low",
            "recommended_action": "keep_lane",
            "reason_codes": [],
            "matched_entity_id": None,
            "lane_change": {
                "left": {"is_safe": True, "reason_codes": []},
                "right": {"is_safe": True, "reason_codes": []},
            },
        }
        started = time.perf_counter()
        proposal = pipeline.predict_proposal(
            batch,
            request_id=f"scene2-audit-{command['id']}",
            frame_id=f"scene2_{args.frame}",
            candidate_entity_ids=[[]],
            world_state={"objects": [], "ego": {"speed_mps": speed_mps}},
            risk_assessment=risk,
            stream_id=str(command["id"]),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        expected = sorted(EXPECTED_FIRST_ACTIONS[str(command["id"])])
        rows.append(
            {
                "command_id": command["id"],
                "text": text,
                "steps": command["steps"],
                "parser_status": parsed.get("parse_result", {}).get("status"),
                "parser_confidence": parsed.get("parse_result", {}).get("confidence"),
                "expected_first_actions": expected,
                "proposal": proposal,
                "first_action_compatible": proposal["action"] in expected,
                "inference_latency_ms": round(latency_ms, 3),
            }
        )
    compatible = sum(row["first_action_compatible"] for row in rows)
    payload = {
        "schema_version": "scene2_vla_adaptation_audit/1.0",
        "scope": "offline single-frame adaptation diagnostic; not a competition score",
        "frame": args.frame,
        "input_mode": "raw Chinese text + exact-frame four-view RGB + vehicle/weather state",
        "checkpoint": str(args.checkpoint),
        "command_count": len(rows),
        "first_action_compatible_count": compatible,
        "first_action_compatible_rate": compatible / len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
