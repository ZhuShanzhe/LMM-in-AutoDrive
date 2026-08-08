from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightweight_vla_adapter.src.contracts import SensorTensorBatch
from lightweight_vla_adapter.src.decision_adapter import LightweightDecisionAdapter
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one offline VLA adapter sample")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--tensor-batch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    return parser.parse_args()


def build_model(config: dict) -> LightweightDecisionAdapter:
    return LightweightDecisionAdapter(
        camera_channels=config["camera_channels"],
        lidar_channels=config["lidar_channels"],
        candidate_dim=config["candidate_dim"],
        ego_dim=config["ego_dim"],
        intent_dim=config["intent_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        bev_grid=tuple(config["bev_grid"]),
        environment_dim=int(config.get("environment_dim", 12)),
        num_camera_views=int(config.get("num_camera_views", 4)),
        raw_camera_token_grid=tuple(
            config.get("raw_camera_token_grid", (2, 2))
        ),
        require_raw_camera=bool(config.get("require_raw_camera", False)),
        use_raw_camera=bool(config.get("use_raw_camera", True)),
        use_environment=bool(config.get("use_environment", True)),
        use_candidate_entities=bool(
            config.get("use_candidate_entities", True)
        ),
        use_structured_bev=bool(config.get("use_structured_bev", True)),
        fuse_structured_visual_risk=bool(
            config.get("fuse_structured_visual_risk", False)
        ),
        condition_decision_on_visual_risk=bool(
            config.get("condition_decision_on_visual_risk", False)
        ),
        speed_cap_environment_index=config.get("speed_cap_environment_index"),
        use_temporal_risk=bool(config.get("use_temporal_risk", False)),
        risk_type_count=int(config.get("risk_type_count", 6)),
    )


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with Path(args.request_json).open("r", encoding="utf-8") as handle:
        request = json.load(handle)
    tensors = torch.load(args.tensor_batch, map_location="cpu", weights_only=True)
    batch = SensorTensorBatch(
        camera_bev=tensors["camera_bev"],
        lidar_bev=tensors["lidar_bev"],
        ego_features=tensors["ego_features"],
        candidate_features=tensors["candidate_features"],
        candidate_mask=tensors["candidate_mask"],
        intent_tokens=tensors["intent_tokens"],
        intent_mask=tensors["intent_mask"],
        camera_images=tensors.get("camera_images"),
        camera_view_mask=tensors.get("camera_view_mask"),
        environment_features=tensors.get("environment_features"),
    )
    pipeline = LightweightVLAPipeline.from_checkpoint(
        build_model(config),
        args.checkpoint,
        model_name=config["model_name"],
        device=args.device,
        dtype={
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.precision],
    )
    proposal, plan_state, control_decision = pipeline.decide(
        batch,
        request["driving_intent"],
        request["world_state"],
        request["semantic_alignment"],
        request["risk_assessment"],
        candidate_entity_ids=request["candidate_entity_ids"],
        prior_state=request.get("prior_state"),
        feedback=request.get("feedback"),
    )
    output = {
        "vla_proposal": proposal,
        "control_plan_state": plan_state,
        "control_decision": control_decision,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
