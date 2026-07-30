from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "models" / "lightweight_vla_adapter" / "v10" / "model.pt"
)
DEFAULT_LANGUAGE_MODEL = (
    REPO_ROOT / "models" / "modernbert-drive-command-compositional"
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene_understanding.src.driving_intent_alignment import align_driving_intent
from scene_understanding.src.risk_interface import assess_scene_risk
from structured_command_parser import ModernBertCommandService

from lightweight_vla_adapter.scripts.train_student import build_model
from lightweight_vla_adapter.src.intent_encoder import ModernBertIntentEncoder
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a VLA checkpoint on synchronized CARLA captures"
    )
    parser.add_argument("--capture-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--language-model", default=str(DEFAULT_LANGUAGE_MODEL))
    parser.add_argument(
        "--instruction",
        default="Keep the current lane.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    command_parser = ModernBertCommandService(
        args.language_model,
        device=args.device,
    )
    command_parser.warmup()
    driving_intent = command_parser.parse_text(
        args.instruction,
        request_id="carla-capture-vla",
        modality="TEXT",
    )
    for step in driving_intent["intent"]["steps"]:
        if step["action"] in {"KEEP_LANE", "PROCEED", "FOLLOW"}:
            step["on_blocked"] = "WAIT_FOR_SAFE"
    del command_parser
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    encoder = ModernBertIntentEncoder.from_pretrained(
        args.language_model,
        output_hidden_size=config["intent_dim"],
        freeze_backbone=True,
        dtype=dtype,
    ).to(device=device, dtype=dtype).eval()
    encoded = tokenizer(
        driving_intent["input"]["normalized_text"],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        intent_tokens, intent_mask = encoder(**encoded)

    pipeline = LightweightVLAPipeline.from_checkpoint(
        build_model(config),
        args.checkpoint,
        model_name=config["model_name"],
        device=device,
        dtype=dtype,
    )
    rasterizer = StructuredBEVRasterizer(
        height=64,
        width=64,
        max_candidates=config["max_candidates"],
    )
    records = [
        json.loads(line)
        for line in Path(args.capture_index).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("capture index is empty")

    details = []
    proposal_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    emergency_frames = 0
    emergency_guarded = 0
    model_emergency = 0
    latencies = []
    prior_state = None
    for index, record in enumerate(records):
        world_state = json.loads(
            Path(record["world_state_path"]).read_text(encoding="utf-8")
        )
        risk = assess_scene_risk(world_state)
        alignment = align_driving_intent(driving_intent, world_state)
        batch, entity_ids = rasterizer.build(
            world_state,
            intent_tokens=intent_tokens.detach().cpu(),
            intent_mask=intent_mask.detach().cpu(),
        )
        if index == 0:
            pipeline.warmup(batch, iterations=30)
        proposal, prior_state, final_decision = pipeline.decide(
            batch,
            driving_intent,
            world_state,
            alignment,
            risk,
            candidate_entity_ids=entity_ids,
            prior_state=prior_state,
        )
        recommended = risk["recommended_action"]
        is_emergency = recommended == "emergency_brake"
        if is_emergency:
            emergency_frames += 1
            model_emergency += int(proposal["action"] == "emergency_brake")
            emergency_guarded += int(
                final_decision["action"] == "emergency_brake"
            )
        proposal_counts[proposal["action"]] += 1
        final_counts[final_decision["action"]] += 1
        reason_counts[final_decision["reason"]] += 1
        latencies.append(float(proposal["latency_ms"]))
        details.append(
            {
                "frame_id": world_state["frame_id"],
                "risk_level": risk["risk_level"],
                "recommended_action": recommended,
                "proposal": proposal,
                "final_decision": final_decision,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    details_path = output.with_suffix(".frames.jsonl")
    details_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n" for item in details
        ),
        encoding="utf-8",
    )
    result = {
        "frames": len(details),
        "instruction": args.instruction,
        "parse_status": driving_intent["parse_result"]["status"],
        "parsed_actions": [
            step["action"] for step in driving_intent["intent"]["steps"]
        ],
        "on_blocked_policies": [
            step["on_blocked"] for step in driving_intent["intent"]["steps"]
        ],
        "proposal_action_counts": dict(proposal_counts),
        "final_action_counts": dict(final_counts),
        "final_reason_counts": dict(reason_counts),
        "emergency_risk_frames": emergency_frames,
        "model_emergency_recall": (
            None if emergency_frames == 0 else model_emergency / emergency_frames
        ),
        "safety_gate_emergency_recall": (
            None if emergency_frames == 0 else emergency_guarded / emergency_frames
        ),
        "adapter_latency_ms": {
            "mean": statistics.fmean(latencies),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "checkpoint": args.checkpoint,
        "capture_index": args.capture_index,
        "details": str(details_path),
    }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
