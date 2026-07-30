"""Run one real-model Scene 2 frame without starting CARLA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
CARLA_ROOT = ROOT / "experiment" / "CARLA"
for path in (ROOT, CARLA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scene2_closed_loop import Scene2ClosedLoop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = Scene2ClosedLoop(
        intents_path=CARLA_ROOT
        / "configs"
        / "scene_2_expected_driving_intents.json",
        intent_token_cache=CARLA_ROOT / "outputs" / "scene2_intent_tokens.pt",
        yolop_root=args.models / "external" / "YOLOP",
        yolo11_weights=args.models
        / "scene_understanding"
        / "yolo11s_specialized_carla_v1"
        / "weights"
        / "best.pt",
        vla_dir=args.models / "lightweight_vla_adapter" / "v10",
    )
    runtime.activate("s2_t05_cmd_01")
    world_state = json.loads(
        (
            ROOT
            / "scene_understanding"
            / "schemas"
            / "examples"
            / "world_state.example.json"
        ).read_text(encoding="utf-8")
    )
    result = runtime.process(
        world_state=world_state,
        image=Image.new("RGB", (640, 640)),
        route_progress_m=0.0,
        route_length_m=8000.0,
    )
    compact = {
        "request_id": result["driving_intent"]["request_id"],
        "perception_tracks": len(result["perception_frame"]["tracks"]),
        "alignment_status": result["semantic_alignment"]["alignment_status"],
        "risk_level": result["risk_assessment"]["risk_level"],
        "vla_proposal": result["vla_proposal"],
        "control_decision": result["control_decision"],
        "plan_status": result["control_plan_state"]["plan_status"],
        "latency_ms": result["latency_ms"],
        "provenance": result["provenance"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
