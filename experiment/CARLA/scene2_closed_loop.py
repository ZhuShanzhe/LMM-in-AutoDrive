"""Reusable Scene 2 perception, alignment, VLA, FSM, and decision runtime."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from control.step_completion import StepCompletionEvaluator
from perception_fusion_adapter import fuse_perception_frame
from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.pipeline import LightweightVLAPipeline
from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer
from scene_understanding.core.risk_assessment import assess_world_state
from scene_understanding.realtime_perception.composite_backend import (
    CompositePanopticBackend,
)
from scene_understanding.realtime_perception.pipeline import (
    RealtimePerceptionPipeline,
)
from scene_understanding.realtime_perception.tracker import ByteTrackAdapter
from scene_understanding.realtime_perception.ultralytics_backend import (
    UltralyticsTrafficDetector,
)
from scene_understanding.realtime_perception.yolop_backend import (
    YolopPanopticBackend,
)
from scene_understanding.src.driving_intent_alignment import (
    align_driving_intent,
)


class Scene2ClosedLoop:
    """Own all learned and deterministic modules after DrivingIntent."""

    def __init__(
        self,
        *,
        intents_path: Path,
        intent_token_cache: Path,
        yolop_root: Path,
        yolo11_weights: Path,
        vla_dir: Path,
        device: str = "cuda",
        image_size: int = 640,
        perception_stride: int = 1,
    ) -> None:
        payload = json.loads(intents_path.read_text(encoding="utf-8"))
        self.intents = {
            item["request_id"]: item for item in payload["driving_intents"]
        }
        cache = torch.load(intent_token_cache, map_location="cpu", weights_only=True)
        self.intent_tokens = cache["records"]
        missing_tokens = sorted(set(self.intents) - set(self.intent_tokens))
        if missing_tokens:
            raise ValueError(f"intent token cache is missing {missing_tokens}")
        self.parser_latency_ms = dict(cache.get("latency_ms", {}))
        self.device = device
        self.perception_stride = max(1, int(perception_stride))
        self.frame_count = 0
        self.active_intent: dict[str, Any] | None = None
        self.plan_state: dict[str, Any] | None = None
        self.latest_perception: dict[str, Any] | None = None
        self.speed_setpoint_kmh: float | None = None
        self.completion = StepCompletionEvaluator()

        road_detector = YolopPanopticBackend(
            yolop_root,
            device=device,
            image_size=image_size,
        )
        object_detector = UltralyticsTrafficDetector(
            yolo11_weights,
            device=device,
            image_size=image_size,
        )
        self.perception = RealtimePerceptionPipeline(
            CompositePanopticBackend(road_detector, object_detector),
            ByteTrackAdapter(frame_rate=20),
        )
        config = json.loads(
            (vla_dir / "student_base.json").read_text(encoding="utf-8")
        )
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.vla = LightweightVLAPipeline.from_checkpoint(
            build_model(config),
            str(vla_dir / "model.pt"),
            model_name=config["model_name"],
            device=device,
            dtype=dtype,
        )
        self.rasterizer = StructuredBEVRasterizer(
            max_candidates=int(config["max_candidates"])
        )
        sample_record = next(iter(self.intent_tokens.values()))
        sample_batch, _ = self.rasterizer.build(
            {
                "objects": [],
                "ego": {},
                "environment": {},
            },
            intent_tokens=sample_record["tokens"],
            intent_mask=sample_record["mask"],
        )
        self.vla.warmup(sample_batch, iterations=3)

    def activate(self, request_id: str) -> dict[str, Any]:
        if request_id not in self.intents:
            raise KeyError(f"unknown Scene 2 request_id {request_id!r}")
        if self.active_intent is not None:
            self.vla.reset_temporal_state(self.active_intent["request_id"])
        self.active_intent = self.intents[request_id]
        self.plan_state = None
        self.completion.reset(request_id)
        return self.active_intent

    def active_step(self) -> Mapping[str, Any] | None:
        if self.active_intent is None:
            return None
        step_id = (
            self.plan_state.get("active_step_id")
            if self.plan_state is not None
            else self.active_intent["intent"]["steps"][0]["step_id"]
        )
        return next(
            (
                step
                for step in self.active_intent["intent"]["steps"]
                if step["step_id"] == step_id
            ),
            None,
        )

    def process(
        self,
        *,
        world_state: dict[str, Any],
        image: Any,
        projection_record: dict[str, Any] | None = None,
        planner_target_location: Mapping[str, Any] | None = None,
        lateral_diagnostics: Mapping[str, Any] | None = None,
        route_progress_m: float = 0.0,
        route_length_m: float | None = None,
    ) -> dict[str, Any]:
        if self.active_intent is None:
            raise RuntimeError("activate a DrivingIntent before processing frames")
        timings: dict[str, float] = {}
        started_total = time.perf_counter()
        step_before_decision = self.active_step()
        if (
            step_before_decision is not None
            and step_before_decision.get("action") == "SET_SPEED"
            and step_before_decision.get("parameters", {}).get(
                "target_speed_mps"
            )
            is not None
        ):
            self.speed_setpoint_kmh = (
                float(
                    step_before_decision["parameters"][
                        "target_speed_mps"
                    ]
                )
                * 3.6
            )

        if self.frame_count % self.perception_stride == 0:
            started = time.perf_counter()
            self.latest_perception = self.perception.process_image(
                image=image,
                frame_id=world_state["frame_id"],
                source="carla",
                camera_name="front_rgb",
                timestamp_s=world_state.get("timestamp_s"),
                world_state=world_state,
            )
            timings["perception_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
        else:
            timings["perception_ms"] = 0.0

        decision_world_state = world_state
        fusion_audit: dict[str, Any] = {
            "status": "SKIPPED",
            "reason": "projection_unavailable",
            "matched_count": 0,
        }
        if (
            projection_record is not None
            and self.latest_perception is not None
            and self.latest_perception.get("frame_id")
            == world_state.get("frame_id")
        ):
            started = time.perf_counter()
            decision_world_state, fusion_audit = fuse_perception_frame(
                world_state,
                self.latest_perception,
                projection_record,
            )
            fusion_audit["status"] = "FUSED"
            timings["visual_fusion_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
        else:
            timings["visual_fusion_ms"] = 0.0
            if projection_record is not None:
                fusion_audit["reason"] = "perception_frame_not_current"

        started = time.perf_counter()
        alignment = align_driving_intent(
            self.active_intent,
            decision_world_state,
        )
        timings["semantic_alignment_ms"] = (
            time.perf_counter() - started
        ) * 1000.0

        started = time.perf_counter()
        risk = assess_world_state(decision_world_state)
        timings["risk_ms"] = (time.perf_counter() - started) * 1000.0

        feedback = self.completion.evaluate(
            self.active_intent,
            self.plan_state,
            decision_world_state,
            alignment,
            lateral_diagnostics=lateral_diagnostics,
            route_progress_m=route_progress_m,
            route_length_m=route_length_m,
        )
        token_record = self.intent_tokens[self.active_intent["request_id"]]
        batch, candidate_ids = self.rasterizer.build(
            decision_world_state,
            intent_tokens=token_record["tokens"],
            intent_mask=token_record["mask"],
        )
        started = time.perf_counter()
        proposal, self.plan_state, decision = self.vla.decide(
            batch,
            self.active_intent,
            decision_world_state,
            alignment,
            risk,
            candidate_entity_ids=candidate_ids,
            prior_state=self.plan_state,
            feedback=feedback,
            planner_target_location=planner_target_location,
        )
        speed_setpoint_applied = False
        if (
            self.speed_setpoint_kmh is not None
            and decision.get("decision_status") == "READY"
            and decision.get("action")
            in {
                "keep_lane",
                "accelerate",
                "lane_change_left",
                "lane_change_right",
                "turn_left",
                "turn_right",
            }
            and risk.get("recommended_action")
            not in {"decelerate", "emergency_brake"}
        ):
            decision = dict(decision)
            decision["target_speed_kmh"] = round(
                float(self.speed_setpoint_kmh),
                6,
            )
            speed_setpoint_applied = True
        timings["vla_fsm_decision_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        timings["vla_model_ms"] = float(proposal["latency_ms"])
        timings["frame_pipeline_ms"] = (
            time.perf_counter() - started_total
        ) * 1000.0
        timings = {key: round(value, 3) for key, value in timings.items()}
        self.frame_count += 1
        return {
            "driving_intent": self.active_intent,
            "perception_frame": self.latest_perception,
            "visual_fusion_audit": fusion_audit,
            "semantic_alignment": alignment,
            "risk_assessment": risk,
            "vla_proposal": proposal,
            "control_plan_state": self.plan_state,
            "control_decision": decision,
            "step_feedback": feedback,
            "speed_setpoint": {
                "target_speed_kmh": self.speed_setpoint_kmh,
                "applied": speed_setpoint_applied,
            },
            "latency_ms": {
                "parser_precomputed": float(
                    self.parser_latency_ms.get(
                        self.active_intent["request_id"], 0.0
                    )
                ),
                **timings,
            },
            "provenance": {
                "qwen_scene_model_realtime": False,
                "perception": "YOLOP+YOLO11s+ByteTrack",
                "scene_metrics": "CARLA WorldState",
                "vla": self.vla.model_name,
                "safety_gate": "canonical_rule_fsm",
            },
        }
