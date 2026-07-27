"""Non-blocking Qwen keyframe worker following scene_understanding's contract."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scene_understanding.core.prepare_carla_samples import append_jsonl, build_manifest_record
from scene_understanding.core.qwen_scene_service import (
    LatestFrameWorker,
    QwenSceneConfig,
    QwenSceneService,
)
from scene_understanding.core.visual_semantic_fusion import fuse_visual_semantics


class AsyncQwenBridge:
    """Submit latest capture records without ever waiting in the control loop."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        model_path: str | Path,
        prompt_path: str | Path,
        max_age_s: float = 15.0,
        max_new_tokens: int = 768,
        min_visual_tokens: int = 256,
        max_visual_tokens: int = 512,
    ) -> None:
        if max_age_s < 0:
            raise ValueError("max_age_s must be non-negative")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        service = QwenSceneService(QwenSceneConfig(
            Path(model_path), max_new_tokens=max_new_tokens,
            min_visual_tokens=min_visual_tokens, max_visual_tokens=max_visual_tokens,
        ))
        self.worker = LatestFrameWorker(service)
        self.max_age_s = max_age_s
        self._captures: dict[str, dict[str, Any]] = {}
        self._handled_frame_ids: set[str] = set()
        self.result_path = self.output_dir / "qwen_async_results.jsonl"
        self.world_path = self.output_dir / "qwen_async_enriched_world_states.jsonl"
        self.audit_path = self.output_dir / "qwen_async_visual_fusion_audits.jsonl"

    def submit_capture(self, capture: Mapping[str, Any]) -> dict[str, Any] | None:
        if capture.get("status") != "captured":
            return None
        record = build_manifest_record(
            capture, base_dir=Path(capture["image_path"]).parent,
            prompt_template=self.prompt_template,
        )
        self._captures[record["frame_id"]] = dict(capture)
        self.worker.submit(record)
        return {"status": "submitted", "frame_id": record["frame_id"]}

    def poll(self) -> dict[str, Any] | None:
        result = self.worker.latest(max_age_seconds=self.max_age_s)
        if result is None or result.get("frame_id") in self._handled_frame_ids:
            return None
        frame_id = str(result["frame_id"])
        self._handled_frame_ids.add(frame_id)
        append_jsonl(self.result_path, result)
        capture = self._captures.get(frame_id)
        response: dict[str, Any] = {
            "status": result.get("status"),
            "frame_id": frame_id,
            "service_elapsed_seconds": result.get("service_elapsed_seconds"),
        }
        if result.get("status") != "valid" or capture is None:
            return response
        try:
            world = json.loads(Path(capture["world_state_path"]).read_text(encoding="utf-8"))
            projection = json.loads(Path(capture["projection_path"]).read_text(encoding="utf-8"))
            enriched, audit = fuse_visual_semantics(world, result, projection)
            append_jsonl(self.world_path, enriched)
            append_jsonl(self.audit_path, audit)
            response.update({
                "status": "fused",
                "matched_count": audit["matched_count"],
                "world_state": enriched,
                "fusion_audit": audit,
            })
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            response.update({"status": "fusion_error", "error_type": type(exc).__name__, "error": str(exc)})
        return response

    def stats(self) -> dict[str, int]:
        return self.worker.stats()

    def close(self) -> None:
        self.worker.close(wait=False)
