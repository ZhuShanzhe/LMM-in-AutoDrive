"""Summarize competition-facing evidence from a Scene 2 pipeline log."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.pipeline_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    run_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    commands = defaultdict(list)
    for record in records:
        commands[str(record["command_id"])].append(record)

    completed = {
        command_id: any(
            frame.get("plan_status") == "COMPLETED" for frame in frames
        )
        for command_id, frames in commands.items()
    }
    alignment_targets: dict[tuple[str, str], bool] = {}
    visual_alignment_targets: dict[tuple[str, str], bool] = {}
    fused_frames = 0
    detector_matches = 0
    unmatched_projected = 0
    unmatched_tracks = 0
    for command_id, frames in commands.items():
        for frame in frames:
            fusion = frame.get("visual_fusion_audit", {})
            visual_world_ids = {
                str(match["world_object_id"])
                for match in fusion.get("matches", [])
                if isinstance(match, dict) and match.get("world_object_id")
            }
            if fusion.get("status") == "FUSED":
                fused_frames += 1
                detector_matches += int(fusion.get("matched_count", 0))
                unmatched_projected += len(
                    fusion.get("unmatched_world_object_ids", [])
                )
                unmatched_tracks += len(
                    fusion.get("unmatched_visual_object_ids", [])
                )
            for item in frame["semantic_alignment"]["step_alignments"]:
                if item.get("alignment_required") is not True:
                    continue
                key = (command_id, str(item["step_id"]))
                alignment_targets[key] = (
                    alignment_targets.get(key, False)
                    or item.get("alignment_success") is True
                )
                matched_entity = item.get("matched_entity")
                matched_entity_id = (
                    str(matched_entity.get("entity_id"))
                    if isinstance(matched_entity, dict)
                    and matched_entity.get("entity_id")
                    else None
                )
                if (
                    isinstance(matched_entity, dict)
                    and matched_entity.get("entity_type") == "actor"
                ):
                    visual_alignment_targets[key] = (
                        visual_alignment_targets.get(key, False)
                        or (
                            item.get("alignment_success") is True
                            and matched_entity_id in visual_world_ids
                        )
                    )

    pipeline_latencies = [
        float(frame["latency_ms"]["frame_pipeline_ms"]) for frame in records
    ]
    perception_latencies = [
        float(frame["latency_ms"]["perception_ms"])
        for frame in records
        if float(frame["latency_ms"]["perception_ms"]) > 0
    ]
    vla_latencies = [
        float(frame["latency_ms"]["vla_model_ms"]) for frame in records
    ]
    steer_angles = [
        abs(float(frame["lateral_diagnostics"]["commanded_wheel_angle_deg"]))
        for frame in records
        if isinstance(frame.get("lateral_diagnostics"), dict)
        and "commanded_wheel_angle_deg" in frame["lateral_diagnostics"]
    ]
    result: dict[str, Any] = {
        "schema_version": "scene2_competition_metrics/v1",
        "evidence": {
            "pipeline_log": str(args.pipeline_log.resolve()),
            "run_summary": str(args.summary.resolve()),
            "frames": len(records),
        },
        "scene_task_completion": {
            "commands_observed": len(commands),
            "commands_completed": sum(completed.values()),
            "completion_rate_percent": round(
                100.0 * sum(completed.values()) / 15.0, 3
            ),
            "by_command": completed,
            "collision_count": run_summary.get("collision_count"),
            "lane_invasion_count": run_summary.get("lane_invasion_count"),
        },
        "multimodal_semantic_alignment": {
            "definition": (
                "unique required DrivingIntent targets matched at least once "
                "during that command"
            ),
            "matched_targets": sum(alignment_targets.values()),
            "required_targets": len(alignment_targets),
            "accuracy_percent": round(
                100.0
                * sum(alignment_targets.values())
                / max(1, len(alignment_targets)),
                3,
            ),
            "uses_carla_metadata_semantic_proxy": True,
            "competition_claim_valid": False,
            "limitation": (
                "Object color and role matching still includes CARLA actor "
                "metadata. The asynchronous Qwen scene model is disabled, so "
                "this is integration evidence rather than the competition "
                "multimodal semantic-alignment accuracy."
            ),
            "actor_targets_with_detector_support": sum(
                visual_alignment_targets.values()
            ),
            "matched_actor_targets_observed": len(
                visual_alignment_targets
            ),
        },
        "detector_projection_grounding": {
            "definition": (
                "one-to-one YOLO/ByteTrack track matches against synchronized "
                "CARLA 3D-box projections"
            ),
            "fused_frames": fused_frames,
            "matched_pairs": detector_matches,
            "projected_objects": detector_matches + unmatched_projected,
            "detector_tracks": detector_matches + unmatched_tracks,
            "projected_object_recall_percent": round(
                100.0
                * detector_matches
                / max(1, detector_matches + unmatched_projected),
                3,
            ),
            "track_precision_percent": round(
                100.0
                * detector_matches
                / max(1, detector_matches + unmatched_tracks),
                3,
            ),
            "competition_claim_valid": False,
            "limitation": (
                "This measures detector-to-simulator projection grounding, "
                "not open-vocabulary VLM semantic accuracy."
            ),
        },
        "decision_latency_ms": {
            "frame_pipeline_mean": round(mean(pipeline_latencies), 3)
            if pipeline_latencies
            else None,
            "frame_pipeline_p95": round(percentile(pipeline_latencies, 0.95), 3)
            if pipeline_latencies
            else None,
            "perception_mean": round(mean(perception_latencies), 3)
            if perception_latencies
            else None,
            "vla_model_mean": round(mean(vla_latencies), 3)
            if vla_latencies
            else None,
            "vla_model_p95": round(percentile(vla_latencies, 0.95), 3)
            if vla_latencies
            else None,
            "meets_150ms_p95": bool(
                pipeline_latencies
                and percentile(pipeline_latencies, 0.95) <= 150.0
            ),
        },
        "lane_change_control": {
            "maximum_commanded_wheel_angle_deg": round(max(steer_angles), 3)
            if steer_angles
            else None,
            "within_5deg_limit": bool(
                steer_angles and max(steer_angles) <= 5.001
            ),
        },
        "asr_accuracy": {
            "measured": False,
            "reason": "This run begins from post-ASR text commands.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
