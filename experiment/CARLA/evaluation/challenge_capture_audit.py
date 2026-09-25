"""Index three recorded model-input runs against independent CARLA truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmark.catalog import load_catalog
from evaluation.model_rig_replay import ModelRigReplayDataset
from evaluation.sensor_replay import DatasetValidationError, _load_jsonl


SCENES = ("scene_1", "scene_2", "scene_3")
REQUIRED_MODEL_SENSORS = ("front", "left", "right", "rear", "lidar")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_indices(available: int, count: int) -> list[int]:
    if available < count:
        raise DatasetValidationError(
            f"need {count} recorded decisions, found only {available}"
        )
    if count == 1:
        return [available // 2]
    return [index * (available - 1) // (count - 1) for index in range(count)]


def _task_coverage_indices(scene: str, frames: list, truth: dict, source_sha256: str) -> dict[str, int]:
    catalog = load_catalog(scene)
    if catalog.source_sha256 != source_sha256:
        raise DatasetValidationError(f"{scene}: assessment uses a different scene configuration")
    chosen = {}
    progress = []
    for frame in frames:
        value = truth[frame.simulation_frame].get("route_s_m")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise DatasetValidationError(f"{scene}: missing or invalid route progress in independent truth")
        progress.append(float(value))
    for index, task in enumerate(catalog.tasks):
        end = task.end_m
        if end is None:
            end = (catalog.tasks[index + 1].activate_m
                   if index + 1 < len(catalog.tasks) else catalog.route_length_m)
        candidates = [position for position, value in enumerate(progress)
                      if task.activate_m <= value < end]
        if not candidates:
            raise DatasetValidationError(
                f"{scene}: no recorded model decision in task interval {task.task_id}"
            )
        chosen[task.task_id] = candidates[len(candidates) // 2]
    return chosen


def audit_scene(scene: str, run_root: Path, count: int, *, require_task_coverage: bool = True) -> dict:
    root = run_root.expanduser().resolve()
    assessment = root / "benchmark"
    try:
        metadata = json.loads((assessment / "manifest.json").read_text(encoding="utf-8"))
        outcome = json.loads((assessment / "run_outcome.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DatasetValidationError(f"{scene}: missing or invalid closed assessment") from error
    if not isinstance(metadata, dict) or metadata.get("scene_id") != scene:
        raise DatasetValidationError(f"{scene}: assessment scene identity mismatch")
    if not isinstance(metadata.get("source_sha256"), str) or len(metadata["source_sha256"]) != 64:
        raise DatasetValidationError(f"{scene}: assessment source fingerprint missing")
    if not isinstance(outcome, dict) or outcome.get("status") not in {
        "RECORDED_CHECKS_PASSED", "CHECK_FAILED", "INCOMPLETE_EVIDENCE", "ASSESSMENT_ERROR",
    }:
        raise DatasetValidationError(f"{scene}: assessment outcome missing or invalid")

    truth_path = assessment / "episode_truth.jsonl"
    truth = {}
    for row in _load_jsonl(truth_path):
        frame = row.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame in truth:
            raise DatasetValidationError(f"{scene}: invalid or duplicate truth frame")
        timestamp = row.get("sim_time_s")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise DatasetValidationError(f"{scene}: invalid truth timestamp")
        truth[frame] = row

    inputs = ModelRigReplayDataset(root / "model_inputs")
    frames = list(inputs)
    for frame in frames:
        if not set(REQUIRED_MODEL_SENSORS).issubset(frame.artifacts):
            raise DatasetValidationError(
                f"{scene}: incomplete four-view and LiDAR input at decision frame {frame.simulation_frame}"
            )
        sensor_frames = {
            frame.sensor_metadata[name]["simulation_frame"] for name in REQUIRED_MODEL_SENSORS
        }
        sensor_times = [
            float(frame.sensor_metadata[name]["timestamp_s"]) for name in REQUIRED_MODEL_SENSORS
        ]
        if len(sensor_frames) != 1 or max(sensor_times) - min(sensor_times) > 1e-3:
            raise DatasetValidationError(
                f"{scene}: multimodal sensors are not synchronized at decision frame {frame.simulation_frame}"
            )
        record = truth.get(frame.simulation_frame)
        if record is None or not math.isclose(
            float(record["sim_time_s"]), frame.timestamp_s, rel_tol=0.0, abs_tol=1e-3
        ):
            raise DatasetValidationError(
                f"{scene}: missing or mistimed independent truth for decision frame {frame.simulation_frame}"
            )
    mandatory = (_task_coverage_indices(scene, frames, truth, metadata["source_sha256"])
                 if require_task_coverage else {})
    selected_indices = set(mandatory.values())
    if len(selected_indices) > count:
        raise DatasetValidationError(f"{scene}: frame quota cannot cover all task intervals")
    remaining = [index for index in range(len(frames)) if index not in selected_indices]
    needed = count - len(selected_indices)
    if needed:
        selected_indices.update(
            remaining[index] for index in _selected_indices(len(remaining), needed)
        )
    selected = [frames[index] for index in sorted(selected_indices)]
    input_manifest = inputs.integrity_manifest()
    input_hashes = {row["simulation_frame"]: row["input_sha256"] for row in input_manifest["frames"]}
    return {
        "scene_id": scene,
        "run_root": str(root),
        "model_input_root": str(inputs.root),
        "recorded_decisions": len(frames),
        "input_dataset_sha256": input_manifest["dataset_sha256"],
        "assessment_source_sha256": metadata.get("source_sha256"),
        "assessment_status": outcome.get("status"),
        "truth_file_sha256": _file_hash(truth_path),
        "task_coverage": {
            "status": "ALL_TASK_INTERVALS_SAMPLED" if require_task_coverage else "NOT_CHECKED",
            "task_ids": list(mandatory),
            "scope": "at least one decision frame in each task distance interval; not task success",
        },
        "selected": [
            {
                "simulation_frame": frame.simulation_frame,
                "timestamp_s": frame.timestamp_s,
                "input_sha256": input_hashes[frame.simulation_frame],
                "truth_sha256": hashlib.sha256(json.dumps(
                    truth[frame.simulation_frame], sort_keys=True, allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
            }
            for frame in selected
        ],
    }


def audit_captures(roots: dict[str, Path], frame_count: int = 1000,
                   *, require_task_coverage: bool = True) -> dict:
    if set(roots) != set(SCENES) or type(frame_count) is not int or frame_count < len(SCENES):
        raise ValueError("provide all three scenes and at least three frames")
    base, remainder = divmod(frame_count, len(SCENES))
    scenes = [
        audit_scene(scene, Path(roots[scene]), base + (index < remainder),
                    require_task_coverage=require_task_coverage)
        for index, scene in enumerate(SCENES)
    ]
    return {
        "schema_version": "challenge_same_source_selection/1.0",
        "frame_count": frame_count,
        "selection": "task-interval frames plus evenly spaced evaluation frames; replay full source sequence for stateful models",
        "truth_in_policy_inputs": False,
        "closed_loop_success": None,
        "scenes": scenes,
    }


def validate_replay_selection(path: Path, scene: str, dataset: ModelRigReplayDataset,
                              manifest: dict) -> set[int]:
    """Bind selected evaluation frames to the exact full replay input and truth file."""
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DatasetValidationError("invalid same-source selection file") from error
    if not isinstance(report, dict) or report.get("schema_version") != "challenge_same_source_selection/1.0":
        raise DatasetValidationError("unsupported same-source selection schema")
    matches = [row for row in report.get("scenes", [])
               if isinstance(row, dict) and row.get("scene_id") == scene]
    if len(matches) != 1:
        raise DatasetValidationError("selection has no unique scene: " + scene)
    entry = matches[0]
    if entry.get("input_dataset_sha256") != manifest["dataset_sha256"]:
        raise DatasetValidationError("selection uses a different model-input capture")
    assessment = dataset.root.parent / "benchmark"
    try:
        metadata = json.loads((assessment / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DatasetValidationError("replay assessment manifest missing or invalid") from error
    if (not isinstance(metadata, dict) or metadata.get("scene_id") != scene
            or metadata.get("source_sha256") != entry.get("assessment_source_sha256")):
        raise DatasetValidationError("replay assessment scene or source changed")
    truth_path = assessment / "episode_truth.jsonl"
    if not truth_path.is_file() or _file_hash(truth_path) != entry.get("truth_file_sha256"):
        raise DatasetValidationError("selection independent truth changed or disappeared")
    hashes = {row["simulation_frame"]: row["input_sha256"] for row in manifest["frames"]}
    selected = entry.get("selected")
    if not isinstance(selected, list) or not selected:
        raise DatasetValidationError("selection contains no evaluation frames")
    ids: set[int] = set()
    for row in selected:
        frame = row.get("simulation_frame") if isinstance(row, dict) else None
        if (type(frame) is not int or frame in ids or hashes.get(frame) != row.get("input_sha256")):
            raise DatasetValidationError("selection frame is duplicate or has changed")
        ids.add(frame)
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for scene in SCENES:
        parser.add_argument("--" + scene.replace("_", "-"), type=Path, required=True,
                            help="Completed run directory containing model_inputs/ and benchmark/")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    roots = {scene: getattr(args, scene) for scene in SCENES}
    report = audit_captures(roots, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Indexed {report['frame_count']} verified input/truth frames across three scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
