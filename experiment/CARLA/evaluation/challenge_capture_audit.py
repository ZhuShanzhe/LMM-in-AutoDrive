"""Index three recorded model-input runs against independent CARLA truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from evaluation.model_rig_replay import ModelRigReplayDataset
from evaluation.sensor_replay import DatasetValidationError, _load_jsonl


SCENES = ("scene_1", "scene_2", "scene_3")


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


def audit_scene(scene: str, run_root: Path, count: int) -> dict:
    root = run_root.expanduser().resolve()
    assessment = root / "benchmark"
    try:
        metadata = json.loads((assessment / "manifest.json").read_text(encoding="utf-8"))
        outcome = json.loads((assessment / "run_outcome.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DatasetValidationError(f"{scene}: missing or invalid closed assessment") from error
    if metadata.get("scene_id") != scene:
        raise DatasetValidationError(f"{scene}: assessment scene identity mismatch")

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
        record = truth.get(frame.simulation_frame)
        if record is None or not math.isclose(
            float(record["sim_time_s"]), frame.timestamp_s, rel_tol=0.0, abs_tol=1e-3
        ):
            raise DatasetValidationError(
                f"{scene}: missing or mistimed independent truth for decision frame {frame.simulation_frame}"
            )
    selected = [frames[index] for index in _selected_indices(len(frames), count)]
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


def audit_captures(roots: dict[str, Path], frame_count: int = 1000) -> dict:
    if set(roots) != set(SCENES) or type(frame_count) is not int or frame_count < len(SCENES):
        raise ValueError("provide all three scenes and at least three frames")
    base, remainder = divmod(frame_count, len(SCENES))
    scenes = [
        audit_scene(scene, Path(roots[scene]), base + (index < remainder))
        for index, scene in enumerate(SCENES)
    ]
    return {
        "schema_version": "challenge_same_source_selection/1.0",
        "frame_count": frame_count,
        "selection": "evenly spaced evaluation frames; replay full source sequence for stateful models",
        "truth_in_policy_inputs": False,
        "closed_loop_success": None,
        "scenes": scenes,
    }


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
