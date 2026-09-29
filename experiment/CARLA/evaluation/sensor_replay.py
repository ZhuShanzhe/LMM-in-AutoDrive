"""Validate and replay frame-synchronized CARLA sensor datasets."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Mapping, Sequence


REQUIRED_MODALITIES = (
    "front_rgb",
    "left_rgb",
    "right_rgb",
    "rear_rgb",
    "lidar",
)


class DatasetValidationError(ValueError):
    """Raised when a recorded dataset violates the replay contract."""


@dataclass(frozen=True)
class ReplayFrame:
    """Policy-facing sensor references and allowlisted ego telemetry only."""
    simulation_frame: int
    timestamp_s: float
    scene_id: str
    artifacts: Mapping[str, Path]
    vehicle_state: Mapping[str, Any]
    driving_intent_request_id: str | None
    driving_intent: Mapping[str, Any] | None = None
    sensor_calibration: Mapping[str, Any] | None = None
    command_context: Mapping[str, Any] | None = None


def policy_vehicle_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Do not propagate evaluator fields or future additions to WorldState."""
    ego = state.get("ego")
    if not isinstance(ego, Mapping):
        raise DatasetValidationError("world state has no ego telemetry object")
    result: dict[str, Any] = {}
    fields = {
        "location": ("x", "y", "z"),
        "rotation": ("pitch", "yaw", "roll"),
        "velocity_mps": ("x", "y", "z"),
        "acceleration_mps2": ("x", "y", "z"),
        "angular_velocity_deg_s": ("x", "y", "z"),
        "control": ("throttle", "brake", "steer", "reverse", "hand_brake", "gear"),
    }
    if "speed_kmh" in ego:
        result["speed_kmh"] = _finite_number(ego["speed_kmh"], "speed_kmh")
    for name, keys in fields.items():
        if name not in ego:
            continue
        source = ego[name]
        if not isinstance(source, Mapping):
            raise DatasetValidationError("invalid ego telemetry: " + name)
        values = {}
        for key in keys:
            if key not in source:
                raise DatasetValidationError("missing ego telemetry: " + name + "." + key)
            value = source[key]
            if name == "control" and key in ("reverse", "hand_brake"):
                if not isinstance(value, bool):
                    raise DatasetValidationError("invalid control boolean: " + key)
                values[key] = value
            else:
                values[key] = _finite_number(value, name + "." + key)
        result[name] = values
    if "speed_kmh" not in result:
        raise DatasetValidationError("missing ego telemetry: speed_kmh")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DatasetValidationError("non-finite or non-numeric " + label)
    return float(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetValidationError("missing required log: {0}".format(path))
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetValidationError(
                    "{0}:{1}: invalid JSON: {2}".format(
                        path, line_number, error
                    )
                ) from error
            if not isinstance(payload, dict):
                raise DatasetValidationError(
                    "{0}:{1}: record must be an object".format(
                        path, line_number
                    )
                )
            records.append(payload)
    return records


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise DatasetValidationError(
            "artifact path escapes dataset root: {0}".format(relative_path)
        ) from error
    return candidate


def _default_artifacts(frame: int) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": (
                "rgb/{0}/{1:08d}.png".format(name, frame)
                if name != "lidar"
                else "lidar/{0:08d}.ply".format(frame)
            )
        }
        for name in REQUIRED_MODALITIES
    }


class SynchronizedReplayDataset:
    """Read exact-frame RGB, LiDAR and vehicle-state bundles.

    Legacy Scene 2 captures without explicit artifact paths are supported via
    the stable on-disk naming convention used by ExactFrameSensorSuite.
    """

    def __init__(self, root: str | Path, *, require_files: bool = True,
                 require_calibration: bool = False) -> None:
        self.root = Path(root).expanduser().resolve()
        self.require_files = bool(require_files)
        calibration_path = self.root / "sensor_calibration.json"
        self.calibration = None
        if calibration_path.is_file():
            try:
                self.calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as error:
                raise DatasetValidationError("cannot read sensor calibration") from error
            if not isinstance(self.calibration, dict) or self.calibration.get("schema_version") != "carla_sensor_calibration/1.0":
                raise DatasetValidationError("unsupported sensor calibration schema")
            sensors = self.calibration.get("sensors")
            if not isinstance(sensors, dict) or any(name not in sensors for name in REQUIRED_MODALITIES):
                raise DatasetValidationError("incomplete sensor calibration")
        elif require_calibration:
            raise DatasetValidationError("missing sensor_calibration.json")
        bundles = _load_jsonl(self.root / "multimodal_frame_bundle.jsonl")
        states = _load_jsonl(self.root / "world_state.jsonl")
        self._states = self._index_unique(states, "simulation_frame", "state")
        self._intents: dict[str, Mapping[str, Any]] = {}
        intent_path = self.root / "driving_intent.jsonl"
        if intent_path.is_file():
            for record in _load_jsonl(intent_path):
                request_id = record.get("request_id")
                if not isinstance(request_id, str) or not request_id.strip():
                    raise DatasetValidationError("intent has no nonempty request_id")
                if request_id in self._intents:
                    raise DatasetValidationError("duplicate intent request_id: " + request_id)
                self._intents[request_id] = record
        command_path = self.root / "command_context.jsonl"
        self._commands = (
            self._index_unique(_load_jsonl(command_path), "simulation_frame", "command")
            if command_path.is_file() else {}
        )
        self._frames = self._build_frames(bundles)
        if command_path.is_file() and set(self._commands) != {
            frame.simulation_frame for frame in self._frames
        }:
            raise DatasetValidationError("command context frames do not match bundles")

    def _command_context(self, frame: int, timestamp: float) -> dict[str, Any] | None:
        if not self._commands:
            return None
        record = self._commands.get(frame)
        if record is None:
            raise DatasetValidationError(f"missing command context for frame {frame}")
        command_time = _finite_number(record.get("timestamp_s"), "command timestamp")
        if not math.isclose(command_time, timestamp, rel_tol=0.0, abs_tol=1e-6):
            raise DatasetValidationError("command and bundle timestamps disagree")
        if record.get("source") != "scene_schedule_not_model_parse":
            raise DatasetValidationError("unsupported command context source")
        commands = record.get("commands")
        if not isinstance(commands, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str) or not item["id"].strip()
            or not isinstance(item.get("text"), str) or not item["text"].strip()
            for item in commands
        ):
            raise DatasetValidationError("invalid command context list")
        return {"source": record["source"], "commands": [
            {"id": item["id"], "text": item["text"]} for item in commands
        ]}

    def _resolve_intent(self, request_id: Any, frame: int, timestamp: float) -> dict[str, Any] | None:
        if request_id is None:
            return None
        if not isinstance(request_id, str) or request_id not in self._intents:
            raise DatasetValidationError("unresolved driving intent: " + str(request_id))
        record = self._intents[request_id]
        issued_frame = record.get("simulation_frame")
        if isinstance(issued_frame, bool) or not isinstance(issued_frame, int) or issued_frame < 0:
            raise DatasetValidationError("intent has invalid simulation_frame")
        issued_time = _finite_number(record.get("timestamp_s"), "intent timestamp")
        if issued_frame > frame or issued_time > timestamp + 1e-6:
            raise DatasetValidationError("intent is from a future frame or timestamp")
        input_record = record.get("input")
        if not isinstance(input_record, Mapping) or not isinstance(input_record.get("raw_text"), str):
            raise DatasetValidationError("intent has no input.raw_text")
        if not isinstance(record.get("intent"), Mapping):
            raise DatasetValidationError("intent has no structured intent object")
        # Issuance metadata is allowed; route progress and evaluator extras are not.
        return deepcopy({key: record[key] for key in (
            "schema_version", "request_id", "simulation_frame", "timestamp_s",
            "input", "intent", "parse_result",
        ) if key in record})

    @staticmethod
    def _index_unique(
        records: Sequence[Mapping[str, Any]], key: str, label: str
    ) -> dict[int, Mapping[str, Any]]:
        indexed: dict[int, Mapping[str, Any]] = {}
        for record in records:
            if key not in record:
                raise DatasetValidationError(
                    "{0} record has no {1}".format(label, key)
                )
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DatasetValidationError(f"invalid {label} {key}")
            if value in indexed:
                raise DatasetValidationError(
                    "duplicate {0} frame: {1}".format(label, value)
                )
            indexed[value] = record
        return indexed

    def _build_frames(
        self, bundles: Sequence[Mapping[str, Any]]
    ) -> tuple[ReplayFrame, ...]:
        frames: list[ReplayFrame] = []
        seen: set[int] = set()
        previous_frame = -1
        previous_timestamp = float("-inf")
        for bundle in bundles:
            frame = int(bundle.get("simulation_frame", -1))
            if frame in seen:
                raise DatasetValidationError(
                    "duplicate multimodal frame: {0}".format(frame)
                )
            if frame <= previous_frame:
                raise DatasetValidationError(
                    "multimodal frames are not strictly increasing"
                )
            synchronization = bundle.get("synchronization", {})
            if bundle.get("status") != "COMPLETE" or not synchronization.get(
                "exact", False
            ):
                raise DatasetValidationError(
                    "frame {0} is not an exact complete bundle".format(frame)
                )
            if synchronization.get("adjacent_frame_fill_used", False):
                raise DatasetValidationError(
                    "frame {0} uses adjacent-frame filling".format(frame)
                )
            state = self._states.get(frame)
            if state is None:
                raise DatasetValidationError(
                    "frame {0} has no exact vehicle state".format(frame)
                )
            state_timestamp = _finite_number(state.get("timestamp_s"), "state timestamp")
            timestamp = _finite_number(bundle.get("timestamp_s", state_timestamp), "bundle timestamp")
            if not math.isclose(timestamp, state_timestamp, rel_tol=0.0, abs_tol=1e-6):
                raise DatasetValidationError("bundle and state timestamps disagree")
            if timestamp <= previous_timestamp:
                raise DatasetValidationError(
                    "timestamps are not strictly increasing"
                )
            artifact_records = bundle.get("artifacts") or _default_artifacts(frame)
            paths: dict[str, Path] = {}
            names = (*REQUIRED_MODALITIES, *(("lidar_raw",) if "lidar_raw" in artifact_records else ()))
            for name in names:
                item = artifact_records.get(name)
                if not isinstance(item, Mapping) or not item.get("path"):
                    raise DatasetValidationError(
                        "frame {0} is missing artifact metadata for {1}".format(
                            frame, name
                        )
                    )
                path = _resolve_artifact(self.root, str(item["path"]))
                if self.require_files and not path.is_file():
                    raise DatasetValidationError(
                        "frame {0} artifact is missing: {1}".format(frame, path)
                    )
                if name == "lidar_raw":
                    if item.get("encoding") != "float32_le_xyzi":
                        raise DatasetValidationError("unsupported raw LiDAR encoding")
                    if self.require_files and path.stat().st_size % 16:
                        raise DatasetValidationError("invalid raw LiDAR byte length")
                paths[name] = path
            frames.append(
                ReplayFrame(
                    simulation_frame=frame,
                    timestamp_s=timestamp,
                    scene_id=str(bundle.get("scene_id", "")),
                    artifacts=paths,
                    vehicle_state=policy_vehicle_state(state),
                    driving_intent_request_id=bundle.get(
                        "driving_intent_request_id"
                    ),
                    driving_intent=self._resolve_intent(
                        bundle.get("driving_intent_request_id"), frame, timestamp
                    ),
                    sensor_calibration=self.calibration,
                    command_context=self._command_context(frame, timestamp),
                )
            )
            seen.add(frame)
            previous_frame = frame
            previous_timestamp = timestamp
        if not frames:
            raise DatasetValidationError("dataset contains no complete frames")
        return tuple(frames)

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[ReplayFrame]:
        return iter(self._frames)

    def replay(
        self,
        consumer: Callable[[ReplayFrame], Any],
        *,
        speed: float = 0.0,
        start_index: int = 0,
        max_frames: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> int:
        """Replay frames deterministically; speed=0 disables wall-clock waits."""

        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError("speed must be non-negative")
        selected = self._frames[max(0, int(start_index)) :]
        if max_frames is not None:
            selected = selected[: max(0, int(max_frames))]
        previous_timestamp: float | None = None
        count = 0
        for frame in selected:
            if speed > 0.0 and previous_timestamp is not None:
                delay = max(0.0, frame.timestamp_s - previous_timestamp) / speed
                if delay:
                    sleep(delay)
            consumer(frame)
            previous_timestamp = frame.timestamp_s
            count += 1
        return count

    def summary(self) -> dict[str, Any]:
        first, last = self._frames[0], self._frames[-1]
        return {
            "schema_version": "carla_sensor_replay_summary/1.0",
            "dataset_root": str(self.root),
            "scene_ids": sorted({frame.scene_id for frame in self._frames}),
            "frame_count": len(self._frames),
            "first_simulation_frame": first.simulation_frame,
            "last_simulation_frame": last.simulation_frame,
            "duration_s": round(last.timestamp_s - first.timestamp_s, 6),
            "required_modalities": list(REQUIRED_MODALITIES),
            "exact_frame_synchronization": True,
            "artifact_files_verified": self.require_files,
            "vehicle_state_contract": "ego_telemetry_allowlist/1.0",
            "evaluation_truth_in_policy_payload": False,
            "frames_with_instruction": sum(frame.driving_intent is not None for frame in self._frames),
            "frames_with_scheduled_text": sum(bool(frame.command_context and frame.command_context["commands"])
                                              for frame in self._frames),
            "sensor_calibration_available": self.calibration is not None,
            "frames_with_raw_lidar": sum("lidar_raw" in frame.artifacts for frame in self._frames),
        }

    def integrity_manifest(self) -> dict[str, Any]:
        """Return byte-level identity evidence for same-source evaluation."""

        if not self.require_files:
            raise DatasetValidationError(
                "integrity manifests require artifact file verification"
            )
        dataset_digest = hashlib.sha256()
        calibration_digest = hashlib.sha256(json.dumps(
            self.calibration, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        dataset_digest.update(calibration_digest.encode("ascii"))
        records: list[dict[str, Any]] = []
        for frame in self._frames:
            artifacts: dict[str, Any] = {}
            for name in sorted(frame.artifacts):
                path = frame.artifacts[name]
                artifact_digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        artifact_digest.update(chunk)
                value = {
                    "path": path.relative_to(self.root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": artifact_digest.hexdigest(),
                }
                artifacts[name] = value
                dataset_digest.update(
                    json.dumps(
                        [frame.simulation_frame, name, value],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            state_digest = hashlib.sha256(
                json.dumps(
                    self._states[frame.simulation_frame],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            dataset_digest.update(state_digest.encode("ascii"))
            instruction_digest = hashlib.sha256(json.dumps(
                [frame.simulation_frame, frame.timestamp_s, frame.scene_id,
                 frame.driving_intent, frame.command_context],
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            dataset_digest.update(instruction_digest.encode("ascii"))
            records.append(
                {
                    "simulation_frame": frame.simulation_frame,
                    "timestamp_s": frame.timestamp_s,
                    "vehicle_state_sha256": state_digest,
                    "instruction_context_sha256": instruction_digest,
                    "artifacts": artifacts,
                }
            )
        return {
            "schema_version": "carla_same_source_manifest/1.4",
            "sensor_calibration_sha256": calibration_digest,
            "dataset_sha256": dataset_digest.hexdigest(),
            "summary": self.summary(),
            "frames": records,
        }


def _write_replay_log(path: Path, dataset: SynchronizedReplayDataset, speed: float) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        def consume(frame: ReplayFrame) -> None:
            handle.write(
                json.dumps(
                    {
                        "simulation_frame": frame.simulation_frame,
                        "timestamp_s": frame.timestamp_s,
                        "scene_id": frame.scene_id,
                        "artifacts": {
                            name: str(value)
                            for name, value in frame.artifacts.items()
                        },
                        "vehicle_state": frame.vehicle_state,
                        "driving_intent_request_id": (
                            frame.driving_intent_request_id
                        ),
                        "driving_intent": frame.driving_intent,
                        "command_context": frame.command_context,
                        "sensor_calibration": frame.sensor_calibration,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        return dataset.replay(consume, speed=speed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate or replay a synchronized CARLA capture"
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--speed", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--integrity-manifest",
        type=Path,
        help="Write per-artifact SHA-256 evidence for same-source comparisons.",
    )
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--require-calibration", action="store_true")
    args = parser.parse_args(argv)

    dataset = SynchronizedReplayDataset(
        args.dataset, require_files=not args.metadata_only,
        require_calibration=args.require_calibration,
    )
    summary = dataset.summary()
    if args.output is not None:
        summary["replayed_frames"] = _write_replay_log(
            args.output, dataset, args.speed
        )
    if args.integrity_manifest is not None:
        manifest = dataset.integrity_manifest()
        args.integrity_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.integrity_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["dataset_sha256"] = manifest["dataset_sha256"]
        summary["integrity_manifest"] = str(args.integrity_manifest.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
