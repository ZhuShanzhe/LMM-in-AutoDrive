"""Replay recorded model-consumed inputs without exposing recorded answers."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from evaluation.sensor_replay import DatasetValidationError, _load_jsonl, _resolve_artifact, _finite_number


@dataclass(frozen=True)
class NativeReplayFrame:
    simulation_frame: int
    timestamp_s: float
    artifacts: Mapping[str, Path]
    sensor_metadata: Mapping[str, Any]
    context: Mapping[str, Any]
    sensor_calibration: Mapping[str, Any]
    driving_intent_request_id: str | None


CONTEXT_FIELDS = (
    "timestamp_s", "command", "driving_intent", "plan_step_id", "vehicle_state_tensor",
    "environment_state_tensor", "camera_view_mask", "modality_mask",
    "camera_preprocessing",
    "radar_corridor_route_points", "raw_forward_radar", "rear_radar",
)


def decode_sensor(frame: NativeReplayFrame, name: str):
    """Return RGB HWC uint8 or raw float32 Nx4; no model-specific resizing."""
    import numpy as np
    metadata = frame.sensor_metadata[name]
    path = frame.artifacts[name]
    if metadata["encoding"] == "uint8_bgra":
        image = np.fromfile(path, dtype=np.uint8).reshape(metadata["height"], metadata["width"], 4)
        return image[:, :, 2::-1].copy()
    values = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    if not np.isfinite(values).all():
        raise DatasetValidationError("non-finite raw sensor values")
    return values


class ModelRigReplayDataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        summary = json.loads((self.root / "capture_summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "RECORDED" or summary.get("errors") or summary.get("missing_consumed_sensor_records"):
            raise DatasetValidationError("native capture is incomplete or invalid")
        calibration = json.loads((self.root / "sensor_calibration.json").read_text(encoding="utf-8"))
        if calibration.get("schema_version") != "carla_sensor_calibration/1.0":
            raise DatasetValidationError("unsupported native calibration")
        index = {}
        for row in _load_jsonl(self.root / "sensor_frames.jsonl"):
            key = (row["sensor"], row["simulation_frame"])
            if key in index:
                raise DatasetValidationError("duplicate native sensor frame")
            index[key] = row
        self._frames = []
        previous_frame, previous_time = -1, float("-inf")
        for row in _load_jsonl(self.root / "consumed_inputs.jsonl"):
            if row.get("schema_version") != "model_consumed_inputs/1.0" or row.get("unavailable_sensors"):
                raise DatasetValidationError("unsupported or incomplete consumed input")
            frame = row["simulation_frame"]
            timestamp = _finite_number(row["context"].get("timestamp_s"), "decision timestamp")
            if isinstance(frame, bool) or not isinstance(frame, int) or frame <= previous_frame or timestamp <= previous_time:
                raise DatasetValidationError("decisions must be strictly ordered")
            references = row["sensor_frames"]
            if not references:
                raise DatasetValidationError("no consumed sensors")
            paths, metadata = {}, {}
            for name, selected in references.items():
                if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0 or selected > frame:
                    raise DatasetValidationError("invalid consumed sensor frame")
                item = index.get((name, selected))
                if item is None:
                    raise DatasetValidationError("missing consumed sensor: " + name)
                if name not in calibration.get("sensors", {}):
                    raise DatasetValidationError("missing sensor calibration: " + name)
                sensor_time = _finite_number(item.get("timestamp_s"), "sensor timestamp")
                if sensor_time > timestamp + 1e-6:
                    raise DatasetValidationError("future sensor timestamp")
                path = _resolve_artifact(self.root, item["path"])
                if not path.is_file() or path.stat().st_size != item["size_bytes"]:
                    raise DatasetValidationError("missing or truncated sensor artifact")
                encoding = item["encoding"]
                if encoding == "uint8_bgra":
                    width, height = item.get("width"), item.get("height")
                    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
                        raise DatasetValidationError("invalid camera dimensions")
                    if path.stat().st_size != width * height * 4:
                        raise DatasetValidationError("invalid BGRA byte length")
                elif encoding in ("float32_le_xyzi", "float32_le_velocity_azimuth_altitude_depth"):
                    if path.stat().st_size % 16:
                        raise DatasetValidationError("invalid raw sensor byte length")
                else:
                    raise DatasetValidationError("unknown native sensor encoding")
                paths[name], metadata[name] = path, deepcopy(item)
            context = {key: deepcopy(row["context"][key]) for key in CONTEXT_FIELDS if key in row["context"]}
            intent = context.get("driving_intent") or {}
            self._frames.append(NativeReplayFrame(frame, timestamp, paths, metadata, context,
                                                  deepcopy(calibration), intent.get("request_id")))
            previous_frame, previous_time = frame, timestamp
        if not self._frames:
            raise DatasetValidationError("native capture has no model decisions")

    def __len__(self):
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames)

    def integrity_manifest(self):
        digest = hashlib.sha256()
        records, cached = [], {}
        for frame in self:
            artifacts = {}
            for name, path in sorted(frame.artifacts.items()):
                if path not in cached:
                    file_hash = hashlib.sha256()
                    with path.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            file_hash.update(block)
                    cached[path] = file_hash.hexdigest()
                artifacts[name] = cached[path]
            payload = dict(simulation_frame=frame.simulation_frame, timestamp_s=frame.timestamp_s,
                           artifacts=artifacts, sensor_metadata=frame.sensor_metadata,
                           context=frame.context, calibration=frame.sensor_calibration)
            frame_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                                   separators=(",", ":")).encode("utf-8")).hexdigest()
            digest.update(frame_hash.encode("ascii"))
            records.append(dict(simulation_frame=frame.simulation_frame, input_sha256=frame_hash))
        return dict(schema_version="model_rig_same_source/1.0", dataset_sha256=digest.hexdigest(),
                    frames=records, scope="consumed inputs only; recorded answers excluded")
