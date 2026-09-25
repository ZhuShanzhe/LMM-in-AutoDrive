"""Optional lossless recording of the existing model rig, not extra sensors."""

import json
import shutil
from pathlib import Path
import threading
import time


class ModelRigCapture:
    def __init__(self, root, *, max_raw_bytes=20*1024**3, min_free_bytes=2*1024**3):
        if type(max_raw_bytes) is not int or max_raw_bytes<=0 or type(min_free_bytes) is not int or min_free_bytes<0:
            raise ValueError('capture byte limits must be integers; max positive and reserve nonnegative')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self._lock = threading.Lock()
        self._index = (self.root / "sensor_frames.jsonl").open("w", encoding="utf-8")
        self._decisions = (self.root / "consumed_inputs.jsonl").open("w", encoding="utf-8")
        self._decision_frames = set()
        self._references = set()
        self._seen = set()
        self._errors = []
        self._count = 0
        self._write_ms = 0.0
        self._closed = False
        self._sensors = {}
        self._raw_bytes = 0
        self._max_raw_bytes = max_raw_bytes
        self._min_free_bytes = min_free_bytes
        self._last_space_check = -float('inf')
        self._halted = False

    def _halt(self, reason, **details):
        self._halted = True
        self._errors.append(dict(stage='storage_guard', error=reason, **details))
        try:
            (self.root/'capture_stopped.json').write_text(json.dumps(self._errors[-1]),encoding='utf-8')
        except OSError:
            pass

    def _storage_available(self, size):
        if self._raw_bytes + size > self._max_raw_bytes:
            self._halt('raw_byte_budget_exceeded',raw_bytes=self._raw_bytes,next_bytes=size)
            return False
        now=time.monotonic()
        if now-self._last_space_check>=1:
            free=shutil.disk_usage(self.root).free
            self._last_space_check=now
            if free-size<self._min_free_bytes:
                self._halt('disk_free_reserve_reached',free_bytes=free)
                return False
        return True

    def register(self, name, actor, mounting):
        from evaluation.sensor_calibration import sensor_record
        with self._lock:
            self._sensors[name] = sensor_record(actor, mounting)
            self._sensors[name]["artifact_format"] = "raw callback bytes; encoding specified per sensor_frames.jsonl record"

    def record(self, name, measurement, encoding, **metadata):
        """Synchronous disk I/O is explicit; capture errors never fabricate data."""
        started = time.perf_counter()
        with self._lock:
            if self._closed or self._halted:
                return
            try:
                frame = int(measurement.frame)
                key = (name, frame)
                if key in self._seen:
                    raise ValueError("duplicate sensor frame")
                data = bytes(measurement.raw_data)
                if not self._storage_available(len(data)):
                    return
                path = self.root / name / ("%08d.bin" % frame)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
                self._raw_bytes += len(data)
                transform = getattr(measurement, "transform", None)
                row = dict(sensor=name, simulation_frame=frame,
                           timestamp_s=float(measurement.timestamp),
                           path=path.relative_to(self.root).as_posix(), encoding=encoding,
                           size_bytes=len(data), **metadata)
                if transform is not None:
                    row["sensor_to_world_at_capture"] = transform.get_matrix()
                self._index.write(json.dumps(row, allow_nan=False) + "\n")
                self._index.flush()
                self._seen.add(key)
                self._count += 1
            except Exception as error:
                self._errors.append(dict(sensor=name, frame=getattr(measurement, "frame", None),
                                         error=str(error), error_type=type(error).__name__))
                if isinstance(error,OSError):
                    self._halt('storage_io_error',message=str(error))
            finally:
                self._write_ms += (time.perf_counter() - started) * 1000

    def close(self):
        from evaluation.sensor_calibration import calibration_document
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._index.close()
            self._decisions.close()
            missing = sorted(self._references - self._seen)
            (self.root / "sensor_calibration.json").write_text(
                json.dumps(calibration_document(self._sensors), indent=2), encoding="utf-8")
            (self.root / "capture_summary.json").write_text(json.dumps({
                "schema_version": "model_rig_raw_capture/1.0",
                "status": "INVALID" if self._errors or missing or not self._count else "RECORDED",
                "sensor_records": self._count, "errors": self._errors,
                "raw_bytes_written": self._raw_bytes,
                "storage_limits": {"max_raw_bytes": self._max_raw_bytes, "min_free_bytes": self._min_free_bytes},
                "capture_halted": self._halted,
                "decision_records": len(self._decision_frames),
                "missing_consumed_sensor_records": [dict(sensor=name, frame=frame) for name, frame in missing],
                "write_ms_total": self._write_ms,
                "timing_note": "synchronous recording adds callback latency; not uninstrumented performance",
                "synchronization": "individual sensor timestamps; not an exact-frame bundle claim",
            }, indent=2), encoding="utf-8")

    def record_decision(self, *, decision, context, camera_names, sensor_frame,
                        lidar_enabled, radar_observations):
        """Record selected frames, not the latest callback at logging time."""
        with self._lock:
            if self._closed or self._halted:
                return
            try:
                frame = int(decision["simulation_frame"])
                if frame in self._decision_frames:
                    raise ValueError("duplicate decision frame")
                references = {name: int(sensor_frame) for name in camera_names}
                if lidar_enabled:
                    references["lidar"] = int(sensor_frame)
                unavailable = []
                for direction, observation in radar_observations.items():
                    selected = int(observation.get("sensor_frame", -1))
                    if selected < 0:
                        unavailable.append("radar_" + direction)
                    else:
                        references["radar_" + direction] = selected
                if any(value < 0 or value > frame for value in references.values()):
                    raise ValueError("invalid or future consumed sensor frame")
                row = dict(schema_version="model_consumed_inputs/1.0",
                           simulation_frame=frame, sensor_frames=references,
                           unavailable_sensors=unavailable, context=context, decision=decision)
                self._decisions.write(json.dumps(row, allow_nan=False) + "\n")
                self._decisions.flush()
                self._references.update(references.items())
                self._decision_frames.add(frame)
                if unavailable:
                    self._errors.append(dict(frame=frame, error="consumed radar unavailable",
                                             sensors=unavailable))
            except Exception as error:
                self._errors.append(dict(error_type=type(error).__name__, error=str(error),
                                         stage="record_decision"))
