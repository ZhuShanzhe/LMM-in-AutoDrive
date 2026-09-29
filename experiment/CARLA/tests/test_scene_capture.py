import json
from types import SimpleNamespace as NS

import pytest

from evaluation.scene_capture import SceneCaptureSession
from evaluation.sensor_replay import REQUIRED_MODALITIES, SynchronizedReplayDataset


def _vector(x=0, y=0, z=0):
    return NS(x=x, y=y, z=z)


class FakeSensors:
    def __init__(self, world, ego, registry, root, tick, **kwargs):
        self.root = root
        self.tick = tick
        self.complete = True

    def start(self):
        (self.root / "sensor_calibration.json").write_text(json.dumps({
            "schema_version": "carla_sensor_calibration/1.0",
            "sensors": {name: {} for name in REQUIRED_MODALITIES},
        }))

    def wait_for_stable_phase(self, frames, stride, timeout):
        return frames[-1], frames

    def wait_for_frame(self, frame, timeout):
        if not self.complete:
            return False, {}
        for name in REQUIRED_MODALITIES:
            if name == "lidar":
                directory = self.root / "lidar"
                directory.mkdir(exist_ok=True)
                (directory / f"{frame:08d}.ply").write_bytes(b"ply\n")
                (directory / f"{frame:08d}.xyzi.bin").write_bytes(bytes(16))
            else:
                directory = self.root / "rgb" / name
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{frame:08d}.png").write_bytes(b"png")
        return True, {name: frame for name in REQUIRED_MODALITIES}

    def summary(self):
        return {"required_modalities": list(REQUIRED_MODALITIES)}


def _ego():
    return NS(
        id=17,
        get_transform=lambda: NS(location=_vector(1, 2, 3),
                                 rotation=NS(pitch=0, yaw=90, roll=0)),
        get_velocity=lambda: _vector(10),
        get_acceleration=lambda: _vector(),
        get_angular_velocity=lambda: _vector(),
        get_control=lambda: NS(throttle=.3, brake=0, steer=.1,
                               reverse=False, hand_brake=False, gear=1),
    )


def test_standard_capture_replays_only_policy_telemetry(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.scene_capture.ExactFrameSensorSuite", FakeSensors)
    world = NS(tick=lambda: 1)
    capture = SceneCaptureSession(world, _ego(), tmp_path / "scene", "scene_1", .05,
                                  stride=1)
    capture.start()
    frame = capture.phase_frame + 1
    snapshot = NS(frame=frame, timestamp=NS(elapsed_seconds=.35))
    assert capture.observe(snapshot, 1400.0,
                           commands=[{"id": "c01", "text": "Keep lane."}])
    capture.close()

    dataset = SynchronizedReplayDataset(capture.root, require_calibration=True)
    replay = next(iter(dataset))
    assert len(dataset) == 1
    assert replay.vehicle_state["speed_kmh"] == 36.0
    assert replay.vehicle_state["location"]["x"] == 1.0
    assert "route_s_m" not in replay.vehicle_state
    assert "actor_id" not in replay.vehicle_state
    assert "lidar_raw" in replay.artifacts
    assert replay.command_context == {
        "source": "scene_schedule_not_model_parse",
        "commands": [{"id": "c01", "text": "Keep lane."}],
    }
    assert replay.driving_intent is None
    assert json.loads((capture.root / "capture_summary.json").read_text())["recorded_frames"] == 1


def test_missing_sensor_fails_without_publishing_complete_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.scene_capture.ExactFrameSensorSuite", FakeSensors)
    capture = SceneCaptureSession(NS(tick=lambda: 1), _ego(), tmp_path / "scene",
                                  "scene_3", .05, stride=1)
    capture.start()
    capture.sensors.complete = False
    with pytest.raises(RuntimeError, match="incomplete synchronized sensors"):
        capture.observe(NS(frame=capture.phase_frame + 1,
                           timestamp=NS(elapsed_seconds=.35)), 3100.0)
    capture.close()
    assert not (capture.root / "multimodal_frame_bundle.jsonl").read_text()
    assert json.loads((capture.root / "capture_summary.json").read_text())["status"] == "INVALID"


def test_capture_does_not_reuse_existing_logs(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.scene_capture.ExactFrameSensorSuite", FakeSensors)
    root = tmp_path / "scene"
    root.mkdir()
    (root / "world_state.jsonl").write_text("existing\n")
    capture = SceneCaptureSession(NS(tick=lambda: 1), _ego(), root, "scene_1", .05)
    with pytest.raises(FileExistsError):
        capture.start()
    assert (root / "world_state.jsonl").read_text() == "existing\n"
