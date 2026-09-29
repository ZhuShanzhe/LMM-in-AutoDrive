import json
from pathlib import Path

import pytest

from evaluation.sensor_replay import (
    DatasetValidationError,
    REQUIRED_MODALITIES,
    SynchronizedReplayDataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dataset(tmp_path: Path, frames=(10, 14)) -> Path:
    states = []
    bundles = []
    for index, frame in enumerate(frames):
        timestamp = 1.0 + index * 0.2
        states.append(
            {
                "simulation_frame": frame,
                "timestamp_s": timestamp,
                "ego": {"speed_kmh": 30.0},
            }
        )
        artifacts = {}
        for name in REQUIRED_MODALITIES:
            relative = (
                "rgb/{0}/{1:08d}.png".format(name, frame)
                if name != "lidar"
                else "lidar/{0:08d}.ply".format(frame)
            )
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")
            artifacts[name] = {"path": relative}
        bundles.append(
            {
                "scene_id": "scene_2",
                "simulation_frame": frame,
                "timestamp_s": timestamp,
                "status": "COMPLETE",
                "synchronization": {
                    "exact": True,
                    "adjacent_frame_fill_used": False,
                },
                "artifacts": artifacts,
                "driving_intent_request_id": "intent-1",
            }
        )
    _write_jsonl(tmp_path / "world_state.jsonl", states)
    _write_jsonl(tmp_path / "driving_intent.jsonl", [{
        "schema_version": "1.2.0", "request_id": "intent-1",
        "simulation_frame": frames[0], "timestamp_s": 1.0,
        "input": {"raw_text": "Keep the current lane."},
        "intent": {"steps": []},
        "parse_result": {"source": "competition_schedule"},
        "route_s_m": 900,
    }])
    _write_jsonl(tmp_path / "multimodal_frame_bundle.jsonl", bundles)
    return tmp_path


def test_replay_preserves_frame_order_and_recorded_timing(tmp_path: Path) -> None:
    dataset = SynchronizedReplayDataset(_dataset(tmp_path))
    observed = []
    sleeps = []
    count = dataset.replay(
        lambda frame: observed.append(frame.simulation_frame),
        speed=2.0,
        sleep=sleeps.append,
    )
    assert count == 2
    assert observed == [10, 14]
    assert sleeps == pytest.approx([0.1])
    assert dataset.summary()["exact_frame_synchronization"] is True


def test_replay_rejects_missing_artifact(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    (root / "lidar" / "00000010.ply").unlink()
    with pytest.raises(DatasetValidationError, match="artifact is missing"):
        SynchronizedReplayDataset(root)


def test_replay_rejects_inexact_bundle(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    bundles = [
        json.loads(line)
        for line in (root / "multimodal_frame_bundle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    bundles[0]["synchronization"]["exact"] = False
    _write_jsonl(root / "multimodal_frame_bundle.jsonl", bundles)
    with pytest.raises(DatasetValidationError, match="not an exact complete"):
        SynchronizedReplayDataset(root)


def test_replay_rejects_artifact_path_escape(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    bundles = [
        json.loads(line)
        for line in (root / "multimodal_frame_bundle.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    bundles[0]["artifacts"]["front_rgb"]["path"] = "../outside.png"
    _write_jsonl(root / "multimodal_frame_bundle.jsonl", bundles)
    with pytest.raises(DatasetValidationError, match="escapes dataset root"):
        SynchronizedReplayDataset(root)


def test_integrity_manifest_is_deterministic_and_content_sensitive(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path)
    first = SynchronizedReplayDataset(root).integrity_manifest()
    second = SynchronizedReplayDataset(root).integrity_manifest()
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert len(first["frames"]) == 2

    (root / "lidar" / "00000010.ply").write_bytes(b"changed")
    changed = SynchronizedReplayDataset(root).integrity_manifest()
    assert changed["dataset_sha256"] != first["dataset_sha256"]


def test_policy_replay_excludes_truth_but_manifest_keeps_it(tmp_path):
    root = _dataset(tmp_path)
    initial = SynchronizedReplayDataset(root).integrity_manifest()
    path = root / "world_state.jsonl"
    states = [json.loads(line) for line in path.read_text().splitlines()]
    for state in states:
        state.update(lane={"lane_id": -2}, safety={"collision_count": 1}, actors=[123])
        state["ego"].update(actor_id=99, speed_limit_kmh=60,
                            location={"x": 1, "y": 2, "z": 3, "target_actor": 123})
    _write_jsonl(path, states)
    dataset = SynchronizedReplayDataset(root)
    observed = []
    dataset.replay(observed.append)
    assert observed[0].vehicle_state == {
        "speed_kmh": 30.0, "location": {"x": 1.0, "y": 2.0, "z": 3.0}
    }
    assert dataset.integrity_manifest()["dataset_sha256"] != initial["dataset_sha256"]


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), 1.0, 0.9])
def test_replay_rejects_bad_timestamps(tmp_path, timestamp):
    root = _dataset(tmp_path)
    for filename in ("world_state.jsonl", "multimodal_frame_bundle.jsonl"):
        path = root / filename
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["timestamp_s"] = timestamp
        _write_jsonl(path, rows)
    with pytest.raises(DatasetValidationError):
        SynchronizedReplayDataset(root)


def test_replay_rejects_state_bundle_time_disagreement(tmp_path):
    root = _dataset(tmp_path)
    path = root / "world_state.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["timestamp_s"] = 1.01
    _write_jsonl(path, rows)
    with pytest.raises(DatasetValidationError, match="timestamps disagree"):
        SynchronizedReplayDataset(root)


def test_replay_links_instruction_and_preserves_provenance(tmp_path):
    dataset = SynchronizedReplayDataset(_dataset(tmp_path))
    frames = list(dataset)
    assert frames[1].driving_intent["input"]["raw_text"] == "Keep the current lane."
    assert frames[1].driving_intent["parse_result"]["source"] == "competition_schedule"
    assert "route_s_m" not in frames[1].driving_intent
    assert frames[0].driving_intent is not frames[1].driving_intent


@pytest.mark.parametrize("fault", ["missing", "duplicate", "future_frame", "future_time"])
def test_replay_rejects_invalid_instruction_reference(tmp_path, fault):
    root = _dataset(tmp_path)
    path = root / "driving_intent.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "missing":
        path.unlink()
    else:
        if fault == "duplicate":
            records.append(records[0])
        elif fault == "future_frame":
            records[0]["simulation_frame"] = 11
        else:
            records[0]["timestamp_s"] = 1.1
        _write_jsonl(path, records)
    with pytest.raises(DatasetValidationError):
        SynchronizedReplayDataset(root)


def test_instruction_change_changes_same_source_hash(tmp_path):
    root = _dataset(tmp_path)
    before = SynchronizedReplayDataset(root).integrity_manifest()
    path = root / "driving_intent.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["input"]["raw_text"] = "Slow down."
    _write_jsonl(path, records)
    after = SynchronizedReplayDataset(root).integrity_manifest()
    assert before["dataset_sha256"] != after["dataset_sha256"]


def test_no_command_frame_does_not_inherit_last_instruction(tmp_path):
    root = _dataset(tmp_path)
    path = root / "multimodal_frame_bundle.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["driving_intent_request_id"] = None
    _write_jsonl(path, records)
    assert list(SynchronizedReplayDataset(root))[1].driving_intent is None


def test_scheduled_text_is_frame_local_and_hashed(tmp_path):
    root = _dataset(tmp_path)
    path = root / "command_context.jsonl"
    rows = [
        {"simulation_frame": 10, "timestamp_s": 1.0,
         "source": "scene_schedule_not_model_parse",
         "commands": [{"id": "c01", "text": "Slow down."}],
         "route_s_m": 700},
        {"simulation_frame": 14, "timestamp_s": 1.2,
         "source": "scene_schedule_not_model_parse", "commands": []},
    ]
    _write_jsonl(path, rows)
    dataset = SynchronizedReplayDataset(root)
    frames = list(dataset)
    assert frames[0].command_context == {
        "source": "scene_schedule_not_model_parse",
        "commands": [{"id": "c01", "text": "Slow down."}],
    }
    assert frames[1].command_context["commands"] == []
    assert dataset.summary()["frames_with_scheduled_text"] == 1
    before = dataset.integrity_manifest()["dataset_sha256"]
    rows[0]["commands"][0]["text"] = "Stop."
    _write_jsonl(path, rows)
    assert SynchronizedReplayDataset(root).integrity_manifest()["dataset_sha256"] != before


@pytest.mark.parametrize("fault", ["missing", "duplicate", "future_time", "bad_source"])
def test_replay_rejects_invalid_scheduled_text(tmp_path, fault):
    root = _dataset(tmp_path)
    rows = [
        {"simulation_frame": frame, "timestamp_s": timestamp,
         "source": "scene_schedule_not_model_parse", "commands": []}
        for frame, timestamp in [(10, 1.0), (14, 1.2)]
    ]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "future_time":
        rows[0]["timestamp_s"] = 1.1
    else:
        rows[0]["source"] = "parsed_model"
    _write_jsonl(root / "command_context.jsonl", rows)
    with pytest.raises(DatasetValidationError):
        SynchronizedReplayDataset(root)


def test_missing_calibration_is_explicit_and_can_be_required(tmp_path):
    root = _dataset(tmp_path)
    assert not SynchronizedReplayDataset(root).summary()["sensor_calibration_available"]
    with pytest.raises(DatasetValidationError, match="missing sensor_calibration"):
        SynchronizedReplayDataset(root, require_calibration=True)


def test_raw_lidar_is_replayed_and_included_in_hash(tmp_path):
    import struct
    from evaluation.raw_lidar import load_xyzi
    root = _dataset(tmp_path)
    path = root / "multimodal_frame_bundle.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    raw = root / "lidar" / "00000010.xyzi.bin"
    raw.write_bytes(struct.pack("<4f", 1, 2, 3, .75))
    rows[0]["artifacts"]["lidar_raw"] = {
        "path": "lidar/00000010.xyzi.bin", "encoding": "float32_le_xyzi"}
    _write_jsonl(path, rows)
    dataset = SynchronizedReplayDataset(root)
    assert load_xyzi(list(dataset)[0].artifacts["lidar_raw"])[0].tolist() == [1, 2, 3, .75]
    before = dataset.integrity_manifest()["dataset_sha256"]
    raw.write_bytes(struct.pack("<4f", 1, 2, 3, .25))
    assert dataset.integrity_manifest()["dataset_sha256"] != before
    raw.write_bytes(b"broken")
    with pytest.raises(DatasetValidationError, match="byte length"):
        SynchronizedReplayDataset(root)


def test_capture_preserves_original_xyzi(tmp_path):
    import struct
    from types import SimpleNamespace
    from evaluation.multimodal import ExactFrameSensorSuite
    suite = ExactFrameSensorSuite(None, None, None, tmp_path, .05)
    raw = struct.pack("<4f", 10, 2, -1, .5)
    measurement = SimpleNamespace(frame=10, raw_data=raw,
        save_to_disk=lambda path: Path(path).write_bytes(b"ply"))
    suite._save_lidar(tmp_path, measurement)
    assert (tmp_path / "00000010.xyzi.bin").read_bytes() == raw
    assert suite.latest_frames["lidar"] == 10
