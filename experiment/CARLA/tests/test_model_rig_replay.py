import json
from types import SimpleNamespace
import pytest
from evaluation.model_rig_capture import ModelRigCapture
from evaluation.model_rig_replay import ModelRigReplayDataset, decode_sensor
from evaluation.replay_benchmark import run
from evaluation.sensor_replay import DatasetValidationError


def capture(root):
    writer = ModelRigCapture(root)
    actor = SimpleNamespace(type_id="sensor.camera.rgb", attributes={
        "image_size_x": "1", "image_size_y": "1", "fov": "90"})
    writer.register("front", actor, SimpleNamespace(get_matrix=lambda: [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    writer.record("front", SimpleNamespace(frame=10, timestamp=.5, raw_data=bytes([3, 2, 1, 255])),
                  "uint8_bgra", width=1, height=1)
    writer.record_decision(decision={"simulation_frame": 11, "answer": "SECRET"},
        context={"timestamp_s": .55, "vehicle_state_tensor": [[1]], "unexpected_answer": "SECRET"},
        camera_names=["front"], sensor_frame=10, lidar_enabled=False, radar_observations={})
    writer.close()


def test_native_replay_decodes_without_exposing_answers(tmp_path):
    root = tmp_path / "capture"
    capture(root)
    dataset = ModelRigReplayDataset(root)
    frame = list(dataset)[0]
    assert decode_sensor(frame, "front").tolist() == [[[1, 2, 3]]]
    assert not hasattr(frame, "decision")
    assert "unexpected_answer" not in frame.context
    before = dataset.integrity_manifest()["dataset_sha256"]
    path = root / "consumed_inputs.jsonl"
    row = json.loads(path.read_text())
    row["decision"]["answer"] = "DIFFERENT"
    path.write_text(json.dumps(row) + "\n")
    assert ModelRigReplayDataset(root).integrity_manifest()["dataset_sha256"] == before
    adapter = SimpleNamespace(predict=lambda item: {"rgb": decode_sensor(item, "front").tolist()})
    summary = run(dataset, adapter, tmp_path / "result", adapter_id="fixture", adapter_config={})
    assert summary["successful_outputs"] == 1


def test_native_replay_rejects_missing_bytes(tmp_path):
    root = tmp_path / "capture"
    capture(root)
    (root / "front/00000010.bin").write_bytes(b"bad")
    with pytest.raises(DatasetValidationError, match="truncated"):
        ModelRigReplayDataset(root)
