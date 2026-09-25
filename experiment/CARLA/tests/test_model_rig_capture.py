import json
from types import SimpleNamespace
import pytest
from evaluation.model_rig_capture import ModelRigCapture


def test_records_original_bytes_timestamp_and_duplicate_failure(tmp_path):
    root = tmp_path / "capture"
    recorder = ModelRigCapture(root)
    sample = SimpleNamespace(frame=10, timestamp=.5, raw_data=b"1234")
    recorder.record("front", sample, "uint8_bgra", width=1, height=1)
    recorder.record("front", sample, "uint8_bgra", width=1, height=1)
    recorder.close()
    recorder.close()
    assert (root / "front/00000010.bin").read_bytes() == b"1234"
    row = json.loads((root / "sensor_frames.jsonl").read_text())
    assert row["timestamp_s"] == .5
    assert row["encoding"] == "uint8_bgra"
    summary = json.loads((root / "capture_summary.json").read_text())
    assert summary["status"] == "INVALID"
    assert summary["sensor_records"] == 1
    assert len(summary["errors"]) == 1


def test_empty_capture_not_success_and_existing_directory_protected(tmp_path):
    root = tmp_path / "empty"
    recorder = ModelRigCapture(root)
    recorder.close()
    assert json.loads((root / "capture_summary.json").read_text())["status"] == "INVALID"
    with pytest.raises(FileExistsError):
        ModelRigCapture(root)


def test_consumed_frames_are_not_replaced_by_newer_callbacks(tmp_path):
    root = tmp_path / "selected"
    recorder = ModelRigCapture(root)
    for name, frame in (("front", 10), ("lidar", 10), ("radar_front", 9),
                        ("radar_rear", 8), ("radar_front", 11)):
        recorder.record(name, SimpleNamespace(frame=frame, timestamp=frame * .05,
                                              raw_data=b"1234"), "fixture")
    recorder.record_decision(decision={"simulation_frame": 11, "control_decision": {"action": "stop"}},
        context={"timestamp_s": .55}, camera_names=["front"], sensor_frame=10,
        lidar_enabled=True, radar_observations={"front": {"sensor_frame": 9}, "rear": {"sensor_frame": 8}})
    recorder.close()
    row = json.loads((root / "consumed_inputs.jsonl").read_text())
    assert row["sensor_frames"]["radar_front"] == 9
    assert row["sensor_frames"]["front"] == 10
    summary = json.loads((root / "capture_summary.json").read_text())
    assert summary["status"] == "RECORDED"
    assert summary["decision_records"] == 1


def test_missing_consumed_file_invalidates_capture(tmp_path):
    root = tmp_path / "missing"
    recorder = ModelRigCapture(root)
    recorder.record_decision(decision={"simulation_frame": 11}, context={},
        camera_names=["front"], sensor_frame=10, lidar_enabled=True,
        radar_observations={"front": {"sensor_frame": -1}})
    recorder.close()
    summary = json.loads((root / "capture_summary.json").read_text())
    assert summary["status"] == "INVALID"
    assert {item["sensor"] for item in summary["missing_consumed_sensor_records"]} == {"front", "lidar"}
    assert summary["errors"][0]["error"] == "consumed radar unavailable"


def test_storage_budget_halts_without_repeated_errors(tmp_path):
    root=tmp_path/'limited'
    recorder=ModelRigCapture(root,max_raw_bytes=4,min_free_bytes=0)
    for frame in range(10):
        recorder.record('front',SimpleNamespace(frame=frame,timestamp=frame*.05,raw_data=b'1234'),'fixture')
    recorder.close()
    summary=json.loads((root/'capture_summary.json').read_text())
    assert summary['status']=='INVALID'
    assert summary['raw_bytes_written']==4 and summary['sensor_records']==1
    assert len(summary['errors'])==1 and summary['capture_halted']
    assert (root/'capture_stopped.json').exists()


def test_disk_reserve_stops_before_writing_frame(tmp_path,monkeypatch):
    import evaluation.model_rig_capture as module
    monkeypatch.setattr(module.shutil,'disk_usage',lambda path:SimpleNamespace(free=8))
    root=tmp_path/'reserve'
    recorder=ModelRigCapture(root,min_free_bytes=6)
    recorder.record('front',SimpleNamespace(frame=0,timestamp=0,raw_data=b'1234'),'fixture')
    recorder.close()
    summary=json.loads((root/'capture_summary.json').read_text())
    assert summary['raw_bytes_written']==0
    assert summary['errors'][0]['error']=='disk_free_reserve_reached'
