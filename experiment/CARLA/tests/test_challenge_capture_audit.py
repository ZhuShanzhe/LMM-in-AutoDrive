import json
from types import SimpleNamespace

import pytest

from evaluation.challenge_capture_audit import audit_captures
from evaluation.model_rig_capture import ModelRigCapture
from evaluation.sensor_replay import DatasetValidationError


def make_run(root, scene, *, truth_frame=11, truth_time=.55, missing_sensor=None):
    inputs = root / "model_inputs"
    writer = ModelRigCapture(inputs)
    camera = SimpleNamespace(type_id="sensor.camera.rgb", attributes={
        "image_size_x": "1", "image_size_y": "1", "fov": "90",
    })
    mounting = SimpleNamespace(get_matrix=lambda: [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    ])
    cameras = ("front", "left", "right", "rear")
    for name in cameras:
        if name == missing_sensor:
            continue
        writer.register(name, camera, mounting)
        writer.record(name, SimpleNamespace(frame=10, timestamp=.5,
                      raw_data=bytes([3, 2, 1, 255])), "uint8_bgra", width=1, height=1)
    if missing_sensor != "lidar":
        lidar = SimpleNamespace(type_id="sensor.lidar.ray_cast", attributes={})
        writer.register("lidar", lidar, mounting)
        writer.record("lidar", SimpleNamespace(frame=10, timestamp=.5,
                      raw_data=bytes(16)), "float32_le_xyzi")
    writer.record_decision(decision={"simulation_frame": 11, "answer": "SECRET"},
        context={"timestamp_s": .55, "vehicle_state_tensor": [[1]]},
        camera_names=[name for name in cameras if name != missing_sensor],
        sensor_frame=10, lidar_enabled=missing_sensor != "lidar",
        radar_observations={})
    writer.close()
    assessment = root / "benchmark"
    assessment.mkdir()
    (assessment / "manifest.json").write_text(json.dumps({"scene_id": scene,
        "source_sha256": "a" * 64}), encoding="utf-8")
    (assessment / "run_outcome.json").write_text(json.dumps({"status": "RECORDED_CHECKS_PASSED"}),
        encoding="utf-8")
    (assessment / "episode_truth.jsonl").write_text(json.dumps({"frame": truth_frame,
        "sim_time_s": truth_time, "private_actor_truth": "SECRET"}) + "\n", encoding="utf-8")
    return root


def test_three_scene_selection_keeps_truth_separate(tmp_path):
    roots = {scene: make_run(tmp_path / scene, scene)
             for scene in ("scene_1", "scene_2", "scene_3")}
    report = audit_captures(roots, 3)
    assert report["frame_count"] == 3
    assert report["truth_in_policy_inputs"] is False
    assert [row["scene_id"] for row in report["scenes"]] == list(roots)
    for row in report["scenes"]:
        assert len(row["selected"]) == 1
        assert "SECRET" not in json.dumps(row)
        assert len(row["selected"][0]["truth_sha256"]) == 64


@pytest.mark.parametrize("fault", ("missing", "mistimed", "wrong_scene", "unbound"))
def test_audit_rejects_unmatched_evidence(tmp_path, fault):
    roots = {scene: make_run(tmp_path / scene, scene,
             truth_frame=12 if fault == "missing" and scene == "scene_2" else 11,
             truth_time=.6 if fault == "mistimed" and scene == "scene_2" else .55)
             for scene in ("scene_1", "scene_2", "scene_3")}
    if fault == "wrong_scene":
        path = roots["scene_2"] / "benchmark" / "manifest.json"
        path.write_text(json.dumps({"scene_id": "scene_1"}), encoding="utf-8")
    if fault == "unbound":
        path = roots["scene_2"] / "benchmark" / "manifest.json"
        path.write_text(json.dumps({"scene_id": "scene_2"}), encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        audit_captures(roots, 3)


def test_audit_requires_real_recorded_frames(tmp_path):
    roots = {scene: make_run(tmp_path / scene, scene)
             for scene in ("scene_1", "scene_2", "scene_3")}
    with pytest.raises(DatasetValidationError, match="need"):
        audit_captures(roots, 1000)


@pytest.mark.parametrize("missing_sensor", ("left", "lidar"))
def test_audit_rejects_incomplete_model_modalities(tmp_path, missing_sensor):
    roots = {scene: make_run(tmp_path / scene, scene,
             missing_sensor=missing_sensor if scene == "scene_2" else None)
             for scene in ("scene_1", "scene_2", "scene_3")}
    with pytest.raises(DatasetValidationError, match="incomplete four-view"):
        audit_captures(roots, 3)
