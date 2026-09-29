import json
import shutil
from types import SimpleNamespace

import pytest

from benchmark.catalog import load_catalog
from evaluation.challenge_capture_audit import (
    _task_coverage_indices, audit_captures, validate_replay_selection,
)
from evaluation.model_rig_capture import ModelRigCapture
from evaluation.model_rig_replay import ModelRigReplayDataset
from evaluation.replay_benchmark import run
from evaluation.sensor_replay import DatasetValidationError, SynchronizedReplayDataset


def make_run(root, scene, *, truth_frame=11, truth_time=.55, missing_sensor=None,
             decisions=1):
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
        if name != missing_sensor:
            writer.register(name, camera, mounting)
    if missing_sensor != "lidar":
        lidar = SimpleNamespace(type_id="sensor.lidar.ray_cast", attributes={})
        writer.register("lidar", lidar, mounting)
    for index in range(decisions):
        sensor_frame = 10 + 10 * index
        sensor_time = .5 + .5 * index
        for name in cameras:
            if name != missing_sensor:
                writer.record(name, SimpleNamespace(frame=sensor_frame, timestamp=sensor_time,
                              raw_data=bytes([3, 2, 1, 255])), "uint8_bgra", width=1, height=1)
        if missing_sensor != "lidar":
            writer.record("lidar", SimpleNamespace(frame=sensor_frame, timestamp=sensor_time,
                          raw_data=bytes(16)), "float32_le_xyzi")
        writer.record_decision(decision={"simulation_frame": sensor_frame + 1,
                               "answer": "SECRET"},
            context={"timestamp_s": sensor_time + .05, "vehicle_state_tensor": [[1]]},
            camera_names=[name for name in cameras if name != missing_sensor],
            sensor_frame=sensor_frame, lidar_enabled=missing_sensor != "lidar",
            radar_observations={})
    writer.close()
    assessment = root / "benchmark"
    assessment.mkdir()
    (assessment / "manifest.json").write_text(json.dumps({"scene_id": scene,
        "source_sha256": "a" * 64}), encoding="utf-8")
    (assessment / "run_outcome.json").write_text(json.dumps({"status": "RECORDED_CHECKS_PASSED"}),
        encoding="utf-8")
    (assessment / "episode_truth.jsonl").write_text("".join(
        json.dumps({"frame": truth_frame + 10 * index,
                    "sim_time_s": truth_time + .5 * index,
                    "private_actor_truth": "SECRET"}) + "\n"
        for index in range(decisions)), encoding="utf-8")
    return root


def make_standard_run(root, scene, *, truth_x=1, time_offset=0, nested=False):
    sensor_root = root / "multimodal" if nested else root
    sensor_root.mkdir(parents=True)
    sensors = ("front_rgb", "left_rgb", "right_rgb", "rear_rgb", "lidar")
    (sensor_root / "sensor_calibration.json").write_text(json.dumps({
        "schema_version": "carla_sensor_calibration/1.0",
        "sensors": {name: {} for name in sensors},
    }))
    artifacts = {}
    for name in sensors:
        path = (f"rgb/{name}/00000011.png" if name != "lidar" else "lidar/00000011.ply")
        file = sensor_root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"sensor")
        artifacts[name] = {"path": path}
    (sensor_root / "lidar/00000011.xyzi.bin").write_bytes(bytes(16))
    artifacts["lidar_raw"] = {"path": "lidar/00000011.xyzi.bin", "encoding": "float32_le_xyzi"}
    (sensor_root / "multimodal_frame_bundle.jsonl").write_text(json.dumps({
        "simulation_frame": 11, "scene_id": scene, "timestamp_s": .55,
        "status": "COMPLETE", "synchronization": {"exact": True},
        "artifacts": artifacts,
    }) + "\n")
    (sensor_root / "world_state.jsonl").write_text(json.dumps({
        "simulation_frame": 11, "timestamp_s": .55,
        "ego": {"actor_id": 17, "speed_kmh": 35,
                 "location": {"x": 1, "y": 2, "z": 3}},
    }) + "\n")
    assessment = root / "benchmark"
    assessment.mkdir()
    (assessment / "manifest.json").write_text(json.dumps({
        "scene_id": scene, "source_sha256": "a" * 64,
    }))
    (assessment / "run_outcome.json").write_text(json.dumps({"status": "CHECK_FAILED"}))
    (assessment / "episode_truth.jsonl").write_text(json.dumps({
        "frame": 11, "sim_time_s": .55 + time_offset,
        "actors": {"17": {"x": truth_x, "y": 2, "z": 3}},
    }) + "\n")
    return root


def test_three_scene_selection_keeps_truth_separate(tmp_path):
    roots = {scene: make_run(tmp_path / scene, scene)
             for scene in ("scene_1", "scene_2", "scene_3")}
    report = audit_captures(roots, 3, require_task_coverage=False)
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
        audit_captures(roots, 3, require_task_coverage=False)


def test_audit_requires_real_recorded_frames(tmp_path):
    roots = {scene: make_run(tmp_path / scene, scene)
             for scene in ("scene_1", "scene_2", "scene_3")}
    with pytest.raises(DatasetValidationError, match="need"):
        audit_captures(roots, 1000, require_task_coverage=False)


@pytest.mark.parametrize("missing_sensor", ("left", "lidar"))
def test_audit_rejects_incomplete_model_modalities(tmp_path, missing_sensor):
    roots = {scene: make_run(tmp_path / scene, scene,
             missing_sensor=missing_sensor if scene == "scene_2" else None)
             for scene in ("scene_1", "scene_2", "scene_3")}
    with pytest.raises(DatasetValidationError, match="incomplete four-view"):
        audit_captures(roots, 3, require_task_coverage=False)


def test_selection_replays_all_decisions_before_scoring_sparse_frames(tmp_path):
    roots = {scene: make_run(tmp_path / scene, scene, decisions=3)
             for scene in ("scene_1", "scene_2", "scene_3")}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(audit_captures(roots, 3,
                         require_task_coverage=False)), encoding="utf-8")
    dataset = ModelRigReplayDataset(roots["scene_1"] / "model_inputs")

    class StatefulAdapter:
        def reset(self):
            self.calls = []

        def predict(self, frame):
            self.calls.append(frame.simulation_frame)
            return {"history_length": len(self.calls)}

    adapter = StatefulAdapter()
    summary = run(dataset, adapter, tmp_path / "replay", adapter_id="fixture",
                  adapter_config={}, selection_path=path, scene_id="scene_1")
    rows = [json.loads(line) for line in (tmp_path / "replay" / "predictions.jsonl").read_text().splitlines()]
    assert summary["status"] == "COMPLETED"
    assert adapter.calls == [11, 21, 31]
    assert [row["selected_for_evaluation"] for row in rows] == [False, True, False]
    assert rows[1]["output"]["history_length"] == 2
    assert summary["evaluation_selection"]["successful_outputs"] == 1

    relocated = tmp_path / "relocated"
    shutil.copytree(roots["scene_1"], relocated)
    relocated_dataset = ModelRigReplayDataset(relocated / "model_inputs")
    assert validate_replay_selection(path, "scene_1", relocated_dataset,
                                     relocated_dataset.integrity_manifest()) == {21}

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["scenes"][0]["input_dataset_sha256"] = "wrong"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="different model-input"):
        validate_replay_selection(path, "scene_1", dataset, dataset.integrity_manifest())


@pytest.mark.parametrize("scene", ("scene_1", "scene_2", "scene_3"))
def test_task_interval_selection_requires_real_route_coverage(scene):
    catalog = load_catalog(scene)
    frames = [SimpleNamespace(simulation_frame=index) for index in range(len(catalog.tasks))]
    truth = {index: {"route_s_m": task.activate_m + 1}
             for index, task in enumerate(catalog.tasks)}
    chosen = _task_coverage_indices(scene, frames, truth, catalog.source_sha256)
    assert set(chosen) == {task.task_id for task in catalog.tasks}
    with pytest.raises(DatasetValidationError, match=catalog.tasks[-1].task_id):
        _task_coverage_indices(scene, frames[:-1], truth, catalog.source_sha256)


def test_standard_sensor_selection_is_not_labeled_model_input(tmp_path):
    roots = {scene: make_standard_run(tmp_path / scene, scene, time_offset=10)
             for scene in ("scene_1", "scene_2", "scene_3")}
    report = audit_captures(roots, 3, require_task_coverage=False,
                            capture_format="synchronized")
    assert report["capture_format"] == "synchronized"
    assert all(not scene["model_input_equivalent"] for scene in report["scenes"])
    path = tmp_path / "standard_selection.json"
    path.write_text(json.dumps(report))
    dataset = SynchronizedReplayDataset(roots["scene_1"], require_calibration=True)

    class Adapter:
        def predict(self, frame):
            return {"speed": frame.vehicle_state["speed_kmh"]}

    summary = run(dataset, Adapter(), tmp_path / "standard_replay", adapter_id="fixture",
                  adapter_config={}, selection_path=path, scene_id="scene_1")
    assert summary["status"] == "COMPLETED"
    assert summary["evaluation_selection"]["successful_outputs"] == 1


def test_standard_capture_rejects_wrong_truth_pose(tmp_path):
    roots = {scene: make_standard_run(tmp_path / scene, scene,
             truth_x=5 if scene == "scene_2" else 1)
             for scene in ("scene_1", "scene_2", "scene_3")}
    with pytest.raises(DatasetValidationError, match="ego pose"):
        audit_captures(roots, 3, require_task_coverage=False,
                       capture_format="synchronized")


def test_standard_capture_rejects_invalid_session_summary(tmp_path):
    roots = {scene: make_standard_run(tmp_path / scene, scene)
             for scene in ("scene_1", "scene_2", "scene_3")}
    (roots["scene_1"] / "capture_summary.json").write_text(json.dumps({
        "status": "INVALID", "recorded_frames": 1,
    }))
    with pytest.raises(DatasetValidationError, match="marked invalid"):
        audit_captures(roots, 3, require_task_coverage=False,
                       capture_format="synchronized")


def test_nested_standard_capture_keeps_assessment_outside_sensor_root(tmp_path):
    roots = {scene: make_standard_run(tmp_path / scene, scene, nested=True)
             for scene in ("scene_1", "scene_2", "scene_3")}
    report = audit_captures(roots, 3, require_task_coverage=False,
                            capture_format="synchronized")
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(report))
    dataset = SynchronizedReplayDataset(roots["scene_3"] / "multimodal")
    assert validate_replay_selection(path, "scene_3", dataset,
                                     dataset.integrity_manifest()) == {11}
