import json
import pytest
from evaluation.replay_benchmark import run, run_repeated
from evaluation.sensor_replay import ReplayFrame


class Dataset:
    def __init__(self):
        self.frames = [ReplayFrame(i, i * .1, "scene_1", {}, {"speed_kmh": 30}, None)
                       for i in range(3)]

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def integrity_manifest(self):
        return {"dataset_sha256": "fixture"}


class Adapter:
    def __init__(self):
        self.syncs = 0
        self.closed = False

    def reset(self):
        self.calls = 0

    def synchronize(self):
        self.syncs += 1

    def predict(self, frame):
        self.calls += 1
        frame.vehicle_state["speed_kmh"] = 0
        return {"action": "KEEP_LANE"}

    def close(self):
        self.closed = True


def test_records_predictions_timing_and_preserves_inputs(tmp_path):
    dataset, adapter = Dataset(), Adapter()
    output = tmp_path / "run"
    summary = run(dataset, adapter, output, adapter_id="fixture", adapter_config={}, max_frames=2)
    assert summary["status"] == "COMPLETED"
    assert summary["attempted_frames"] == 2
    assert summary["closed_loop_success"] is None
    assert summary["latency_ms"]["p95"] >= 0
    assert adapter.closed and adapter.syncs == 4
    assert dataset.frames[0].vehicle_state["speed_kmh"] == 30
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [row["simulation_frame"] for row in rows] == [0, 1]


def test_invalid_model_output_stops_and_closes(tmp_path):
    adapter = Adapter()
    adapter.predict = lambda frame: {"speed": float("nan")}
    summary = run(Dataset(), adapter, tmp_path / "bad", adapter_id="fixture", adapter_config={})
    assert summary["status"] == "FAILED"
    assert summary["failed_frames"] == 1
    assert summary["attempted_frames"] == 1
    assert adapter.closed


def test_never_overwrites_existing_run(tmp_path):
    with pytest.raises(FileExistsError):
        run(Dataset(), Adapter(), tmp_path, adapter_id="fixture", adapter_config={})


def test_repeated_models_are_fresh_and_outputs_comparable(tmp_path):
    adapters=[]
    def factory(config):
        adapter=Adapter()
        adapters.append(adapter)
        return adapter
    result=run_repeated(Dataset(),factory,tmp_path/'suite',repetitions=2,
                        adapter_id='fixture',adapter_config={})
    assert result['status']=='COMPLETED'
    assert result['exact_predictions_repeatable'] is True
    assert len(adapters)==2 and all(a.closed for a in adapters)


def test_initialization_failure_stops_repeated_suite(tmp_path):
    def factory(config):
        raise RuntimeError('missing weights')
    result=run_repeated(Dataset(),factory,tmp_path/'suite',repetitions=3,
                        adapter_id='fixture',adapter_config={})
    assert result['status']=='FAILED' and len(result['runs'])==1
    assert result['exact_predictions_repeatable'] is None
