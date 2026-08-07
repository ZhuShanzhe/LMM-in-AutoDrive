from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from carla_multiview_sensor import SynchronizedMultiviewCameraRig
from control.structured_vla_scene_bridge_policy import StructuredVlaSceneBridgePolicy


class _Measurement:
    def __init__(self, points: np.ndarray) -> None:
        self.raw_data = points.astype(np.float32).tobytes()


def test_real_lidar_points_are_rasterized_into_nonzero_bev() -> None:
    measurement = _Measurement(
        np.asarray(
            [
                [10.0, 0.0, 0.5, 0.8],
                [10.1, 0.1, 1.0, 0.4],
                [-30.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    bev = SynchronizedMultiviewCameraRig._rasterize_lidar(measurement)
    assert bev.shape == (4, 64, 64)
    assert bev.dtype.is_floating_point
    assert int((bev[0] > 0).sum()) >= 1
    assert float(bev[1].max()) > 0.0
    assert float(bev[3].max()) > 0.0


def test_policy_strips_actor_raster_but_preserves_physical_lidar() -> None:
    batch = SimpleNamespace(
        camera_bev=torch.ones(1, 8, 4, 4),
        lidar_bev=torch.ones(1, 4, 4, 4),
        candidate_features=torch.ones(1, 2, 12),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    batch, entity_ids = StructuredVlaSceneBridgePolicy._strip_privileged_inputs(
        batch, [["actor-1", "actor-2"]]
    )
    assert int(torch.count_nonzero(batch.camera_bev)) == 0
    assert int(torch.count_nonzero(batch.candidate_features)) == 0
    assert int(torch.count_nonzero(batch.candidate_mask)) == 0
    assert int(torch.count_nonzero(batch.lidar_bev)) > 0
    assert entity_ids == [[]]


def test_policy_temporal_state_contains_no_world_objects() -> None:
    state = StructuredVlaSceneBridgePolicy._sensor_policy_state(
        {
            "frame_id": "carla_10",
            "timestamp_s": 1.5,
            "ego": {"speed_mps": 4.0},
            "objects": [{"entity_id": "privileged"}],
        }
    )
    assert state == {
        "frame_id": "carla_10",
        "timestamp_s": 1.5,
        "ego": {"speed_mps": 4.0},
        "objects": [],
    }


def test_longitudinal_proposal_marks_below_speed_setpoint_as_deceleration() -> None:
    proposal = {"action": "keep_lane", "target_speed_kmh": 20.0}
    world_state = {"ego": {"speed_mps": 10.0}}

    normalized = StructuredVlaSceneBridgePolicy._normalize_longitudinal_proposal(
        proposal, world_state
    )

    assert normalized["action"] == "decelerate"
    assert normalized["target_speed_kmh"] == 20.0
    assert proposal["action"] == "keep_lane"


def test_longitudinal_proposal_keeps_consistent_action() -> None:
    proposal = {"action": "keep_lane", "target_speed_kmh": 35.0}
    world_state = {"ego": {"speed_mps": 10.0}}

    normalized = StructuredVlaSceneBridgePolicy._normalize_longitudinal_proposal(
        proposal, world_state
    )

    assert normalized == proposal


def test_temporal_state_is_scoped_to_atomic_substep() -> None:
    first = {"request_id": "compound-01", "source_step_id": "step_01"}
    second = {"request_id": "compound-01", "source_step_id": "step_02"}

    assert StructuredVlaSceneBridgePolicy._temporal_stream_id(first) == (
        "compound-01:step_01"
    )
    assert StructuredVlaSceneBridgePolicy._temporal_stream_id(second) == (
        "compound-01:step_02"
    )


def test_sparse_sensor_cadence_uses_bounded_complete_bundle_age() -> None:
    assert StructuredVlaSceneBridgePolicy._minimum_sensor_frame(100, 3) == 97
    assert StructuredVlaSceneBridgePolicy._minimum_sensor_frame(1, 3) == 0
    assert StructuredVlaSceneBridgePolicy._minimum_sensor_frame(None, 3) is None
