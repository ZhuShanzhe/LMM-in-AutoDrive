#!/usr/bin/env python3
"""Focused offline contract test for the Bench2Drive sensor adapter."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np


def _add_runtime_paths() -> None:
    bench = Path(
        os.environ.get("BENCH2DRIVE_ROOT", "/root/autodl-tmp/Bench2Drive-submission")
    )
    repo = Path(
        os.environ.get(
            "REPOSITORY_ROOT",
            "/root/autodl-tmp/LMM-in-AutoDrive-basic-track",
        )
    )
    carla = Path(os.environ.get("CARLA_ROOT", "/root/autodl-tmp/CARLA_0.9.16"))
    for path in (
        bench / "leaderboard",
        bench / "scenario_runner",
        carla / "PythonAPI",
        carla / "PythonAPI/carla",
        repo,
        repo / "experiment/CARLA",
        repo / "basic_track_submission/bench2drive/team_code",
    ):
        sys.path.insert(0, str(path))


def main() -> None:
    _add_runtime_paths()
    from leaderboard.autoagents.agent_wrapper import validate_sensor_configuration
    from leaderboard.autoagents.autonomous_agent import Track
    from universal_vla_agent import (
        UniversalThreeSceneVLAAgent,
        _Bench2DriveSensorRig,
    )

    frame = 42
    bgra = np.zeros((224, 224, 4), dtype=np.uint8)
    bgra[:, :, 2] = 255
    lidar = np.array(
        [[10.0, 0.0, 0.5, 1.0], [20.0, 2.0, 0.2, 0.5]],
        dtype=np.float32,
    )
    radar = np.array([[12.0, 0.0, 0.0, -4.0]], dtype=np.float32)
    data = {
        name: (frame, bgra.copy())
        for name in ("front", "left", "right", "rear")
    }
    data.update(
        {
            "lidar": (frame, lidar),
            "front_radar": (frame, radar),
            "rear_radar": (frame, np.empty((0, 4), dtype=np.float32)),
        }
    )
    rig = _Bench2DriveSensorRig()
    rig.update(data)
    observed_frame, images, mask, lidar_bev, wait_ms = rig.latest_multisensor(
        minimum_frame=frame
    )
    assert observed_frame == frame
    assert images.shape == (1, 4, 3, 224, 224)
    assert mask.tolist() == [[True, True, True, True]]
    assert lidar_bev.shape == (1, 4, 64, 64)
    assert wait_ms == 0.0
    front = rig.latest_radar("front", maximum_frame=frame)
    assert front["closing_candidate_count"] == 1
    assert front["nearest_closing_distance_m"] == 12.0
    agent = object.__new__(UniversalThreeSceneVLAAgent)
    sensors = agent.sensors()
    assert len(sensors) == 7
    validate_sensor_configuration(sensors, Track.SENSORS, "SENSORS")
    print("BENCH2DRIVE_ADAPTER_CONTRACT_OK")


if __name__ == "__main__":
    main()
