from types import SimpleNamespace
import numpy as np
import pytest
import torch
from evaluation.model_rig_preprocessing import build_sensor_batch
from carla_multiview_sensor import SynchronizedMultiviewCameraRig


def frame_fixture(tmp_path):
    image = tmp_path / "front.bin"
    image.write_bytes(bytes([3, 2, 1, 255]) * 4)
    points = np.array([[10, 0, 1, .8], [15, 1, -.2, .4]], dtype=np.float32)
    lidar = tmp_path / "lidar.bin"
    lidar.write_bytes(points.tobytes())
    context = dict(camera_preprocessing=dict(width=1, height=1, resize_front=True,
                                            implementation="camera_rgb_tensor/1.0"),
                   camera_view_mask=[[True, False, False, False]],
                   modality_mask=dict(text=True, front_rgb=True, left_rgb=False, right_rgb=False,
                                      rear_rgb=False, lidar_bev=True, vehicle_state=True, environment_state=True),
                   vehicle_state_tensor=[[1., 2.]], environment_state_tensor=[[3., 4.]])
    frame = SimpleNamespace(simulation_frame=10, timestamp_s=.5, context=context,
                            artifacts={"front": image, "lidar": lidar}, sensor_metadata={
                                "front": dict(encoding="uint8_bgra", width=2, height=2),
                                "lidar": dict(encoding="float32_le_xyzi")})
    return frame, points


def test_preprocessing_matches_online_lidar_and_camera(tmp_path):
    frame, points = frame_fixture(tmp_path)
    batch = build_sensor_batch(frame, text_tokens=torch.zeros(1, 2, 768),
                               text_mask=torch.ones(1, 2, dtype=torch.bool))
    online = SynchronizedMultiviewCameraRig._rasterize_lidar(SimpleNamespace(raw_data=points.tobytes()))
    assert torch.equal(batch.lidar_bev[0], online)
    assert batch.front_rgb.flatten().tolist() == [1, 2, 3]
    assert batch.front_rgb.dtype == torch.uint8
    assert not batch.left_rgb.any()
    assert batch.camera_view_mask.tolist() == [[True, False, False, False]]


def test_enabled_modality_not_silently_filled(tmp_path):
    frame, _ = frame_fixture(tmp_path)
    del frame.artifacts["lidar"]
    with pytest.raises(ValueError, match="missing enabled LiDAR"):
        build_sensor_batch(frame, text_tokens=torch.zeros(1, 2, 768), text_mask=torch.ones(1, 2))
