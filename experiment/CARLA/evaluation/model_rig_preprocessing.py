"""Build the team's sensor batch using the online rig's preprocessing code."""

from types import SimpleNamespace
import numpy as np
import torch

from carla_multiview_sensor import CAMERA_ORDER, SynchronizedMultiviewCameraRig, camera_rgb_tensor
from evaluation.model_rig_replay import decode_sensor


def build_sensor_batch(frame, *, text_tokens, text_mask):
    """Language features must come from the real encoder; never synthesize them."""
    from lightweight_vla_adapter.src.unified_sensor_batch import UnifiedSensorBatch, MODALITY_KEYS
    context = frame.context
    settings = context.get("camera_preprocessing")
    if not isinstance(settings, dict) or settings.get("implementation") != "camera_rgb_tensor/1.0":
        raise ValueError("missing or unsupported recorded camera preprocessing")
    width, height = settings.get("width"), settings.get("height")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise ValueError("invalid recorded camera dimensions")
    if not isinstance(settings.get("resize_front"), bool):
        raise ValueError("missing recorded front resize mode")
    raw_mask = np.asarray(context["camera_view_mask"])
    if raw_mask.shape != (1, 4) or raw_mask.dtype != np.bool_:
        raise ValueError("camera mask must be bool [1,4]")
    modalities = context["modality_mask"]
    if any(type(modalities.get(key)) is not bool for key in MODALITY_KEYS):
        raise ValueError("incomplete or invalid modality mask")
    views = {}
    for index, name in enumerate(CAMERA_ORDER):
        available = bool(raw_mask[0, index])
        if modalities[name + "_rgb"] != available:
            raise ValueError("camera mask and modality mask disagree")
        if available:
            if name not in frame.artifacts:
                raise ValueError("missing enabled camera: " + name)
            rgb = decode_sensor(frame, name)
            tensor = camera_rgb_tensor(rgb, width, height, resize=name == "front" and settings["resize_front"])
            if tuple(tensor.shape) != (3, height, width):
                raise ValueError("camera dimensions disagree with online configuration")
        else:
            tensor = torch.zeros((3, height, width), dtype=torch.uint8)
        views[name + "_rgb"] = tensor.unsqueeze(0)
    if modalities["lidar_bev"]:
        if "lidar" not in frame.artifacts:
            raise ValueError("missing enabled LiDAR")
        points = decode_sensor(frame, "lidar")
        if frame.sensor_metadata["lidar"]["encoding"] != "float32_le_xyzi":
            raise ValueError("LiDAR must use original XYZI")
        lidar = SynchronizedMultiviewCameraRig._rasterize_lidar(
            SimpleNamespace(raw_data=np.asarray(points, dtype=np.float32).tobytes())).unsqueeze(0)
    else:
        lidar = torch.zeros((1, 4, 64, 64), dtype=torch.float32)
    tensors = {}
    for name in ("vehicle_state", "environment_state"):
        tensor = torch.tensor(context[name + "_tensor"], dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[0] != 1 or not torch.isfinite(tensor).all():
            raise ValueError("invalid recorded " + name)
        tensors[name] = tensor
    if text_tokens is None or text_mask is None:
        raise ValueError("real language encoder tokens and mask required")
    batch = UnifiedSensorBatch(
        text_tokens=text_tokens, text_mask=text_mask,
        lidar_bev=lidar, camera_view_mask=torch.from_numpy(raw_mask.copy()),
        modality_mask=dict(modalities), frame_id="carla_" + str(frame.simulation_frame),
        timestamp_s=frame.timestamp_s, **views, **tensors)
    batch.validate()
    return batch
