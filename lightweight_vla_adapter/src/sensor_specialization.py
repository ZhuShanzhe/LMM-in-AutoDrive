"""Exact eval-only constant-input specialization; never infer free space."""

import torch
from torch import nn


class SensorSpecializedBEVEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self._cache = {}
        self._weight_stamp = None
        self.calls = {"full": 0, "zero_camera": 0, "zero_both": 0}

    @property
    def output_grid(self):
        return self.encoder.output_grid

    @property
    def hidden_size(self):
        return self.encoder.hidden_size

    def clear_cache(self):
        self._cache.clear()
        self._weight_stamp = None

    def train(self, mode=True):
        self.clear_cache()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def forward(self, camera_bev, lidar_bev):
        if camera_bev.ndim != 4 or lidar_bev.ndim != 4:
            raise ValueError("BEV inputs must have shape [B,C,H,W]")
        if camera_bev.shape[0] != lidar_bev.shape[0]:
            raise ValueError("camera and LiDAR BEV batch sizes differ")
        # Training/grad-enabled calls always retain the original graph.
        if self.training or torch.is_grad_enabled():
            self.calls["full"] += 1
            return self.encoder(camera_bev, lidar_bev)
        stamp = tuple((id(p), p._version) for p in (*self.encoder.parameters(),*self.encoder.buffers()))
        if stamp != self._weight_stamp:
            self.clear_cache()
            self._weight_stamp = stamp
        if bool(torch.count_nonzero(camera_bev)):
            self.calls["full"] += 1
            return self.encoder(camera_bev, lidar_bev)
        key = (tuple(camera_bev.shape), tuple(lidar_bev.shape), camera_bev.device,
               lidar_bev.device, camera_bev.dtype, lidar_bev.dtype)
        saved = self._cache.setdefault(key, {})
        if len(self._cache)>16:
            self._cache={key:saved}
        if not bool(torch.count_nonzero(lidar_bev)):
            if "tokens" not in saved:
                saved["tokens"] = self.encoder(camera_bev, lidar_bev).detach().clone()
            self.calls["zero_both"] += 1
            # Do not expose the mutable cached tensor to downstream callers.
            return saved["tokens"].clone()
        if "camera" not in saved:
            saved["camera"] = self.encoder.camera_stem(camera_bev).detach().clone()
        camera = saved["camera"]
        lidar = self.encoder.lidar_stem(lidar_bev)
        if camera.shape[-2:] != lidar.shape[-2:]:
            lidar = nn.functional.interpolate(lidar, size=camera.shape[-2:],
                                             mode="bilinear", align_corners=False)
        fused = self.encoder.pool(self.encoder.fusion(torch.cat([camera, lidar], dim=1)))
        self.calls["zero_camera"] += 1
        return fused.flatten(2).transpose(1, 2).contiguous()


def install_sensor_specialization(model):
    """Install after loading/fusing weights; serialized source stays unchanged."""
    if model.training:
        raise ValueError("Specialization requires an eval model")
    if isinstance(model.bev_encoder, SensorSpecializedBEVEncoder):
        return model
    model.bev_encoder = SensorSpecializedBEVEncoder(model.bev_encoder).eval()
    return model
