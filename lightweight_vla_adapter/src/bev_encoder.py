from __future__ import annotations

import torch
from torch import nn


def _groups(channels: int) -> int:
    for value in (16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class _ConvStem(nn.Sequential):
    def __init__(self, input_channels: int, hidden_size: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, hidden_size // 2, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden_size // 2), hidden_size // 2),
            nn.GELU(),
            nn.Conv2d(hidden_size // 2, hidden_size, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden_size), hidden_size),
            nn.GELU(),
        )


class LightweightBEVEncoder(nn.Module):
    """Fuse precomputed camera and LiDAR BEV maps into compact memory tokens.

    Raw camera lifting and point-cloud voxelization stay outside this module.
    Inputs may come from BEVFusion, PointPillars/LSS, or the CARLA structured
    rasterizer used by the integration tests.
    """

    def __init__(
        self,
        camera_channels: int,
        lidar_channels: int,
        hidden_size: int = 256,
        output_grid: tuple[int, int] = (8, 8),
    ) -> None:
        super().__init__()
        if min(camera_channels, lidar_channels, hidden_size) <= 0:
            raise ValueError("BEV channel counts and hidden_size must be positive")
        self.camera_stem = _ConvStem(camera_channels, hidden_size)
        self.lidar_stem = _ConvStem(lidar_channels, hidden_size)
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, 1, bias=False),
            nn.GroupNorm(_groups(hidden_size), hidden_size),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_grid)
        self.output_grid = output_grid
        self.hidden_size = hidden_size

    def forward(
        self, camera_bev: torch.Tensor, lidar_bev: torch.Tensor
    ) -> torch.Tensor:
        if camera_bev.ndim != 4 or lidar_bev.ndim != 4:
            raise ValueError("BEV inputs must have shape [B,C,H,W]")
        if camera_bev.shape[0] != lidar_bev.shape[0]:
            raise ValueError("camera and LiDAR BEV batch sizes differ")
        camera = self.camera_stem(camera_bev)
        lidar = self.lidar_stem(lidar_bev)
        if camera.shape[-2:] != lidar.shape[-2:]:
            lidar = nn.functional.interpolate(
                lidar,
                size=camera.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        fused = self.pool(self.fusion(torch.cat([camera, lidar], dim=1)))
        return fused.flatten(2).transpose(1, 2).contiguous()
