"""Lossless image storage boundary for executed CARLA trajectory collection."""

import torch

IMAGE_STORAGE_VERSION = 'raw_rgb_uint8/1.0'


def store_rgb(images):
    if images.ndim not in (4,5) or images.shape[-3] != 3:
        raise ValueError('Expected [V,3,H,W] or [B,V,3,H,W] RGB images')
    if images.dtype == torch.uint8:
        return images.detach().cpu().clone()
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError('Expected uint8 RGB or finite float RGB in [0,1]')
    if images.min() < 0 or images.max() > 1:
        raise ValueError('Float RGB must be explicitly normalized to [0,1]')
    return (images.detach().cpu()*255).round().to(torch.uint8)


def image_audit(images):
    stored=store_rgb(images)
    return dict(dtype=str(stored.dtype),shape=list(stored.shape),
        min=int(stored.min()),max=int(stored.max()),
        unique_values=int(stored.unique().numel()),
        zero_or_255_fraction=float(((stored==0)|(stored==255)).float().mean()))
