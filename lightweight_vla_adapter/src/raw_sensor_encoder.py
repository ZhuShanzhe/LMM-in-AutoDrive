from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiviewImageEncoder(nn.Module):
    """Encode synchronized raw RGB views into compact visual tokens.

    The ImageNet initialization helper is deliberately separate from
    ``__init__``.  Runtime construction therefore never downloads weights;
    released checkpoints are fully self-contained.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        num_views: int = 4,
        token_grid: tuple[int, int] = (2, 2),
    ) -> None:
        super().__init__()
        from torchvision.models import mobilenet_v3_small

        backbone = mobilenet_v3_small(weights=None)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(token_grid)
        self.projection = nn.Linear(576, hidden_size)
        self.view_embeddings = nn.Parameter(
            torch.empty(num_views, hidden_size)
        )
        nn.init.normal_(self.view_embeddings, std=0.02)
        self.num_views = int(num_views)
        self.token_grid = tuple(int(value) for value in token_grid)
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    def load_imagenet_initialization(self) -> None:
        """Load the official torchvision MobileNetV3-Small initialization."""
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        initialized = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.backbone.load_state_dict(initialized.features.state_dict())

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("camera_images must have shape [B,V,3,H,W]")
        batch, views = images.shape[:2]
        if views != self.num_views:
            raise ValueError(
                f"expected {self.num_views} camera views, received {views}"
            )
        pixels = images.reshape(batch * views, *images.shape[2:])
        if not pixels.is_floating_point():
            pixels = pixels.float().div_(255.0)
        else:
            pixels = pixels.clamp(0.0, 1.0)
        pixels = F.interpolate(
            pixels,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        pixels = (pixels - self.image_mean) / self.image_std
        features = self.pool(self.backbone(pixels))
        features = features.flatten(2).transpose(1, 2)
        tokens_per_view = features.shape[1]
        tokens = self.projection(features).reshape(
            batch, views, tokens_per_view, -1
        )
        tokens = tokens + self.view_embeddings.view(1, views, 1, -1)
        tokens = tokens.reshape(batch, views * tokens_per_view, -1)
        if view_mask is None:
            view_mask = torch.ones(
                (batch, views), dtype=torch.bool, device=tokens.device
            )
        view_mask = view_mask.to(device=tokens.device, dtype=torch.bool)
        token_mask = view_mask.unsqueeze(-1).expand(
            batch, views, tokens_per_view
        ).reshape(batch, views * tokens_per_view)
        return tokens, token_mask
