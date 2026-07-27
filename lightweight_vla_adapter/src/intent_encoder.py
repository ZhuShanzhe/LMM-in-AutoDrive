from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class ModernBertIntentEncoder(nn.Module):
    """Reuse the parser's ModernBERT backbone as the language feature tower."""

    def __init__(
        self,
        backbone: nn.Module,
        input_hidden_size: int,
        output_hidden_size: int = 256,
        *,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(input_hidden_size, output_hidden_size),
            nn.LayerNorm(output_hidden_size),
        )
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        output_hidden_size: int = 256,
        freeze_backbone: bool = True,
        attn_implementation: str = "sdpa",
        dtype: torch.dtype | None = None,
    ) -> "ModernBertIntentEncoder":
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_path)
        backbone = AutoModel.from_pretrained(
            model_path,
            config=config,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        return cls(
            backbone,
            int(config.hidden_size),
            output_hidden_size,
            freeze_backbone=freeze_backbone,
        )

    def train(self, mode: bool = True) -> "ModernBertIntentEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, **model_inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            output = self.backbone(**model_inputs)
            hidden = output.last_hidden_state
        tokens = self.projection(hidden)
        mask = model_inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(
                tokens.shape[:2], dtype=torch.bool, device=tokens.device
            )
        else:
            mask = mask.to(dtype=torch.bool)
        return tokens, mask
