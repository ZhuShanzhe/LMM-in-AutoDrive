from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .modernbert_labels import (
    ACTION_LABELS,
    CATEGORY_LABELS,
    CHANGE_LABELS,
    DIRECTION_LABELS,
    STATUS_LABELS,
    URGENCY_LABELS,
)


HEADS_FILE = "multitask_heads.pt"


class ModernBertDrivingModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                "actions": nn.Linear(hidden_size, len(ACTION_LABELS)),
                "status": nn.Linear(hidden_size, len(STATUS_LABELS)),
                "category": nn.Linear(hidden_size, len(CATEGORY_LABELS)),
                "urgency": nn.Linear(hidden_size, len(URGENCY_LABELS)),
                "directions": nn.Linear(hidden_size, len(DIRECTION_LABELS)),
                "change": nn.Linear(hidden_size, len(CHANGE_LABELS)),
            }
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        dropout: float = 0.1,
        attn_implementation: str = "sdpa",
        dtype: torch.dtype | None = None,
    ) -> "ModernBertDrivingModel":
        path = Path(model_path)
        config = AutoConfig.from_pretrained(path)
        backbone = AutoModel.from_pretrained(
            path,
            config=config,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        model = cls(backbone, config.hidden_size, dropout=dropout)
        heads_path = path / HEADS_FILE
        if heads_path.is_file():
            state = torch.load(heads_path, map_location="cpu", weights_only=True)
            model.heads.load_state_dict(state)
        return model

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output, safe_serialization=True)
        torch.save(self.heads.state_dict(), output / HEADS_FILE)

    def forward(self, **model_inputs: Any) -> dict[str, torch.Tensor]:
        outputs = self.backbone(**model_inputs)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return {name: head(pooled) for name, head in self.heads.items()}
