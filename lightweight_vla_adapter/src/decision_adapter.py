from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .bev_encoder import LightweightBEVEncoder
from .contracts import ACTION_LABELS, VLA_PROPOSAL_SCHEMA_VERSION
from .raw_sensor_encoder import MultiviewImageEncoder


class CrossAttentionBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        attended, _ = self.attention(
            normalized_query,
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


@dataclass
class AdapterOutput:
    action_logits: torch.Tensor
    target_speed_kmh: torch.Tensor
    target_lane_logits: torch.Tensor
    target_pointer_logits: torch.Tensor
    confidence_logits: torch.Tensor
    confidence: torch.Tensor
    decision_embedding: torch.Tensor
    visual_risk_logits: torch.Tensor


class LightweightDecisionAdapter(nn.Module):
    """Four-to-six-layer scene-conditioned high-level driving policy."""

    def __init__(
        self,
        *,
        camera_channels: int,
        lidar_channels: int,
        candidate_dim: int,
        ego_dim: int,
        intent_dim: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        bev_grid: tuple[int, int] = (8, 8),
        environment_dim: int = 12,
        num_camera_views: int = 4,
        raw_camera_token_grid: tuple[int, int] = (2, 2),
        require_raw_camera: bool = False,
        use_raw_camera: bool = True,
        use_environment: bool = True,
        use_structured_bev: bool = True,
    ) -> None:
        super().__init__()
        if not 4 <= num_layers <= 6:
            raise ValueError("num_layers must be between 4 and 6")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.bev_encoder = LightweightBEVEncoder(
            camera_channels,
            lidar_channels,
            hidden_size,
            output_grid=bev_grid,
        )
        self.intent_projection = nn.Linear(intent_dim, hidden_size)
        self.candidate_projection = nn.Linear(candidate_dim, hidden_size)
        self.ego_projection = nn.Linear(ego_dim, hidden_size)
        self.environment_projection = nn.Linear(environment_dim, hidden_size)
        self.raw_camera_encoder = MultiviewImageEncoder(
            hidden_size,
            num_views=num_camera_views,
            token_grid=raw_camera_token_grid,
        )
        self.query_tokens = nn.Parameter(torch.empty(2, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.layers = nn.ModuleList(
            CrossAttentionBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.action_head = nn.Linear(hidden_size, len(ACTION_LABELS))
        self.speed_head = nn.Linear(hidden_size, 1)
        self.lane_head = nn.Linear(hidden_size, 3)
        self.confidence_head = nn.Linear(hidden_size, 1)
        self.visual_risk_head = nn.Linear(hidden_size, 3)
        self.target_query = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.require_raw_camera = bool(require_raw_camera)
        self.use_raw_camera = bool(use_raw_camera)
        self.use_environment = bool(use_environment)
        self.use_structured_bev = bool(use_structured_bev)

    def forward(
        self,
        *,
        camera_bev: torch.Tensor,
        lidar_bev: torch.Tensor,
        ego_features: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        intent_tokens: torch.Tensor,
        intent_mask: torch.Tensor,
        camera_images: torch.Tensor | None = None,
        camera_view_mask: torch.Tensor | None = None,
        environment_features: torch.Tensor | None = None,
    ) -> AdapterOutput:
        batch = camera_bev.shape[0]
        if self.require_raw_camera and camera_images is None:
            raise ValueError("this checkpoint requires synchronized raw camera input")
        bev_tokens = (
            self.bev_encoder(camera_bev, lidar_bev)
            if self.use_structured_bev
            else camera_bev.new_zeros((batch, 0, self.hidden_size))
        )
        if camera_images is not None and self.use_raw_camera:
            raw_camera_tokens, raw_camera_mask = self.raw_camera_encoder(
                camera_images, camera_view_mask
            )
        else:
            raw_camera_tokens = camera_bev.new_zeros(
                (batch, 0, self.hidden_size)
            )
            raw_camera_mask = torch.zeros(
                (batch, 0), dtype=torch.bool, device=camera_bev.device
            )
        if raw_camera_tokens.shape[1] > 0:
            valid_visual = raw_camera_mask.sum(dim=1, keepdim=True).clamp_min(1)
            visual_summary = (
                raw_camera_tokens * raw_camera_mask.unsqueeze(-1)
            ).sum(dim=1) / valid_visual
        else:
            visual_summary = camera_bev.new_zeros(
                (batch, self.hidden_size)
            )
        intent_tokens = self.intent_projection(intent_tokens)
        candidate_tokens = self.candidate_projection(candidate_features)
        ego_token = self.ego_projection(ego_features).unsqueeze(1)
        if self.use_environment:
            if environment_features is None:
                environment_features = ego_features.new_zeros(
                    (batch, self.environment_projection.in_features)
                )
            environment_token = self.environment_projection(
                environment_features
            ).unsqueeze(1)
        else:
            environment_token = ego_features.new_zeros(
                (batch, 0, self.hidden_size)
            )

        candidate_mask = candidate_mask.to(dtype=torch.bool)
        intent_mask = intent_mask.to(dtype=torch.bool)
        memory = torch.cat(
            [
                bev_tokens,
                raw_camera_tokens,
                intent_tokens,
                candidate_tokens,
                ego_token,
                environment_token,
            ],
            dim=1,
        )
        bev_mask = torch.zeros(
            (batch, bev_tokens.shape[1]), dtype=torch.bool, device=memory.device
        )
        ego_mask = torch.zeros((batch, 1), dtype=torch.bool, device=memory.device)
        memory_padding_mask = torch.cat(
            [
                bev_mask,
                ~raw_camera_mask,
                ~intent_mask,
                ~candidate_mask,
                ego_mask,
                torch.zeros(
                    (batch, environment_token.shape[1]),
                    dtype=torch.bool,
                    device=memory.device,
                ),
            ],
            dim=1,
        )
        query = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        valid_intent_count = intent_mask.sum(dim=1, keepdim=True).clamp_min(1)
        intent_summary = (
            intent_tokens * intent_mask.unsqueeze(-1)
        ).sum(dim=1) / valid_intent_count
        query = query + intent_summary.unsqueeze(1)
        for layer in self.layers:
            query = layer(query, memory, memory_padding_mask)

        decision_token = query[:, 0]
        target_token = query[:, 1]
        target_pointer_logits = torch.einsum(
            "bd,bnd->bn",
            self.target_query(target_token),
            candidate_tokens,
        ) / math.sqrt(self.hidden_size)
        target_pointer_logits = target_pointer_logits.masked_fill(
            ~candidate_mask, torch.finfo(target_pointer_logits.dtype).min
        )
        confidence_logits = self.confidence_head(decision_token).squeeze(-1)
        return AdapterOutput(
            action_logits=self.action_head(decision_token),
            target_speed_kmh=torch.sigmoid(self.speed_head(decision_token)).squeeze(-1)
            * 100.0,
            target_lane_logits=self.lane_head(decision_token),
            target_pointer_logits=target_pointer_logits,
            confidence_logits=confidence_logits,
            confidence=torch.sigmoid(confidence_logits),
            decision_embedding=decision_token,
            visual_risk_logits=self.visual_risk_head(visual_summary),
        )


def decode_proposal(
    output: AdapterOutput,
    *,
    request_id: str,
    frame_id: str,
    candidate_entity_ids: Sequence[Sequence[str]],
    model_name: str,
    latency_ms: float,
) -> list[dict]:
    action_indices = output.action_logits.argmax(dim=-1).tolist()
    lane_indices = output.target_lane_logits.argmax(dim=-1).tolist()
    pointer_indices = output.target_pointer_logits.argmax(dim=-1).tolist()
    speeds = output.target_speed_kmh.detach().cpu().tolist()
    confidences = output.confidence.detach().cpu().tolist()
    proposals: list[dict] = []
    lane_labels = (None, "left", "right")
    for batch_index, action_index in enumerate(action_indices):
        action = ACTION_LABELS[action_index]
        lane = lane_labels[lane_indices[batch_index]]
        if action == "lane_change_left":
            lane = "left"
        elif action == "lane_change_right":
            lane = "right"
        elif not action.startswith("lane_change_"):
            lane = None
        entities = list(candidate_entity_ids[batch_index])
        target_entity_id = (
            entities[pointer_indices[batch_index]]
            if entities and pointer_indices[batch_index] < len(entities)
            else None
        )
        proposals.append(
            {
                "schema_version": VLA_PROPOSAL_SCHEMA_VERSION,
                "request_id": request_id,
                "frame_id": frame_id,
                "action": action,
                "target_speed_kmh": round(float(speeds[batch_index]), 6),
                "target_lane": lane,
                "target_location": None,
                "target_entity_id": target_entity_id,
                "confidence": round(float(confidences[batch_index]), 6),
                "model": model_name,
                "latency_ms": round(float(latency_ms), 6),
            }
        )
    return proposals
