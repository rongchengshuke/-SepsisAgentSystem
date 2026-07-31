from __future__ import annotations

import torch
import torch.nn as nn


class Figure1TokenTransformerFusion(nn.Module):
    """Five-stream token fusion over [EHR, note, reasoning, image, static]."""

    expects_modalities_list = True

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        ffn_multiplier: int = 4,
        num_modalities: int = 5,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.modality_embedding = nn.Parameter(torch.zeros(1, num_modalities, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ffn_multiplier,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.pool = nn.Linear(embed_dim, 1)
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, modalities: list[torch.Tensor]) -> torch.Tensor:
        if len(modalities) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(modalities)}")
        tokens = torch.stack(modalities, dim=1) + self.modality_embedding
        encoded = self.encoder(tokens)
        weights = torch.softmax(self.pool(encoded).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1)
        residual = tokens.mean(dim=1)
        return self.norm(self.out(pooled) + residual)


class GatedFusionBlock(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 3),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, ehr: torch.Tensor, reasoning: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([ehr, reasoning, static], dim=-1)
        return self.out(joined * self.gate(joined))


class CARERAttentionFusion(nn.Module):
    """CARER-style fusion: use EHR as query over clinical-reasoning tokens."""

    uses_reasoning_tokens = True

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.local_norm = nn.LayerNorm(embed_dim)
        self.local_dropout = nn.Dropout(dropout)
        self.reasoning_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, ehr: torch.Tensor, reasoning: torch.Tensor | dict[str, torch.Tensor], static: torch.Tensor) -> torch.Tensor:
        key_padding_mask = None
        if isinstance(reasoning, dict):
            attention_mask = reasoning.get("attention_mask")
            reasoning = reasoning["tokens"]
            if attention_mask is not None:
                key_padding_mask = attention_mask == 0
        if reasoning.dim() == 2:
            reasoning = reasoning.unsqueeze(1)
        local = self.local_dropout(self.local_norm(ehr + static))
        local_query = local.unsqueeze(1)
        attended_reasoning, _ = self.reasoning_attention(
            query=local_query,
            key=reasoning,
            value=reasoning,
            key_padding_mask=key_padding_mask,
        )
        attended_reasoning = attended_reasoning.squeeze(1)
        return self.out(torch.cat([local, attended_reasoning], dim=-1))


class EmergeCrossAttentionFusion(nn.Module):
    """EMERGE-style bidirectional cross-attention between EHR and report text."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.text_to_ehr = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ehr_to_text = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.static_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_ehr = nn.LayerNorm(embed_dim)
        self.norm_text = nn.LayerNorm(embed_dim)
        self.out = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, ehr: torch.Tensor, reasoning: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        ehr_seq = ehr.unsqueeze(1)
        text_seq = reasoning.unsqueeze(1)
        text_conditioned, _ = self.text_to_ehr(query=text_seq, key=ehr_seq, value=ehr_seq)
        ehr_conditioned, _ = self.ehr_to_text(query=ehr_seq, key=text_seq, value=text_seq)
        text_conditioned = self.norm_text(text_conditioned.squeeze(1) + reasoning)
        ehr_conditioned = self.norm_ehr(ehr_conditioned.squeeze(1) + ehr)
        return self.out(torch.cat([ehr_conditioned, text_conditioned, self.static_proj(static)], dim=-1))


class EmergeMAGFusion(nn.Module):
    """MAG-style auxiliary modulation from report and static embeddings."""

    def __init__(self, embed_dim: int, dropout: float = 0.25):
        super().__init__()
        self.gate = nn.Linear(embed_dim * 3, embed_dim)
        self.adjust = nn.Linear(embed_dim * 2, embed_dim)
        self.beta = nn.Parameter(torch.ones(()))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ehr: torch.Tensor, reasoning: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        aux = torch.cat([reasoning, static], dim=-1)
        weight = torch.sigmoid(self.gate(torch.cat([ehr, reasoning, static], dim=-1)))
        adjust = self.adjust(aux) * weight
        scale = torch.norm(ehr, dim=-1, keepdim=True) / torch.clamp(
            torch.norm(adjust, dim=-1, keepdim=True), min=1e-6
        )
        alpha = torch.clamp(scale * self.beta, max=1.0)
        return self.dropout(self.norm(ehr + alpha * adjust))


class EmergeTokenTransformerFusion(nn.Module):
    """EMERGE token-transformer fusion over [EHR, report, static] tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.25,
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ffn_multiplier,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.pool = nn.Linear(embed_dim, 1)
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ehr: torch.Tensor, reasoning: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([ehr, reasoning, static], dim=1) + self.modality_embedding
        encoded = self.encoder(tokens)
        weights = torch.softmax(self.pool(encoded).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1)
        residual = tokens.mean(dim=1)
        return self.norm(self.out(pooled) + residual)


class ConcatFusionBlock(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, ehr: torch.Tensor, reasoning: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.out(torch.cat([ehr, reasoning, static], dim=-1))


def build_fusion(
    backend: str,
    embed_dim: int,
    num_heads: int,
    dropout: float,
) -> nn.Module:
    if backend == "figure1_token_transformer":
        return Figure1TokenTransformerFusion(embed_dim, num_heads=num_heads, dropout=dropout)
    if backend == "gated":
        return GatedFusionBlock(embed_dim, dropout)
    if backend == "carer_attention":
        return CARERAttentionFusion(embed_dim, num_heads=num_heads, dropout=dropout)
    if backend == "concat":
        return ConcatFusionBlock(embed_dim, dropout)
    if backend == "emerge_cross_attention":
        return EmergeCrossAttentionFusion(embed_dim, num_heads=num_heads, dropout=dropout)
    if backend == "emerge_mag":
        return EmergeMAGFusion(embed_dim, dropout=dropout)
    if backend == "emerge_token_transformer":
        return EmergeTokenTransformerFusion(embed_dim, num_heads=num_heads, dropout=dropout)
    raise ValueError(f"Unsupported fusion backend: {backend}")
