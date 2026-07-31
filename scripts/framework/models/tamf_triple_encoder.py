from __future__ import annotations

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Upstream TAMF-compatible time embedding."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(t))


class MaskGuidedSequenceEmbedding(nn.Module):
    """Upstream TAMF-compatible per-source encoder."""

    def __init__(self, input_dim: int, mask_dim: int, time_dim: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim, embed_dim)
        self.value_embedding = nn.Linear(input_dim, embed_dim)
        self.mask_embedding = nn.Linear(mask_dim, embed_dim)
        self.self_attention_x = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.self_attention_m = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, m: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_embed = self.time_embedding(t)
        x_embed = self.value_embedding(x) + time_embed
        m_embed = self.mask_embedding(m) + time_embed
        x_out, _ = self.self_attention_x(x_embed, x_embed, x_embed)
        m_out, _ = self.self_attention_m(m_embed, m_embed, m_embed)
        x_out, _ = self.cross_attention(x_out, m_out, x_out)
        return x_out, m_out


class CrossSourceFusion(nn.Module):
    """Upstream TAMF-compatible cross-source fusion."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        fused_input = torch.cat(embeddings, dim=1)
        out, _ = self.self_attention(fused_input, fused_input, fused_input)
        return out


class UpstreamCompatibleTripleTAMFEncoder(nn.Module):
    """
    Encoder-only wrapper matching the upstream TAMF encoder naming/layout so
    encoder weights can be loaded directly from a TAMF checkpoint.
    """

    def __init__(
        self,
        source_dims: list[int],
        embed_dim: int = 128,
        num_heads: int = 4,
        time_dims: list[int] | None = None,
        mask_dims: list[int] | None = None,
    ):
        super().__init__()
        time_dims = time_dims or [1] * len(source_dims)
        mask_dims = mask_dims or list(source_dims)
        self.sequence_embeddings = nn.ModuleList(
            [
                MaskGuidedSequenceEmbedding(
                    input_dim=input_dim,
                    mask_dim=mask_dim,
                    time_dim=time_dim,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                )
                for input_dim, mask_dim, time_dim in zip(source_dims, mask_dims, time_dims)
            ]
        )
        self.cross_source_fusion = CrossSourceFusion(embed_dim, num_heads)

    def forward(
        self,
        x_list: list[torch.Tensor],
        m_list: list[torch.Tensor],
        t_list: list[torch.Tensor],
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        embeddings = [seq_embed(x, m, t) for seq_embed, x, m, t in zip(self.sequence_embeddings, x_list, m_list, t_list)]
        source_sequences = [x_out for x_out, _ in embeddings]
        mask_sequences = [m_out for _, m_out in embeddings]
        fused_sequence = self.cross_source_fusion(source_sequences)
        source_embeddings = [seq.mean(dim=1) for seq in source_sequences]
        ehr_embedding = fused_sequence.mean(dim=1)
        return {
            "ehr_embedding": ehr_embedding,
            "source_embeddings": source_embeddings,
            "source_sequences": source_sequences,
            "mask_sequences": mask_sequences,
            "fused_tokens": fused_sequence,
        }


class SimplifiedMaskGuidedSourceEncoder(nn.Module):
    """Previous lightweight pooled encoder retained for backward compatibility."""

    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.value_proj = nn.Linear(input_dim, embed_dim)
        self.mask_proj = nn.Linear(input_dim, embed_dim)
        self.time_proj = nn.Linear(1, embed_dim)
        self.value_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mask_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mask_guided_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, masks: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_embed = self.time_proj(times)
        value_embed = self.value_proj(values) + time_embed
        mask_embed = self.mask_proj(masks) + time_embed
        value_out, _ = self.value_self_attn(value_embed, value_embed, value_embed)
        mask_out, _ = self.mask_self_attn(mask_embed, mask_embed, mask_embed)
        guided, _ = self.mask_guided_attn(value_out, mask_out, value_out)
        encoded = self.norm(value_out + self.dropout(guided))
        pooled = encoded.mean(dim=1)
        return encoded, pooled


class SimplifiedTripleTAMFEncoder(nn.Module):
    """Previous project-local approximation kept so old checkpoints still load."""

    def __init__(
        self,
        source_dims: list[int],
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.source_encoders = nn.ModuleList(
            [SimplifiedMaskGuidedSourceEncoder(dim, embed_dim, num_heads, dropout) for dim in source_dims]
        )
        self.cross_source_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x_list: list[torch.Tensor],
        m_list: list[torch.Tensor],
        t_list: list[torch.Tensor],
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        source_sequences = []
        source_pooled = []
        for encoder, values, masks, times in zip(self.source_encoders, x_list, m_list, t_list):
            encoded, pooled = encoder(values, masks, times)
            source_sequences.append(encoded)
            source_pooled.append(pooled)

        source_tokens = torch.stack(source_pooled, dim=1)
        fused_tokens, _ = self.cross_source_attn(source_tokens, source_tokens, source_tokens)
        fused_tokens = self.norm(source_tokens + fused_tokens)
        ehr_embedding = fused_tokens.mean(dim=1)
        return {
            "ehr_embedding": ehr_embedding,
            "source_embeddings": source_pooled,
            "source_sequences": source_sequences,
            "fused_tokens": fused_tokens,
        }


def build_tamf_encoder(
    backend: str,
    source_dims: list[int],
    embed_dim: int,
    num_heads: int,
    dropout: float,
) -> nn.Module:
    if backend == "tamf_upstream_compatible":
        return UpstreamCompatibleTripleTAMFEncoder(source_dims, embed_dim=embed_dim, num_heads=num_heads)
    if backend == "pooled_simplified":
        return SimplifiedTripleTAMFEncoder(source_dims, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
    raise ValueError(f"Unsupported ehr_encoder backend: {backend}")


class TripleTAMFEncoder(UpstreamCompatibleTripleTAMFEncoder):
    """
    Default alias kept for compatibility with existing imports.
    This now points to the upstream-compatible encoder layout.
    """

