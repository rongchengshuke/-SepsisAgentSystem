from __future__ import annotations

import torch
import torch.nn as nn

from scripts.framework.models.fusion import build_fusion
from scripts.framework.models.image_encoders import build_image_encoder
from scripts.framework.models.tamf_triple_encoder import build_tamf_encoder
from scripts.framework.models.text_encoders import build_text_encoder


class StaticEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        return self.net(static)


class CARERTAMFModel(nn.Module):
    """Figure-1 framework with Figure-2 replacing the multimodal encoder."""

    def __init__(
        self,
        source_dims: list[int],
        static_dim: int,
        num_binary_tasks: int,
        num_regression_tasks: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        ehr_encoder_internal_embed_dim: int | None = None,
        ehr_encoder_internal_num_heads: int | None = None,
        note_text_encoder: str = "hash",
        note_text_model_name: str | None = None,
        note_text_cache_dir: str | None = None,
        note_text_max_length: int = 256,
        note_text_freeze: bool = False,
        note_text_freeze_first_n_layers: int = 9,
        reasoning_text_encoder: str = "hash",
        reasoning_text_model_name: str | None = None,
        reasoning_text_cache_dir: str | None = None,
        reasoning_text_max_length: int = 256,
        reasoning_text_freeze: bool = False,
        reasoning_text_freeze_first_n_layers: int = 9,
        image_encoder: str = "empty",
        image_model_name: str | None = None,
        image_model_weights: str | None = None,
        image_vendor_dir: str = "vendor/CheXzero",
        image_trainable: bool = False,
        fusion_backend: str = "gated",
        ehr_encoder_backend: str = "tamf_upstream_compatible",
        text_encoder: str | None = None,
        text_model_name: str | None = None,
        text_cache_dir: str | None = None,
        text_max_length: int | None = None,
        text_freeze: bool | None = None,
        text_freeze_first_n_layers: int | None = None,
    ):
        super().__init__()
        ehr_encoder_internal_embed_dim = ehr_encoder_internal_embed_dim or embed_dim
        ehr_encoder_internal_num_heads = ehr_encoder_internal_num_heads or num_heads
        # Backward compatibility for checkpoints saved before note/reasoning branches
        # were split into separate constructor arguments.
        if text_encoder is not None:
            if note_text_encoder == "hash":
                note_text_encoder = text_encoder
            if reasoning_text_encoder == "hash":
                reasoning_text_encoder = text_encoder
        if text_model_name is not None:
            if note_text_model_name is None:
                note_text_model_name = text_model_name
            if reasoning_text_model_name is None:
                reasoning_text_model_name = text_model_name
        if text_cache_dir is not None:
            if note_text_cache_dir is None:
                note_text_cache_dir = text_cache_dir
            if reasoning_text_cache_dir is None:
                reasoning_text_cache_dir = text_cache_dir
        if text_max_length is not None:
            if note_text_max_length == 256:
                note_text_max_length = text_max_length
            if reasoning_text_max_length == 256:
                reasoning_text_max_length = text_max_length
        if text_freeze is not None:
            if note_text_freeze is False:
                note_text_freeze = text_freeze
            if reasoning_text_freeze is False:
                reasoning_text_freeze = text_freeze
        if text_freeze_first_n_layers is not None:
            if note_text_freeze_first_n_layers == 9:
                note_text_freeze_first_n_layers = text_freeze_first_n_layers
            if reasoning_text_freeze_first_n_layers == 9:
                reasoning_text_freeze_first_n_layers = text_freeze_first_n_layers
        self.ehr_encoder_backend = ehr_encoder_backend
        self.ehr_encoder = build_tamf_encoder(
            backend=ehr_encoder_backend,
            source_dims=source_dims,
            embed_dim=ehr_encoder_internal_embed_dim,
            num_heads=ehr_encoder_internal_num_heads,
            dropout=dropout,
        )
        self.ehr_output_proj = (
            nn.Identity()
            if ehr_encoder_internal_embed_dim == embed_dim
            else nn.Sequential(
                nn.Linear(ehr_encoder_internal_embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        )
        self.static_encoder = StaticEncoder(static_dim, embed_dim, dropout)
        self.note_encoder = build_text_encoder(
            note_text_encoder,
            output_dim=embed_dim,
            model_name=note_text_model_name,
            cache_dir=note_text_cache_dir,
            max_length=note_text_max_length,
            freeze=note_text_freeze,
            freeze_first_n_layers=note_text_freeze_first_n_layers,
        )
        self.reasoning_encoder = build_text_encoder(
            reasoning_text_encoder,
            output_dim=embed_dim,
            model_name=reasoning_text_model_name,
            cache_dir=reasoning_text_cache_dir,
            max_length=reasoning_text_max_length,
            freeze=reasoning_text_freeze,
            freeze_first_n_layers=reasoning_text_freeze_first_n_layers,
        )
        self.image_encoder = build_image_encoder(
            image_encoder,
            output_dim=embed_dim,
            vendor_dir=image_vendor_dir,
            model_name_or_path=image_model_name,
            weights_path=image_model_weights,
            trainable=image_trainable,
        )

        self.fusion = build_fusion(fusion_backend, embed_dim, num_heads, dropout)
        self.binary_head = nn.Linear(embed_dim, num_binary_tasks)
        self.num_regression_tasks = num_regression_tasks
        self.regression_head = (
            nn.Linear(embed_dim, num_regression_tasks)
            if num_regression_tasks > 0
            else None
        )

    def forward(
        self,
        x_list: list[torch.Tensor],
        m_list: list[torch.Tensor],
        t_list: list[torch.Tensor],
        static: torch.Tensor,
        note_texts: list[str] | None = None,
        reasoning_texts: list[str] | None = None,
        image_paths: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        ehr_outputs = self.ehr_encoder(x_list, m_list, t_list)
        ehr_embedding = self.ehr_output_proj(ehr_outputs["ehr_embedding"])
        static_embedding = self.static_encoder(static)
        if note_texts is None:
            note_texts = [""] * static.shape[0]
        if reasoning_texts is None:
            reasoning_texts = [""] * static.shape[0]
        if image_paths is None:
            image_paths = [""] * static.shape[0]
        note_outputs = self.note_encoder.encode(note_texts) if hasattr(self.note_encoder, "encode") else {
            "embedding": self.note_encoder(note_texts),
            "tokens": self.note_encoder(note_texts).unsqueeze(1),
        }
        note_embedding = note_outputs["embedding"].to(static.device)

        if hasattr(self.reasoning_encoder, "encode"):
            reasoning_outputs = self.reasoning_encoder.encode(reasoning_texts)
        else:
            reasoning_embedding_fallback = self.reasoning_encoder(reasoning_texts)
            reasoning_outputs = {
                "embedding": reasoning_embedding_fallback,
                "tokens": reasoning_embedding_fallback.unsqueeze(1),
            }
        reasoning_embedding = reasoning_outputs["embedding"].to(static.device)
        reasoning_tokens = reasoning_outputs["tokens"].to(static.device)
        reasoning_attention_mask = reasoning_outputs.get("attention_mask")
        image_outputs = self.image_encoder.encode(image_paths) if hasattr(self.image_encoder, "encode") else {
            "embedding": self.image_encoder(image_paths),
        }
        image_embedding = image_outputs["embedding"].to(static.device)

        if getattr(self.fusion, "expects_modalities_list", False):
            fused = self.fusion([ehr_embedding, note_embedding, reasoning_embedding, image_embedding, static_embedding])
        else:
            multimodal_aux = torch.stack([note_embedding, reasoning_embedding, image_embedding], dim=0).mean(dim=0)
            reasoning_for_fusion = (
                {
                    "tokens": reasoning_tokens,
                    "attention_mask": reasoning_attention_mask,
                }
                if getattr(self.fusion, "uses_reasoning_tokens", False)
                else multimodal_aux
            )
            fused = self.fusion(ehr_embedding, reasoning_for_fusion, static_embedding)
        return {
            "binary_logits": self.binary_head(fused),
            "regression": (
                self.regression_head(fused)
                if self.regression_head is not None
                else fused.new_empty((fused.shape[0], 0))
            ),
            "fused_embedding": fused,
            "ehr_embedding": ehr_embedding,
            "note_embedding": note_embedding,
            "reasoning_embedding": reasoning_embedding,
            "reasoning_tokens": reasoning_tokens,
            "image_embedding": image_embedding,
            "static_embedding": static_embedding,
        }
