from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class HashTextEncoder(nn.Module):
    """Dependency-light placeholder for clinical reasoning embeddings.

    It is deterministic, cheap, and intentionally replaceable with BioClinicalBERT,
    PubMedBERT, MiniLM, or any sentence-transformers model.
    """

    def __init__(self, output_dim: int = 128, buckets: int = 4096):
        super().__init__()
        self.output_dim = output_dim
        self.buckets = buckets
        self.proj = nn.Linear(buckets, output_dim)

    def _vectorize_one(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self.buckets, dtype=torch.float32)
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.buckets
            vec[idx] += 1.0
        norm = vec.norm(p=2)
        if norm > 0:
            vec = vec / norm
        return vec

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.proj.weight.device
        vectors = torch.stack([self._vectorize_one(text) for text in texts], dim=0).to(device)
        return self.proj(vectors)

    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        embedding = self.forward(texts)
        return {"embedding": embedding, "tokens": embedding.unsqueeze(1)}


class EmptyTextEncoder(nn.Module):
    """No-op text branch used when a text source should feed the LLM only."""

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def forward(self, texts: list[str]) -> torch.Tensor:
        return torch.zeros((len(texts), self.output_dim), dtype=torch.float32, device=self._device_anchor.device)

    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        embedding = self.forward(texts)
        return {"embedding": embedding, "tokens": embedding.unsqueeze(1)}


EmptyReasoningEncoder = EmptyTextEncoder


class TfidfSvdTextEncoder(nn.Module):
    """EMERGE-style lightweight report encoder.

    The encoder is lazy-fitted on first use so smoke tests can run without a
    separate fitting stage. For real training, call `fit(all_train_reports)`
    before constructing dataloaders/evaluation.
    """

    def __init__(
        self,
        output_dim: int = 128,
        max_features: int = 30000,
        ngram_range: tuple[int, int] = (1, 2),
    ):
        super().__init__()
        self.output_dim = output_dim
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            strip_accents="unicode",
            min_df=1,
        )
        self.svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfSvdTextEncoder":
        clean = _ensure_nonempty_texts(texts)
        matrix = self.vectorizer.fit_transform(clean)
        max_components = min(self.output_dim, max(1, matrix.shape[0] - 1), max(1, matrix.shape[1] - 1))
        if max_components >= 2:
            self.svd = TruncatedSVD(n_components=max_components, random_state=42)
            self.svd.fit(matrix)
        else:
            self.svd = None
        self._fitted = True
        return self

    def forward(self, texts: list[str]) -> torch.Tensor:
        if not self._fitted:
            self.fit(texts)
        clean = _ensure_nonempty_texts(texts)
        matrix = self.vectorizer.transform(clean)
        if self.svd is not None:
            emb = self.svd.transform(matrix).astype(np.float32)
        else:
            emb = matrix.toarray().astype(np.float32)
        emb = normalize(emb, norm="l2", axis=1, copy=False)
        emb = _pad_or_trim(emb, self.output_dim)
        return torch.from_numpy(emb)

    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        embedding = self.forward(texts)
        return {"embedding": embedding, "tokens": embedding.unsqueeze(1)}


class TransformerClsTextEncoder(nn.Module):
    """Pretrained LM encoder for CARER clinical reasoning text."""

    def __init__(
        self,
        model_name: str,
        output_dim: int = 128,
        cache_dir: str | Path | None = None,
        max_length: int = 2048,
        freeze: bool = False,
        freeze_first_n_layers: int = 9,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        hidden_size = int(getattr(self.model.config, "hidden_size"))
        self.proj = nn.Linear(hidden_size, output_dim)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        elif freeze_first_n_layers > 0:
            self._freeze_first_encoder_layers(freeze_first_n_layers)

    def _freeze_first_encoder_layers(self, layer_count: int) -> None:
        encoder_layers = self._find_encoder_layers()
        if encoder_layers is None:
            return
        for layer in list(encoder_layers)[:layer_count]:
            for param in layer.parameters():
                param.requires_grad = False

    def _find_encoder_layers(self) -> nn.ModuleList | list[nn.Module] | None:
        for attr_path in (
            ("longformer", "encoder", "layer"),
            ("bert", "encoder", "layer"),
            ("roberta", "encoder", "layer"),
            ("encoder", "layer"),
            ("encoder", "layers"),
        ):
            module = self.model
            for attr in attr_path:
                if not hasattr(module, attr):
                    module = None
                    break
                module = getattr(module, attr)
            if module is not None:
                return module
        return None

    def forward(self, texts: list[str]) -> torch.Tensor:
        clean = _ensure_nonempty_texts(texts)
        device = next(self.parameters()).device
        encoded = self.tokenizer(
            clean,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.set_grad_enabled(any(param.requires_grad for param in self.model.parameters())):
            output = self.model(**encoded)
        cls = output.last_hidden_state[:, 0, :]
        return self.proj(cls)

    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        clean = _ensure_nonempty_texts(texts)
        device = next(self.parameters()).device
        encoded = self.tokenizer(
            clean,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.set_grad_enabled(any(param.requires_grad for param in self.model.parameters())):
            output = self.model(**encoded)
        token_embeddings = self.proj(output.last_hidden_state)
        return {
            "embedding": token_embeddings[:, 0, :],
            "tokens": token_embeddings,
            "attention_mask": encoded.get("attention_mask"),
        }


def _ensure_nonempty_texts(texts: list[str]) -> list[str]:
    clean = [" ".join(str(text).split()) for text in texts]
    return [text if text else "none" for text in clean]


def _pad_or_trim(matrix: np.ndarray, dim: int) -> np.ndarray:
    if matrix.shape[1] == dim:
        return matrix.astype(np.float32)
    if matrix.shape[1] > dim:
        return matrix[:, :dim].astype(np.float32)
    pad_width = dim - matrix.shape[1]
    return np.pad(matrix, ((0, 0), (0, pad_width)), mode="constant").astype(np.float32)


def build_text_encoder(
    encoder_name: str,
    output_dim: int,
    model_name: str | None = None,
    cache_dir: str | Path | None = None,
    max_length: int = 512,
    freeze: bool = False,
    freeze_first_n_layers: int = 9,
) -> nn.Module:
    if encoder_name == "empty":
        return EmptyTextEncoder(output_dim)
    if encoder_name == "hash":
        return HashTextEncoder(output_dim)
    if encoder_name == "tfidf_svd":
        return TfidfSvdTextEncoder(output_dim)
    if encoder_name in {"clinical_longformer", "clinicalbert", "transformer_cls", "bioclinicalbert"}:
        if not model_name:
            raise ValueError(f"model_name is required for text encoder {encoder_name}")
        return TransformerClsTextEncoder(
            model_name=model_name,
            output_dim=output_dim,
            cache_dir=cache_dir,
            max_length=max_length,
            freeze=freeze,
            freeze_first_n_layers=freeze_first_n_layers,
        )
    raise ValueError(f"Unsupported text encoder: {encoder_name}")
