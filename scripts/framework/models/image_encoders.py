from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, SiglipModel


def _load_chexzero_clip_class(vendor_dir: Path):
    model_py = vendor_dir / "model.py"
    spec = importlib.util.spec_from_file_location("chexzero_model_module", model_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load CheXzero model definition from {model_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CLIP


class EmptyImageEncoder(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim

    def encode(self, image_paths: list[str]) -> dict[str, torch.Tensor]:
        embedding = torch.zeros((len(image_paths), self.output_dim), dtype=torch.float32)
        return {"embedding": embedding}

    def forward(self, image_paths: list[str]) -> torch.Tensor:
        return self.encode(image_paths)["embedding"]


class CheXzeroImageEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 128,
        vendor_dir: str | Path = "vendor/CheXzero",
        weights_path: str | Path | None = None,
        trainable: bool = False,
    ):
        super().__init__()
        vendor_dir = Path(vendor_dir)
        CLIP = _load_chexzero_clip_class(vendor_dir)
        self.visual_dim = 768
        self.model = CLIP(
            embed_dim=768,
            image_resolution=320,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=16,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        )
        self.weights_path = str(weights_path) if weights_path else ""
        self.weight_status = "random_init"
        if weights_path:
            resolved = Path(weights_path)
            if resolved.exists():
                state_dict = torch.load(resolved, map_location="cpu")
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
                self.weight_status = "loaded"
        self.proj = nn.Linear(self.visual_dim, output_dim)
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        self.trainable = trainable
        self.register_buffer("pixel_mean", torch.tensor([101.48761, 101.48761, 101.48761], dtype=torch.float32).view(3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([83.43944, 83.43944, 83.43944], dtype=torch.float32).view(3, 1, 1))

    def _load_one(self, image_path: str) -> torch.Tensor:
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new("RGB", (320, 320), color=(255, 255, 255))
        image = image.resize((320, 320), resample=Image.BICUBIC)
        tensor = pil_to_tensor(image).float()
        pixel_mean = self.pixel_mean.to(device=tensor.device)
        pixel_std = self.pixel_std.to(device=tensor.device)
        return (tensor - pixel_mean) / pixel_std

    def encode(self, image_paths: list[str]) -> dict[str, torch.Tensor]:
        device = self.proj.weight.device
        batch = torch.stack([self._load_one(path) for path in image_paths], dim=0).to(device)
        with torch.set_grad_enabled(self.trainable):
            features = self.model.encode_image(batch)
        embedding = self.proj(features.float())
        return {
            "embedding": embedding,
        }

    def forward(self, image_paths: list[str]) -> torch.Tensor:
        return self.encode(image_paths)["embedding"]


class XraySigLIPImageEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 128,
        model_name_or_path: str | Path = "models/encoders/image_encoder/xray_siglip/hf_cache",
        trainable: bool = False,
    ):
        super().__init__()
        model_name_or_path = str(model_name_or_path)
        self.model = SiglipModel.from_pretrained(model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.visual_dim = int(getattr(self.model.config.vision_config, "hidden_size", 1024))
        self.proj = nn.Linear(self.visual_dim, output_dim)
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        self.trainable = trainable

    @staticmethod
    def _load_one(image_path: str) -> Image.Image:
        if image_path and Path(image_path).exists():
            return Image.open(image_path).convert("RGB")
        return Image.new("RGB", (384, 384), color=(255, 255, 255))

    def encode(self, image_paths: list[str]) -> dict[str, torch.Tensor]:
        device = self.proj.weight.device
        images = [self._load_one(path) for path in image_paths]
        processed = self.processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device)
        with torch.set_grad_enabled(self.trainable):
            features = self.model.get_image_features(pixel_values=pixel_values)
        embedding = self.proj(features.float())
        return {"embedding": embedding}

    def forward(self, image_paths: list[str]) -> torch.Tensor:
        return self.encode(image_paths)["embedding"]


def build_image_encoder(
    encoder_name: str,
    output_dim: int,
    vendor_dir: str | Path,
    model_name_or_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    trainable: bool = False,
) -> nn.Module:
    if encoder_name == "empty":
        return EmptyImageEncoder(output_dim)
    if encoder_name == "xray_siglip":
        return XraySigLIPImageEncoder(
            output_dim=output_dim,
            model_name_or_path=model_name_or_path or "models/encoders/image_encoder/xray_siglip/hf_cache",
            trainable=trainable,
        )
    if encoder_name == "chexzero":
        return CheXzeroImageEncoder(
            output_dim=output_dim,
            vendor_dir=vendor_dir,
            weights_path=weights_path,
            trainable=trainable,
        )
    raise ValueError(f"Unsupported image encoder: {encoder_name}")
