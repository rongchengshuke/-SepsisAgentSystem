from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.data.triple_dataset import TripleWindowDataset, collate_triple_windows
from scripts.framework.models.carer_tamf import CARERTAMFModel
from scripts.framework.utils.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--note-csv", default=None)
    parser.add_argument("--image-manifest-csv", default=None)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--note-text-encoder", default=None)
    parser.add_argument("--note-text-model-name", default=None)
    parser.add_argument("--note-text-cache-dir", default=None)
    parser.add_argument("--note-text-max-length", type=int, default=None)
    parser.add_argument("--reasoning-text-encoder", default=None)
    parser.add_argument("--reasoning-text-model-name", default=None)
    parser.add_argument("--reasoning-text-cache-dir", default=None)
    parser.add_argument("--reasoning-text-max-length", type=int, default=None)
    parser.add_argument("--image-encoder", default=None)
    parser.add_argument("--image-model-name", default=None)
    parser.add_argument("--image-model-weights", default=None)
    parser.add_argument("--fusion-backend", default=None)
    parser.add_argument("--ehr-encoder-backend", default=None)
    parser.add_argument("--ehr-encoder-weights", default=None)
    parser.add_argument("--lambda-alignment", type=float, default=None)
    parser.add_argument("--checkpoint-path", default="models/checkpoints/trained_pipeline/latest.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_indices(total: int, train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[int]]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train = indices[:train_end]
    val = indices[train_end:val_end]
    test = indices[val_end:]
    if not val and test:
        val = test[:1]
        test = test[1:]
    if not test and val:
        test = val[-1:]
        val = val[:-1]
    return {"train": train, "val": val, "test": test}


def make_loader(
    dataset: TripleWindowDataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_triple_windows)


def placeholder_reasoning_texts(sample_ids: list[str]) -> list[str]:
    return [f"elderly sepsis sample {sample_id}" for sample_id in sample_ids]


def placeholder_note_texts(sample_ids: list[str]) -> list[str]:
    return [f"synthetic clinical note for {sample_id}" for sample_id in sample_ids]


def carer_alignment_loss(ehr_embedding: torch.Tensor, reasoning_embedding: torch.Tensor) -> torch.Tensor:
    """CARER cross-view alignment loss between local EHR and global reasoning views.

    The loss matches the in-batch pairwise similarity structure of the two
    representation spaces, following CARER's Frobenius-norm objective.
    """
    ehr = F.normalize(ehr_embedding, p=2, dim=-1)
    reasoning = F.normalize(reasoning_embedding, p=2, dim=-1)
    ehr_similarity = ehr @ ehr.T
    reasoning_similarity = reasoning @ reasoning.T
    batch_size = ehr.shape[0]
    return ((ehr_similarity - reasoning_similarity) ** 2).sum() / (batch_size**2)


def zero_loss_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def resolve_model_ref(value: str | None) -> str | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)
    project_candidate = PROJECT_DIR / value
    if project_candidate.exists():
        return str(project_candidate)
    return value


def build_model(
    dataset: TripleWindowDataset,
    cfg: dict,
    note_text_encoder: str,
    note_text_model_name: str | None,
    note_text_cache_dir: str | None,
    note_text_max_length: int | None,
    reasoning_text_encoder: str,
    reasoning_text_model_name: str | None,
    reasoning_text_cache_dir: str | None,
    reasoning_text_max_length: int | None,
    image_encoder: str,
    image_model_name: str | None,
    image_model_weights: str | None,
    fusion_backend: str,
    ehr_encoder_backend: str,
    device: torch.device,
) -> CARERTAMFModel:
    source_dims = [dataset.source_dims[source] for source in ["lab", "vital", "treatment"]]
    model = CARERTAMFModel(
        source_dims=source_dims,
        static_dim=dataset.static_dim,
        num_binary_tasks=dataset.num_binary_tasks,
        num_regression_tasks=dataset.num_regression_tasks,
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
        ehr_encoder_internal_embed_dim=int(cfg["model"].get("ehr_encoder_internal_embed_dim", cfg["model"]["embed_dim"])),
        ehr_encoder_internal_num_heads=int(cfg["model"].get("ehr_encoder_internal_num_heads", cfg["model"]["num_heads"])),
        note_text_encoder=note_text_encoder,
        note_text_model_name=resolve_model_ref(note_text_model_name),
        note_text_cache_dir=note_text_cache_dir,
        note_text_max_length=note_text_max_length or int(cfg["model"].get("note_text_max_length", 256)),
        note_text_freeze=bool(cfg["model"].get("note_text_freeze", False)),
        note_text_freeze_first_n_layers=int(cfg["model"].get("note_text_freeze_first_n_layers", 9)),
        reasoning_text_encoder=reasoning_text_encoder,
        reasoning_text_model_name=resolve_model_ref(reasoning_text_model_name),
        reasoning_text_cache_dir=reasoning_text_cache_dir,
        reasoning_text_max_length=reasoning_text_max_length or int(cfg["model"].get("reasoning_text_max_length", 256)),
        reasoning_text_freeze=bool(cfg["model"].get("reasoning_text_freeze", False)),
        reasoning_text_freeze_first_n_layers=int(cfg["model"].get("reasoning_text_freeze_first_n_layers", 9)),
        image_encoder=image_encoder,
        image_model_name=resolve_model_ref(image_model_name),
        image_model_weights=resolve_model_ref(image_model_weights),
        image_vendor_dir=str(resolve_path(PROJECT_DIR, cfg["model"].get("image_vendor_dir", "vendor/CheXzero"))),
        image_trainable=bool(cfg["model"].get("image_trainable", False)),
        fusion_backend=fusion_backend,
        ehr_encoder_backend=ehr_encoder_backend,
    )
    return model.to(device)


def load_ehr_encoder_weights(
    model: CARERTAMFModel,
    weights_path: str | None,
) -> dict[str, list[str]] | None:
    if not weights_path:
        return None
    resolved = resolve_path(PROJECT_DIR, weights_path)
    payload = torch.load(resolved, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    missing_keys, unexpected_keys = model.ehr_encoder.load_state_dict(state_dict, strict=False)
    return {
        "path": [str(resolved)],
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "sample_id": batch["sample_id"],
        "note_texts": batch.get("note_texts", []),
        "reasoning_texts": batch.get("reasoning_texts", []),
        "image_paths": batch.get("image_paths", []),
        "x_list": [tensor.to(device) for tensor in batch["x_list"]],
        "m_list": [tensor.to(device) for tensor in batch["m_list"]],
        "t_list": [tensor.to(device) for tensor in batch["t_list"]],
        "static": batch["static"].to(device),
        "binary_targets": batch["binary_targets"].to(device),
        "regression_targets": batch["regression_targets"].to(device),
    }


def run_epoch(
    model: CARERTAMFModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    lambda_alignment: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)
    bce = nn.BCEWithLogitsLoss()
    reg = nn.SmoothL1Loss()
    total_loss = 0.0
    total_bin = 0.0
    total_reg = 0.0
    total_align = 0.0
    total_samples = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(
                x_list=batch["x_list"],
                m_list=batch["m_list"],
                t_list=batch["t_list"],
                static=batch["static"],
                note_texts=batch.get("note_texts") or placeholder_note_texts(batch["sample_id"]),
                reasoning_texts=batch.get("reasoning_texts") or placeholder_reasoning_texts(batch["sample_id"]),
                image_paths=batch.get("image_paths") or [""] * len(batch["sample_id"]),
            )
            loss_bin = bce(outputs["binary_logits"], batch["binary_targets"])
            if batch["regression_targets"].numel() > 0:
                loss_reg = reg(outputs["regression"], batch["regression_targets"])
            else:
                loss_reg = zero_loss_like(outputs["binary_logits"])
            loss_align = carer_alignment_loss(outputs["ehr_embedding"], outputs["reasoning_embedding"])
            loss = loss_bin + loss_reg + lambda_alignment * loss_align
            if training:
                loss.backward()
                optimizer.step()
        batch_size = batch["static"].shape[0]
        total_loss += float(loss.item()) * batch_size
        total_bin += float(loss_bin.item()) * batch_size
        total_reg += float(loss_reg.item()) * batch_size
        total_align += float(loss_align.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return {"loss": 0.0, "binary_loss": 0.0, "regression_loss": 0.0, "alignment_loss": 0.0}
    return {
        "loss": total_loss / total_samples,
        "binary_loss": total_bin / total_samples,
        "regression_loss": total_reg / total_samples,
        "alignment_loss": total_align / total_samples,
    }


def export_component_weights(model: CARERTAMFModel, checkpoint_path: Path) -> dict[str, str]:
    export_targets = {
        "ehr_encoder": PROJECT_DIR / "models" / "encoders" / "multimodal_encoder" / "weights" / f"{checkpoint_path.stem}_ehr_encoder.pt",
        "note_encoder": PROJECT_DIR / "models" / "encoders" / "text_encoder" / "weights" / f"{checkpoint_path.stem}_note_encoder.pt",
        "reasoning_encoder": PROJECT_DIR / "models" / "encoders" / "reasoning_encoder" / "weights" / f"{checkpoint_path.stem}_reasoning_encoder.pt",
        "image_encoder": PROJECT_DIR / "models" / "encoders" / "image_encoder" / "weights" / f"{checkpoint_path.stem}_image_encoder.pt",
        "static_encoder": PROJECT_DIR / "models" / "encoders" / "static_encoder" / "weights" / f"{checkpoint_path.stem}_static_encoder.pt",
        "fusion": PROJECT_DIR / "models" / "fusion_heads" / "weights" / f"{checkpoint_path.stem}_fusion.pt",
        "binary_head": PROJECT_DIR / "models" / "fusion_heads" / "weights" / f"{checkpoint_path.stem}_binary_head.pt",
        "regression_head": PROJECT_DIR / "models" / "fusion_heads" / "weights" / f"{checkpoint_path.stem}_regression_head.pt",
    }
    modules = {
        "ehr_encoder": model.ehr_encoder,
        "note_encoder": model.note_encoder,
        "reasoning_encoder": model.reasoning_encoder,
        "image_encoder": model.image_encoder,
        "static_encoder": model.static_encoder,
        "fusion": model.fusion,
        "binary_head": model.binary_head,
        "regression_head": model.regression_head,
    }
    exported = {}
    for key, target in export_targets.items():
        if modules[key] is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(modules[key].state_dict(), target)
        exported[key] = str(target)
    return exported


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(PROJECT_DIR / args.config)
    processed_dir = resolve_path(PROJECT_DIR, args.processed_dir or cfg["data"]["processed_dir"])
    if not (processed_dir / "all_triples.parquet").exists():
        raise SystemExit("Run scripts/data/build_triples.py first.")

    dataset = TripleWindowDataset(
        processed_dir=processed_dir,
        tasks_binary=cfg["model"]["tasks"]["binary"],
        tasks_regression=cfg["model"]["tasks"]["regression"],
        max_samples=args.max_samples,
        reasoning_csv=resolve_path(PROJECT_DIR, cfg["data"]["reasoning_csv"]),
        note_csv=resolve_path(PROJECT_DIR, args.note_csv or cfg["data"].get("note_csv", "")),
        image_manifest_csv=resolve_path(PROJECT_DIR, args.image_manifest_csv or cfg["data"].get("image_manifest_csv", "")),
    )
    splits = split_indices(len(dataset), args.train_ratio, args.val_ratio, args.seed)
    if not splits["train"]:
        raise SystemExit("Training split is empty; increase --max-samples.")

    train_loader = make_loader(dataset, splits["train"], args.batch_size, shuffle=True)
    val_loader = make_loader(dataset, splits["val"], args.batch_size, shuffle=False)
    device = choose_device(args.device)
    note_text_encoder = args.note_text_encoder or cfg["model"].get("note_text_encoder", "hash")
    note_text_model_name = args.note_text_model_name or cfg["model"].get("note_text_model_name")
    note_text_cache_dir = args.note_text_cache_dir
    note_text_max_length = args.note_text_max_length
    reasoning_text_encoder = args.reasoning_text_encoder or cfg["model"].get("reasoning_text_encoder", "hash")
    reasoning_text_model_name = args.reasoning_text_model_name or cfg["model"].get("reasoning_text_model_name")
    reasoning_text_cache_dir = args.reasoning_text_cache_dir
    reasoning_text_max_length = args.reasoning_text_max_length
    image_encoder = args.image_encoder or cfg["model"].get("image_encoder", "empty")
    image_model_name = args.image_model_name or cfg["model"].get("image_model_name")
    image_model_weights = args.image_model_weights or cfg["model"].get("image_model_weights")
    fusion_backend = args.fusion_backend or cfg["model"]["fusion_backend"]
    ehr_encoder_backend = args.ehr_encoder_backend or cfg["model"].get("ehr_encoder_backend", "tamf_upstream_compatible")
    ehr_encoder_weights = args.ehr_encoder_weights or cfg["model"].get("ehr_encoder_weights")
    lambda_alignment = (
        args.lambda_alignment
        if args.lambda_alignment is not None
        else float(cfg["model"].get("alignment", {}).get("lambda", 0.0))
    )
    model = build_model(
        dataset,
        cfg,
        note_text_encoder,
        note_text_model_name,
        note_text_cache_dir,
        note_text_max_length,
        reasoning_text_encoder,
        reasoning_text_model_name,
        reasoning_text_cache_dir,
        reasoning_text_max_length,
        image_encoder,
        image_model_name,
        image_model_weights,
        fusion_backend,
        ehr_encoder_backend,
        device,
    )
    ehr_weight_report = load_ehr_encoder_weights(model, ehr_encoder_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_path = resolve_path(PROJECT_DIR, args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: list[dict[str, float]] = []
    exported_paths: dict[str, str] = {}

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, lambda_alignment)
        val_metrics = (
            run_epoch(model, val_loader, optimizer=None, device=device, lambda_alignment=lambda_alignment)
            if splits["val"]
            else train_metrics
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_binary_loss": train_metrics["binary_loss"],
            "train_regression_loss": train_metrics["regression_loss"],
            "train_alignment_loss": train_metrics["alignment_loss"],
            "val_loss": val_metrics["loss"],
            "val_binary_loss": val_metrics["binary_loss"],
            "val_regression_loss": val_metrics["regression_loss"],
            "val_alignment_loss": val_metrics["alignment_loss"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_metrics["loss"] <= best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_args": {
                        "source_dims": [dataset.source_dims[source] for source in ["lab", "vital", "treatment"]],
                        "static_dim": dataset.static_dim,
                        "num_binary_tasks": dataset.num_binary_tasks,
                        "num_regression_tasks": dataset.num_regression_tasks,
                        "embed_dim": cfg["model"]["embed_dim"],
                        "num_heads": cfg["model"]["num_heads"],
                        "dropout": cfg["model"]["dropout"],
                        "ehr_encoder_internal_embed_dim": int(cfg["model"].get("ehr_encoder_internal_embed_dim", cfg["model"]["embed_dim"])),
                        "ehr_encoder_internal_num_heads": int(cfg["model"].get("ehr_encoder_internal_num_heads", cfg["model"]["num_heads"])),
                        "note_text_encoder": note_text_encoder,
                        "note_text_model_name": resolve_model_ref(note_text_model_name),
                        "note_text_cache_dir": note_text_cache_dir,
                        "note_text_max_length": note_text_max_length or int(cfg["model"].get("note_text_max_length", 256)),
                        "note_text_freeze": bool(cfg["model"].get("note_text_freeze", False)),
                        "note_text_freeze_first_n_layers": int(cfg["model"].get("note_text_freeze_first_n_layers", 9)),
                        "reasoning_text_encoder": reasoning_text_encoder,
                        "reasoning_text_model_name": resolve_model_ref(reasoning_text_model_name),
                        "reasoning_text_cache_dir": reasoning_text_cache_dir,
                        "reasoning_text_max_length": reasoning_text_max_length or int(cfg["model"].get("reasoning_text_max_length", 256)),
                        "reasoning_text_freeze": bool(cfg["model"].get("reasoning_text_freeze", False)),
                        "reasoning_text_freeze_first_n_layers": int(cfg["model"].get("reasoning_text_freeze_first_n_layers", 9)),
                        "image_encoder": image_encoder,
                        "image_model_name": resolve_model_ref(image_model_name),
                        "image_model_weights": resolve_model_ref(image_model_weights),
                        "image_vendor_dir": str(resolve_path(PROJECT_DIR, cfg["model"].get("image_vendor_dir", "vendor/CheXzero"))),
                        "image_trainable": bool(cfg["model"].get("image_trainable", False)),
                        "fusion_backend": fusion_backend,
                        "ehr_encoder_backend": ehr_encoder_backend,
                    },
                    "alignment": {
                        "type": "carer_similarity_matrix",
                        "lambda": lambda_alignment,
                    },
                    "tasks_binary": cfg["model"]["tasks"]["binary"],
                    "tasks_regression": cfg["model"]["tasks"]["regression"],
                    "dataset_meta": {
                        "processed_dir": str(processed_dir),
                        "max_samples": args.max_samples,
                        "splits": splits,
                    },
                    "ehr_encoder_init": ehr_weight_report,
                    "history": history,
                },
                checkpoint_path,
            )
            exported_paths = export_component_weights(model, checkpoint_path)

    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(
        json.dumps(
            {
                "device": str(device),
                "train_samples": len(splits["train"]),
                "val_samples": len(splits["val"]),
                "test_samples": len(splits["test"]),
                "checkpoint": str(checkpoint_path),
                "component_weights": exported_paths,
                "ehr_encoder_init": ehr_weight_report,
                "alignment": {
                    "type": "carer_similarity_matrix",
                    "lambda": lambda_alignment,
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
