from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_DIR = Path(__file__).resolve().parents[2]
VENDOR_DIR = PROJECT_DIR / "vendor" / "TAMF" / "deep_learning"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(VENDOR_DIR))

from model import MultiSourceModel, contrastive_loss, focal_loss, reconstruction_loss
from scripts.framework.data.triple_dataset import TripleWindowDataset
from scripts.framework.utils.config import load_config, resolve_path


SOURCE_ORDER = ["lab", "vital", "treatment"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 TAMF training with explicit train/test split")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary-task-index", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lambda-focal", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=0.5)
    parser.add_argument("--lambda-contrast", type=float, default=0.3)
    parser.add_argument("--contrast-margin", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-beta", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--resume-split-membership", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


class TAMFProcessedDataset(Dataset):
    def __init__(self, base_dataset: TripleWindowDataset, binary_task_index: int = 0):
        self.base_dataset = base_dataset
        self.binary_task_index = binary_task_index

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        item = self.base_dataset[index]
        target = item["binary_targets"][self.binary_task_index : self.binary_task_index + 1]
        return {
            "sample_id": item["sample_id"],
            "x_list": item["x_list"],
            "m_list": item["m_list"],
            "t_list": item["t_list"],
            "static": item["static"],
            "target": target,
        }


def collate_tamf_batch(batch: list[dict]) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    source_count = len(batch[0]["x_list"])
    x_list = []
    m_list = []
    t_list = []
    for source_idx in range(source_count):
        x_list.append(torch.stack([item["x_list"][source_idx] for item in batch], dim=1))
        m_list.append(torch.stack([item["m_list"][source_idx] for item in batch], dim=1))
        t_list.append(torch.stack([item["t_list"][source_idx] for item in batch], dim=1))
    sample_ids = [item["sample_id"] for item in batch]
    static = torch.stack([item["static"] for item in batch], dim=0)
    target = torch.stack([item["target"] for item in batch], dim=0)
    return sample_ids, x_list, m_list, t_list, static, target


def split_indices(total: int, test_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    test_count = max(1, int(total * test_ratio))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return train_indices, test_indices


def load_split_membership(base_dataset: TripleWindowDataset, split_csv: Path) -> tuple[list[int], list[int]]:
    df = pd.read_csv(split_csv)
    sample_id_to_index = {sample_id: idx for idx, sample_id in enumerate(base_dataset.sample_ids)}
    train_indices: list[int] = []
    test_indices: list[int] = []
    for row in df.itertuples(index=False):
        sample_id = row.sample_id
        if sample_id not in sample_id_to_index:
            continue
        if row.split == "train":
            train_indices.append(sample_id_to_index[sample_id])
        elif row.split == "test":
            test_indices.append(sample_id_to_index[sample_id])
    if not train_indices or not test_indices:
        raise ValueError(f"Invalid split membership file: {split_csv}")
    return train_indices, test_indices


def make_loader(
    dataset: Dataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    worker_count = max(0, int(num_workers))
    loader_kwargs = {
        "dataset": Subset(dataset, indices),
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_tamf_batch,
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
    }
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(
        **loader_kwargs,
    )


def move_batch(batch, device: str):
    sample_ids, x_list, m_list, t_list, static, target = batch
    return (
        sample_ids,
        [x.to(device) for x in x_list],
        [m.to(device) for m in m_list],
        [t.to(device) for t in t_list],
        static.to(device),
        target.to(device),
    )


def compute_batch_loss(model, x_list, m_list, t_list, static, target, args: argparse.Namespace):
    classification_output, embeddings, reconstructions = model(x_list, m_list, t_list, static)
    focal = focal_loss(classification_output, target, gamma=args.focal_gamma, beta=args.focal_beta)
    recon = reconstruction_loss(reconstructions, x_list, m_list)
    pooled_embeddings = [torch.mean(x_out, dim=0) for x_out, _ in embeddings]
    contrast = contrastive_loss(pooled_embeddings, margin=args.contrast_margin)
    total = args.lambda_focal * focal + args.lambda_recon * recon + args.lambda_contrast * contrast
    return total, focal, recon, contrast, classification_output


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def run_epoch(
    model: MultiSourceModel,
    loader: DataLoader,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
):
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    total_focal = 0.0
    total_recon = 0.0
    total_contrast = 0.0
    total_samples = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    prediction_rows: list[dict[str, float | str | int]] = []

    for batch_idx, raw_batch in enumerate(loader, start=1):
        sample_ids, x_list, m_list, t_list, static, target = move_batch(raw_batch, args.device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, focal, recon, contrast, classification_output = compute_batch_loss(
                model, x_list, m_list, t_list, static, target, args
            )
            if training:
                loss.backward()
                optimizer.step()

        probs = classification_output.detach().squeeze(-1).cpu().numpy()
        targets = target.detach().squeeze(-1).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        for sample_id, prob, label, pred in zip(sample_ids, probs, targets, preds):
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "probability": float(prob),
                    "target": int(label),
                    "prediction": int(pred),
                }
            )

        batch_size = target.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_focal += float(focal.item()) * batch_size
        total_recon += float(recon.item()) * batch_size
        total_contrast += float(contrast.item()) * batch_size
        total_samples += batch_size
        all_probs.append(probs)
        all_targets.append(targets)

        if training and batch_idx % args.log_interval == 0:
            logging.info(
                "Train Batch [%s/%s] - Loss: %.4f (Focal: %.4f, Recon: %.4f, Contrast: %.4f)",
                batch_idx,
                len(loader),
                float(loss.item()),
                float(focal.item()),
                float(recon.item()),
                float(contrast.item()),
            )

    y_score = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets).astype(int)
    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        "loss": total_loss / total_samples,
        "focal_loss": total_focal / total_samples,
        "recon_loss": total_recon / total_samples,
        "contrast_loss": total_contrast / total_samples,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "positive_rate": float(y_true.mean()),
        "pred_positive_rate": float(y_pred.mean()),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "auprc": safe_auprc(y_true, y_score),
    }
    return metrics, prediction_rows


def save_split_files(
    base_dataset: TripleWindowDataset,
    train_indices: list[int],
    test_indices: list[int],
    save_dir: Path,
    args: argparse.Namespace,
) -> None:
    rows = []
    for split_name, indices in (("train", train_indices), ("test", test_indices)):
        for idx in indices:
            rows.append(
                {
                    "split": split_name,
                    "index": idx,
                    "sample_id": base_dataset.sample_ids[idx],
                }
            )
    pd.DataFrame(rows).to_csv(save_dir / "split_membership.csv", index=False)

    summary = {
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "total_samples": len(base_dataset),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "binary_task_index": args.binary_task_index,
    }
    (save_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (PROJECT_DIR / save_dir).resolve()
    setup_logging(save_dir)

    cfg = load_config(PROJECT_DIR / args.config)
    processed_dir = resolve_path(PROJECT_DIR, args.processed_dir or cfg["data"]["processed_dir"])
    base_dataset = TripleWindowDataset(
        processed_dir=processed_dir,
        tasks_binary=cfg["model"]["tasks"]["binary"],
        tasks_regression=cfg["model"]["tasks"]["regression"],
        max_samples=None,
        reasoning_csv=resolve_path(PROJECT_DIR, cfg["data"]["reasoning_csv"]),
    )
    resume_checkpoint_path = None
    if args.resume_checkpoint:
        resume_checkpoint_path = Path(args.resume_checkpoint)
        if not resume_checkpoint_path.is_absolute():
            resume_checkpoint_path = (PROJECT_DIR / resume_checkpoint_path).resolve()

    resume_split_path = None
    if args.resume_split_membership:
        resume_split_path = Path(args.resume_split_membership)
        if not resume_split_path.is_absolute():
            resume_split_path = (PROJECT_DIR / resume_split_path).resolve()
    elif resume_checkpoint_path is not None:
        inferred_split = resume_checkpoint_path.parent / "split_membership.csv"
        if inferred_split.exists():
            resume_split_path = inferred_split

    if resume_split_path is not None:
        train_indices, test_indices = load_split_membership(base_dataset, resume_split_path)
    else:
        train_indices, test_indices = split_indices(len(base_dataset), args.test_ratio, args.seed)
    stage1_dataset = TAMFProcessedDataset(base_dataset, binary_task_index=args.binary_task_index)

    train_loader = make_loader(
        stage1_dataset,
        train_indices,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    test_loader = make_loader(
        stage1_dataset,
        test_indices,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    source_dims = [base_dataset.source_dims[source] for source in SOURCE_ORDER]
    model = MultiSourceModel(
        input_dims=source_dims,
        mask_dims=source_dims,
        time_dims=[1] * len(source_dims),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        static_dim=base_dataset.static_dim,
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    previous_best_test_loss = None
    previous_best_epoch = None
    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        previous_metrics = checkpoint.get("test_metrics") or {}
        if "loss" in previous_metrics and previous_metrics["loss"] is not None:
            previous_best_test_loss = float(previous_metrics["loss"])
            previous_best_epoch = int(checkpoint["epoch"])

    logging.info("Resolved processed_dir: %s", processed_dir)
    logging.info("Source dims: %s", source_dims)
    logging.info("Train/Test split: %s / %s", len(train_indices), len(test_indices))
    logging.info("Training args: %s", vars(args))
    if resume_checkpoint_path is not None:
        logging.info("Resume checkpoint: %s", resume_checkpoint_path)
        logging.info("Resume split_membership: %s", resume_split_path)
        logging.info("Resuming from epoch %s", start_epoch)
    save_split_files(base_dataset, train_indices, test_indices, save_dir, args)

    history: list[dict[str, float | int | None]] = []
    best_state = checkpoint if resume_checkpoint_path is not None else None
    best_test_loss = previous_best_test_loss
    best_epoch = previous_best_epoch
    best_predictions = None

    final_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, final_epoch + 1):
        logging.info("")
        logging.info("Epoch %s/%s", epoch, final_epoch)
        train_metrics, _ = run_epoch(model, train_loader, args, optimizer=optimizer)
        test_metrics, test_predictions = run_epoch(model, test_loader, args, optimizer=None)
        epoch_record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"test_{k}": v for k, v in test_metrics.items()}}
        history.append(epoch_record)
        logging.info("Train metrics: %s", json.dumps(train_metrics, ensure_ascii=False))
        logging.info("Test metrics: %s", json.dumps(test_metrics, ensure_ascii=False))

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "args": vars(args),
            "source_dims": source_dims,
            "processed_dir": str(processed_dir),
            "resume_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path is not None else None,
        }
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        if best_test_loss is None or test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]
            best_state = checkpoint
            best_epoch = epoch
            best_predictions = test_predictions

    pd.DataFrame(history).to_csv(save_dir / "metrics_history.csv", index=False)
    if best_state is not None:
        torch.save(best_state, save_dir / "best_test_loss.pt")
    if best_predictions is not None:
        pd.DataFrame(best_predictions).to_csv(save_dir / "best_test_predictions.csv", index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_test_loss": best_test_loss,
        "run_dir": str(save_dir),
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Best summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
