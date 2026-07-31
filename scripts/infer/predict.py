from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.data.triple_dataset import TripleWindowDataset, collate_triple_windows
from scripts.framework.models.carer_tamf import CARERTAMFModel
from scripts.framework.utils.config import load_config, resolve_path


def infer_ehr_encoder_backend(model_state_dict: dict[str, torch.Tensor]) -> str:
    if any(key.startswith("ehr_encoder.sequence_embeddings.") for key in model_state_dict):
        return "tamf_upstream_compatible"
    if any(key.startswith("ehr_encoder.source_encoders.") for key in model_state_dict):
        return "pooled_simplified"
    return "tamf_upstream_compatible"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint-path", default="models/checkpoints/trained_pipeline/latest.pt")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--note-csv", default=None)
    parser.add_argument("--image-manifest-csv", default=None)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-csv", default="outputs/predictions.csv")
    return parser.parse_args()


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
    return {"train": train, "val": val, "test": test, "all": indices}


def placeholder_reasoning_texts(sample_ids: list[str]) -> list[str]:
    return [f"elderly sepsis sample {sample_id}" for sample_id in sample_ids]


def placeholder_note_texts(sample_ids: list[str]) -> list[str]:
    return [f"synthetic clinical note for {sample_id}" for sample_id in sample_ids]


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


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_DIR / args.config)
    processed_dir = resolve_path(PROJECT_DIR, args.processed_dir or cfg["data"]["processed_dir"])
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
    indices = splits[args.split]
    if not indices:
        raise SystemExit(f"Split {args.split} is empty.")

    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, collate_fn=collate_triple_windows)
    checkpoint_path = resolve_path(PROJECT_DIR, args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_args"].setdefault(
        "ehr_encoder_backend",
        infer_ehr_encoder_backend(checkpoint["model_state_dict"]),
    )
    original_note_encoder = checkpoint["model_args"].get("note_text_encoder")
    checkpoint["model_args"]["note_text_encoder"] = "empty"
    checkpoint["model_args"]["note_text_model_name"] = None
    checkpoint.setdefault("runtime_overrides", {})["note_text_encoder"] = {
        "from": original_note_encoder,
        "to": "empty",
        "reason": "raw admission record is LLM input only; encode LLM reasoning output downstream",
    }
    current_source_dims = [dataset.source_dims[source] for source in ["lab", "vital", "treatment"]]
    checkpoint_source_dims = checkpoint["model_args"]["source_dims"]
    if list(checkpoint_source_dims) != list(current_source_dims):
        raise SystemExit(
            f"Checkpoint source_dims {checkpoint_source_dims} do not match current dataset source_dims {current_source_dims}. "
            "Use a compatible checkpoint or retrain with the current processed data."
        )
    device = choose_device(args.device)
    model = CARERTAMFModel(**checkpoint["model_args"]).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(
            "checkpoint_compat_warning:",
            {
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
                "runtime_overrides": checkpoint.get("runtime_overrides", {}),
            },
            flush=True,
        )
    model.eval()

    output_csv = resolve_path(PROJECT_DIR, args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    binary_tasks = checkpoint["tasks_binary"]
    regression_tasks = checkpoint["tasks_regression"]
    fieldnames = ["sample_id"]
    for task in binary_tasks:
        fieldnames.extend([f"{task}_prob", f"{task}_target"])
    for task in regression_tasks:
        fieldnames.extend([f"{task}_pred", f"{task}_target"])

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for raw_batch in loader:
                batch = move_batch(raw_batch, device)
                outputs = model(
                    x_list=batch["x_list"],
                    m_list=batch["m_list"],
                    t_list=batch["t_list"],
                    static=batch["static"],
                    note_texts=batch.get("note_texts") or placeholder_note_texts(batch["sample_id"]),
                    reasoning_texts=batch.get("reasoning_texts") or placeholder_reasoning_texts(batch["sample_id"]),
                    image_paths=batch.get("image_paths") or [""] * len(batch["sample_id"]),
                )
                binary_prob = torch.sigmoid(outputs["binary_logits"]).cpu()
                regression_pred = outputs["regression"].cpu()
                binary_targets = batch["binary_targets"].cpu()
                regression_targets = batch["regression_targets"].cpu()

                for row_idx, sample_id in enumerate(batch["sample_id"]):
                    row = {"sample_id": sample_id}
                    for task_idx, task in enumerate(binary_tasks):
                        row[f"{task}_prob"] = float(binary_prob[row_idx, task_idx])
                        row[f"{task}_target"] = float(binary_targets[row_idx, task_idx])
                    for task_idx, task in enumerate(regression_tasks):
                        row[f"{task}_pred"] = float(regression_pred[row_idx, task_idx])
                        row[f"{task}_target"] = float(regression_targets[row_idx, task_idx])
                    writer.writerow(row)

    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"predictions: {output_csv}", flush=True)
    print(f"rows: {len(indices)}", flush=True)


if __name__ == "__main__":
    main()
