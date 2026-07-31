from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_INDEX = PROJECT_DIR / "data" / "processed_dataset_outputs" / "sample_index.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "splits" / "stage_pipeline_90_10_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index-csv", default=str(DEFAULT_SAMPLE_INDEX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-train-size", type=int, default=1000)
    parser.add_argument("--reuse-existing-stage1-split", action="store_true")
    parser.add_argument(
        "--existing-stage1-split-csv",
        default=str(PROJECT_DIR / "vendor" / "TAMF" / "deep_learning" / "checkpoints" / "stage1_split_B_saved" / "split_membership.csv"),
    )
    return parser.parse_args()


def random_split(sample_index: pd.DataFrame, train_ratio: float, seed: int) -> pd.DataFrame:
    indices = list(range(len(sample_index)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_count = int(len(indices) * train_ratio)
    train_indices = set(indices[:train_count])
    split_rows = []
    for idx, row in sample_index.reset_index(drop=True).iterrows():
        split_rows.append(
            {
                "split": "train" if idx in train_indices else "test",
                "index": idx,
                "sample_id": row["sample_id"],
                "stay_id": int(row["stay_id"]),
                "window_start": int(row["window_start"]),
                "window_end": int(row["window_end"]),
            }
        )
    return pd.DataFrame(split_rows)


def stage2_memory_subset(train_df: pd.DataFrame, subset_size: int, seed: int) -> pd.DataFrame:
    if subset_size >= len(train_df):
        return train_df.copy()
    return train_df.sample(n=subset_size, random_state=seed).sort_values("index").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    sample_index_path = Path(args.sample_index_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_index = pd.read_csv(sample_index_path)
    if args.reuse_existing_stage1_split and Path(args.existing_stage1_split_csv).exists():
        split_df = pd.read_csv(args.existing_stage1_split_csv)
        split_df = split_df.merge(sample_index, on="sample_id", how="left")
        split_df = split_df[["split", "index", "sample_id", "stay_id", "window_start", "window_end"]].copy()
    else:
        split_df = random_split(sample_index, train_ratio=args.train_ratio, seed=args.seed)

    train_df = split_df[split_df["split"] == "train"].copy().reset_index(drop=True)
    test_df = split_df[split_df["split"] == "test"].copy().reset_index(drop=True)
    memory_df = stage2_memory_subset(train_df, subset_size=args.memory_train_size, seed=args.seed)
    memory_df = memory_df.copy()
    memory_df["split"] = "memory_train"

    split_df.to_csv(output_dir / "split_membership.csv", index=False)
    train_df.to_csv(output_dir / "stage1_train.csv", index=False)
    test_df.to_csv(output_dir / "stage1_test.csv", index=False)
    memory_df.to_csv(output_dir / "stage2_memory_train1000.csv", index=False)

    summary = {
        "sample_index_csv": str(sample_index_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "reuse_existing_stage1_split": bool(args.reuse_existing_stage1_split and Path(args.existing_stage1_split_csv).exists()),
        "total_samples": int(len(split_df)),
        "stage1_train_samples": int(len(train_df)),
        "stage1_test_samples": int(len(test_df)),
        "stage2_memory_train_samples": int(len(memory_df)),
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
