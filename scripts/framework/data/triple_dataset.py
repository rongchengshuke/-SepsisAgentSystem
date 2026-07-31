from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from scripts.framework.data.feature_schema import SOURCE_FEATURES, STATIC_FEATURES


SOURCE_ORDER = ["lab", "vital", "treatment"]


class TripleWindowDataset(Dataset):
    """Window-level tensors derived from long-form triples.

    Each source is converted back to a fixed [time, variable] grid so the
    TAMF-style encoder can model values, masks, and time per source.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        tasks_binary: list[str],
        tasks_regression: list[str],
        max_samples: int | None = None,
        reasoning_csv: str | Path | None = None,
        note_csv: str | Path | None = None,
        image_manifest_csv: str | Path | None = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.tasks_binary = tasks_binary
        self.tasks_regression = tasks_regression

        self.sample_index = pd.read_csv(self.processed_dir / "sample_index.csv")
        if max_samples is not None:
            self.sample_index = self.sample_index.head(max_samples).copy()
        self.sample_ids = self.sample_index["sample_id"].tolist()

        self.labels = pd.read_csv(self.processed_dir / "labels.csv").set_index("sample_id")
        self.static = pd.read_csv(self.processed_dir / "static.csv").set_index("sample_id")
        self.triples = pd.read_parquet(self.processed_dir / "all_triples.parquet")
        self.triples = self.triples[self.triples["sample_id"].isin(self.sample_ids)]
        self.reasoning_map: dict[str, str] = {}
        self.note_map: dict[str, str] = {}
        self.image_map: dict[str, str] = {}
        if reasoning_csv is not None:
            reasoning_path = Path(reasoning_csv)
            if reasoning_path.exists():
                reasoning_df = pd.read_csv(reasoning_path)
                if {"sample_id", "reasoning_text"}.issubset(reasoning_df.columns):
                    self.reasoning_map = dict(zip(reasoning_df["sample_id"], reasoning_df["reasoning_text"]))
        if note_csv is not None:
            note_path = Path(note_csv)
            if note_path.exists():
                note_df = pd.read_csv(note_path)
                if {"sample_id", "note_text"}.issubset(note_df.columns):
                    self.note_map = dict(zip(note_df["sample_id"], note_df["note_text"]))
        if image_manifest_csv is not None:
            image_manifest_path = Path(image_manifest_csv)
            if image_manifest_path.exists():
                image_df = pd.read_csv(image_manifest_path)
                if {"sample_id", "image_path"}.issubset(image_df.columns):
                    self.image_map = dict(zip(image_df["sample_id"], image_df["image_path"]))

        # Keep a fixed feature schema per source so training/inference use the same
        # variable order even when a given sample subset never observes some variables.
        self.source_features = {
            source: list(SOURCE_FEATURES[source])
            for source in SOURCE_ORDER
        }
        self.feature_to_idx = {
            source: {feature: idx for idx, feature in enumerate(features)}
            for source, features in self.source_features.items()
        }
        self.max_time = int(self.triples["time"].max()) + 1 if len(self.triples) else 24
        self.static_cols = [c for c in STATIC_FEATURES if c in self.static.columns]

        self.value_stats = {}
        for source in SOURCE_ORDER:
            src = self.triples[self.triples["source"] == source]
            stats = src.groupby("variable")["value"].agg(["mean", "std"])
            stats["std"] = stats["std"].replace(0, 1).fillna(1)
            self.value_stats[source] = stats

        self.grouped = {
            sample_id: frame
            for sample_id, frame in self.triples.groupby("sample_id", sort=False)
        }

    @property
    def source_dims(self) -> dict[str, int]:
        return {source: len(features) for source, features in self.source_features.items()}

    @property
    def static_dim(self) -> int:
        return len(self.static_cols)

    @property
    def num_binary_tasks(self) -> int:
        return len(self.tasks_binary)

    @property
    def num_regression_tasks(self) -> int:
        return len(self.tasks_regression)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _empty_source(self, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dim = len(self.source_features[source])
        values = np.zeros((self.max_time, dim), dtype=np.float32)
        masks = np.zeros((self.max_time, dim), dtype=np.float32)
        times = np.linspace(0, 1, self.max_time, dtype=np.float32).reshape(self.max_time, 1)
        return values, masks, times

    def _build_source_grid(
        self, sample_triples: pd.DataFrame, source: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values, masks, times = self._empty_source(source)
        src = sample_triples[sample_triples["source"] == source]
        feature_to_idx = self.feature_to_idx[source]
        stats = self.value_stats[source]
        for row in src.itertuples(index=False):
            if row.variable not in feature_to_idx:
                continue
            t = int(row.time)
            if t < 0 or t >= self.max_time:
                continue
            j = feature_to_idx[row.variable]
            mean = float(stats.loc[row.variable, "mean"]) if row.variable in stats.index else 0.0
            std = float(stats.loc[row.variable, "std"]) if row.variable in stats.index else 1.0
            values[t, j] = (float(row.value) - mean) / std
            masks[t, j] = 1.0
        return values, masks, times

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_id = self.sample_ids[index]
        sample_triples = self.grouped.get(sample_id, self.triples.iloc[0:0])

        x_list, m_list, t_list = [], [], []
        for source in SOURCE_ORDER:
            values, masks, times = self._build_source_grid(sample_triples, source)
            x_list.append(torch.from_numpy(values))
            m_list.append(torch.from_numpy(masks))
            t_list.append(torch.from_numpy(times))

        static_values = self.static.loc[sample_id, self.static_cols].astype("float32").to_numpy()
        binary = self.labels.loc[sample_id, self.tasks_binary].fillna(0).astype("float32").to_numpy()
        regression = self.labels.loc[sample_id, self.tasks_regression].fillna(0).astype("float32").to_numpy()

        return {
            "sample_id": sample_id,
            "x_list": x_list,
            "m_list": m_list,
            "t_list": t_list,
            "static": torch.from_numpy(static_values),
            "binary_targets": torch.from_numpy(binary),
            "regression_targets": torch.from_numpy(regression),
            "note_text": self.note_map.get(sample_id, ""),
            "reasoning_text": self.reasoning_map.get(sample_id, ""),
            "image_path": self.image_map.get(sample_id, ""),
        }


def collate_triple_windows(batch: list[dict]) -> dict[str, torch.Tensor | list[str] | list[torch.Tensor]]:
    source_count = len(batch[0]["x_list"])
    x_list = []
    m_list = []
    t_list = []
    for source_idx in range(source_count):
        x_list.append(torch.stack([item["x_list"][source_idx] for item in batch], dim=0))
        m_list.append(torch.stack([item["m_list"][source_idx] for item in batch], dim=0))
        t_list.append(torch.stack([item["t_list"][source_idx] for item in batch], dim=0))
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "x_list": x_list,
        "m_list": m_list,
        "t_list": t_list,
        "static": torch.stack([item["static"] for item in batch], dim=0),
        "binary_targets": torch.stack([item["binary_targets"] for item in batch], dim=0),
        "regression_targets": torch.stack([item["regression_targets"] for item in batch], dim=0),
        "note_texts": [item.get("note_text", "") for item in batch],
        "reasoning_texts": [item.get("reasoning_text", "") for item in batch],
        "image_paths": [item.get("image_path", "") for item in batch],
    }
