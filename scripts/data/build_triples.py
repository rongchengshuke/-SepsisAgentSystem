from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.data.triple_builder import build_processed_dataset
from scripts.framework.utils.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-patients", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_DIR / args.config)
    mimiciv_dir = resolve_path(PROJECT_DIR, cfg["data"]["mimiciv_dir"])
    output_dir = resolve_path(PROJECT_DIR, cfg["data"]["processed_dir"])
    outlier_range_path = None
    if cfg["data"].get("outlier_range_path"):
        outlier_range_path = resolve_path(PROJECT_DIR, cfg["data"]["outlier_range_path"])
    stats = build_processed_dataset(
        mimiciv_dir=mimiciv_dir,
        output_dir=output_dir,
        outlier_range_path=outlier_range_path,
        min_age=cfg["data"]["cohort"]["min_age"],
        min_los_icu_day=cfg["data"]["cohort"]["min_los_icu_day"],
        max_los_icu_day=cfg["data"]["cohort"]["max_los_icu_day"],
        anchor_year_groups=cfg["data"]["cohort"].get("anchor_year_groups"),
        length_hours=cfg["data"]["windows"]["length_hours"],
        step_hours=cfg["data"]["windows"]["step_hours"],
        min_start_hour=cfg["data"]["windows"]["min_start_hour"],
        max_windows_per_stay=cfg["data"]["windows"]["max_windows_per_stay"],
        max_patients=args.max_patients,
    )
    print(stats)


if __name__ == "__main__":
    main()
