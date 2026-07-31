from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.utils.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_compact_los1_7_year1119.yaml")
    parser.add_argument("--input-csv", default=None, help="Legacy direct dynamic CSV input. Original flow ignores this.")
    parser.add_argument("--outlier-range-csv", default=None)
    parser.add_argument("--output-csv", default="data/reasoning/mimic_dynamic_llm_imputed.csv")
    parser.add_argument("--output-static-csv", default="data/reasoning/mimic_static_llm_preprocessed.csv")
    parser.add_argument("--summary-json", default="data/reasoning/mimic_dynamic_llm_imputed_summary.json")
    parser.add_argument(
        "--mode",
        choices=["original_flow", "dynamic_only"],
        default="original_flow",
        help="original_flow follows 3.dynamic_combine.py + 4.preprocess.py; dynamic_only keeps the previous direct CSV path.",
    )
    return parser.parse_args()


def remove_outliers(data: pd.DataFrame, outlier_range_csv: Path) -> pd.DataFrame:
    if not outlier_range_csv.exists():
        raise FileNotFoundError(f"outlier range file not found: {outlier_range_csv}")
    ranges = pd.read_csv(outlier_range_csv).rename(columns={"Unnamed: 0": "index_name"})
    result = data.copy()
    result.columns = [str(column).lower() for column in result.columns]
    for row in ranges.itertuples(index=False):
        column = str(row.index_name).lower()
        if column not in result.columns:
            continue
        result.loc[
            (result[column] > float(row.upper_bound)) | (result[column] < float(row.lower_bound)),
            column,
        ] = np.nan
    return result


def add_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if {"pao2", "fio2"}.issubset(result.columns):
        result["pao2fio2"] = (result["pao2"] / result["fio2"]).round(2)
    if {"neutrophils", "lymphocytes"}.issubset(result.columns):
        result["NLR"] = (result["neutrophils"] / result["lymphocytes"]).round(2)
    if {"platelet", "neutrophils", "lymphocytes"}.issubset(result.columns):
        result["SII"] = (result["platelet"] * result["neutrophils"] / result["lymphocytes"]).round(2)
    if {"platelet", "lymphocytes"}.issubset(result.columns):
        result["PLR"] = (result["platelet"] / result["lymphocytes"]).round(2)
    return result


def cal_gnri(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    required = {"height", "gender", "albumin", "weight"}
    if not required.issubset(result.columns):
        return result
    ideal_weight_male = result["height"] - 100 - ((result["height"] - 150) / 4)
    ideal_weight_female = result["height"] - 100 - ((result["height"] - 150) / 2.5)
    ideal_weight = np.where(result["gender"] == 1, ideal_weight_male, ideal_weight_female)
    result["gnri"] = ((14.89 * result["albumin"]) + (41.7 * (result["weight"] / ideal_weight))).round(2)
    return result


def cal_vasopressors(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    pressor_cols = ["dobutamine", "dopamine", "epinephrine", "norepinephrine"]
    if all(column in result.columns for column in pressor_cols):
        result["vasopressors"] = np.where(
            (result["dobutamine"] == 1)
            | (result["dopamine"] == 1)
            | (result["epinephrine"] == 1)
            | (result["norepinephrine"] == 1),
            1,
            0,
        )
        result.drop(columns=pressor_cols, inplace=True)
    return result


def preprocess_original(data: pd.DataFrame, outlier_range_csv: Path) -> pd.DataFrame:
    result = remove_outliers(data, outlier_range_csv)
    if {"ventilation", "fio2"}.issubset(result.columns):
        result["fio2"] = np.where(result["ventilation"] == 0, 21, result["fio2"])

    result = result.sort_values(["stay_id", "hr"]).reset_index(drop=True)
    result = result.groupby("stay_id", group_keys=False).apply(lambda x: x.ffill().bfill())

    fill_columns = [
        column
        for column in result.columns
        if column not in {"stay_id", "hr"} and pd.api.types.is_numeric_dtype(result[column])
    ]
    result[fill_columns] = result[fill_columns].fillna(result[fill_columns].median(numeric_only=True))
    result = add_derived_features(result)
    result = cal_vasopressors(result)
    result = cal_gnri(result)
    return result


def prepare_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    result = cohort.copy()
    result.columns = [str(column).lower() for column in result.columns]
    if "gender" in result.columns:
        if result["gender"].dtype == object:
            result["gender"] = result["gender"].map({"M": 1, "F": 0, "m": 1, "f": 0})
        result["gender"] = result["gender"].fillna(result["gender"].mode()[0])
    if "bmi" in result.columns:
        result["bmi"] = result["bmi"].fillna(result["bmi"].median())
    if "height" in result.columns:
        result["height"] = result["height"].fillna(result["height"].median())
    return result


def inject_raw_bnp_if_missing(lab: pd.DataFrame, mimiciv_dir: Path | None) -> tuple[pd.DataFrame, str | None]:
    if "bnp" in lab.columns or mimiciv_dir is None:
        return lab, None
    raw_lab_path = mimiciv_dir / "lab_test.csv"
    if not raw_lab_path.exists():
        return lab, None
    raw_bnp = pd.read_csv(raw_lab_path, usecols=["stay_id", "hr", "bnp"])
    result = lab.merge(raw_bnp, on=["stay_id", "hr"], how="left")
    return result, str(raw_lab_path)


def build_combined_dynamic(masked_dir: Path, mimiciv_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    vital = pd.read_csv(masked_dir / "mimic_vital.csv")
    lab = pd.read_csv(masked_dir / "mimic_lab.csv")
    lab, raw_bnp_source = inject_raw_bnp_if_missing(lab, mimiciv_dir)
    treatment = pd.read_csv(masked_dir / "mimic_treatment.csv")
    dynamic = vital.merge(lab, on=["stay_id", "hr"], how="inner")
    dynamic = dynamic.merge(treatment, on=["stay_id", "hr"], how="inner")
    dynamic = dynamic.sort_values(["stay_id", "hr"]).reset_index(drop=True)
    metadata = {
        "raw_bnp_source": raw_bnp_source,
        "bnp_in_combined_dynamic": "bnp" in dynamic.columns,
    }
    return dynamic[dynamic["hr"] != 0].copy(), metadata


def build_original_flow(
    masked_dir: Path,
    outlier_range_csv: Path,
    mimiciv_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cohort = prepare_cohort(pd.read_csv(masked_dir / "mimic_cohort.csv"))
    dynamic, combine_metadata = build_combined_dynamic(masked_dir, mimiciv_dir)
    # Same intent as the original comments in 4.preprocess.py: weight is treated
    # as dynamic, while height/gender come from the cohort table for GNRI.
    if "weight" in cohort.columns and "weight" in dynamic.columns:
        cohort = cohort.drop(columns=["weight"])
    cohort_columns = cohort.columns.tolist()
    merged = dynamic.merge(cohort, on="stay_id", how="left")
    preprocessed = preprocess_original(merged, outlier_range_csv).reset_index(drop=True)

    dynamic_preprocessed = preprocessed.drop(
        columns=[column for column in cohort_columns if column != "stay_id"],
        errors="ignore",
    )
    static_base = cohort[["stay_id", "age", "gender", "bmi", "cci_score"]].copy()

    if "gnri" in dynamic_preprocessed.columns:
        first_24h_gnri = (
            dynamic_preprocessed.sort_values(["stay_id", "hr"])
            .groupby("stay_id")
            .head(24)
            .groupby("stay_id")["gnri"]
            .mean()
            .reset_index()
        )
    else:
        first_24h_gnri = pd.DataFrame({"stay_id": static_base["stay_id"], "gnri": np.nan})
    static_preprocessed = first_24h_gnri.merge(static_base, on="stay_id", how="left")
    if "gnri" in dynamic_preprocessed.columns:
        dynamic_preprocessed = dynamic_preprocessed.drop(columns=["gnri"])

    summary = {
        "mode": "original_flow",
        "cohort_rows": int(len(cohort)),
        "combined_dynamic_rows_after_drop_hr0": int(len(dynamic)),
        "preprocessed_rows": int(len(preprocessed)),
        "cohort_columns_removed_from_dynamic": [column for column in cohort_columns if column != "stay_id"],
        **combine_metadata,
    }
    return dynamic_preprocessed, static_preprocessed, summary


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_DIR / args.config)
    masked_dir = resolve_path(PROJECT_DIR, cfg["data"]["masked_mimic_dir"])
    mimiciv_dir = resolve_path(PROJECT_DIR, cfg["data"]["mimiciv_dir"]) if cfg["data"].get("mimiciv_dir") else None
    input_csv = resolve_path(PROJECT_DIR, args.input_csv) if args.input_csv else masked_dir / "mimic_dynamic.csv"
    outlier_range_csv = (
        resolve_path(PROJECT_DIR, args.outlier_range_csv)
        if args.outlier_range_csv
        else resolve_path(PROJECT_DIR, cfg["data"].get("outlier_range_path", "../outlier_range_check.csv"))
    )
    output_csv = resolve_path(PROJECT_DIR, args.output_csv)
    output_static_csv = resolve_path(PROJECT_DIR, args.output_static_csv)
    summary_json = resolve_path(PROJECT_DIR, args.summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_static_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "original_flow":
        raw_for_missing, raw_combine_metadata = build_combined_dynamic(masked_dir, mimiciv_dir)
        before_missing = raw_for_missing.isna().sum().to_dict()
        imputed, static_preprocessed, flow_summary = build_original_flow(masked_dir, outlier_range_csv, mimiciv_dir)
        flow_summary = {**raw_combine_metadata, **flow_summary}
        static_preprocessed.to_csv(output_static_csv, index=False, encoding="utf-8")
    else:
        raw = pd.read_csv(input_csv)
        before_missing = raw.isna().sum().to_dict()
        imputed = preprocess_original(raw, outlier_range_csv)
        static_preprocessed = None
        flow_summary = {"mode": "dynamic_only"}
    after_missing = imputed.isna().sum().to_dict()
    imputed.to_csv(output_csv, index=False, encoding="utf-8")

    summary = {
        **flow_summary,
        "input_csv": str(input_csv),
        "source_files": (
            {
                "cohort": str(masked_dir / "mimic_cohort.csv"),
                "vital": str(masked_dir / "mimic_vital.csv"),
                "lab": str(masked_dir / "mimic_lab.csv"),
                "raw_bnp": flow_summary.get("raw_bnp_source"),
                "treatment": str(masked_dir / "mimic_treatment.csv"),
            }
            if args.mode == "original_flow"
            else {"dynamic": str(input_csv)}
        ),
        "outlier_range_csv": str(outlier_range_csv),
        "output_csv": str(output_csv),
        "output_static_csv": str(output_static_csv) if static_preprocessed is not None else None,
        "rows": int(len(imputed)),
        "columns": list(imputed.columns),
        "before_missing_total": int(sum(int(v) for v in before_missing.values())),
        "after_missing_total": int(sum(int(v) for v in after_missing.values())),
        "tracked_after_missing": {
            key: int(after_missing.get(key, 0))
            for key in [
                "heart_rate",
                "sbp",
                "mbp",
                "spo2",
                "glucose",
                "creatinine",
                "bnp",
                "platelet",
                "neutrophils",
                "lymphocytes",
                "vasopressors",
                "ventilation",
            ]
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
