from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.framework.data.feature_schema import SOURCE_FEATURES, STATIC_FEATURES


LABEL_COLUMNS = [
    "death_hosp",
    "death_icu",
    "death_leavehp_in_30d",
    "los_icu_day",
    "los_hospital_day",
    "deathtime_icu_hour",
    "los_icu_admit_day",
    "sepsis_time",
]

SOFA_SUBSCORES = [
    "respiration",
    "coagulation",
    "liver",
    "cardiovascular",
    "cns",
    "renal",
]

OUTLIER_ALIASES = {
    "be": "baseexcess",
    "pafi": "pao2fio2ratio",
}


def load_elder_sepsis_cohort(
    mimiciv_dir: Path,
    min_age: int = 65,
    min_los_icu_day: float = 1,
    max_los_icu_day: float = 30,
    anchor_year_groups: list[str] | None = None,
    max_patients: int | None = None,
) -> pd.DataFrame:
    cohort = pd.read_csv(mimiciv_dir / "study_cohort.csv")
    mask = (
        (cohort["age"] >= min_age)
        & cohort["los_icu_day"].between(min_los_icu_day, max_los_icu_day)
        & cohort["death_hosp"].notna()
        & cohort["sepsis_time"].notna()
    )
    # Keep year-group filtering optional so older preprocessing behavior remains available.
    if anchor_year_groups:
        normalized_groups = {str(item).strip() for item in anchor_year_groups if str(item).strip()}
        if normalized_groups and "anchor_year_group" in cohort.columns:
            mask = mask & cohort["anchor_year_group"].astype(str).str.strip().isin(normalized_groups)
    cohort = cohort.loc[mask].copy()
    if max_patients is not None:
        cohort = cohort.sort_values("stay_id").head(max_patients).copy()

    cohort["gender"] = cohort["gender"].map({"M": 1, "F": 0}).fillna(0)
    cohort["bmi"] = cohort["bmi"].fillna(cohort["bmi"].median())
    cohort["cci_score"] = cohort["cci_score"].fillna(cohort["cci_score"].median())
    return cohort


def load_outlier_ranges(outlier_range_path: Path | None) -> pd.DataFrame:
    if outlier_range_path is None or not outlier_range_path.exists():
        return pd.DataFrame(columns=["variable", "upper_bound", "lower_bound"])

    ranges = pd.read_csv(outlier_range_path)
    first_col = ranges.columns[0]
    ranges = ranges.rename(columns={first_col: "variable"})
    ranges["variable"] = (
        ranges["variable"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(OUTLIER_ALIASES)
    )
    ranges["upper_bound"] = pd.to_numeric(ranges["upper_bound"], errors="coerce")
    ranges["lower_bound"] = pd.to_numeric(ranges["lower_bound"], errors="coerce")
    ranges = ranges.dropna(subset=["variable", "upper_bound", "lower_bound"])
    ranges = ranges.drop_duplicates(subset=["variable"], keep="last").reset_index(drop=True)
    return ranges


def add_compact_treatment_features(treatment_df: pd.DataFrame) -> pd.DataFrame:
    prepared = treatment_df.copy()
    vasopressor_cols = [
        col
        for col in ["dobutamine", "dopamine", "epinephrine", "norepinephrine"]
        if col in prepared.columns
    ]
    if vasopressor_cols:
        vasopressor_values = prepared[vasopressor_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        prepared["vasopressors"] = (vasopressor_values.max(axis=1) > 0).astype("float32")
    return prepared


def compute_gnri_static(cohort: pd.DataFrame, lab_df: pd.DataFrame) -> pd.Series:
    if "albumin" not in lab_df.columns:
        return pd.Series(np.nan, index=cohort.index, dtype="float32")

    lab_albumin = lab_df[["stay_id", "hr", "albumin"]].copy()
    lab_albumin["stay_id"] = pd.to_numeric(lab_albumin["stay_id"], errors="coerce").astype("Int64")
    lab_albumin["hr"] = pd.to_numeric(lab_albumin["hr"], errors="coerce")
    lab_albumin["albumin"] = pd.to_numeric(lab_albumin["albumin"], errors="coerce")
    lab_albumin = lab_albumin.dropna(subset=["stay_id", "hr", "albumin"])

    first_24h_albumin = (
        lab_albumin[lab_albumin["hr"] <= 24]
        .groupby("stay_id")["albumin"]
        .mean()
    )
    stay_albumin = lab_albumin.groupby("stay_id")["albumin"].mean()
    albumin_by_stay = first_24h_albumin.combine_first(stay_albumin)

    height = pd.to_numeric(cohort.get("height"), errors="coerce")
    weight = pd.to_numeric(cohort.get("weight"), errors="coerce")
    gender = pd.to_numeric(cohort.get("gender"), errors="coerce")
    albumin = cohort["stay_id"].map(albumin_by_stay)

    ideal_weight_male = height - 100 - ((height - 150) / 4.0)
    ideal_weight_female = height - 100 - ((height - 150) / 2.5)
    ideal_weight = np.where(gender == 1, ideal_weight_male, ideal_weight_female)
    ideal_weight = pd.Series(ideal_weight, index=cohort.index, dtype="float64").replace(0, np.nan)

    gnri = (14.89 * albumin) + (41.7 * (weight / ideal_weight))
    return gnri.astype("float32")


def apply_outlier_ranges(
    source_name: str,
    source_df: pd.DataFrame,
    outlier_ranges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if outlier_ranges.empty:
        return source_df, pd.DataFrame(
            columns=[
                "source",
                "variable",
                "actual_column",
                "lower_bound",
                "upper_bound",
                "non_null_before",
                "outlier_count",
                "outlier_ratio",
            ]
        )

    cleaned = source_df.copy()
    column_lookup = {str(col).strip().lower(): col for col in cleaned.columns}
    summary_rows: list[dict[str, object]] = []

    for row in outlier_ranges.itertuples(index=False):
        if row.variable not in column_lookup:
            continue
        actual_col = column_lookup[row.variable]
        values = pd.to_numeric(cleaned[actual_col], errors="coerce")
        outlier_mask = values.notna() & (
            (values < float(row.lower_bound)) | (values > float(row.upper_bound))
        )
        outlier_count = int(outlier_mask.sum())
        non_null_before = int(values.notna().sum())
        if outlier_count > 0:
            cleaned.loc[outlier_mask, actual_col] = np.nan
        summary_rows.append(
            {
                "source": source_name,
                "variable": row.variable,
                "actual_column": actual_col,
                "lower_bound": float(row.lower_bound),
                "upper_bound": float(row.upper_bound),
                "non_null_before": non_null_before,
                "outlier_count": outlier_count,
                "outlier_ratio": outlier_count / non_null_before if non_null_before else 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["outlier_count", "outlier_ratio", "variable"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return cleaned, summary


def make_windows_for_stay(
    stay_id: int,
    max_hour: int,
    length_hours: int,
    step_hours: int,
    min_start_hour: int,
    max_windows_per_stay: int | None,
) -> list[dict[str, int | str]]:
    windows = []
    first_end = min_start_hour + length_hours
    for window_end in range(first_end, max_hour + 1, step_hours):
        window_start = window_end - length_hours
        windows.append(
            {
                "sample_id": f"{stay_id}_{window_start}_{window_end}",
                "stay_id": stay_id,
                "window_start": window_start,
                "window_end": window_end,
            }
        )
        if max_windows_per_stay and len(windows) >= max_windows_per_stay:
            break
    return windows


def build_sample_index(
    cohort: pd.DataFrame,
    dynamic_max_hr: pd.Series,
    length_hours: int,
    step_hours: int,
    min_start_hour: int,
    max_windows_per_stay: int | None,
) -> pd.DataFrame:
    rows = []
    for stay_id in cohort["stay_id"].astype(int):
        if stay_id not in dynamic_max_hr.index:
            continue
        max_hour = int(dynamic_max_hr.loc[stay_id])
        rows.extend(
            make_windows_for_stay(
                stay_id=stay_id,
                max_hour=max_hour,
                length_hours=length_hours,
                step_hours=step_hours,
                min_start_hour=min_start_hour,
                max_windows_per_stay=max_windows_per_stay,
            )
        )
    return pd.DataFrame(rows)


def melt_source_to_triples(
    source_name: str,
    source_df: pd.DataFrame,
    sample_index: pd.DataFrame,
    keep_zero: bool,
) -> pd.DataFrame:
    features = [c for c in SOURCE_FEATURES[source_name] if c in source_df.columns]
    if not features:
        raise ValueError(f"No configured features found for source {source_name}")

    source_df = source_df[["stay_id", "hr", *features]].copy()
    source_df["stay_id"] = source_df["stay_id"].astype(int)
    source_df["hr"] = source_df["hr"].astype(int)

    merged = sample_index.merge(source_df, on="stay_id", how="inner")
    merged = merged[
        (merged["hr"] >= merged["window_start"])
        & (merged["hr"] < merged["window_end"])
    ].copy()
    merged["time"] = merged["hr"] - merged["window_start"]

    id_cols = ["sample_id", "stay_id", "window_start", "window_end", "time"]
    triples = merged.melt(
        id_vars=id_cols,
        value_vars=features,
        var_name="variable",
        value_name="value",
    )
    triples["source"] = source_name
    triples["observed"] = triples["value"].notna().astype("int8")
    triples = triples[triples["observed"] == 1].copy()
    if not keep_zero:
        triples = triples[triples["value"] != 0].copy()
    triples["value"] = triples["value"].astype("float32")
    return triples[
        [
            "sample_id",
            "stay_id",
            "window_start",
            "window_end",
            "time",
            "source",
            "variable",
            "value",
            "observed",
        ]
    ]


def build_labels_and_static(
    cohort: pd.DataFrame,
    sample_index: pd.DataFrame,
    lab_df: pd.DataFrame,
    vital_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = sample_index.merge(
        cohort[["stay_id", *LABEL_COLUMNS]],
        on="stay_id",
        how="left",
    )
    labels["los_in_icu"] = (labels["los_icu_day"] - labels["window_end"] / 24.0).clip(lower=0)
    hospital_los_from_icu_days = (
        labels["los_hospital_day"] - labels["los_icu_admit_day"].fillna(0)
    ).clip(lower=0)
    labels["los_in_hospital"] = (
        hospital_los_from_icu_days - labels["window_end"] / 24.0
    ).clip(lower=0)
    labels["los_in_icu_class"] = (labels["los_in_icu"] >= 2).astype("int8")
    labels["los_in_hospital_class"] = (labels["los_in_hospital"] >= 2).astype("int8")
    labels["death_delta"] = np.where(
        labels["deathtime_icu_hour"].notna(),
        labels["deathtime_icu_hour"] - labels["window_end"],
        np.nan,
    ).astype("float32")

    for horizon in (6, 12, 24, 48):
        labels[f"death_{horizon}"] = (
            (labels["death_hosp"] == 1)
            & labels["deathtime_icu_hour"].notna()
            & (labels["window_end"] + horizon >= labels["deathtime_icu_hour"])
        ).astype("int8")

    for col in ["sofa", *[f"{score}_label" for score in SOFA_SUBSCORES]]:
        labels[col] = 0.0

    vital_cols = ["stay_id", "hr", "sofa", *[c for c in SOFA_SUBSCORES if c in vital_df.columns]]
    vital_for_labels = vital_df[vital_cols].copy()
    vital_for_labels["stay_id"] = vital_for_labels["stay_id"].astype(int)
    vital_for_labels["hr"] = vital_for_labels["hr"].astype(int)
    for col in [c for c in vital_for_labels.columns if c not in {"stay_id", "hr"}]:
        vital_for_labels[col] = pd.to_numeric(vital_for_labels[col], errors="coerce")

    for stay_id, stay_vitals in vital_for_labels.groupby("stay_id", sort=False):
        label_idx = labels.index[labels["stay_id"].astype(int) == int(stay_id)]
        if len(label_idx) == 0:
            continue

        stay_vitals = (
            stay_vitals.sort_values("hr")
            .drop_duplicates(subset=["hr"], keep="last")
            .set_index("hr")
        )
        hrs = stay_vitals.index.to_numpy()
        if len(hrs) == 0:
            continue

        last_sofa = stay_vitals["sofa"].dropna()
        fallback_sofa = float(last_sofa.iloc[-1]) if not last_sofa.empty else 0.0

        for idx in label_idx:
            window_end = int(labels.at[idx, "window_end"])
            sofa_hr = window_end + 25
            if sofa_hr in stay_vitals.index and pd.notna(stay_vitals.at[sofa_hr, "sofa"]):
                labels.at[idx, "sofa"] = float(stay_vitals.at[sofa_hr, "sofa"])
            else:
                labels.at[idx, "sofa"] = fallback_sofa

            future_vitals = stay_vitals[(hrs > window_end) & (hrs <= window_end + 24)]
            for score in SOFA_SUBSCORES:
                label_col = f"{score}_label"
                if score not in stay_vitals.columns:
                    labels.at[idx, label_col] = 0.0
                    continue
                if not future_vitals.empty:
                    max_value = future_vitals[score].max(skipna=True)
                    labels.at[idx, label_col] = float(max_value) if pd.notna(max_value) else 0.0
                elif window_end in stay_vitals.index and pd.notna(stay_vitals.at[window_end, score]):
                    labels.at[idx, label_col] = float(stay_vitals.at[window_end, score])
                else:
                    labels.at[idx, label_col] = 0.0

    cohort = cohort.copy()
    cohort["gnri"] = compute_gnri_static(cohort, lab_df)
    static_cols = [c for c in STATIC_FEATURES if c in cohort.columns]
    static = sample_index[["sample_id", "stay_id"]].merge(
        cohort[["stay_id", *static_cols]], on="stay_id", how="left"
    )
    for col in static_cols:
        static[col] = static[col].fillna(static[col].median())

    return labels, static


def write_vocab(triples: pd.DataFrame, output_dir: Path) -> None:
    vocab = (
        triples[["source", "variable"]]
        .drop_duplicates()
        .sort_values(["source", "variable"])
        .reset_index(drop=True)
    )
    vocab.insert(0, "variable_id", np.arange(len(vocab), dtype=np.int32))
    vocab.to_csv(output_dir / "variable_vocab.csv", index=False)


def build_processed_dataset(
    mimiciv_dir: Path,
    output_dir: Path,
    outlier_range_path: Path | None = None,
    min_age: int = 65,
    min_los_icu_day: float = 1,
    max_los_icu_day: float = 30,
    anchor_year_groups: list[str] | None = None,
    length_hours: int = 24,
    step_hours: int = 4,
    min_start_hour: int = 1,
    max_windows_per_stay: int | None = 8,
    max_patients: int | None = None,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_elder_sepsis_cohort(
        mimiciv_dir=mimiciv_dir,
        min_age=min_age,
        min_los_icu_day=min_los_icu_day,
        max_los_icu_day=max_los_icu_day,
        anchor_year_groups=anchor_year_groups,
        max_patients=max_patients,
    )
    cohort.to_csv(output_dir / "cohort_elder_sepsis.csv", index=False)

    source_frames = {
        "lab": pd.read_csv(mimiciv_dir / "lab_test.csv"),
        "vital": pd.read_csv(mimiciv_dir / "vital_uo_sofa.csv"),
        "treatment": add_compact_treatment_features(pd.read_csv(mimiciv_dir / "treatment.csv")),
    }
    outlier_ranges = load_outlier_ranges(outlier_range_path)
    stay_ids = set(cohort["stay_id"].astype(int))
    outlier_summaries = []
    for source, df in source_frames.items():
        filtered = df[df["stay_id"].isin(stay_ids)].copy()
        cleaned, summary = apply_outlier_ranges(source, filtered, outlier_ranges)
        source_frames[source] = cleaned
        if not summary.empty:
            outlier_summaries.append(summary)

    if outlier_summaries:
        outlier_summary = pd.concat(outlier_summaries, ignore_index=True)
        outlier_summary.to_csv(output_dir / "outlier_cleanup_summary.csv", index=False)
    else:
        outlier_summary = pd.DataFrame()

    dynamic_max_hr = pd.concat(
        [df.groupby("stay_id")["hr"].max() for df in source_frames.values()],
        axis=1,
    ).max(axis=1)
    sample_index = build_sample_index(
        cohort=cohort,
        dynamic_max_hr=dynamic_max_hr,
        length_hours=length_hours,
        step_hours=step_hours,
        min_start_hour=min_start_hour,
        max_windows_per_stay=max_windows_per_stay,
    )
    sample_index.to_csv(output_dir / "sample_index.csv", index=False)

    all_triples = []
    keep_zero_by_source = {"lab": False, "vital": True, "treatment": True}
    for source, df in source_frames.items():
        triples = melt_source_to_triples(
            source_name=source,
            source_df=df,
            sample_index=sample_index,
            keep_zero=keep_zero_by_source[source],
        )
        triples.to_parquet(output_dir / f"{source}_triples.parquet", index=False)
        all_triples.append(triples)

    triples_all = pd.concat(all_triples, ignore_index=True)
    triples_all.to_parquet(output_dir / "all_triples.parquet", index=False)
    write_vocab(triples_all, output_dir)

    labels, static = build_labels_and_static(
        cohort=cohort,
        lab_df=source_frames["lab"],
        sample_index=sample_index,
        vital_df=source_frames["vital"],
    )
    labels.to_csv(output_dir / "labels.csv", index=False)
    static.to_csv(output_dir / "static.csv", index=False)

    return {
        "cohort_rows": len(cohort),
        "samples": len(sample_index),
        "triples": len(triples_all),
        "variables": triples_all["variable"].nunique(),
        "outliers_masked": int(outlier_summary["outlier_count"].sum()) if not outlier_summary.empty else 0,
    }
