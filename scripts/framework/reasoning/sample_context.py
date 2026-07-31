from __future__ import annotations

from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "stay_id",
    "hr",
    "ventilation",
    "vasopressors",
    "heart_rate",
    "sbp",
    "mbp",
    "resp_rate",
    "temperature",
    "spo2",
    "urineoutput",
    "gcs",
    "respiration",
    "coagulation",
    "liver",
    "cardiovascular",
    "cns",
    "renal",
    "sofa",
    "albumin",
    "bicarbonate",
    "bun",
    "calcium",
    "chloride",
    "creatinine",
    "glucose",
    "sodium",
    "potassium",
    "inr",
    "pt",
    "ptt",
    "hemoglobin",
    "platelet",
    "wbc",
    "alt",
    "ast",
    "bilirubin",
    "pao2",
    "fio2",
    "ph",
    "baseexcess",
    "lactate",
    "magnesium",
    "lymphocytes",
    "neutrophils",
    "pao2fio2",
    "bnp",
    "NLR",
    "SII",
    "PLR",
]

STATIC_COLUMNS = ["stay_id", "gnri", "age", "gender", "bmi", "cci_score"]

PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_LLM_IMPUTED_DYNAMIC_CSV = PROJECT_DIR / "data" / "reasoning" / "mimic_dynamic_llm_imputed.csv"
DEFAULT_LLM_PREPROCESSED_STATIC_CSV = PROJECT_DIR / "data" / "reasoning" / "mimic_static_llm_preprocessed.csv"

# Keep dense bedside signals on the last row of the window, but pull sparse
# lab values from the last non-null observation within the window.
SPARSE_WINDOW_FIELDS = [
    "albumin",
    "bun",
    "creatinine",
    "bilirubin",
    "platelet",
    "wbc",
    "ph",
    "pao2fio2",
    "inr",
    "pt",
    "lactate",
    "lymphocytes",
    "neutrophils",
]

DYNAMIC_SERIES_FIELDS = [
    {
        "key": "heart_rate",
        "label": "Heart rate",
        "unit": "beats/min",
        "reference_range": "60-100 beats/min",
    },
    {
        "key": "sbp",
        "label": "Systolic blood pressure",
        "unit": "mmHg",
        "reference_range": "90-140 mmHg",
    },
    {
        "key": "mbp",
        "label": "Mean blood pressure",
        "unit": "mmHg",
        "reference_range": ">=65 mmHg in septic shock resuscitation",
    },
    {
        "key": "resp_rate",
        "label": "Respiratory rate",
        "unit": "breaths/min",
        "reference_range": "12-20 breaths/min",
    },
    {
        "key": "temperature",
        "label": "Temperature",
        "unit": "C",
        "reference_range": "36.0-37.5 C",
    },
    {
        "key": "spo2",
        "label": "SpO2",
        "unit": "%",
        "reference_range": "95-100%",
    },
    {
        "key": "urineoutput",
        "label": "Urine output",
        "unit": "mL/hour",
        "reference_range": ">=0.5 mL/kg/hour is often used as an adequate perfusion target",
    },
    {
        "key": "gcs",
        "label": "GCS",
        "unit": "score",
        "reference_range": "15 normal",
    },
    {
        "key": "sofa",
        "label": "SOFA score",
        "unit": "score",
        "reference_range": "higher score indicates worse organ dysfunction",
    },
    {
        "key": "lactate",
        "label": "Lactate",
        "unit": "mmol/L",
        "reference_range": "<2 mmol/L",
    },
    {
        "key": "inr",
        "label": "INR",
        "unit": "ratio",
        "reference_range": "0.8-1.2",
    },
    {
        "key": "pt",
        "label": "Prothrombin time",
        "unit": "s",
        "reference_range": "11-13.5 s",
    },
    {
        "key": "platelet",
        "label": "Platelets",
        "unit": "10^9/L",
        "reference_range": "150-400 x10^9/L",
    },
    {
        "key": "albumin",
        "label": "Albumin",
        "unit": "g/dL",
        "reference_range": "3.5-5.0 g/dL",
    },
    {
        "key": "creatinine",
        "label": "Creatinine",
        "unit": "mg/dL",
        "reference_range": "0.7-1.3 mg/dL",
    },
    {
        "key": "glucose",
        "label": "Blood glucose",
        "unit": "mg/dL",
        "reference_range": "70-180 mg/dL is a common ICU target range",
    },
    {
        "key": "bun",
        "label": "BUN",
        "unit": "mg/dL",
        "reference_range": "7-20 mg/dL",
    },
    {
        "key": "pao2fio2",
        "label": "PaO2/FiO2 ratio",
        "unit": "ratio",
        "reference_range": ">3.0 if stored as PaO2/FiO2 divided by 100; <=3.0 suggests impaired oxygenation",
    },
    {
        "key": "bilirubin",
        "label": "Bilirubin",
        "unit": "mg/dL",
        "reference_range": "0.1-1.2 mg/dL",
    },
    {
        "key": "bnp",
        "label": "BNP",
        "unit": "pg/mL",
        "reference_range": "<100 pg/mL",
    },
    {
        "key": "wbc",
        "label": "WBC",
        "unit": "10^9/L",
        "reference_range": "4-11 x10^9/L",
    },
    {
        "key": "neutrophils",
        "label": "Neutrophils",
        "unit": "10^9/L",
        "reference_range": "1.5-8.0 x10^9/L",
    },
    {
        "key": "lymphocytes",
        "label": "Lymphocytes",
        "unit": "10^9/L",
        "reference_range": "1.0-4.0 x10^9/L",
    },
    {
        "key": "vasopressors",
        "label": "Vasopressor use",
        "unit": "binary/status",
        "reference_range": "0 means not used; 1 means used",
    },
    {
        "key": "ventilation",
        "label": "Mechanical ventilation",
        "unit": "binary/status",
        "reference_range": "0 means not used; 1 means used",
    },
]

QUERY_SERIES_FIELDS = [
    "heart_rate",
    "sbp",
    "mbp",
    "spo2",
    "resp_rate",
    "sofa",
    "lactate",
    "creatinine",
    "platelet",
    "wbc",
    "neutrophils",
    "lymphocytes",
    "vasopressors",
    "ventilation",
]


class ReasoningContextBuilder:
    def __init__(
        self,
        processed_dir: str | Path,
        masked_mimic_dir: str | Path,
        llm_dynamic_csv: str | Path | None = None,
        llm_static_csv: str | Path | None = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.masked_mimic_dir = Path(masked_mimic_dir)
        self.sample_index = pd.read_csv(self.processed_dir / "sample_index.csv").set_index("sample_id")
        dynamic_csv = Path(llm_dynamic_csv) if llm_dynamic_csv else DEFAULT_LLM_IMPUTED_DYNAMIC_CSV
        if not dynamic_csv.exists():
            dynamic_csv = self.masked_mimic_dir / "mimic_dynamic.csv"
        self.dynamic_source_path = dynamic_csv
        self.dynamic = pd.read_csv(
            dynamic_csv,
            usecols=lambda c: c in SUMMARY_COLUMNS,
        )
        self.static = pd.read_csv(
            self._resolve_static_csv(llm_static_csv),
            usecols=lambda c: c in STATIC_COLUMNS,
        ).set_index("stay_id")
        self.dynamic_groups = {
            int(stay_id): frame.sort_values("hr").reset_index(drop=True)
            for stay_id, frame in self.dynamic.groupby("stay_id", sort=False)
        }

    def _resolve_static_csv(self, llm_static_csv: str | Path | None) -> Path:
        static_csv = Path(llm_static_csv) if llm_static_csv else DEFAULT_LLM_PREPROCESSED_STATIC_CSV
        if not static_csv.exists():
            static_csv = self.masked_mimic_dir / "mimic_static.csv"
        self.static_source_path = static_csv
        return static_csv

    def build(self, sample_id: str) -> dict[str, object]:
        row = self.sample_index.loc[sample_id]
        stay_id = int(row["stay_id"])
        start = int(row["window_start"])
        end = int(row["window_end"])
        dynamic = self.dynamic_groups[stay_id]
        window = dynamic[(dynamic["hr"] >= start) & (dynamic["hr"] < end)].copy()
        if window.empty:
            raise ValueError(f"No window data for sample {sample_id}")
        latest = window.sort_values("hr").iloc[-1]
        static = self.static.loc[stay_id] if stay_id in self.static.index else pd.Series(dtype="object")
        sparse_values = self._extract_sparse_window_values(window, SPARSE_WINDOW_FIELDS)
        hourly_window = self._hourly_window(window, start, end)
        summary = {
            "sample_id": sample_id,
            "window_start": start,
            "window_end": end,
            "time_points": [f"hour {hour}" for hour in range(start, end)],
            "age": self._fmt(static.get("age")),
            "gender": "male" if static.get("gender") == 1 else "female",
            "bmi": self._fmt(static.get("bmi")),
            "cci_score": self._fmt(static.get("cci_score")),
            "gnri": self._fmt(static.get("gnri")),
            "hr": self._fmt(latest.get("heart_rate")),
            "sbp": self._fmt(latest.get("sbp")),
            "mbp": self._fmt(latest.get("mbp")),
            "resp_rate": self._fmt(latest.get("resp_rate")),
            "temperature": self._fmt(latest.get("temperature")),
            "spo2": self._fmt(latest.get("spo2")),
            "urineoutput": self._fmt(latest.get("urineoutput")),
            "gcs": self._fmt(latest.get("gcs")),
            "sofa": self._fmt(latest.get("sofa")),
            # Old behavior directly used latest.get(...) for labs. We keep that
            # bedside-vitals rule above, but switch sparse labs to last non-null.
            "lactate": self._fmt(sparse_values.get("lactate")),
            "creatinine": self._fmt(sparse_values.get("creatinine")),
            "bilirubin": self._fmt(sparse_values.get("bilirubin")),
            "platelet": self._fmt(sparse_values.get("platelet")),
            "wbc": self._fmt(sparse_values.get("wbc")),
            "ph": self._fmt(sparse_values.get("ph")),
            "pao2fio2": self._fmt(sparse_values.get("pao2fio2")),
            "inr": self._fmt(sparse_values.get("inr")),
            "vasopressors": self._fmt(latest.get("vasopressors")),
            "ventilation": self._fmt(latest.get("ventilation")),
        }
        summary["dynamic_series"] = self._build_dynamic_series(hourly_window)
        query = self._build_query(summary)
        return {
            "summary": summary,
            "query": query,
            "patient_summary_text": format_patient_summary(summary),
            "compact_patient_summary_text": format_compact_patient_summary(summary),
        }

    @staticmethod
    def _fmt(value: object) -> str:
        if pd.isna(value):
            return "missing"
        if isinstance(value, (float, int)):
            return f"{float(value):.2f}"
        return str(value)

    @classmethod
    def _fmt_series(cls, value: object) -> str:
        if pd.isna(value):
            return "missing"
        if isinstance(value, (float, int)):
            return f"{float(value):.2f}"
        return str(value)

    @staticmethod
    def _hourly_window(window: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
        hourly = window.sort_values("hr").groupby("hr", as_index=True).last()
        return hourly.reindex(range(start, end))

    @classmethod
    def _series_values(cls, hourly_window: pd.DataFrame, column: str) -> list[str]:
        if column not in hourly_window.columns:
            return ["missing"] * len(hourly_window)
        return [cls._fmt_series(value) for value in hourly_window[column].tolist()]

    @classmethod
    def _build_dynamic_series(cls, hourly_window: pd.DataFrame) -> dict[str, dict[str, object]]:
        series: dict[str, dict[str, object]] = {}
        for spec in DYNAMIC_SERIES_FIELDS:
            key = str(spec["key"])
            series[key] = {
                "label": spec["label"],
                "unit": spec["unit"],
                "reference_range": spec["reference_range"],
                "values": cls._series_values(hourly_window, key),
            }
        return series

    @staticmethod
    def _numeric_observations(values: list[str]) -> list[float]:
        numbers: list[float] = []
        for value in values:
            if value == "missing":
                continue
            try:
                numbers.append(float(value))
            except ValueError:
                continue
        return numbers

    @classmethod
    def _trend_label(cls, values: list[str]) -> str:
        numbers = cls._numeric_observations(values)
        if len(numbers) < 2:
            return "insufficient observations"
        delta = numbers[-1] - numbers[0]
        spread = max(numbers) - min(numbers)
        scale = max(abs(numbers[0]), 1.0)
        if abs(delta) / scale >= 0.2:
            return "rising" if delta > 0 else "declining"
        if spread / max(abs(sum(numbers) / len(numbers)), 1.0) >= 0.2:
            return "fluctuating"
        return "stable"

    @classmethod
    def _series_query_fragment(cls, key: str, payload: dict[str, object]) -> str:
        values = list(payload.get("values", []))
        numbers = cls._numeric_observations(values)
        if not numbers:
            return f"{key} trend insufficient observations"
        return (
            f"{key} trend {cls._trend_label(values)} "
            f"last {numbers[-1]:.2f} min {min(numbers):.2f} max {max(numbers):.2f}"
        )

    @classmethod
    def _build_query(cls, summary: dict[str, object]) -> str:
        series = summary.get("dynamic_series", {})
        fragments = [
            "elderly sepsis prognosis 24h multivariate time-series",
            f"age {summary['age']}",
            f"gender {summary['gender']}",
            f"cci {summary['cci_score']}",
            f"gnri {summary['gnri']}",
        ]
        if isinstance(series, dict):
            for key in QUERY_SERIES_FIELDS:
                payload = series.get(key)
                if isinstance(payload, dict):
                    fragments.append(cls._series_query_fragment(key, payload))
        return " | ".join(fragments)

    @staticmethod
    def _last_non_null(window: pd.DataFrame, column: str) -> object:
        if column not in window.columns:
            return pd.NA
        non_null = window.loc[window[column].notna(), column]
        if non_null.empty:
            return pd.NA
        return non_null.iloc[-1]

    @classmethod
    def _extract_sparse_window_values(
        cls,
        window: pd.DataFrame,
        columns: list[str],
    ) -> dict[str, object]:
        return {column: cls._last_non_null(window, column) for column in columns}


def format_patient_summary(summary: dict[str, object]) -> str:
    series = summary.get("dynamic_series")
    if isinstance(series, dict) and series:
        lines = [
            "Patient Record",
            "Static structured data:",
            f"- Sample ID: {summary['sample_id']}",
            f"- Window: hour {summary['window_start']} to {summary['window_end']}",
            f"- Age: {summary['age']} years",
            f"- Gender: {summary['gender']}",
            f"- BMI: {summary['bmi']} kg/m^2",
            f"- CCI score: {summary['cci_score']} (>=5 indicates heavy comorbidity burden)",
            f"- GNRI score: {summary['gnri']} (<=92 indicates significant malnutrition risk)",
            "",
            (
                "Dynamic time-series data: past 24 hours when the window length is 24, "
                "ordered by the hourly time points below. Each dynamic variable is represented "
                "as a list; missing means no observed value at that hour."
            ),
            f"- Time points: {', '.join(str(item) for item in summary.get('time_points', []))}",
            "",
            "Reference ranges for interpretation:",
        ]
        for key in [str(spec["key"]) for spec in DYNAMIC_SERIES_FIELDS]:
            payload = series.get(key)
            if not isinstance(payload, dict):
                continue
            lines.append(f"- {payload['label']}: {payload['reference_range']} ({payload['unit']})")
        lines.append("")
        lines.append("Dynamic variable values:")
        for key in [str(spec["key"]) for spec in DYNAMIC_SERIES_FIELDS]:
            payload = series.get(key)
            if not isinstance(payload, dict):
                continue
            values = ", ".join(str(value) for value in payload.get("values", []))
            lines.append(
                f"- {payload['label']}: \"{values}\". "
                f"Unit: {payload['unit']}. Reference range: {payload['reference_range']}."
            )
        return "\n".join(lines)
    return format_compact_patient_summary(summary)


def format_compact_patient_summary(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            f"Sample ID: {summary['sample_id']}",
            f"Window: hour {summary['window_start']} to {summary['window_end']}",
            f"Demographics: age {summary['age']}, gender {summary['gender']}, BMI {summary['bmi']}, CCI {summary['cci_score']}, GNRI {summary['gnri']}",
            f"Hemodynamics: heart_rate {summary['hr']}, SBP {summary['sbp']}, MBP {summary['mbp']}, vasopressors {summary['vasopressors']}",
            f"Respiratory: resp_rate {summary['resp_rate']}, temperature {summary['temperature']}, SpO2 {summary['spo2']}, ventilation {summary['ventilation']}, PaO2/FiO2 {summary['pao2fio2']}",
            f"Organ dysfunction: GCS {summary['gcs']}, SOFA {summary['sofa']}, lactate {summary['lactate']}, creatinine {summary['creatinine']}, bilirubin {summary['bilirubin']}, platelet {summary['platelet']}, WBC {summary['wbc']}, INR {summary['inr']}, pH {summary['ph']}, urineoutput {summary['urineoutput']}",
        ]
    )
