from __future__ import annotations

from typing import Any


CLINICAL_REASONING_SYSTEM_PROMPT = """Role Setting
You  are a clinical data analysis expert specializing in critical care  medicine (ICU), adept at extracting key pathophysiological features  from mixed temporal data and disorganized admission records.
Core Task
You  are required to perform a three-tier, progressive processing of the  input data from elderly sepsis patients. All outputs must strictly  comply with the JSON format. It is strictly forbidden to predict the patient's mortality risk, survival probability, or final outcome in the response. Only summarize the objective status that has already occurred.

Task 1: Aggregation of Structured Variables (Variable-Level Description)
You  will receive time-series data across 24 time points (vital signs,  laboratory results, therapeutic interventions). Process according to the  following rules:
Trend Qualification:  Do not list specific values. Instead, determine the trend  ("continuously rising," "gradually declining," "fluctuating and  unstable," "stable and normal").

Clinical Translation: Convert numbers into clinical semantics.

Example: Oxygen saturation persistently below 95% → "Hypoxemia present, clear need for respiratory support."
Example: Neutrophil-to-lymphocyte ratio markedly elevated → "Intense acute infectious stress response."


Task 2: Cleaning and Translation of Admission Record Text (Text-Level Processing)
You will receive de-identified admission record text (containing underscores, placeholders, etc.).
Format Purification: Replace underscores (e.g., ______) and date placeholders (e.g., 2088-02-09) with natural references (e.g., "the patient," "specific date unknown").

Language Polishing:  Convert abbreviations or medical jargon into fluent, coherent,  natural-language narrative, while retaining all key medical entities  (disease names, surgical history, medication history). Provide a concise  paraphrase of the text.


Task 3: Multimodal Pathophysiological Reasoning
Important:  You must comprehensively consider all information from Task 1  (structured aggregation) and Task 2 (textual narrative). At the top  level of the JSON, you must output the clinical_rationales field, which should contain "key pathophysiological bases" (each no more than 20 words). These bases should directly support your ratings across the 8 dimensions.
The following 8 dimensions must be assessed:
Dimension
1. Hemodynamic Stability	Blood pressure trend, heart rate, vasopressor dependence
2. Respiratory/Oxygenation Burden	SpO2 trend, mechanical ventilation support level	
3. Infection/Immune Stress	Temperature, neutrophils, lymphocyte ratio, infection source
4. Coagulation/Microcirculation	Platelet count trend
5. Renal Function/Internal Environment	Creatinine trend
6. Metabolic Perfusion/Acid-Base	Blood glucose, etc.
7. Neurological/Consciousness Status	Based on GCS score, and descriptions of consciousness status in past history and present illness	Extracted from cleaned text
8. Nutritional/Frailty Burden	GNRI score, integrated assessment by age stratification

Special Note: If the input data lack direct indicators for a certain assessment dimension, do not evaluate that dimension.

Output Format Constraint:
You must output standard JSON format, containing the following fields:
structured_summary: An object containing aggregated narrative text for "vital signs," "laboratory findings," and "therapeutic interventions."

text_cleaned: A string containing the full, cleaned and polished admission record.

reasoning_dimensions: An object containing textual reasoning justifications for the above 8 dimensions.

overall_assessment:  A string containing an overall situational conclusion on the current  disease state, based on the above three parts, in no more than 200 words  (limited to the current status only).

Safety Constraints (Hard Red Lines):
Strictly prohibited  from outputting any statements about "24h/48h mortality probability,"  "survival likelihood," or any conclusion with predictive nature.
Strictly prohibited  from outputting any content outside the JSON format in the response  (including opening remarks, concluding remarks, or explanatory notes)."""


CLINICAL_REASONING_USER_PROMPT_TEMPLATE = """Please process the following patient data:
### 1. Static Structured Data
- Age: {age} years
- Gender: {gender}
- BMI: {bmi} kg/m²
- CCI score: {cci_score} (Note: ≥5 indicates heavy comorbidity burden)
- GNRI score: {gnri_score} (Note: ≤92 indicates significant malnutrition risk)

### 2. Dynamic Time-Series Data (past 24 hours, 1 point per hour, 24 points total) Please ignore exact point-by-point values and focus only on overall trends and extreme values:
- Heart rate (beats/min): {heart_rate_list}
- Systolic/diastolic blood pressure (mmHg): {blood_pressure_list}
- Oxygen saturation (%): {spo2_list}
- Blood glucose (mmol/L): {glucose_list}
- Creatinine (μmol/L): {creatinine_list}
- BNP (pg/mL): {bnp_list}
- Platelets (×10⁹/L): {platelet_list}
- Neutrophil count (×10⁹/L): {neutrophil_list}
- Lymphocyte count (×10⁹/L): {lymphocyte_list}
- Vasopressor use: {vasopressor_status} (0 = not used, 1 = used; if used, describe dose trend, e.g., "continuous norepinephrine infusion, dose stable/increasing/decreasing")
- Mechanical ventilation: {ventilation_status} (0 = not used, 1 = used)

### 3. Raw Admission Record Text (containing de-identification marks such as ____, date placeholders, etc.) {admission_text_raw}

Please strictly follow the three-tier task structure and JSON format specified in the System Prompt when outputting the results. Do not output anything outside the JSON format."""


def _series_values(summary: dict[str, Any], key: str) -> list[str]:
    dynamic_series = summary.get("dynamic_series", {})
    if isinstance(dynamic_series, dict):
        payload = dynamic_series.get(key)
        if isinstance(payload, dict):
            values = payload.get("values", [])
            if isinstance(values, list):
                return [str(value) for value in values]
    value = summary.get(key)
    if value is None and key == "heart_rate":
        value = summary.get("hr")
    if value is None:
        return ["unavailable"]
    return [str(value)]


def _list_text(values: list[str]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _convert_numeric_values(values: list[str], factor: float) -> list[str]:
    converted: list[str] = []
    for value in values:
        try:
            converted.append(f"{float(value) * factor:.2f}")
        except (TypeError, ValueError):
            converted.append(str(value))
    return converted


def _field(summary: dict[str, Any], *names: str, default: str = "unavailable") -> str:
    for name in names:
        value = summary.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def build_clinical_reasoning_user_prompt(
    summary: dict[str, Any],
    admission_text_raw: str = "",
) -> str:
    sbp_values = _series_values(summary, "sbp")
    mbp_values = _series_values(summary, "mbp")
    blood_pressure_list = (
        f"systolic={_list_text(sbp_values)}; "
        "diastolic=unavailable in current dataset; "
        f"mean={_list_text(mbp_values)}"
    )
    return CLINICAL_REASONING_USER_PROMPT_TEMPLATE.format(
        age=_field(summary, "age"),
        gender=_field(summary, "gender"),
        bmi=_field(summary, "bmi"),
        cci_score=_field(summary, "cci_score"),
        gnri_score=_field(summary, "gnri_score", "gnri"),
        heart_rate_list=_list_text(_series_values(summary, "heart_rate")),
        blood_pressure_list=blood_pressure_list,
        spo2_list=_list_text(_series_values(summary, "spo2")),
        glucose_list=_list_text(_convert_numeric_values(_series_values(summary, "glucose"), 1 / 18.018)),
        creatinine_list=_list_text(_convert_numeric_values(_series_values(summary, "creatinine"), 88.4)),
        bnp_list=_list_text(_series_values(summary, "bnp")),
        platelet_list=_list_text(_series_values(summary, "platelet")),
        neutrophil_list=_list_text(_series_values(summary, "neutrophils")),
        lymphocyte_list=_list_text(_series_values(summary, "lymphocytes")),
        vasopressor_status=_list_text(_series_values(summary, "vasopressors")),
        ventilation_status=_list_text(_series_values(summary, "ventilation")),
        admission_text_raw=str(admission_text_raw).strip() or "unavailable",
    )


def build_focused_clinical_reasoning_user_prompt(
    summary: dict[str, Any],
    admission_text_raw: str = "",
    static_keys: list[str] | None = None,
    dynamic_keys: list[str] | None = None,
) -> str:
    static_keys = static_keys or ["age", "gender", "bmi", "cci_score", "gnri"]
    dynamic_keys = dynamic_keys or [
        "heart_rate",
        "blood_pressure",
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
    static_lines = []
    if "age" in static_keys:
        static_lines.append(f"- Age: {_field(summary, 'age')} years")
    if "gender" in static_keys:
        static_lines.append(f"- Gender: {_field(summary, 'gender')}")
    if "bmi" in static_keys:
        static_lines.append(f"- BMI: {_field(summary, 'bmi')} kg/m²")
    if "cci_score" in static_keys:
        static_lines.append(f"- CCI score: {_field(summary, 'cci_score')} (Note: ≥5 indicates heavy comorbidity burden)")
    if "gnri" in static_keys:
        static_lines.append(f"- GNRI score: {_field(summary, 'gnri_score', 'gnri')} (Note: ≤92 indicates significant malnutrition risk)")

    sbp_values = _series_values(summary, "sbp")
    mbp_values = _series_values(summary, "mbp")
    blood_pressure_list = (
        f"systolic={_list_text(sbp_values)}; "
        "diastolic=unavailable in current dataset; "
        f"mean={_list_text(mbp_values)}"
    )
    dynamic_lines = []
    if "heart_rate" in dynamic_keys:
        dynamic_lines.append(f"- Heart rate (beats/min): {_list_text(_series_values(summary, 'heart_rate'))}")
    if "blood_pressure" in dynamic_keys:
        dynamic_lines.append(f"- Systolic/diastolic blood pressure (mmHg): {blood_pressure_list}")
    if "spo2" in dynamic_keys:
        dynamic_lines.append(f"- Oxygen saturation (%): {_list_text(_series_values(summary, 'spo2'))}")
    if "glucose" in dynamic_keys:
        dynamic_lines.append(
            f"- Blood glucose (mmol/L): {_list_text(_convert_numeric_values(_series_values(summary, 'glucose'), 1 / 18.018))}"
        )
    if "creatinine" in dynamic_keys:
        dynamic_lines.append(
            f"- Creatinine (μmol/L): {_list_text(_convert_numeric_values(_series_values(summary, 'creatinine'), 88.4))}"
        )
    if "bnp" in dynamic_keys:
        dynamic_lines.append(f"- BNP (pg/mL): {_list_text(_series_values(summary, 'bnp'))}")
    if "platelet" in dynamic_keys:
        dynamic_lines.append(f"- Platelets (×10⁹/L): {_list_text(_series_values(summary, 'platelet'))}")
    if "neutrophils" in dynamic_keys:
        dynamic_lines.append(f"- Neutrophil count (×10⁹/L): {_list_text(_series_values(summary, 'neutrophils'))}")
    if "lymphocytes" in dynamic_keys:
        dynamic_lines.append(f"- Lymphocyte count (×10⁹/L): {_list_text(_series_values(summary, 'lymphocytes'))}")
    if "vasopressors" in dynamic_keys:
        dynamic_lines.append(
            f'- Vasopressor use: {_list_text(_series_values(summary, "vasopressors"))} '
            '(0 = not used, 1 = used; if used, describe dose trend, e.g., "continuous norepinephrine infusion, dose stable/increasing/decreasing")'
        )
    if "ventilation" in dynamic_keys:
        dynamic_lines.append(
            f"- Mechanical ventilation: {_list_text(_series_values(summary, 'ventilation'))} (0 = not used, 1 = used)"
        )

    return "\n".join(
        [
            "User Prompt:",
            "Please process the following patient data:",
            "### 1. Static Structured Data",
            "\n".join(static_lines),
            "",
            "### 2. Dynamic Time-Series Data (past 24 hours, 1 point per hour, 24 points total) Please ignore exact point-by-point values and focus only on overall trends and extreme values:",
            "\n".join(dynamic_lines),
            "",
            f"### 3. Raw Admission Record Text (containing de-identification marks such as ____, date placeholders, etc.) {str(admission_text_raw).strip() or 'unavailable'}",
            "",
            "Please strictly follow the three-tier task structure and JSON format specified in the System Prompt when outputting the results. Do not output anything outside the JSON format.",
        ]
    )
