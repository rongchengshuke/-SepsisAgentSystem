FULL_LAB_FEATURES = [
    "albumin",
    "aniongap",
    "bicarbonate",
    "bun",
    "calcium",
    "chloride",
    "creatinine",
    "glucose",
    "sodium",
    "potassium",
    "fibrinogen",
    "inr",
    "pt",
    "ptt",
    "hematocrit",
    "hemoglobin",
    "platelet",
    "wbc",
    "alt",
    "ast",
    "bilirubin",
    "pao2",
    "paco2",
    "fio2",
    "pao2fio2ratio",
    "ph",
    "baseexcess",
    "lactate",
    "eGFR",
    "troponin",
    "magnesium",
    "bnp",
    "lymphocytes",
    "neutrophils",
    "alkaline_phosphatase",
]

FULL_VITAL_FEATURES = [
    "heart_rate",
    "sbp",
    "mbp",
    "resp_rate",
    "temperature",
    "spo2",
    "weight",
    "urineoutput",
    "gcs",
    "respiration",
    "coagulation",
    "liver",
    "cardiovascular",
    "cns",
    "renal",
    "sofa",
]

FULL_TREATMENT_FEATURES = [
    "dobutamine",
    "dopamine",
    "epinephrine",
    "norepinephrine",
    "ventilation",
]

FULL_STATIC_FEATURES = [
    "age",
    "gender",
    "bmi",
    "cci_score",
    "myocardial_infarct",
    "congestive_heart_failure",
    "peripheral_vascular_disease",
    "cerebrovascular_disease",
    "dementia",
    "chronic_pulmonary_disease",
    "renal_disease",
    "malignant_cancer",
]

# Active compact schema requested for the current stage. The full definitions above
# are kept so we can switch back later without reconstructing the lists by hand.
LAB_FEATURES = [
    "lactate",
    "inr",
    "pt",
    "platelet",
    "albumin",
    "creatinine",
    "bun",
    "pao2fio2ratio",
    "bilirubin",
    "bnp",
    "wbc",
    "neutrophils",
    "lymphocytes",
]

VITAL_FEATURES = [
    "heart_rate",
    "sbp",
    "mbp",
    "resp_rate",
    "temperature",
    "spo2",
    "weight",
    "urineoutput",
    "gcs",
    "sofa",
]

TREATMENT_FEATURES = [
    "vasopressors",
    "ventilation",
]

STATIC_FEATURES = [
    "age",
    "gender",
    "bmi",
    "cci_score",
    "gnri",
]

SOURCE_FEATURES = {
    "lab": LAB_FEATURES,
    "vital": VITAL_FEATURES,
    "treatment": TREATMENT_FEATURES,
}
