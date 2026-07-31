from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.utils.config import load_config, resolve_path


SECTION_ALIASES = {
    "allergies": "Allergies",
    "chief complaint": "Chief Complaint",
    "history of present illness": "History of Present Illness",
    "past medical history": "Past Medical History",
    "pmh": "Past Medical History",
    "social history": "Social History",
    "family history": "Family History",
}
ALL_HEADINGS = [
    "Name",
    "Admission Date",
    "Date of Birth",
    "Sex",
    "Service",
    "Allergies",
    "Attending",
    "Chief Complaint",
    "Major Surgical or Invasive Procedure",
    "History of Present Illness",
    "Past Medical History",
    "PMH",
    "Social History",
    "Family History",
    "Physical Exam",
    "Pertinent Results",
    "Brief Hospital Course",
    "Medications on Admission",
    "Discharge Medications",
    "Discharge Disposition",
    "Facility",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Followup Instructions",
]
HEADING_PATTERN = re.compile(
    r"(?P<head>" + "|".join(re.escape(item) for item in ALL_HEADINGS) + r")\s*:",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build history-only clinical note table aligned to sample_id")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-csv", default="data/multimodal/history_only_notes.csv")
    parser.add_argument("--missing-text", default="\u65e0")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_excluded_sections(text: str) -> str:
    text = re.split(r"Physical Exam\s*\(On the floor\)\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"Physical Exam\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"Pertinent Results\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"Brief Hospital Course\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"Discharge Diagnosis\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_history_only(raw_text: str, missing_text: str) -> str:
    text = normalize_text(raw_text)
    if not text:
        return missing_text

    sections: dict[str, str] = {}
    matches = list(HEADING_PATTERN.finditer(text))
    for idx, match in enumerate(matches):
        heading_raw = match.group("head")
        canonical = SECTION_ALIASES.get(heading_raw.lower())
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = re.sub(r"\s+", " ", text[start:end]).strip()
        if canonical and content:
            previous = sections.get(canonical, "")
            if len(content) > len(previous):
                sections[canonical] = content

    ordered_keys = [
        "Allergies",
        "Chief Complaint",
        "History of Present Illness",
        "Past Medical History",
        "Social History",
        "Family History",
    ]
    blocks: list[str] = []
    for key in ordered_keys:
        content = sections.get(key, "")
        content = re.sub(r"\s+", " ", content).strip()
        if content:
            blocks.append(f"{key}: {content}")

    if blocks:
        joined = "\n\n".join(blocks)
        cleaned = strip_excluded_sections(joined)
        return cleaned if cleaned else missing_text

    # Fallback: keep only the pre-hospital-course prefix if structured section
    # headings were not parsed cleanly.
    prefix = strip_excluded_sections(text)
    return prefix[:2000] if prefix else missing_text


def load_note_lookup(notes_csv: Path, missing_text: str) -> dict[int, str]:
    lookup: dict[int, str] = {}
    with notes_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                stay_id = int(row.get("stay_id", ""))
            except ValueError:
                continue
            history_only = extract_history_only(str(row.get("text", "")), missing_text)
            current = lookup.get(stay_id, "")
            if len(history_only) > len(current):
                lookup[stay_id] = history_only
    return lookup


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_DIR / args.config)
    processed_dir = resolve_path(PROJECT_DIR, cfg["data"]["processed_dir"])
    mimiciv_dir = resolve_path(PROJECT_DIR, cfg["data"]["mimiciv_dir"])
    output_csv = resolve_path(PROJECT_DIR, args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sample_index = pd.read_csv(processed_dir / "sample_index.csv")
    note_lookup = load_note_lookup(mimiciv_dir / "discharge_note.csv", args.missing_text)

    rows: list[dict[str, str | int]] = []
    matched = 0
    missing = 0
    for row in sample_index.itertuples(index=False):
        stay_id = int(row.stay_id)
        note_text = note_lookup.get(stay_id, args.missing_text)
        if note_text == args.missing_text:
            missing += 1
        else:
            matched += 1
        rows.append(
            {
                "sample_id": row.sample_id,
                "stay_id": stay_id,
                "note_text": note_text,
            }
        )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "stay_id", "note_text"])
        writer.writeheader()
        writer.writerows(rows)

    matched_stays = sample_index["stay_id"].isin(note_lookup.keys()).sum()
    print(f"output_csv: {output_csv}", flush=True)
    print(f"sample_rows: {len(rows)}", flush=True)
    print(f"sample_rows_with_note: {matched}", flush=True)
    print(f"sample_rows_missing_note: {missing}", flush=True)
    print(f"matched_stay_rows: {matched_stays}", flush=True)
    print(f"unique_note_stays: {len(note_lookup)}", flush=True)


if __name__ == "__main__":
    main()
