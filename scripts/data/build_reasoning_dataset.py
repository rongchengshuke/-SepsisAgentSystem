from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.reasoning import build_reasoner
from scripts.framework.reasoning.retriever import TfidfKnowledgeBase
from scripts.framework.reasoning.sample_context import ReasoningContextBuilder
from scripts.framework.utils.config import (
    load_config,
    resolve_experience_store_config,
    resolve_path,
    resolve_reasoning_llm_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend", choices=["deepseek", "glm", "qwen", "mock"], default="glm")
    parser.add_argument("--workflow", default=None)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sample-ids-csv", default=None)
    parser.add_argument("--split-membership-csv", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--disable-self-evolution", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    return parser.parse_args()


def render_knowledge(results: list[dict[str, str | float]]) -> str:
    blocks = []
    for idx, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Knowledge {idx}] {result['title']}",
                    f"Source: {result['source']}",
                    str(result["text"]),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_mock_reasoning(summary: dict[str, object], retrieved: list[dict[str, str | float]]) -> str:
    evidence = ", ".join(str(item["title"]) for item in retrieved[:2]) if retrieved else "retrieved sepsis knowledge"
    return (
        f"This elderly sepsis window shows current illness severity through SOFA {summary['sofa']}, "
        f"hemodynamics with MBP {summary['mbp']} and vasopressor exposure {summary['vasopressors']}, "
        f"and respiratory burden with SpO2 {summary['spo2']} and ventilation status {summary['ventilation']}. "
        f"Laboratory signals including lactate {summary['lactate']}, creatinine {summary['creatinine']}, bilirubin {summary['bilirubin']}, "
        f"platelet {summary['platelet']}, and INR {summary['inr']} suggest the current degree of organ dysfunction should be monitored closely. "
        f"Baseline reserve is limited by age {summary['age']}, comorbidity burden {summary['cci_score']}, and GNRI {summary['gnri']}. "
        f"Using {evidence}, the overall trajectory appears to depend on whether perfusion, oxygenation, and multi-organ dysfunction stabilize over the next 24 to 48 hours."
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_DIR / args.config)
    processed_dir = resolve_path(PROJECT_DIR, cfg["data"]["processed_dir"])
    masked_dir = resolve_path(PROJECT_DIR, cfg["data"]["masked_mimic_dir"])
    llm_dynamic_csv = resolve_path(PROJECT_DIR, cfg["data"]["llm_dynamic_csv"]) if cfg["data"].get("llm_dynamic_csv") else None
    llm_static_csv = resolve_path(PROJECT_DIR, cfg["data"]["llm_static_csv"]) if cfg["data"].get("llm_static_csv") else None
    output_csv = resolve_path(PROJECT_DIR, args.output_csv or cfg["data"]["reasoning_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reasoning_cfg = cfg.get("reasoning", {})
    rag_enabled = bool(cfg.get("rag", {}).get("enabled", False))
    experience_store_path, experience_store_paths = resolve_experience_store_config(PROJECT_DIR, reasoning_cfg)
    llm_cfg = resolve_reasoning_llm_config(cfg, args.backend)

    retriever = TfidfKnowledgeBase(PROJECT_DIR / "knowledge_base" / "docs")
    context_builder = ReasoningContextBuilder(
        processed_dir=processed_dir,
        masked_mimic_dir=masked_dir,
        llm_dynamic_csv=llm_dynamic_csv,
        llm_static_csv=llm_static_csv,
    )
    api_key = llm_cfg["api_key"]
    if args.backend != "mock" and not api_key:
        raise SystemExit(f"{llm_cfg['api_env']} is not set.")
    reasoner = build_reasoner(
        workflow=args.workflow or reasoning_cfg.get("workflow", "single_agent"),
        docs_dir=[PROJECT_DIR / "knowledge_base" / "docs"],
        api_key=api_key,
        model_name=llm_cfg["model_name"],
        base_url=llm_cfg["base_url"],
        backend=args.backend,
        enable_rag=rag_enabled,
        experience_store_path=experience_store_path,
        experience_store_paths=experience_store_paths,
        experience_top_k=int(reasoning_cfg.get("experience_top_k", 3)),
        max_consult_rounds=int(reasoning_cfg.get("max_consult_rounds", 2)),
        enable_self_evolution=(
            False if args.disable_self_evolution else bool(reasoning_cfg.get("enable_self_evolution", True))
        ),
        self_evolution_mode=str(reasoning_cfg.get("self_evolution_mode", "inline")),
        memory_update_policy=str(reasoning_cfg.get("memory_update_policy", "all")),
    )

    if args.sample_ids_csv:
        sample_df = pd.read_csv(args.sample_ids_csv)
    elif args.split_membership_csv:
        sample_df = pd.read_csv(args.split_membership_csv)
        if args.split:
            split_values = {value.strip() for value in str(args.split).split(",") if value.strip()}
            sample_df = sample_df[sample_df["split"].isin(split_values)].copy()
    else:
        sample_df = pd.DataFrame({"sample_id": context_builder.sample_index.index.tolist()})
    if "sample_id" not in sample_df.columns:
        raise SystemExit("sample id source must contain a sample_id column.")
    sample_ids = sample_df["sample_id"].astype(str).tolist()
    if args.max_samples is not None and args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]

    rows: list[dict[str, str]] = []
    completed_ids: set[str] = set()
    if args.resume and output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        if "sample_id" in existing_df.columns:
            completed_ids = set(existing_df["sample_id"].astype(str).tolist())
            rows = existing_df.to_dict("records")

    def flush_rows() -> None:
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "reasoning_text", "retrieved_doc_ids", "retrieved_titles", "backend"],
            )
            writer.writeheader()
            writer.writerows(rows)

    for idx, sample_id in enumerate(sample_ids, start=1):
        if sample_id in completed_ids:
            continue
        context = context_builder.build(sample_id)
        retrieved = retriever.search(context["query"], top_k=args.top_k) if rag_enabled else []
        result = reasoner.invoke(context, top_k=args.top_k)
        reasoning_text = str(result.get("reasoning_text", "")).strip()
        rows.append(
            {
                "sample_id": sample_id,
                "reasoning_text": reasoning_text,
                "retrieved_doc_ids": " | ".join(str(item["doc_id"]) for item in retrieved),
                "retrieved_titles": " | ".join(str(item["title"]) for item in retrieved),
                "backend": args.backend,
            }
        )
        print(f"[{idx}/{len(sample_ids)}] generated reasoning for {sample_id}", flush=True)
        if args.save_every > 0 and (len(rows) % args.save_every == 0):
            flush_rows()
        time.sleep(args.sleep_seconds)

    flush_rows()
    print(f"reasoning_csv: {output_csv}", flush=True)


if __name__ == "__main__":
    main()
