from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.reasoning import build_reasoner
from scripts.framework.reasoning.sample_context import ReasoningContextBuilder
from scripts.framework.reasoning.output_formatting import write_multiagent_readable_output
from scripts.framework.reasoning.retriever import TfidfKnowledgeBase
from scripts.framework.utils.config import (
    load_config,
    resolve_experience_store_config,
    resolve_path,
    resolve_reasoning_llm_config,
)


@dataclass(frozen=True)
class CompareSpec:
    backend: str
    provider: str
    model_id: str
    api_env: str | None
    base_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_compact_los1_7_year1119.yaml")
    parser.add_argument("--samples-jsonl", default="reasoning/sampled_20_cases.jsonl")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--request-note-limit", type=int, default=2400)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--memory-update-policy", default=None)
    parser.add_argument("--max-consult-rounds", type=int, default=None)
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--include-deepseek", action="store_true")
    parser.add_argument("--include-qwen", action="store_true")
    parser.add_argument("--skip-glm", action="store_true")
    return parser.parse_args()


def utc_now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def seconds_between(start_iso: str, end_iso: str) -> float:
    start_dt = datetime.fromisoformat(start_iso)
    end_dt = datetime.fromisoformat(end_iso)
    return round((end_dt - start_dt).total_seconds(), 3)


def load_cases(jsonl_path: Path, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def refresh_case_contexts(cases: list[dict[str, Any]], context_builder: ReasoningContextBuilder) -> list[dict[str, Any]]:
    refreshed_cases: list[dict[str, Any]] = []
    for case in cases:
        sample_id = str(case.get("sample_id", ""))
        updated = dict(case)
        if sample_id in context_builder.sample_index.index:
            context = context_builder.build(sample_id)
            updated["summary_fields"] = context["summary"]
            updated["summary"] = context["summary"]
            updated["query"] = context["query"]
            updated["patient_summary_text"] = context["patient_summary_text"]
        refreshed_cases.append(updated)
    return refreshed_cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_multiagent_stage_artifacts(
    *,
    log_dir: Path,
    sample_id: str,
    record: dict[str, Any],
    agent_context: dict[str, Any],
) -> None:
    doctor_inputs_dir = log_dir / "doctor_inputs" / sample_id
    doctor_outputs_dir = log_dir / "doctor_outputs" / sample_id
    meta_inputs_dir = log_dir / "meta_agent_inputs"
    meta_outputs_dir = log_dir / "meta_agent_outputs"
    traces_dir = log_dir / "sample_traces"
    for folder in [doctor_inputs_dir, doctor_outputs_dir, meta_inputs_dir, meta_outputs_dir, traces_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    doctor_steps: list[dict[str, Any]] = []
    doctor_reviews = record.get("doctor_reviews", [])
    if isinstance(doctor_reviews, list):
        for review in doctor_reviews:
            doctor_id = str(review.get("doctor_id", "unknown_doctor"))
            doctor_input = {
                "sample_id": sample_id,
                "doctor_id": doctor_id,
                "title": review.get("title", ""),
                "focus": review.get("focus", ""),
                "system_prompt": review.get("system_prompt", ""),
                "user_prompt": review.get("user_prompt", ""),
            }
            doctor_output = {
                "sample_id": sample_id,
                "doctor_id": doctor_id,
                "title": review.get("title", ""),
                "review": review.get("review", ""),
            }
            (doctor_inputs_dir / f"{doctor_id}.json").write_text(
                json.dumps(doctor_input, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (doctor_outputs_dir / f"{doctor_id}.json").write_text(
                json.dumps(doctor_output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            doctor_steps.append({"input": doctor_input, "output": doctor_output})

    meta_input = {
        "sample_id": sample_id,
        "system_prompt": record.get("meta_system_prompt", ""),
        "user_prompt": record.get("meta_user_prompt", ""),
        "doctor_outputs_used": [
            {
                "doctor_id": item.get("doctor_id", ""),
                "title": item.get("title", ""),
                "review": item.get("review", ""),
            }
            for item in doctor_reviews
            if isinstance(item, dict)
        ],
        "agent_context": agent_context,
    }
    meta_output = {
        "sample_id": sample_id,
        "meta_report": record.get("meta_report", ""),
        "agent_final_output": record.get("agent_final_output", ""),
        "consultation_feedback": record.get("consultation_feedback", []),
    }
    (meta_inputs_dir / f"{sample_id}.json").write_text(
        json.dumps(meta_input, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (meta_outputs_dir / f"{sample_id}.json").write_text(
        json.dumps(meta_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trace = {
        "sample_id": sample_id,
        "pipeline": [
            {
                "stage": "doctor_parallel_initial_review",
                "description": "Three doctors receive different scoped inputs and produce three independent reviews.",
                "items": doctor_steps,
            },
            {
                "stage": "meta_agent_synthesis",
                "description": "Meta agent receives all doctor outputs and produces the integrated report.",
                "input": meta_input,
                "output": meta_output,
            },
        ],
    }
    (traces_dir / f"{sample_id}.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def slugify(provider: str, model_id: str) -> str:
    return f"{provider.lower()}__{model_id.lower().replace('/', '_').replace(':', '_')}"


def clone_experience_store_paths(
    source_paths: dict[str, Path] | None,
    target_root: Path,
) -> dict[str, Path] | None:
    if not source_paths:
        return None
    target_root.mkdir(parents=True, exist_ok=True)
    cloned: dict[str, Path] = {}
    for key, src in source_paths.items():
        dst = target_root / f"{key}.json"
        if src.exists():
            shutil.copy2(src, dst)
        else:
            dst.write_text(json.dumps({"principles": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        cloned[key] = dst
    return cloned


def render_knowledge(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Knowledge {idx}] {result.get('title', '')}",
                    f"Source: {result.get('source', '')}",
                    str(result.get("text", "")),
                ]
            )
        )
    return "\n\n".join(blocks).strip()


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_DIR / args.config)
    reasoning_cfg = cfg.get("reasoning", {})
    memory_update_policy = str(args.memory_update_policy or reasoning_cfg.get("memory_update_policy", "all"))
    rag_enabled = bool(cfg.get("rag", {}).get("enabled", False))
    global_docs_dir = resolve_path(PROJECT_DIR, cfg.get("rag", {}).get("global_docs_dir", "knowledge_base/global_docs"))
    local_docs_dir = resolve_path(PROJECT_DIR, "knowledge_base/docs")
    samples_jsonl = resolve_path(PROJECT_DIR, args.samples_jsonl)
    processed_dir = resolve_path(PROJECT_DIR, cfg["data"]["processed_dir"])
    masked_mimic_dir = resolve_path(PROJECT_DIR, cfg["data"]["masked_mimic_dir"])
    llm_dynamic_csv = resolve_path(PROJECT_DIR, cfg["data"]["llm_dynamic_csv"]) if cfg["data"].get("llm_dynamic_csv") else None
    llm_static_csv = resolve_path(PROJECT_DIR, cfg["data"]["llm_static_csv"]) if cfg["data"].get("llm_static_csv") else None
    base_experience_store_path, base_experience_store_paths = resolve_experience_store_config(PROJECT_DIR, reasoning_cfg)

    glm_cfg = resolve_reasoning_llm_config(cfg, "glm")
    compare_specs = []
    if args.include_deepseek:
        deepseek_cfg = resolve_reasoning_llm_config(cfg, "deepseek")
        compare_specs.append(
            CompareSpec(
                backend="deepseek",
                provider=deepseek_cfg["provider"],
                model_id=deepseek_cfg["model_name"],
                api_env=deepseek_cfg["api_env"],
                base_url=deepseek_cfg["base_url"],
            )
        )
    if args.include_qwen:
        qwen_cfg = resolve_reasoning_llm_config(cfg, "qwen")
        compare_specs.append(
            CompareSpec(
                backend="qwen",
                provider=qwen_cfg["provider"],
                model_id=qwen_cfg["model_name"],
                api_env=qwen_cfg["api_env"],
                base_url=qwen_cfg["base_url"],
            )
        )
    if not args.skip_glm:
        compare_specs.append(
            CompareSpec(
                backend="glm",
                provider=glm_cfg["provider"],
                model_id=glm_cfg["model_name"],
                api_env=glm_cfg["api_env"],
                base_url=glm_cfg["base_url"],
            )
        )

    if args.log_dir:
        log_dir = resolve_path(PROJECT_DIR, args.log_dir)
    else:
        log_dir = resolve_path(PROJECT_DIR, f"log/multiagent_flash_compare_{utc_now_compact()}")
    pre_rag_dir = log_dir / ("pre_rag_inputs" if rag_enabled else "structured_patient_inputs")
    post_rag_dir = log_dir / ("post_rag_inputs" if rag_enabled else "agent_context_inputs")
    output_dir = log_dir / "outputs"
    readable_dir = log_dir / "readable_outputs"
    memory_dir = log_dir / "memory"
    doctor_inputs_root = log_dir / "doctor_inputs"
    doctor_outputs_root = log_dir / "doctor_outputs"
    meta_inputs_dir = log_dir / "meta_agent_inputs"
    meta_outputs_dir = log_dir / "meta_agent_outputs"
    traces_dir = log_dir / "sample_traces"
    for folder in [
        log_dir,
        pre_rag_dir,
        post_rag_dir,
        output_dir,
        readable_dir,
        memory_dir,
        doctor_inputs_root,
        doctor_outputs_root,
        meta_inputs_dir,
        meta_outputs_dir,
        traces_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    context_builder = ReasoningContextBuilder(
        processed_dir=processed_dir,
        masked_mimic_dir=masked_mimic_dir,
        llm_dynamic_csv=llm_dynamic_csv,
        llm_static_csv=llm_static_csv,
    )
    cases = refresh_case_contexts(load_cases(samples_jsonl, args.max_samples), context_builder)
    retriever = TfidfKnowledgeBase([global_docs_dir, local_docs_dir])

    pre_rag_rows: list[dict[str, Any]] = []
    post_rag_rows: list[dict[str, Any]] = []
    for case in cases:
        sample_id = str(case["sample_id"])
        admission_record_text = str(case.get("admission_record_text") or case.get("note_excerpt") or "")
        retrieved_docs = retriever.search(str(case.get("query", "")), top_k=args.top_k) if rag_enabled else []
        pre_row = {
            "sample_id": sample_id,
            "stay_id": case.get("stay_id"),
            "window_start": case.get("window_start"),
            "window_end": case.get("window_end"),
            "rag_enabled": rag_enabled,
            "query": case.get("query"),
            "patient_summary_text": case.get("patient_summary_text"),
            "summary_fields": case.get("summary_fields", {}),
            "admission_record_text": admission_record_text,
            "note_excerpt": admission_record_text,
        }
        post_row = {
            "sample_id": sample_id,
            "stay_id": case.get("stay_id"),
            "window_start": case.get("window_start"),
            "window_end": case.get("window_end"),
            "workflow": reasoning_cfg.get("workflow", "colacare_evolver"),
            "memory_update_policy": memory_update_policy,
            "rag_enabled": rag_enabled,
            "retrieved_docs": retrieved_docs,
            "patient_summary_text": case.get("patient_summary_text"),
            "admission_record_text": admission_record_text,
            "note_excerpt": admission_record_text,
            "knowledge_text": render_knowledge(retrieved_docs) if rag_enabled else "RAG disabled for this experiment.",
        }
        pre_rag_rows.append(pre_row)
        post_rag_rows.append(post_row)
        (pre_rag_dir / f"{sample_id}.json").write_text(json.dumps(pre_row, ensure_ascii=False, indent=2), encoding="utf-8")
        (post_rag_dir / f"{sample_id}.json").write_text(json.dumps(post_row, ensure_ascii=False, indent=2), encoding="utf-8")

    write_jsonl(log_dir / ("pre_rag_inputs.jsonl" if rag_enabled else "structured_patient_inputs.jsonl"), pre_rag_rows)
    write_jsonl(log_dir / ("post_rag_inputs.jsonl" if rag_enabled else "agent_context_inputs.jsonl"), post_rag_rows)

    registry_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in compare_specs:
        llm_cfg = resolve_reasoning_llm_config(cfg, spec.backend)
        model_slug = slugify(spec.provider, spec.model_id)
        output_path = output_dir / f"{model_slug}.jsonl"
        model_memory_root = memory_dir / model_slug
        if args.disable_memory:
            model_memory_root.mkdir(parents=True, exist_ok=True)
            model_experience_store_paths = None
            model_experience_store_path = model_memory_root / "empty_experience_no_memory.json"
            model_experience_store_path.write_text(
                json.dumps({"principles": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            model_experience_store_paths = clone_experience_store_paths(base_experience_store_paths, model_memory_root)
            model_experience_store_path = (
                model_memory_root / "shared_experience.json"
                if model_experience_store_paths is None
                else model_memory_root / "meta_agent.json"
            )
            if model_experience_store_paths is None:
                if base_experience_store_path.exists():
                    shutil.copy2(base_experience_store_path, model_experience_store_path)
                else:
                    model_experience_store_path.write_text(
                        json.dumps({"principles": []}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

        model_rows: list[dict[str, Any]] = []
        registry_rows.append(
            {
                "provider": spec.provider,
                "backend": spec.backend,
                "model_id": spec.model_id,
                "api_env": spec.api_env,
                "base_url": spec.base_url,
                "has_api_key": bool(llm_cfg["api_key"]),
                "output_file": str(output_path),
                "memory_root": str(model_memory_root),
                "memory_disabled": bool(args.disable_memory),
            }
        )

        if llm_cfg["api_key"]:
            reasoner = build_reasoner(
                workflow=reasoning_cfg.get("workflow", "colacare_evolver"),
                docs_dir=[global_docs_dir, local_docs_dir],
                api_key=llm_cfg["api_key"],
                model_name=llm_cfg["model_name"],
                base_url=llm_cfg["base_url"],
                backend=spec.backend,
                enable_rag=rag_enabled,
                experience_store_path=model_experience_store_path,
                experience_store_paths=model_experience_store_paths,
                experience_top_k=0 if args.disable_memory else int(reasoning_cfg.get("experience_top_k", 3)),
                max_consult_rounds=int(args.max_consult_rounds or reasoning_cfg.get("max_consult_rounds", 2)),
                enable_self_evolution=False if args.disable_memory else bool(reasoning_cfg.get("enable_self_evolution", True)),
                self_evolution_mode=str(reasoning_cfg.get("self_evolution_mode", "post_prediction")),
                memory_update_policy=memory_update_policy,
            )
        else:
            reasoner = None

        for idx, case in enumerate(cases, start=1):
            sample_id = str(case["sample_id"])
            admission_record_text = str(case.get("admission_record_text") or case.get("note_excerpt") or "")
            record: dict[str, Any] = {
                "sample_id": sample_id,
                "stay_id": case.get("stay_id"),
                "provider": spec.provider,
                "backend": spec.backend,
                "model_id": spec.model_id,
                "status": "pending",
                "memory_update_policy": memory_update_policy,
                "response_text": "",
                "agent_final_output": "",
                "agent_final_reasoning_output": "",
                "meta_system_prompt": "",
                "meta_user_prompt": "",
                "meta_report": "",
                "doctor_reviews": [],
                "consultation_feedback": [],
                "evolution_result": {},
                "rag_enabled": rag_enabled,
                "retrieved_docs": retriever.search(str(case.get("query", "")), top_k=args.top_k) if rag_enabled else [],
                "error": "",
                "started_at": datetime.now().isoformat(),
            }
            if reasoner is None:
                record["status"] = "skipped_missing_api_key"
                record["error"] = f"{llm_cfg['api_env']} is not set."
            else:
                try:
                    result = reasoner.invoke(
                        {
                            "summary": dict(case.get("summary_fields", {})),
                            "query": str(case.get("query", "")),
                        },
                        top_k=args.top_k,
                        note_text=admission_record_text[: args.request_note_limit],
                        image_data_url="",
                        image_path="",
                    )
                    record["status"] = "ok"
                    record["response_text"] = str(result.get("reasoning_text", "")).strip()
                    record["agent_final_output"] = str(result.get("final_report", result.get("reasoning_text", ""))).strip()
                    record["agent_final_reasoning_output"] = str(result.get("reasoning_text", "")).strip()
                    record["meta_system_prompt"] = str(result.get("meta_system_prompt", "")).strip()
                    record["meta_user_prompt"] = str(result.get("meta_user_prompt", "")).strip()
                    record["meta_report"] = str(result.get("meta_report", "")).strip()
                    record["doctor_reviews"] = result.get("doctor_reviews", [])
                    record["consultation_feedback"] = result.get("consultation_feedback", [])
                    record["evolution_result"] = result.get("evolution_result", {})
                except Exception as exc:
                    record["status"] = "error"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                time.sleep(args.sleep_seconds)

            record["finished_at"] = datetime.now().isoformat()
            record["duration_seconds"] = seconds_between(record["started_at"], record["finished_at"])
            model_rows.append(record)
            write_jsonl(output_path, model_rows)
            agent_context = next((row for row in post_rag_rows if str(row.get("sample_id")) == sample_id), {})
            write_multiagent_stage_artifacts(
                log_dir=log_dir,
                sample_id=sample_id,
                record=record,
                agent_context=agent_context,
            )
            write_multiagent_readable_output(
                readable_dir / f"{sample_id}__{model_slug}.md",
                record=record,
                agent_context=agent_context,
            )
            print(
                f"[{spec.provider}/{spec.model_id}] {idx}/{len(cases)} {sample_id} -> {record['status']}",
                flush=True,
            )

        ok_count = sum(1 for row in model_rows if row["status"] == "ok")
        skipped_count = sum(1 for row in model_rows if row["status"] == "skipped_missing_api_key")
        error_count = sum(1 for row in model_rows if row["status"] == "error")
        avg_duration = round(
            sum(float(row.get("duration_seconds", 0.0)) for row in model_rows) / max(len(model_rows), 1),
            3,
        )
        summary_rows.append(
            {
                "provider": spec.provider,
                "backend": spec.backend,
                "model_id": spec.model_id,
                "ok_count": ok_count,
                "skipped_count": skipped_count,
                "error_count": error_count,
                "memory_disabled": bool(args.disable_memory),
                "total_duration_seconds": round(
                    sum(float(row.get("duration_seconds", 0.0)) for row in model_rows),
                    3,
                ),
                "avg_duration_seconds": avg_duration,
                "output_file": str(output_path),
                "memory_root": str(model_memory_root),
            }
        )

    (log_dir / "model_registry.json").write_text(
        json.dumps(registry_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (log_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(),
                "log_dir": str(log_dir),
                "sample_count": len(cases),
                "memory_update_policy": memory_update_policy,
                "memory_disabled": bool(args.disable_memory),
                "total_model_duration_seconds": round(
                    sum(float(item.get("total_duration_seconds", 0.0)) for item in summary_rows),
                    3,
                ),
                "models": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "log_dir": str(log_dir),
                "sample_count": len(cases),
                "memory_update_policy": memory_update_policy,
                "memory_disabled": bool(args.disable_memory),
                "models": summary_rows,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
