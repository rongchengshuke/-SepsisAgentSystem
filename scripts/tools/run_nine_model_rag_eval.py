from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.reasoning.retriever import TfidfKnowledgeBase
from scripts.framework.reasoning.sample_context import ReasoningContextBuilder
from scripts.framework.reasoning.prompts import (
    CLINICAL_REASONING_SYSTEM_PROMPT,
    build_clinical_reasoning_user_prompt,
)
from scripts.framework.reasoning.output_formatting import write_single_llm_readable_output
from scripts.framework.utils.config import load_config, resolve_path


SYSTEM_PROMPT = CLINICAL_REASONING_SYSTEM_PROMPT


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_id: str
    api_env: str
    base_url_env: str | None
    default_base_url: str


NINE_MODEL_SPECS = [
    ModelSpec("Zhipu", "glm-4.7-flash", "ZHIPU_API_KEY", None, "https://open.bigmodel.cn/api/paas/v4/"),
    ModelSpec("DeepSeek", "deepseek-v4-pro", "DEEPSEEK_API_KEY", None, "https://api.deepseek.com"),
    ModelSpec("DeepSeek", "deepseek-v4-flash", "DEEPSEEK_API_KEY", None, "https://api.deepseek.com"),
    ModelSpec("Zhipu", "glm-5.2", "ZHIPU_API_KEY", None, "https://open.bigmodel.cn/api/paas/v4/"),
    ModelSpec("Baidu", "ERNIE-5.1", "BAIDU_QIANFAN_API_KEY", "BAIDU_QIANFAN_BASE_URL", "https://qianfan.baidubce.com/v2"),
    ModelSpec("Qwen", "qwen-flash", "QWEN_API_KEY", "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ModelSpec("Qwen", "qwen3.7-max", "DASHSCOPE_API_KEY", "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ModelSpec("Qwen", "qwen3.7-plus", "DASHSCOPE_API_KEY", "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ModelSpec("Qwen", "qwen3.6-plus", "DASHSCOPE_API_KEY", "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ModelSpec("Qwen", "qwen3.6-flash", "DASHSCOPE_API_KEY", "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ModelSpec("Qwen", "qwen3-max", "DASHSCOPE_API_KEY", "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--samples-jsonl", default="reasoning/sampled_20_cases.jsonl")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--providers", default="Zhipu", help="Comma-separated providers to run, e.g. Qwen,Zhipu")
    parser.add_argument("--model-ids", default="glm-4.7-flash", help="Comma-separated model ids to run, e.g. qwen-flash,qwen3.7-plus")
    parser.add_argument("--drop-providers", default=None, help="Comma-separated providers to remove from existing logs")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-wait-seconds", type=float, default=30.0)
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


def build_post_rag_prompt(case: dict[str, Any], retrieved_docs: list[dict[str, Any]]) -> str:
    return build_clinical_reasoning_user_prompt(
        summary=dict(case.get("summary_fields") or case.get("summary") or {}),
        admission_text_raw=str(case.get("admission_record_text") or case.get("note_excerpt") or "").strip(),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def slugify_model(provider: str, model_id: str) -> str:
    return f"{provider.lower()}__{model_id.lower().replace('/', '_').replace(':', '_')}"


def resolve_api_key(spec: ModelSpec) -> str | None:
    if spec.provider == "Zhipu":
        return os.environ.get(spec.api_env) or os.environ.get("ZAI_API_KEY")
    if spec.provider == "Qwen":
        return os.environ.get(spec.api_env) or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIBABA_API_KEY")
    return os.environ.get(spec.api_env)


def resolve_base_url(spec: ModelSpec) -> str:
    if spec.base_url_env:
        return os.environ.get(spec.base_url_env, spec.default_base_url)
    return spec.default_base_url


def parse_name_set(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip().lower() for item in str(raw).split(",") if item.strip()}


def filter_specs(specs: list[ModelSpec], providers: set[str]) -> list[ModelSpec]:
    if not providers:
        return list(specs)
    return [spec for spec in specs if spec.provider.lower() in providers]


def filter_specs_by_model_ids(specs: list[ModelSpec], model_ids: set[str]) -> list[ModelSpec]:
    if not model_ids:
        return list(specs)
    return [spec for spec in specs if spec.model_id.lower() in model_ids]


def load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def invoke_model(
    spec: ModelSpec,
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    request_timeout: float,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout, max_retries=1)
    response = client.chat.completions.create(
        model=spec.model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if str(part).strip()).strip()
    return str(content or "").strip()


def invoke_model_with_retries(
    spec: ModelSpec,
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    request_timeout: float,
    max_retries: int,
    retry_wait_seconds: float,
) -> tuple[str, list[str]]:
    errors: list[str] = []
    attempts = max(1, max_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return invoke_model(
                spec=spec,
                api_key=api_key,
                base_url=base_url,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                request_timeout=request_timeout,
            ), errors
        except Exception as exc:
            error_text = f"attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}"
            errors.append(error_text)
            if attempt >= attempts:
                raise RuntimeError(" | ".join(errors)) from exc
            time.sleep(retry_wait_seconds)


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_DIR / args.config)
    rag_enabled = bool(cfg.get("rag", {}).get("enabled", False))
    samples_jsonl = resolve_path(PROJECT_DIR, args.samples_jsonl)
    processed_dir = resolve_path(PROJECT_DIR, cfg["data"]["processed_dir"])
    masked_mimic_dir = resolve_path(PROJECT_DIR, cfg["data"]["masked_mimic_dir"])
    llm_dynamic_csv = resolve_path(PROJECT_DIR, cfg["data"]["llm_dynamic_csv"]) if cfg["data"].get("llm_dynamic_csv") else None
    llm_static_csv = resolve_path(PROJECT_DIR, cfg["data"]["llm_static_csv"]) if cfg["data"].get("llm_static_csv") else None
    global_docs_dir = resolve_path(PROJECT_DIR, cfg.get("rag", {}).get("global_docs_dir", "knowledge_base/global_docs"))
    local_docs_dir = resolve_path(PROJECT_DIR, "knowledge_base/docs")
    providers_filter = parse_name_set(args.providers)
    model_ids_filter = parse_name_set(args.model_ids)
    dropped_providers = parse_name_set(args.drop_providers)
    selected_specs = filter_specs(NINE_MODEL_SPECS, providers_filter)
    selected_specs = filter_specs_by_model_ids(selected_specs, model_ids_filter)
    replacing_providers = {spec.provider.lower() for spec in selected_specs}

    if args.log_dir:
        log_dir = resolve_path(PROJECT_DIR, args.log_dir)
    else:
        log_dir = resolve_path(PROJECT_DIR, f"log/nine_model_rag_eval_{utc_now_compact()}")
    pre_rag_dir = log_dir / ("pre_rag_inputs" if rag_enabled else "structured_patient_inputs")
    post_rag_dir = log_dir / ("post_rag_inputs" if rag_enabled else "llm_prompt_inputs")
    raw_cases_dir = log_dir / "raw_cases"
    output_dir = log_dir / "outputs"
    readable_dir = log_dir / "readable_outputs"
    for folder in [log_dir, raw_cases_dir, pre_rag_dir, post_rag_dir, output_dir, readable_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    context_builder = ReasoningContextBuilder(
        processed_dir=processed_dir,
        masked_mimic_dir=masked_mimic_dir,
        llm_dynamic_csv=llm_dynamic_csv,
        llm_static_csv=llm_static_csv,
    )
    cases = refresh_case_contexts(load_cases(samples_jsonl, args.max_samples), context_builder)
    retriever = TfidfKnowledgeBase([global_docs_dir, local_docs_dir])
    registry_path = log_dir / "model_registry.json"
    summary_path = log_dir / "run_summary.json"

    pre_rag_rows: list[dict[str, Any]] = []
    post_rag_rows: list[dict[str, Any]] = []
    for case in cases:
        sample_id = str(case["sample_id"])
        (raw_cases_dir / f"{sample_id}.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
        admission_record_text = str(case.get("admission_record_text") or case.get("note_excerpt") or "")
        retrieved_docs = retriever.search(str(case["query"]), top_k=args.top_k) if rag_enabled else []
        pre_row = {
            "sample_id": sample_id,
            "stay_id": case.get("stay_id"),
            "window_start": case.get("window_start"),
            "window_end": case.get("window_end"),
            "rag_enabled": rag_enabled,
            "query": case.get("query"),
            "patient_summary_text": case.get("patient_summary_text"),
            "admission_record_text": admission_record_text,
            "note_excerpt": admission_record_text,
        }
        post_row = {
            "sample_id": sample_id,
            "stay_id": case.get("stay_id"),
            "window_start": case.get("window_start"),
            "window_end": case.get("window_end"),
            "rag_enabled": rag_enabled,
            "retrieved_docs": retrieved_docs,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": build_post_rag_prompt(case, retrieved_docs),
        }
        pre_rag_rows.append(pre_row)
        post_rag_rows.append(post_row)
        (pre_rag_dir / f"{sample_id}.json").write_text(json.dumps(pre_row, ensure_ascii=False, indent=2), encoding="utf-8")
        (post_rag_dir / f"{sample_id}.json").write_text(json.dumps(post_row, ensure_ascii=False, indent=2), encoding="utf-8")

    write_jsonl(log_dir / ("pre_rag_inputs.jsonl" if rag_enabled else "structured_patient_inputs.jsonl"), pre_rag_rows)
    write_jsonl(log_dir / ("post_rag_inputs.jsonl" if rag_enabled else "llm_prompt_inputs.jsonl"), post_rag_rows)
    write_jsonl(log_dir / "raw_cases.jsonl", cases)

    model_registry = []
    run_summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "log_dir": str(log_dir),
        "sample_count": len(cases),
        "models": [],
    }

    for provider_name in dropped_providers:
        pattern = f"{provider_name.lower()}__*.jsonl"
        for stale in output_dir.glob(pattern):
            stale.unlink()

    for spec in selected_specs:
        api_key = resolve_api_key(spec)
        base_url = resolve_base_url(spec)
        model_slug = slugify_model(spec.provider, spec.model_id)
        model_output_path = output_dir / f"{model_slug}.jsonl"
        model_rows: list[dict[str, Any]] = []
        model_meta = {
            "provider": spec.provider,
            "model_id": spec.model_id,
            "api_env": spec.api_env,
            "base_url": base_url,
            "has_api_key": bool(api_key),
            "output_file": str(model_output_path),
        }
        model_registry.append(model_meta)

        for sample_idx, post_row in enumerate(post_rag_rows, start=1):
            sample_id = str(post_row["sample_id"])
            record = {
                "sample_id": sample_id,
                "provider": spec.provider,
                "model_id": spec.model_id,
                "status": "pending",
                "response_text": "",
                "error": "",
                "started_at": datetime.now().isoformat(),
            }
            if not api_key:
                record["status"] = "skipped_missing_api_key"
                record["error"] = f"{spec.api_env} is not set."
            else:
                try:
                    response_text, retry_errors = invoke_model_with_retries(
                        spec=spec,
                        api_key=api_key,
                        base_url=base_url,
                        system_prompt=str(post_row["system_prompt"]),
                        user_prompt=str(post_row["user_prompt"]),
                        temperature=args.temperature,
                        request_timeout=args.request_timeout,
                        max_retries=args.max_retries,
                        retry_wait_seconds=args.retry_wait_seconds,
                    )
                    record["response_text"] = response_text
                    record["status"] = "ok"
                    record["retry_errors"] = retry_errors
                except Exception as exc:
                    record["status"] = "error"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                time.sleep(args.sleep_seconds)
            record["finished_at"] = datetime.now().isoformat()
            record["duration_seconds"] = seconds_between(record["started_at"], record["finished_at"])
            model_rows.append(record)
            write_jsonl(model_output_path, model_rows)
            write_single_llm_readable_output(
                readable_dir / f"{sample_id}__{model_slug}.md",
                record=record,
                system_prompt=str(post_row["system_prompt"]),
                user_prompt=str(post_row["user_prompt"]),
            )
            print(
                f"[{spec.provider}/{spec.model_id}] {sample_idx}/{len(post_rag_rows)} {sample_id} -> {record['status']}",
                flush=True,
            )

        ok_count = sum(1 for row in model_rows if row["status"] == "ok")
        skipped_count = sum(1 for row in model_rows if row["status"] == "skipped_missing_api_key")
        error_count = sum(1 for row in model_rows if row["status"] == "error")
        run_summary["models"].append(
            {
                **model_meta,
                "ok_count": ok_count,
                "skipped_count": skipped_count,
                "error_count": error_count,
            }
        )

    existing_registry = load_json_file(registry_path, [])
    existing_summary = load_json_file(summary_path, {})
    preserved_registry = [
        item
        for item in existing_registry
        if item.get("provider", "").lower() not in replacing_providers
        and item.get("provider", "").lower() not in dropped_providers
    ]
    preserved_summary_models = [
        item
        for item in existing_summary.get("models", [])
        if item.get("provider", "").lower() not in replacing_providers
        and item.get("provider", "").lower() not in dropped_providers
    ]
    merged_registry = preserved_registry + model_registry
    merged_summary = {
        "created_at": existing_summary.get("created_at", run_summary["created_at"]),
        "updated_at": datetime.now().isoformat(),
        "log_dir": str(log_dir),
        "sample_count": len(cases),
        "models": preserved_summary_models + run_summary["models"],
    }

    registry_path.write_text(json.dumps(merged_registry, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(merged_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
