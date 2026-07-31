from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.reasoning.sample_context import ReasoningContextBuilder, format_patient_summary
from scripts.framework.utils.config import load_config, resolve_path


AUDIT_LABS = ["lactate", "inr", "creatinine", "bilirubin", "platelet", "wbc"]


MODEL_CANDIDATES = [
    {
        "rank": 1,
        "provider": "DeepSeek",
        "model_id": "deepseek-v4-pro",
        "status": "current",
        "recommended_use": "主力高质量推理模型，优先用于20例展示与多轮对比。",
        "notes": "DeepSeek V4 正式旗舰，适合做临床长文本和结构化证据综合推理。",
        "source_url": "https://api-docs.deepseek.com/news/news260424/",
    },
    {
        "rank": 2,
        "provider": "DeepSeek",
        "model_id": "deepseek-v4-flash",
        "status": "current",
        "recommended_use": "低成本快速批量生成推理文本，适合首轮筛选。",
        "notes": "同属 V4 系列，速度更快，便于大样本跑通。",
        "source_url": "https://api-docs.deepseek.com/news/news260424/",
    },
    {
        "rank": 3,
        "provider": "Zhipu",
        "model_id": "glm-5.2",
        "status": "current",
        "recommended_use": "国产高能力对照组，适合做结构化病情推理对比。",
        "notes": "智谱官方 GLM-5.2 系列，可作为非 DeepSeek 路线的主力备选。",
        "source_url": "https://open.bigmodel.cn/dev/api/normal-model/glm-5_2",
    },
    {
        "rank": 4,
        "provider": "Baidu",
        "model_id": "ERNIE-5.1",
        "status": "current",
        "recommended_use": "国产中文文本对照组，适合病历摘要和风险解释生成。",
        "notes": "文心大模型 5.1 系列，可作为另一条国产系列基线。",
        "source_url": "https://cloud.baidu.com/doc/WENXINWORKSHOP/s/6mvek1y19",
    },
    {
        "rank": 5,
        "provider": "Qwen",
        "model_id": "qwen3.8-max-preview",
        "status": "preview_token_plan_only",
        "recommended_use": "如果要冲更高上限，可作为高配对照。",
        "notes": "预览版，适合少量高质量实验。",
        "source_url": "https://help.aliyun.com/zh/model-studio/models",
    },
    {
        "rank": 6,
        "provider": "Qwen",
        "model_id": "qwen3.7-max",
        "status": "current",
        "recommended_use": "Qwen 高能力主力候选，适合和 DeepSeek V4-Pro 对照。",
        "notes": "适合正式文本推理实验。",
        "source_url": "https://help.aliyun.com/zh/model-studio/models",
    },
    {
        "rank": 7,
        "provider": "Qwen",
        "model_id": "qwen3.7-plus",
        "status": "current",
        "recommended_use": "能力与成本平衡，适合 20 例正式比较。",
        "notes": "作为中档稳定选择比较合适。",
        "source_url": "https://help.aliyun.com/zh/model-studio/models",
    },
    {
        "rank": 8,
        "provider": "Qwen",
        "model_id": "qwen3.6-plus",
        "status": "current",
        "recommended_use": "平衡档稳定基线，适合批量推理文本生成。",
        "notes": "更适合预算受限时铺量测试。",
        "source_url": "https://help.aliyun.com/zh/model-studio/text-generation-model/",
    },
    {
        "rank": 9,
        "provider": "Qwen",
        "model_id": "qwen3.6-flash",
        "status": "current",
        "recommended_use": "低成本快速跑样本，适合首轮粗筛模型。",
        "notes": "速度优先，便于做 prompt 初筛。",
        "source_url": "https://help.aliyun.com/zh/model-studio/models",
    },
    {
        "rank": 10,
        "provider": "Qwen",
        "model_id": "qwen3-max",
        "status": "current",
        "recommended_use": "Qwen 正式 Max 备选，可和 3.7-max 做版本对照。",
        "notes": "适合做系列内对比。",
        "source_url": "https://help.aliyun.com/zh/model-studio/text-generation-model/",
    },
]


PROMPT_TEMPLATE = """# ColaCare 数据转文本 Prompt 模板

## System
你是一名重症医学 ICU 临床数据分析专家，擅长从老年脓毒症患者的时序数据和入院记录中提取客观病理生理特征。

核心任务：
1. 对结构化变量进行归纳：不要逐点复述数值，要总结趋势、极值和临床语义。
2. 结构化变量解释必须结合输入中提供的正常参考范围。
3. 对入院记录文本进行清洗和自然语言润色，保留关键医学实体。
4. 基于结构化归纳和文本叙述，按 8 个病理生理维度总结当前客观状态。

安全约束：
- 禁止输出未来 12/24/48 小时死亡概率、生存概率或最终结局。
- 禁止输出 risk_level。
- 只能总结已经发生的客观状态。
- 必须输出严格 JSON，不要输出 JSON 以外的内容。

## User
【Patient record with static data and 24-hour dynamic time-series】
{patient_summary_text}

【Patient admission record text】
{admission_record_text}

【External knowledge status】
{knowledge_text}

请严格输出 JSON，字段为：
{{
  "structured_summary": {{
    "vital_signs": "",
    "laboratory_findings": "",
    "therapeutic_interventions": ""
  }},
  "text_cleaned": "",
  "reasoning_dimensions": {{
    "hemodynamic_stability": "",
    "respiratory_oxygenation_burden": "",
    "infection_immune_stress": "",
    "coagulation_microcirculation": "",
    "renal_internal_environment": "",
    "metabolic_perfusion_acid_base": "",
    "neurological_consciousness_status": "",
    "nutrition_frailty_burden": ""
  }},
  "overall_assessment": ""
}}
"""


QUALITY_CHECKLIST = """# 推理质量检查清单
建议从以下维度人工打分（1-5分）：
1. 事实一致性：是否与结构化数据和病历摘要一致。
2. 证据引用：是否点到了真正关键的生命体征、实验室指标和病史。
3. 因果逻辑：是否解释了“为什么这些证据提示风险变化”。
4. 缺失值处理：是否正确识别关键缺失项，而不是硬编。
5. 简洁性：是否适合作为下游文本编码输入。
6. 稳定性：同类病例的输出风格和逻辑是否一致。

额外记录：
- 是否出现明显幻觉
- 是否忽略高危证据
- 是否过度依赖病史叙述而忽视结构化指标
- 是否给出了可复用的临床判断模板
"""


README_TEMPLATE = """# Reasoning Showcase

这个文件夹用于展示“基于 LLM 的多智能体协同推理”前的数据到文本转换效果。
当前结构化输入已改为 ColaCare 风格：静态信息保留单值，生命体征、实验室检查和治疗措施用窗口内小时级时间序列表示。

包含内容：
- `sampled_20_cases.csv`：随机抽取的 20 个窗口级样本
- `sampled_20_cases.jsonl`：20 例结构化转文本结果，病历字段为 `admission_record_text`
- `cases/`：每例单独一个 Markdown，方便人工核查
- `prompt_template.md`：统一提示词模板
- `reasoning_quality_checklist.md`：人工核查推理质量用
- `model_candidates_top10_2026-07-21.csv/.md`：10 个候选模型
- `sampled_20_cases_window_lab_audit.csv`：窗口内化验项最后一次非空值审计表

当前样本来源：
- 数据集：`data/processed_dataset_outputs_compact_los1_7_year1119`
- 抽样随机种子：`{seed}`
- 抽样日期：2026-07-21
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_compact_los1_7_year1119.yaml")
    parser.add_argument("--output-dir", default="reasoning")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_excerpt(text: str, limit: int = 1200) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    return normalized[:limit]


def render_case_markdown(case: dict[str, object]) -> str:
    return "\n".join(
        [
            f"# {case['sample_id']}",
            "",
            "## 基本信息",
            f"- stay_id: {case['stay_id']}",
            f"- window: {case['window_start']} -> {case['window_end']}",
            f"- death_12: {case['death_12']}",
            f"- death_24: {case['death_24']}",
            f"- death_48: {case['death_48']}",
            f"- death_delta: {case['death_delta']}",
            "",
            "## 结构化转文本摘要",
            case["patient_summary_text"],
            "",
            "## 检索查询串",
            case["query"],
            "",
            "## Patient admission record text",
            case["admission_record_text"] or "(empty)",
            "",
            "## 推荐 Prompt 输入片段",
            "```text",
            PROMPT_TEMPLATE.format(
                patient_summary_text=case["patient_summary_text"],
                admission_record_text=case["admission_record_text"] or "(empty)",
                knowledge_text="RAG disabled for this experiment.",
            ),
            "```",
        ]
    )


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_window_lab_audit(
    context_builder: ReasoningContextBuilder,
    sample_id: str,
    stay_id: int,
    window_start: int,
    window_end: int,
) -> dict[str, object]:
    dynamic = context_builder.dynamic_groups[int(stay_id)]
    window = dynamic[(dynamic["hr"] >= int(window_start)) & (dynamic["hr"] < int(window_end))].copy()
    row: dict[str, object] = {
        "sample_id": sample_id,
        "stay_id": int(stay_id),
        "window": f"{int(window_start)}-{int(window_end)}",
    }
    for lab in AUDIT_LABS:
        if lab not in window.columns:
            row[f"{lab}_has_any"] = 0
            row[f"{lab}_last_hr"] = None
            row[f"{lab}_last_val"] = None
            continue
        non_null = window.loc[window[lab].notna(), ["hr", lab]]
        if non_null.empty:
            row[f"{lab}_has_any"] = 0
            row[f"{lab}_last_hr"] = None
            row[f"{lab}_last_val"] = None
        else:
            last = non_null.iloc[-1]
            row[f"{lab}_has_any"] = 1
            row[f"{lab}_last_hr"] = float(last["hr"])
            row[f"{lab}_last_val"] = float(last[lab])
    return row


def export_model_candidates(output_dir: Path) -> None:
    csv_path = output_dir / "model_candidates_top10_2026-07-21.csv"
    md_path = output_dir / "model_candidates_top10_2026-07-21.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "provider", "model_id", "status", "recommended_use", "notes", "source_url"],
        )
        writer.writeheader()
        writer.writerows(MODEL_CANDIDATES)

    lines = [
        "# 10个候选模型（2026-07-21）",
        "",
        "> 说明：前两位仍以 DeepSeek 为主，第 3/4 位改为其他国产系列模型 GLM-5.2 和 ERNIE-5.1，用来做跨系列对照。",
        "",
        "| Rank | Provider | Model ID | Status | 推荐用途 | 备注 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in MODEL_CANDIDATES:
        lines.append(
            f"| {item['rank']} | {item['provider']} | `{item['model_id']}` | {item['status']} | {item['recommended_use']} | {item['notes']} |"
        )
    lines.extend(
        [
            "",
            "## 官方来源",
            f"- DeepSeek V4 发布页：{MODEL_CANDIDATES[0]['source_url']}",
            "- DeepSeek 当前模型列表：https://api-docs.deepseek.com/api/list-models/",
            f"- Zhipu GLM-5.2 官方页：{MODEL_CANDIDATES[2]['source_url']}",
            f"- Baidu ERNIE-5.1 官方页：{MODEL_CANDIDATES[3]['source_url']}",
            f"- Qwen 模型总览：{MODEL_CANDIDATES[4]['source_url']}",
            f"- Qwen 文本生成模型页：{MODEL_CANDIDATES[7]['source_url']}",
        ]
    )
    write_markdown(md_path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_DIR / args.config)
    processed_dir = resolve_path(PROJECT_DIR, cfg["data"]["processed_dir"])
    masked_dir = resolve_path(PROJECT_DIR, cfg["data"]["masked_mimic_dir"])
    llm_dynamic_csv = resolve_path(PROJECT_DIR, cfg["data"]["llm_dynamic_csv"]) if cfg["data"].get("llm_dynamic_csv") else None
    llm_static_csv = resolve_path(PROJECT_DIR, cfg["data"]["llm_static_csv"]) if cfg["data"].get("llm_static_csv") else None
    note_csv = resolve_path(PROJECT_DIR, cfg["data"].get("note_csv", ""))
    output_dir = resolve_path(PROJECT_DIR, args.output_dir)
    cases_dir = output_dir / "cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    for stale_case in cases_dir.glob("*.md"):
        stale_case.unlink()

    sample_index = pd.read_csv(processed_dir / "sample_index.csv")
    labels = pd.read_csv(processed_dir / "labels.csv")
    note_map: dict[str, str] = {}
    if note_csv.exists():
        notes = pd.read_csv(note_csv)
        if {"sample_id", "note_text"}.issubset(notes.columns):
            note_map = dict(zip(notes["sample_id"].astype(str), notes["note_text"].fillna("").astype(str)))

    merged = sample_index.merge(labels, on=["sample_id", "stay_id", "window_start", "window_end"], how="left")
    sampled = merged.sample(n=min(args.sample_size, len(merged)), random_state=args.seed).sort_values(
        ["stay_id", "window_start", "window_end"]
    )

    context_builder = ReasoningContextBuilder(
        processed_dir=processed_dir,
        masked_mimic_dir=masked_dir,
        llm_dynamic_csv=llm_dynamic_csv,
        llm_static_csv=llm_static_csv,
    )

    structured_rows: list[dict[str, object]] = []
    jsonl_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for idx, row in enumerate(sampled.itertuples(index=False), start=1):
        sample_id = str(row.sample_id)
        context = context_builder.build(sample_id)
        summary = context["summary"]
        admission_record_text = clean_excerpt(note_map.get(sample_id, ""))
        audit_rows.append(
            build_window_lab_audit(
                context_builder=context_builder,
                sample_id=sample_id,
                stay_id=int(row.stay_id),
                window_start=int(row.window_start),
                window_end=int(row.window_end),
            )
        )

        structured_rows.append(
            {
                "sample_id": sample_id,
                "stay_id": int(row.stay_id),
                "window_start": int(row.window_start),
                "window_end": int(row.window_end),
                "death_12": int(getattr(row, "death_12", 0)),
                "death_24": int(getattr(row, "death_24", 0)),
                "death_48": int(getattr(row, "death_48", 0)),
                "death_delta": getattr(row, "death_delta", None),
                "sofa": summary.get("sofa", "missing"),
                "mbp": summary.get("mbp", "missing"),
                "spo2": summary.get("spo2", "missing"),
                "lactate": summary.get("lactate", "missing"),
                "creatinine": summary.get("creatinine", "missing"),
                "bilirubin": summary.get("bilirubin", "missing"),
                "platelet": summary.get("platelet", "missing"),
                "wbc": summary.get("wbc", "missing"),
                "inr": summary.get("inr", "missing"),
                "vasopressors": summary.get("vasopressors", "missing"),
                "ventilation": summary.get("ventilation", "missing"),
            }
        )

        jsonl_row = {
            "sample_id": sample_id,
            "stay_id": int(row.stay_id),
            "window_start": int(row.window_start),
            "window_end": int(row.window_end),
            "death_12": int(getattr(row, "death_12", 0)),
            "death_24": int(getattr(row, "death_24", 0)),
            "death_48": int(getattr(row, "death_48", 0)),
            "death_delta": getattr(row, "death_delta", None),
            "query": context["query"],
            "patient_summary_text": str(context.get("patient_summary_text") or format_patient_summary(summary)),
            "summary_fields": summary,
            "admission_record_text": admission_record_text,
            # Kept as a compatibility alias for older evaluation scripts.
            "note_excerpt": admission_record_text,
        }
        jsonl_rows.append(jsonl_row)
        write_markdown(cases_dir / f"{idx:02d}_{sample_id}.md", render_case_markdown(jsonl_row))

    pd.DataFrame(structured_rows).to_csv(output_dir / "sampled_20_cases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(output_dir / "sampled_20_cases_window_lab_audit.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "sampled_20_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in jsonl_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_markdown(output_dir / "prompt_template.md", PROMPT_TEMPLATE)
    write_markdown(output_dir / "reasoning_quality_checklist.md", QUALITY_CHECKLIST)
    write_markdown(output_dir / "README.md", README_TEMPLATE.format(seed=args.seed))
    export_model_candidates(output_dir)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sample_count": len(jsonl_rows),
                "cases_dir": str(cases_dir),
                "processed_dir": str(processed_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
