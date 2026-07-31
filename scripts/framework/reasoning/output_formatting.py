from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def safe_parse_json_object(text: str) -> Any:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def render_value_block(value: Any, level: int = 0) -> str:
    if isinstance(value, dict):
        blocks: list[str] = []
        for key, item in value.items():
            blocks.append(f"{'#' * (level + 3)} {key}\n\n{render_value_block(item, level + 1)}")
        return "\n\n".join(blocks)
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return "\n\n".join(f"- {item}" for item in value)
        return "\n\n".join(
            f"{'#' * (level + 3)} Item {idx}\n\n{render_value_block(item, level + 1)}"
            for idx, item in enumerate(value, start=1)
        )
    return str(value)


def render_response_sections(response_text: str) -> str:
    parsed = safe_parse_json_object(response_text)
    if isinstance(parsed, dict):
        return "\n\n".join(f"## {key}\n\n{render_value_block(value)}" for key, value in parsed.items())
    return f"## raw_response\n\n{str(response_text or '').strip()}"


def write_single_llm_readable_output(
    path: Path,
    *,
    record: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
) -> None:
    lines = [
        f"# {record.get('sample_id', '')}",
        "",
        "## run_info",
        "",
        f"- provider: {record.get('provider', '')}",
        f"- model_id: {record.get('model_id', '')}",
        f"- status: {record.get('status', '')}",
        f"- duration_seconds: {record.get('duration_seconds', '')}",
        "",
        "## system_prompt",
        "",
        "```text",
        str(system_prompt).strip(),
        "```",
        "",
        "## user_prompt",
        "",
        "```text",
        str(user_prompt).strip(),
        "```",
        "",
        "## response",
        "",
        render_response_sections(str(record.get("response_text", ""))),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_multiagent_readable_output(
    path: Path,
    *,
    record: dict[str, Any],
    agent_context: dict[str, Any],
) -> None:
    lines = [
        f"# {record.get('sample_id', '')}",
        "",
        "## run_info",
        "",
        f"- provider: {record.get('provider', '')}",
        f"- model_id: {record.get('model_id', '')}",
        f"- status: {record.get('status', '')}",
        f"- duration_seconds: {record.get('duration_seconds', '')}",
        f"- memory_update_policy: {record.get('memory_update_policy', '')}",
        "",
        "## agent_context_input",
        "",
        render_value_block(agent_context),
        "",
        "## doctor_reviews",
        "",
    ]
    doctor_reviews = record.get("doctor_reviews", [])
    if isinstance(doctor_reviews, list) and doctor_reviews:
        for item in doctor_reviews:
            lines.extend(
                [
                    f"### {item.get('doctor_id', '')} - {item.get('title', '')}",
                    "",
                    f"Focus: {item.get('focus', '')}",
                    "",
                    "#### doctor_input_system_prompt",
                    "",
                    "```text",
                    str(item.get("system_prompt", "")).strip(),
                    "```",
                    "",
                    "#### doctor_input_user_prompt",
                    "",
                    "```text",
                    str(item.get("user_prompt", "")).strip(),
                    "```",
                    "",
                    "#### doctor_output",
                    "",
                    render_response_sections(str(item.get("review", ""))),
                    "",
                ]
            )
    else:
        lines.extend(["(empty)", ""])
    lines.extend(
        [
            "## consultation_feedback",
            "",
            render_value_block(record.get("consultation_feedback", [])),
            "",
            "## meta_agent_input",
            "",
            "### meta_system_prompt",
            "",
            "```text",
            str(record.get("meta_system_prompt", "")).strip(),
            "```",
            "",
            "### meta_user_prompt",
            "",
            "```text",
            str(record.get("meta_user_prompt", "")).strip(),
            "```",
            "",
            "## meta_agent_output",
            "",
            render_response_sections(str(record.get("meta_report", ""))),
            "",
            "## agent_final_output",
            "",
            render_response_sections(str(record.get("agent_final_output", ""))),
            "",
            "## evolution_result",
            "",
            render_value_block(record.get("evolution_result", {})),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
