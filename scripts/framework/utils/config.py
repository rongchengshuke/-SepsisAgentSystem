from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir: str | Path, value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    return Path(base_dir).resolve() / value


def resolve_experience_store_config(
    base_dir: str | Path,
    reasoning_cfg: dict[str, Any],
) -> tuple[Path, dict[str, Path] | None]:
    paths_cfg = reasoning_cfg.get("experience_store_paths")
    if isinstance(paths_cfg, dict) and paths_cfg:
        resolved_paths: dict[str, Path] = {}
        for key, value in paths_cfg.items():
            if value in (None, ""):
                continue
            resolved_paths[str(key)] = resolve_path(base_dir, value)
        if resolved_paths:
            fallback = resolve_path(
                base_dir,
                reasoning_cfg.get("experience_store_path", "data/reasoning/colacare_evolver_experience.json"),
            )
            return fallback, resolved_paths
    return (
        resolve_path(
            base_dir,
            reasoning_cfg.get("experience_store_path", "data/reasoning/colacare_evolver_experience.json"),
        ),
        None,
    )


def resolve_reasoning_llm_config(cfg: dict[str, Any], backend: str) -> dict[str, Any]:
    normalized = str(backend).strip().lower()
    if normalized == "mock":
        return {
            "backend": "mock",
            "provider": "mock",
            "api_key": None,
            "api_env": None,
            "model_name": "",
            "base_url": "",
        }

    if normalized == "deepseek":
        section = cfg["model"]["deepseek"]
        return {
            "backend": "deepseek",
            "provider": "DeepSeek",
            "api_key": os.environ.get("DEEPSEEK_API_KEY"),
            "api_env": "DEEPSEEK_API_KEY",
            "model_name": str(section["model_name"]),
            "base_url": str(section["base_url"]),
        }

    if normalized == "glm":
        section = cfg["model"]["glm"]
        return {
            "backend": "glm",
            "provider": "Zhipu",
            "api_key": os.environ.get("ZHIPU_API_KEY")
            or os.environ.get("ZAI_API_KEY")
            or os.environ.get("GLM_API_KEY"),
            "api_env": "ZHIPU_API_KEY|ZAI_API_KEY|GLM_API_KEY",
            "model_name": str(section["model_name"]),
            "base_url": str(section["base_url"]),
        }

    if normalized == "qwen":
        section = cfg["model"]["qwen"]
        return {
            "backend": "qwen",
            "provider": "Qwen",
            "api_key": os.environ.get("QWEN_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("ALIBABA_API_KEY"),
            "api_env": "QWEN_API_KEY|DASHSCOPE_API_KEY|ALIBABA_API_KEY",
            "model_name": str(section["model_name"]),
            "base_url": str(section["base_url"]),
        }

    raise ValueError(f"Unsupported reasoning backend: {backend}")
