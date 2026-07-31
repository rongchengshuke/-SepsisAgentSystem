from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.utils.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(PROJECT_DIR / "configs/default.yaml")
    model_name = args.model_name or cfg["model"]["text_model_name"]
    cache_dir = resolve_path(PROJECT_DIR, cfg["model"]["text_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    print("downloaded:", model_name)
    print("cache_dir:", cache_dir)
    print("hidden_size:", getattr(model.config, "hidden_size", "unknown"))
    print("tokenizer:", tokenizer.__class__.__name__)


if __name__ == "__main__":
    main()
