from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.framework.models.tamf_triple_encoder import build_tamf_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract encoder-only weights from a stage1 TAMF checkpoint")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--source-dims", nargs=3, type=int, default=[35, 16, 5])
    parser.add_argument("--embed-dim", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    output_path = Path(args.output_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        raw_state = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        raw_state = payload["state_dict"]
    else:
        raw_state = payload

    encoder = build_tamf_encoder(
        backend="tamf_upstream_compatible",
        source_dims=list(args.source_dims),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=0.1,
    )
    valid_keys = set(encoder.state_dict().keys())
    encoder_state = {key: value for key, value in raw_state.items() if key in valid_keys}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_state, output_path)
    print(f"checkpoint: {checkpoint_path}")
    print(f"output_path: {output_path}")
    print(f"saved_keys: {len(encoder_state)}")
    missing = sorted(valid_keys - set(encoder_state.keys()))
    print(f"missing_keys: {len(missing)}")
    if missing:
        print("first_missing:", missing[:10])


if __name__ == "__main__":
    main()
