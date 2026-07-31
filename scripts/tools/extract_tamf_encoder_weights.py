from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to vendor TAMF checkpoint (.pt)")
    parser.add_argument(
        "--output",
        default="models/encoders/multimodal_encoder/weights/tamf_vendor_encoder_only.pt",
        help="Output path for encoder-only weights",
    )
    parser.add_argument(
        "--include-static-proj",
        action="store_true",
        help="Also export static_proj.* if present in the checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_DIR / checkpoint_path).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (PROJECT_DIR / output_path).resolve()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    keep_prefixes = [
        "sequence_embeddings.",
        "cross_source_fusion.",
    ]
    if args.include_static_proj:
        keep_prefixes.append("static_proj.")

    encoder_state = {
        key: value
        for key, value in state_dict.items()
        if any(key.startswith(prefix) for prefix in keep_prefixes)
    }

    payload = {
        "source_checkpoint": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "source_loss": checkpoint.get("loss"),
        "state_dict": encoder_state,
        "included_prefixes": keep_prefixes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print(
        {
            "source_checkpoint": str(checkpoint_path),
            "output": str(output_path),
            "tensor_count": len(encoder_state),
            "sample_keys": list(encoder_state.keys())[:12],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
