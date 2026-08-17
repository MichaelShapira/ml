"""One-time weight-staging script for MiniMax Music 3.

Downloads ONLY the components the diffusers ModularPipeline actually loads
(per modular_model_index.json) into a local directory, ready to be uploaded
to S3 as an uncompressed model artifact for SageMaker hosting.

Why filtered: the Hub repo is ~58 GB, but almost half of it is NOT used by
the diffusers path:
  - qwen_7B/            (~18.5 GB) -- SGLang-Omni serving checkpoint
  - flowmatching_vae.pth (~9.8 GB) -- .pth duplicate of transformer/ for SGLang
  - dav.pth              (~0.5 GB) -- .pth vocoder duplicate
  - figures/, assets/    -- readme images
Downloading them would roughly double the S3 bill, the sync time, AND the
endpoint cold start (SageMaker mirrors the whole prefix to /opt/ml/model).

After running this script:

    weights/
    ├── modular_model_index.json
    ├── condition_encoder/    <- 100 MB
    ├── language_model/       <- 8B Qwen3 global LLM, bf16 (~17.2 GB)
    ├── rvq_depth_decoder/    <- 0.6B local LLM (~1.3 GB)
    ├── scheduler/
    ├── tokenizer/
    ├── transformer/          <- 2.4B flow-matching DiT (~9.7 GB)
    └── vocoder/              <- Flow-VAE decoder

Sync that directory to S3:

    aws s3 sync weights/ s3://<bucket>/minimax-music3/weights/

Then point `PyTorchModel(model_data=...)` at the prefix with
`CompressionType: None` so SageMaker mounts it at /opt/ml/model/
without re-archiving anything.

Usage:
    HF_HUB_ENABLE_HF_TRANSFER=1 python prepare_weights.py [--out weights]

The repo is public (no HF_TOKEN required). If you have one exported it
will be used, which can help with rate limits.

Required deps:
    pip install "huggingface_hub>=0.27" hf_transfer
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import snapshot_download


BASE_REPO = "MiniMaxAI/MiniMax-Music3"

# Exactly the subfolders referenced by modular_model_index.json, plus the
# index itself and each component's config. Everything else in the repo is
# for the SGLang-Omni serving path and dead weight here.
ALLOW_PATTERNS = [
    "modular_model_index.json",
    "condition_encoder/**",
    "language_model/**",
    "rvq_depth_decoder/**",
    "scheduler/**",
    "tokenizer/**",
    "transformer/**",
    "vocoder/**",
]

EXPECTED = [
    "modular_model_index.json",
    "condition_encoder",
    "language_model",
    "rvq_depth_decoder",
    "scheduler",
    "tokenizer",
    "transformer",
    "vocoder",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="weights",
        help="Local directory to download the diffusers-format model into.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")  # optional; the repo is public

    print(f"Downloading {BASE_REPO} (diffusers components only) -> {args.out}")
    print("(Set HF_HUB_ENABLE_HF_TRANSFER=1 + install hf_transfer for a much faster download.)")

    snapshot_download(
        repo_id=BASE_REPO,
        local_dir=args.out,
        token=token,
        allow_patterns=ALLOW_PATTERNS,
    )

    # Sanity check the layout before anyone pays for an S3 sync of a partial tree.
    missing = [p for p in EXPECTED if not os.path.exists(os.path.join(args.out, p))]
    if missing:
        print(f"WARNING: expected components missing from {args.out}: {missing}")
        return 2

    total = 0
    for root, _, files in os.walk(args.out):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print(f"\nDone. Downloaded {total / 1e9:.1f} GB to {args.out} (expected ~29 GB)")
    print("Next step:")
    print(f"  aws s3 sync {args.out}/ s3://<bucket>/minimax-music3/weights/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
