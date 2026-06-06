"""One-time weight-staging script.

Downloads FLUX.2 [klein] 9B (bf16) from the Hub into a local directory in
standard diffusers format, ready to be uploaded to S3 as an uncompressed
model artifact for SageMaker hosting.

After running this script:

    weights/
    ├── model_index.json
    ├── scheduler/
    ├── text_encoder/   <- 8 B Qwen3, bf16
    ├── tokenizer/
    ├── transformer/    <- 9 B flow transformer, bf16
    └── vae/

Sync that directory to S3:

    aws s3 sync weights/ s3://<bucket>/flux2-klein/weights/

Then point `PyTorchModel(model_data=...)` at the prefix with
`CompressionType: None` so SageMaker mounts it at `/opt/ml/model/`
without re-archiving anything.

Usage:
    HF_TOKEN=hf_xxx HF_HUB_ENABLE_HF_TRANSFER=1 python prepare_weights.py [--out weights]

Required deps:
    pip install "huggingface_hub>=0.27" hf_transfer

(huggingface_hub is pre-installed on SageMaker notebook images; only
hf_transfer needs to be added to speed up the ~25 GB download.)
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import snapshot_download


BASE_REPO = "black-forest-labs/FLUX.2-klein-9B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="weights",
        help="Local directory to download the diffusers-format model into.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var is required.", file=sys.stderr)
        return 1

    print(f"Downloading {BASE_REPO} -> {args.out}")
    print("(Set HF_HUB_ENABLE_HF_TRANSFER=1 + install hf_transfer for ~5x faster download.)")

    snapshot_download(
        repo_id=BASE_REPO,
        local_dir=args.out,
        token=token,
        # Skip non-essential preview JPGs and license/readme files.
        ignore_patterns=["*.jpg", "*.jpeg", "*.png", "*.md", "LICENSE*"],
    )

    # Sanity check the layout.
    expected = ["model_index.json", "scheduler", "text_encoder", "tokenizer", "transformer", "vae"]
    missing = [p for p in expected if not os.path.exists(os.path.join(args.out, p))]
    if missing:
        print(f"WARNING: expected components missing from {args.out}: {missing}")
        return 2

    total = 0
    for root, _, files in os.walk(args.out):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print(f"\nDone. Downloaded {total / 1e9:.1f} GB to {args.out}")
    print("Next step:")
    print(f"  aws s3 sync {args.out}/ s3://<bucket>/flux2-klein/weights/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
