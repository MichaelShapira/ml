"""One-time weight staging for SAM 3 -> S3.

SAM 3 (`facebook/sam3`) is a GATED model: you must accept Meta's license on the
model page and use an HF token that has been granted access. This script pulls
only the files the transformers loader needs (config + safetensors + tokenizer)
and skips the large original `sam3.pt` checkpoint, then you sync the result to
S3 once. After that the endpoint loads from S3 with `local_files_only=True` and
never needs a token at runtime.

Usage
-----
    export HF_TOKEN=hf_...                       # access-granted token
    pip install "huggingface_hub>=0.26" hf_transfer
    HF_HUB_ENABLE_HF_TRANSFER=1 python prepare_weights.py --out weights
    aws s3 sync weights/ s3://<bucket>/sam3/weights/

Verify:
    aws s3 ls --recursive s3://<bucket>/sam3/weights/ --human-readable --summarize
Expected: config.json, model.safetensors (~3.4 GB), processor_config.json,
tokenizer files. The endpoint expects `code/inference.py` + `code/requirements.txt`
to also live under the same prefix (the deploy notebook uploads those).
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Stage SAM 3 weights for SageMaker.")
    parser.add_argument("--model-id", default="facebook/sam3", help="HF repo id.")
    parser.add_argument("--out", default="weights", help="Local output directory.")
    parser.add_argument(
        "--include-pt",
        action="store_true",
        help="Also download the original sam3.pt checkpoint (not needed by transformers).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install 'huggingface_hub>=0.26'")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("WARNING: no HF_TOKEN in env. facebook/sam3 is gated and the "
              "download will fail unless you are already logged in.", file=sys.stderr)

    # Allowlist EXACTLY the files the transformers loader needs (~3.44 GB total,
    # almost all of it model.safetensors). This is the robust way to keep the
    # download minimal: the repo also ships sam3.pt (~3.45 GB, the original FAIR
    # checkpoint) which transformers never reads. `--include-pt` overrides this.
    allow_patterns = None if args.include_pt else [
        "config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]

    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=args.out,
        token=token,
        allow_patterns=allow_patterns,
    )
    print(f"\nStaged SAM 3 weights to: {path}")
    print("Next:")
    print(f"  aws s3 sync {args.out}/ s3://<bucket>/sam3/weights/")


if __name__ == "__main__":
    main()
