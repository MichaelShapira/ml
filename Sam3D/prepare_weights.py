"""One-time checkpoint staging for SAM 3D Objects -> S3.

`facebook/sam-3d-objects` is a GATED model: you must accept Meta's license on
the model page (https://huggingface.co/facebook/sam-3d-objects) and use an HF
token that has been granted access. This script downloads the repo's
`checkpoints/` tree (pipeline.yaml + the encoder/decoder/generator ckpts,
~13 GB total) and lays it out so it can be synced to S3 once. After that the
endpoint loads from disk (S3 prefix mounted at /opt/ml/model) with no Hub
access and no token at runtime.

The official setup downloads the whole repo and then `mv`s the nested
`checkpoints/` up one level. We replicate that: everything ends up flat under
`--out`, so `--out/pipeline.yaml` is the config the endpoint points at.

Usage
-----
    export HF_TOKEN=hf_...                              # access-granted token
    pip install "huggingface_hub[cli]>=0.26,<1.0"
    python prepare_weights.py --out checkpoints
    aws s3 sync checkpoints/ s3://<bucket>/sam3d/weights/

Verify:
    aws s3 ls --recursive s3://<bucket>/sam3d/weights/ --human-readable --summarize
Expected (flat under the prefix): pipeline.yaml, slat_encoder.*, slat_decoder_*,
slat_generator.*, ss_encoder.*, ss_decoder.*, ss_generator.* — ~13 GB total.
"""

import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description="Stage SAM 3D Objects checkpoints for SageMaker.")
    parser.add_argument("--model-id", default="facebook/sam-3d-objects", help="HF repo id.")
    parser.add_argument("--out", default="checkpoints", help="Local output directory (becomes the S3 prefix root).")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install 'huggingface_hub[cli]>=0.26,<1.0'")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "WARNING: no HF_TOKEN in env. facebook/sam-3d-objects is gated and the "
            "download will fail unless you are already logged in (`hf auth login`).",
            file=sys.stderr,
        )

    staging = f"{args.out}-download"

    # Pull only the checkpoints/ tree; skip docs/images and the LICENSE/readme noise.
    # The transformers-style loader is NOT used here — the pipeline reads these
    # files by relative path from pipeline.yaml, so we keep the whole checkpoints/ set.
    snapshot_download(
        repo_id=args.model_id,
        local_dir=staging,
        token=token,
        allow_patterns=["checkpoints/*"],
        max_workers=4,
    )

    # Flatten: <staging>/checkpoints/* -> <out>/*  (mirrors `mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}`)
    nested = os.path.join(staging, "checkpoints")
    if not os.path.isdir(nested):
        sys.exit(f"Expected {nested} after download; the repo layout may have changed.")

    os.makedirs(args.out, exist_ok=True)
    for name in os.listdir(nested):
        src = os.path.join(nested, name)
        dst = os.path.join(args.out, name)
        if os.path.exists(dst):
            (shutil.rmtree if os.path.isdir(dst) else os.remove)(dst)
        shutil.move(src, dst)
    shutil.rmtree(staging, ignore_errors=True)

    cfg = os.path.join(args.out, "pipeline.yaml")
    print(f"\nStaged SAM 3D Objects checkpoints to: {os.path.abspath(args.out)}")
    print(f"  pipeline config: {cfg}  ({'OK' if os.path.exists(cfg) else 'MISSING — check the download'})")
    print("Next:")
    print(f"  aws s3 sync {args.out}/ s3://<bucket>/sam3d/weights/")


if __name__ == "__main__":
    main()
