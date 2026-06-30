"""One-time weight staging for NAFNet image modification -> S3.

Pulls the four pretrained checkpoints the endpoint serves and drops them in a
local folder you then `aws s3 sync` to the weights prefix:

    denoise     NAFNet-SIDD-width64.pth   (HF mirror, no token)
    deblur      NAFNet-GoPro-width64.pth  (HF mirror, no token)
    deblur/reds NAFNet-REDS-width64.pth   (HF mirror, no token)
    stereo_sr   NAFSSR-L_4x.pth           (official Google Drive via gdown)

The three NAFNet checkpoints come from the community mirror
`mikestealth/nafnet-models` (mirrors megvii-research/NAFNet's Google Drive
releases) so no Hugging Face token is needed. NAFSSR has no clean HF mirror, so
it is fetched from Megvii's official Google Drive file with gdown.

Usage:
    pip install "huggingface_hub>=0.26" gdown
    python prepare_weights.py --out weights
    aws s3 sync weights/ s3://<bucket>/nafnet/weights/

Verify:
    aws s3 ls --recursive s3://<bucket>/nafnet/weights/ --human-readable --summarize
"""

import argparse
import os
import subprocess
import sys

# HF mirror (megvii-research NAFNet release checkpoints, MIT-mirrored).
HF_REPO = "mikestealth/nafnet-models"
HF_FILES = [
    "NAFNet-SIDD-width64.pth",
    "NAFNet-GoPro-width64.pth",
    "NAFNet-REDS-width64.pth",
]

# Official NAFSSR-L 4x checkpoint (Megvii Google Drive, see docs/StereoSR.md).
NAFSSR_GDRIVE_ID = "1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb"
NAFSSR_FILENAME = "NAFSSR-L_4x.pth"


def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def main():
    parser = argparse.ArgumentParser(description="Stage NAFNet weights for SageMaker.")
    parser.add_argument("--out", default="weights", help="Local output directory.")
    parser.add_argument("--skip-stereo", action="store_true",
                        help="Skip the NAFSSR (stereo SR) Google Drive download.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. NAFNet denoise/deblur from the HF mirror.
    _ensure("huggingface_hub>=0.26", "huggingface_hub")
    from huggingface_hub import hf_hub_download
    for fname in HF_FILES:
        print(f"downloading {fname} from {HF_REPO} ...")
        src = hf_hub_download(repo_id=HF_REPO, filename=fname)
        dst = os.path.join(args.out, fname)
        if os.path.abspath(src) != os.path.abspath(dst):
            import shutil
            shutil.copyfile(src, dst)
        print(f"  -> {dst}")

    # 2. NAFSSR stereo SR from Google Drive.
    if not args.skip_stereo:
        _ensure("gdown", "gdown")
        import gdown
        dst = os.path.join(args.out, NAFSSR_FILENAME)
        print(f"downloading {NAFSSR_FILENAME} from Google Drive ({NAFSSR_GDRIVE_ID}) ...")
        gdown.download(id=NAFSSR_GDRIVE_ID, output=dst, quiet=False)
        print(f"  -> {dst}")
    else:
        print("skipping NAFSSR (stereo SR) download (--skip-stereo).")

    print(f"\nStaged NAFNet weights to: {args.out}")
    print("Next:")
    print(f"  aws s3 sync {args.out}/ s3://<bucket>/nafnet/weights/")


if __name__ == "__main__":
    main()
