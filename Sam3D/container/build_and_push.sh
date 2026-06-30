#!/bin/bash
# Build the SAM 3D Objects BYOC image for linux/amd64 and push it to ECR.
#
# SageMaker notebooks / Studio JupyterLab have NO Docker daemon. AWS's
# recommended client there is Finch (https://runfinch.com), which is Docker-CLI
# compatible. This script auto-detects a builder:
#     BUILDER=finch  (preferred on SageMaker)  ->  finch build + finch push
#     BUILDER=docker (laptop / EC2 with daemon) ->  docker buildx build --push
# Override with:  BUILDER=docker ./build_and_push.sh ...
#
# Usage:
#   ./build_and_push.sh [repo-name] [region] [commit]
#       repo-name  ECR repository name        (default: sam3d-objects)
#       region     AWS region                 (default: us-east-1)
#       commit     sam-3d-objects git commit  (default: main; PIN for reproducibility)
#
# HEADS UP: this image compiles pytorch3d / flash_attn / gsplat from source —
# slow and RAM-hungry. If the notebook host can't handle it (OOM / killed),
# use the CodeBuild path instead (see README "Heavy build" note):
#     pip install sagemaker-studio-image-build
#     sm-docker build . --repository sam3d-objects:latest \
#         --compute-type BUILD_GENERAL1_2XLARGE --build-arg SAM3D_COMMIT=<sha>
set -euo pipefail

REPO_NAME="${1:-sam3d-objects}"
REGION="${2:-us-east-1}"
COMMIT="${3:-main}"
TAG="latest"

# --- pick a builder ---
BUILDER="${BUILDER:-}"
if [ -z "${BUILDER}" ]; then
  if command -v finch >/dev/null 2>&1; then
    BUILDER="finch"
  elif command -v docker >/dev/null 2>&1; then
    BUILDER="docker"
  else
    echo "ERROR: neither 'finch' nor 'docker' found." >&2
    echo "On a SageMaker notebook, install Finch (AWS-recommended) or use the" >&2
    echo "CodeBuild path: pip install sagemaker-studio-image-build && sm-docker build ." >&2
    exit 1
  fi
fi
echo "Builder:   ${BUILDER}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${ECR_URI}/${REPO_NAME}:${TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Account:   ${ACCOUNT_ID}"
echo "Region:    ${REGION}"
echo "Image URI: ${IMAGE_URI}"
echo "Commit:    ${COMMIT}"

# Create the ECR repo if it does not exist.
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}" >/dev/null

if [ "${BUILDER}" = "finch" ]; then
  # Finch needs its VM running on macOS/Windows (and on some managed Linux setups).
  # No-op if Finch runs natively; harmless if already started.
  finch vm status >/dev/null 2>&1 || finch vm init >/dev/null 2>&1 || true
  finch vm start >/dev/null 2>&1 || true

  aws ecr get-login-password --region "${REGION}" \
    | finch login --username AWS --password-stdin "${ECR_URI}"

  # Finch builds with BuildKit; build then push (no buildx --push).
  finch build \
    --platform linux/amd64 \
    --build-arg "SAM3D_COMMIT=${COMMIT}" \
    -t "${IMAGE_URI}" \
    "${SCRIPT_DIR}"
  finch push "${IMAGE_URI}"

elif [ "${BUILDER}" = "docker" ]; then
  aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${ECR_URI}"

  docker buildx build \
    --platform linux/amd64 \
    --build-arg "SAM3D_COMMIT=${COMMIT}" \
    -t "${IMAGE_URI}" \
    --push \
    "${SCRIPT_DIR}"
else
  echo "ERROR: unknown BUILDER '${BUILDER}' (use finch or docker)." >&2
  exit 1
fi

echo ""
echo "Pushed: ${IMAGE_URI}"
echo "Use this as image_uri in sam3d-sagemaker.ipynb."
