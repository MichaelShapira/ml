#!/usr/bin/env bash
# Build the MiniMax-H3 serving image and push it to ECR.
#
# Disk: the sglang base image is 13.7 GB compressed / ~30 GB extracted, so budget
# ~80 GB free on whatever filesystem holds Docker's data-root. Check it with
#   docker info --format '{{.DockerRootDir}}' && df -h $(docker info --format '{{.DockerRootDir}}')
#
# On a SageMaker notebook instance that is the ~100 GB ROOT volume, not the
# volume VolumeSizeInGB sizes, and it usually has 10-20 GB free - so this fails
# unless you relocate the data-root (scripts/notebook-lifecycle-on-start.sh) or
# build in CodeBuild with `sm-docker build`. See the README.
#
# Usage:
#   ./build_and_push.sh [-r REGION] [-n REPO_NAME] [-t TAG] [-i SGLANG_IMAGE] [-c SGLANG_COMMIT]
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
REPO_NAME="minimax-h3-sglang"
TAG="latest"
SGLANG_IMAGE="lmsysorg/sglang:dev"
SGLANG_COMMIT="d257b58e67193780ff8a59ab54b48219b9dc28d2"

while getopts "r:n:t:i:c:h" opt; do
  case "$opt" in
    r) REGION="$OPTARG" ;;
    n) REPO_NAME="$OPTARG" ;;
    t) TAG="$OPTARG" ;;
    i) SGLANG_IMAGE="$OPTARG" ;;
    c) SGLANG_COMMIT="$OPTARG" ;;
    h) sed -n '2,14p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done

cd "$(dirname "$0")"

command -v docker >/dev/null || { echo "docker not found on PATH"; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker daemon not reachable"; exit 1; }

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO_NAME}:${TAG}"

echo "region=${REGION} account=${ACCOUNT}"
echo "image =${IMAGE_URI}"
echo "base  =${SGLANG_IMAGE}"
echo "commit=${SGLANG_COMMIT}"

aws ecr describe-repositories --region "$REGION" --repository-names "$REPO_NAME" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO_NAME" \
       --image-scanning-configuration scanOnPush=true \
       --encryption-configuration encryptionType=AES256 >/dev/null

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

# Pull the base explicitly so a transient Docker Hub failure is obvious rather
# than showing up as a confusing build error 20 minutes in.
docker pull --platform linux/amd64 "$SGLANG_IMAGE"

DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  --build-arg "SGLANG_IMAGE=${SGLANG_IMAGE}" \
  --build-arg "SGLANG_COMMIT=${SGLANG_COMMIT}" \
  -t "$IMAGE_URI" .

docker push "$IMAGE_URI"

echo
echo "pushed: ${IMAGE_URI}"
echo "$IMAGE_URI" > .image_uri
