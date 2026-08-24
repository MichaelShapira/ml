#!/bin/bash
# SageMaker notebook instance lifecycle configuration - ON-START script.
#
# WHY THIS EXISTS
# ---------------
# A notebook instance has TWO volumes:
#
#   /                            ~100 GB, FIXED SIZE, not configurable, and
#                                already largely consumed by the bundled conda
#                                environments. Typically 10-20 GB free.
#   /home/ec2-user/SageMaker     the "ML storage volume". This is what
#                                VolumeSizeInGB sizes (5 GB default, up to 16 TB)
#                                and the only path that persists across
#                                stop/start.
#
# Docker's data-root defaults to /var/lib/docker, which is on the ROOT volume.
# The MiniMax-H3 base image (lmsysorg/sglang:dev) is 13.7 GB compressed and
# ~30 GB extracted, so `docker build` fails with "no space left on device" no
# matter how large you made VolumeSizeInGB. Raising VolumeSizeInGB does not help,
# because it does not touch the root volume.
#
# This script repoints Docker's data-root at the ML volume. It must run
# ON-START rather than on-create: the root volume is rebuilt from the AMI on
# every start, so /etc/docker/daemon.json does not survive a stop.
#
# HOW TO USE
# ----------
#   aws sagemaker create-notebook-instance-lifecycle-config \
#     --notebook-instance-lifecycle-config-name h3-docker-on-ml-volume \
#     --on-start Content=$(base64 -i scripts/notebook-lifecycle-on-start.sh)
#
# then pass --lifecycle-config-name h3-docker-on-ml-volume when creating the
# notebook instance.
#
# NOT NEEDED if you build the image in CodeBuild (`sm-docker build`) or on a
# separate EC2 host. See README.
set -euo pipefail

DOCKER_DIR=/home/ec2-user/SageMaker/docker
echo "=== relocating Docker data-root to ${DOCKER_DIR} ==="

df -h / /home/ec2-user/SageMaker || true

mkdir -p "$DOCKER_DIR"
chown root:root "$DOCKER_DIR"

# Stop the daemon before moving its data-root, or it will keep writing to the old
# location and you get two half-populated stores.
systemctl stop docker 2>/dev/null || service docker stop 2>/dev/null || true

# Merge rather than overwrite: the AMI may already set other daemon options.
mkdir -p /etc/docker
python3 - "$DOCKER_DIR" <<'PY'
import json, os, sys

path = "/etc/docker/daemon.json"
cfg = {}
if os.path.exists(path):
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except ValueError:
        cfg = {}          # malformed file: replace rather than fail the boot
cfg["data-root"] = sys.argv[1]
with open(path, "w") as fh:
    json.dump(cfg, fh, indent=2)
print("wrote", path, "->", json.dumps(cfg))
PY

systemctl start docker 2>/dev/null || service docker start 2>/dev/null || true

# Give the daemon a moment, then prove where it landed. Do not fail the boot if
# Docker is unhappy - a broken lifecycle script leaves the notebook stuck in
# "Pending" and is far more annoying than a failed build.
for _ in $(seq 1 15); do
    if docker info >/dev/null 2>&1; then break; fi
    sleep 2
done
docker info --format 'Docker Root Dir: {{.DockerRootDir}}' 2>/dev/null || echo "WARNING: docker not responding"
df -h / "$DOCKER_DIR" 2>/dev/null || true

echo "=== done ==="
