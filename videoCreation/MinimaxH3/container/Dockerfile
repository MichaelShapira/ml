# MiniMax-H3 on SageMaker: SGLang Diffusion + a thin /ping,/invocations shim.
#
# Base image choice: the AWS SageMaker SGLang DLC. SGLang v0.5.17 is the first
# tagged release that carries the MiniMax-H3 pipeline (verified: absent from
# v0.5.16, present in v0.5.17), and the 2026/08/08 DLC release ships exactly
# that, built with the [all] extra - so the whole diffusion stack (diffusers,
# imageio, av, ...) plus the compiled sgl-kernel/flashinfer set is already in
# the image, CVE-patched by AWS. No source checkout, no editable reinstall.
#
# The DLC lives in AWS's DLC registry (account 763104351884), which exists in
# every region - build_and_push.sh and h3/codebuild.py pass the region-correct
# URI as a build arg and log Docker in to that registry before pulling. The
# default below is us-east-1.
ARG SGLANG_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/sglang:0.5.17-gpu-py312-cu130-ubuntu24.04-sagemaker
FROM ${SGLANG_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # SageMaker mounts read-only-ish paths; keep every cache under /tmp.
    HF_HOME=/tmp/h3-hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8

# ffmpeg/ffprobe: the shim extracts first/last frames and the soundtrack on the
# endpoint. The DLC does not guarantee a system ffmpeg (imageio-ffmpeg's binary
# is python-internal), so install it explicitly.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Record which sglang the image actually serves - handy when reading endpoint
# logs months later.
RUN python3 -c "from importlib.metadata import version; print(version('sglang'))" > /etc/sglang-version \
 && echo "sglang $(cat /etc/sglang-version)"

# Fail the build - not the endpoint 25 minutes into a deploy - if this sglang
# does not actually carry the MiniMax-H3 pipeline.
RUN python3 - <<'PY'
import importlib.util, sys
required = [
    "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3",
    "sglang.multimodal_gen.runtime.models.dits.minimax_h3",
]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("MiniMax-H3 support missing from this sglang build: " + ", ".join(missing))
print("MiniMax-H3 modules present:", *required, sep="\n  ")
PY

RUN python3 -m pip install --no-cache-dir "boto3>=1.34" "hf_transfer>=0.1.6" "huggingface_hub>=0.30"

COPY serve.py /opt/h3/serve.py
COPY serve    /usr/local/bin/serve
# The SageMaker DLC ships its own single-model entrypoint at /usr/bin/serve.
# Overwrite it too, so `serve` resolves to the H3 shim no matter which PATH
# entry wins.
RUN cp /usr/local/bin/serve /usr/bin/serve \
 && chmod +x /usr/local/bin/serve /usr/bin/serve /opt/h3/serve.py \
 && mkdir -p /tmp/h3 /tmp/h3-hf

EXPOSE 8080

# SageMaker hosting runs `docker run <image> serve`. Clear the DLC's inherited
# ENTRYPOINT so that argument is executed directly.
ENTRYPOINT []
CMD ["serve"]
