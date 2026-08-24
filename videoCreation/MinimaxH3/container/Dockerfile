# MiniMax-H3 on SageMaker: SGLang Diffusion + a thin /ping,/invocations shim.
#
# Base image choice: lmsysorg/sglang ships the whole compiled kernel stack
# (sgl-kernel, flashinfer, a matching torch/CUDA) already built for the image's
# CUDA version. Building that from scratch on top of nvidia/cuda works but adds
# ~30 min and a new set of version-triangle failures for no benefit.
#
# MiniMax-H3 support is NOT in any tagged sglang release (verified: absent from
# v0.5.16, present in main), so the build checks out a pinned main commit inside
# the image's bundled source tree and reinstalls it editable with the
# [diffusion] extra. Pin SGLANG_COMMIT for reproducible rebuilds.
ARG SGLANG_IMAGE=lmsysorg/sglang:dev
FROM ${SGLANG_IMAGE}

# Pinned sglang main commit that contains python/sglang/multimodal_gen/**/minimax_h3*.
# Override with --build-arg SGLANG_COMMIT=main to track HEAD.
ARG SGLANG_COMMIT=d257b58e67193780ff8a59ab54b48219b9dc28d2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # SageMaker mounts read-only-ish paths; keep every cache under /tmp.
    HF_HOME=/tmp/h3-hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Locate the sglang source bundled in the base image and move it to the pinned
# commit. Order of attempts matters: (1) checkout from objects already in the
# bundled clone - zero network; (2) plain-HTTPS tarball from codeload - no git
# smart protocol. git-over-https is deliberately NOT used: anonymous fetches
# from CodeBuild's shared IPs get an auth challenge ("could not read Username
# for 'https://github.com'") and there is no terminal to answer it.
RUN set -eux; \
    export GIT_TERMINAL_PROMPT=0; \
    SRC=""; \
    for c in /sgl-workspace/sglang /sglang /opt/sglang /workspace/sglang; do \
      if [ -f "$c/python/pyproject.toml" ]; then SRC="$c"; break; fi; \
    done; \
    if [ -n "$SRC" ]; then \
      git config --global --add safe.directory "$SRC"; \
      git -C "$SRC" checkout --force "${SGLANG_COMMIT}" || SRC=""; \
    fi; \
    if [ -z "$SRC" ]; then \
      echo "pinned commit not in bundled clone; downloading tarball"; \
      SRC=/sgl-workspace/sglang-pinned; \
      rm -rf "$SRC"; mkdir -p "$SRC"; \
      curl -fsSL "https://github.com/sgl-project/sglang/archive/${SGLANG_COMMIT}.tar.gz" \
        | tar -xz -C "$SRC" --strip-components=1; \
    fi; \
    cd "$SRC"; \
    { git rev-parse HEAD 2>/dev/null || echo "${SGLANG_COMMIT}"; } > /etc/sglang-commit; \
    python3 -m pip install --no-cache-dir -e "python[diffusion]"; \
    # The moving :dev base ships flashinfer companion packages (flashinfer-cubin,
    # flashinfer-jit-cache) built for ITS flashinfer, but the pinned sglang commit
    # pins its own flashinfer-python, and flashinfer hard-fails at import when the
    # versions disagree. Matching companion wheels do not exist for every release,
    # so re-align them when possible and drop them otherwise: without them
    # flashinfer falls back to fetching cubins / JIT-compiling at runtime, which
    # is correct (if slower on first use) rather than mismatched binaries.
    FI_VER="$(python3 -c 'from importlib.metadata import version; print(version("flashinfer-python"))')"; \
    for pkg in flashinfer-cubin flashinfer-jit-cache; do \
      if python3 -m pip show "$pkg" >/dev/null 2>&1; then \
        python3 -m pip install --no-cache-dir "$pkg==$FI_VER" \
          || { echo "no $pkg==$FI_VER wheel; removing stale $pkg"; python3 -m pip uninstall -y "$pkg"; }; \
      fi; \
    done; \
    echo "SGLANG_SRC=$SRC" > /etc/sglang-src

# Fail the build - not the endpoint 25 minutes into a deploy - if the pinned
# commit does not actually carry the MiniMax-H3 pipeline.
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
RUN chmod +x /usr/local/bin/serve /opt/h3/serve.py \
 && mkdir -p /tmp/h3 /tmp/h3-hf

EXPOSE 8080

# SageMaker hosting runs `docker run <image> serve`. Clear any inherited
# ENTRYPOINT so that argument is executed directly.
ENTRYPOINT []
CMD ["serve"]
