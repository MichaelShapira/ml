# SAM 3D Objects — SageMaker Bring-Your-Own-Container image.
#
# Reproduces https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md
# (conda env + CUDA-toolkit-compiled pytorch3d / flash_attn / kaolin / gsplat),
# then layers a tiny Flask app that satisfies the SageMaker /ping + /invocations
# contract. Weights are NOT baked in — they are mounted from an S3 prefix at
# /opt/ml/model at deploy time.
#
# Build for linux/amd64 (SageMaker is x86-64). The compile steps are slow and
# RAM-hungry; see container/build_and_push.sh.

FROM condaforge/miniforge3:24.9.2-0

ARG SAM3D_COMMIT=main
# L40S (g6e) is Ada = sm_89. Restrict arch to keep the flash_attn/pytorch3d/gsplat
# builds tractable. Add more (e.g. "8.0;8.6;8.9;9.0") only if you target other GPUs.
ARG TORCH_CUDA_ARCH_LIST="8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV DEBIAN_FRONTEND=noninteractive
# Bound source-build parallelism so flash_attn doesn't OOM the builder.
ENV MAX_JOBS=4
ENV PIP_NO_CACHE_DIR=1

# System libs needed at runtime by opencv / open3d / bpy / rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates build-essential \
        libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 \
        libegl1 libgles2 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git \
    && cd sam-3d-objects \
    && git checkout ${SAM3D_COMMIT}

WORKDIR /opt/sam-3d-objects

# 1) Create the conda env exactly as upstream specifies (CUDA toolkit, gcc, python 3.11).
RUN mamba env create -f environments/default.yml && mamba clean -afy

# All subsequent RUNs execute inside the sam3d-objects env.
SHELL ["mamba", "run", "-n", "sam3d-objects", "/bin/bash", "-c"]

# 2) Core + pytorch3d (two-step, per upstream: p3d's torch pin is broken otherwise).
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
RUN pip install -e '.[dev]'
RUN pip install -e '.[p3d]'

# 3) Inference extras: kaolin (NVIDIA wheels pinned to torch 2.5.1/cu121), gsplat, etc.
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
RUN pip install -e '.[inference]'

# 4) Patch hydra (upstream fix not yet released to PyPI).
RUN ./patching/hydra

# 5) Serving deps (kept minimal; the heavy stack is already in the env).
RUN pip install "flask>=3.0,<4" "gunicorn>=22.0" "boto3>=1.34"

# 6) Our serving code. `notebook/` holds the repo's inference.py + the sam3d_objects
#    import path; we add it to PYTHONPATH so predictor.py can import the pipeline.
ENV PYTHONPATH="/opt/sam-3d-objects:/opt/sam-3d-objects/notebook:${PYTHONPATH}"
ENV SAM3D_REPO=/opt/sam-3d-objects
COPY serve/ /opt/program/
RUN chmod +x /opt/program/serve

# SageMaker calls `docker run <image> serve`. Make /opt/program/serve resolvable
# and ensure it activates the conda env before launching gunicorn.
ENV PATH="/opt/program:${PATH}"
ENV SAGEMAKER_PROGRAM=predictor.py
WORKDIR /opt/program

ENTRYPOINT ["/opt/program/serve"]
