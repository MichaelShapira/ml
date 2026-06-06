"""SageMaker async inference handler for FLUX.2 [klein] 9B.

Supports two modes in a single endpoint:
  - text-to-image: pass `inputs` only.
  - image manipulation / multi-reference editing: also pass `images`
    as a list of base64-encoded PNG/JPEG strings.

Weights layout
--------------
This handler expects the model to already be on disk at `model_dir`
(SageMaker mounts the uncompressed S3 prefix at /opt/ml/model). Standard
diffusers bf16 layout:

    /opt/ml/model/
    ├── model_index.json
    ├── scheduler/
    ├── text_encoder/   (8 B Qwen3, bf16)
    ├── tokenizer/
    ├── transformer/    (9 B flow transformer, bf16)
    └── vae/

See `prepare_weights.py` at the repo root for how to assemble that layout
and sync it to S3.

Memory strategy
---------------
Total bf16 model weights are ~34 GB. That's tight on a 48 GB L40S once
activations and reference-image latents land on the device, so we follow
the model card's recommendation and call `enable_model_cpu_offload()`.
The offload hooks move the text encoder, transformer, and VAE between
CPU and GPU just-in-time during each request. Trade-off: ~7-12 seconds
of PCIe transfer overhead per request on top of ~1s of compute.

For a fully-resident GPU deployment (lower per-request latency) we'd
need bnb 8-bit quantization, which adds CUDA-extension dependency risk
on the SageMaker DLC. Start here, measure, decide.
"""

import base64
import binascii
import gc
import io
import json
import logging
import subprocess
import sys
import time
from typing import List

import torch
from PIL import Image


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _ensure_dependencies():
    """Guarantee a `transformers` compatible with Qwen3 AND both torch stacks.

    Version triangle (holds on BOTH the g6e torch-2.5 and g7e torch-2.7 DLCs):
      - FLUX.2's text encoder is `Qwen3ForCausalLM`, added in transformers >= 4.51.
      - transformers 5.x references `torch.float8_e8m0fnu`, which only exists in
        torch >= 2.7. On the g6e PyTorch 2.5 DLC, 5.x crashes at import with
        `module 'torch' has no attribute 'float8_e8m0fnu'`. On the g7e torch-2.7
        DLC 5.x would import, but we keep a single pin that works on both, so we
        stay in [4.51, 5.0). 4.56.2 has Qwen3 and avoids the float8 reference.

    The SageMaker toolkit's `pip install -r requirements.txt` runs without
    `--upgrade` and an unbounded `>=` could resolve to 5.x, so we re-enforce the
    bounded range here at runtime too (belt-and-suspenders; requirements.txt
    already pins ==4.56.2). Runs once at container start (cold start only).

    NOTE: this function NEVER touches torch. torch is the DLC's CUDA-matched
    build; reinstalling it would break the GPU kernels. We only pin transformers.
    """
    from importlib.metadata import version as _pkg_version
    from packaging.version import parse as _parse

    _TARGET = "transformers==4.56.2"
    _LO, _HI = _parse("4.51.0"), _parse("5.0.0")

    needs_fix = False
    try:
        import transformers  # noqa: F401
        current = _parse(_pkg_version("transformers"))
        if current < _LO or current >= _HI:
            logger.warning("transformers %s outside [4.51, 5.0); pinning %s.", current, _TARGET)
            needs_fix = True
        else:
            logger.info("transformers %s is in the supported range.", current)
    except Exception as exc:
        logger.warning("transformers not importable (%s); installing %s.", exc, _TARGET)
        needs_fix = True

    if needs_fix:
        # --no-deps so pinning transformers can NEVER pull a different torch or
        # disturb already-resolved deps (tokenizers, etc.). transformers 4.56's
        # own deps are already satisfied by the DLC + requirements.txt.
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", _TARGET]
        )
        # Drop any transformers submodules imported before the pin so the new
        # version is what Python resolves from here on.
        for mod in [m for m in list(sys.modules) if m == "transformers" or m.startswith("transformers.")]:
            del sys.modules[mod]
        logger.info("Installed %s and reloaded.", _TARGET)

    # Sanity check: the symbol diffusers needs must now actually import (this
    # forces transformers' lazy loader to resolve the qwen3 module).
    from transformers import Qwen3ForCausalLM  # noqa: F401
    logger.info("Qwen3ForCausalLM import confirmed.")


# Module-level pipeline cache. SageMaker calls model_fn once per worker.
# Typed loosely because Flux2KleinPipeline may not be importable until
# _ensure_dependencies() has run inside model_fn.
pipe = None


# Hard limits to keep VRAM and latency bounded for an interactive endpoint.
_MAX_REFERENCE_IMAGES = 4
_MAX_HEIGHT = 1536
_MAX_WIDTH = 1536
_MIN_SIDE = 256

# FLUX.2 has VAE scale factor 8 and packs latents into 2x2 patches, so the
# actual height/width constraint is divisibility by 16. The pipeline emits a
# warning and silently resizes otherwise.
_DIM_MULTIPLE = 16


def _decode_b64_image(b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 image payload: {exc}") from exc
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise ValueError(f"Could not decode image bytes as PIL image: {exc}") from exc
    return img.convert("RGB")


def _snap_dim(value: int, multiple: int = _DIM_MULTIPLE, lo: int = _MIN_SIDE, hi: int = _MAX_WIDTH) -> int:
    value = max(lo, min(hi, int(value)))
    return (value // multiple) * multiple


def model_fn(model_dir):
    """Load FLUX.2 [klein] 9B from the local mount point.

    Memory mode is chosen by the `BOOTH_INSTANCE_FAMILY` env var (set on the
    endpoint by the deploy notebook):

      - "g6e" (or unset): L40S 48 GB. Total bf16 weights (~34 GB) plus
        activations are tight, so we follow the model card and call
        `enable_model_cpu_offload()` (JIT CPU<->GPU swaps, ~7-12s/request).
      - "g7e": more GPU memory, so keep the whole pipeline resident on CUDA
        (`pipe.to("cuda")`, NO CPU offload) for lower per-request latency.

    `enable_model_cpu_offload()` and `pipe.to("cuda")` are mutually exclusive:
    the former installs hooks that move sub-modules just-in-time, and calling
    `.to()` afterwards breaks them — so we do exactly one based on the family.
    """
    global pipe
    if pipe is not None:
        return pipe

    _ensure_dependencies()
    from diffusers import Flux2KleinPipeline

    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    logger.info("Loading pipeline from %s", model_dir)
    pipe = Flux2KleinPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    logger.info("Pipeline loaded (%.1fs)", time.time() - t0)

    # Pick the memory strategy from the instance family the endpoint runs on.
    import os

    family = os.environ.get("BOOTH_INSTANCE_FAMILY", "g6e").strip().lower()
    t0 = time.time()
    if family == "g7e":
        # Larger-memory GPU: keep everything resident on CUDA, no CPU offload.
        pipe.to("cuda")
        logger.info("Moved pipeline fully to CUDA (no CPU offload) (%.1fs)", time.time() - t0)
    else:
        # L40S (g6e) and any unknown family: JIT CPU offload to fit in 48 GB.
        pipe.enable_model_cpu_offload()
        logger.info("CPU offload enabled (%.1fs)", time.time() - t0)
    return pipe


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)


def predict_fn(data, model):
    prompt = data.get("inputs") or data.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Field 'inputs' (str) is required.")

    # Distilled model: 4 steps, guidance 1.0 are the recommended defaults.
    num_inference_steps = max(1, min(20, int(data.get("num_inference_steps", 4))))
    guidance_scale = max(0.0, min(10.0, float(data.get("guidance_scale", 1.0))))
    seed = data.get("seed")

    # Reference images for editing / multi-reference generation. Optional.
    reference_b64: List[str] = data.get("images") or []
    if reference_b64 and not isinstance(reference_b64, list):
        raise ValueError("Field 'images' must be a list of base64-encoded image strings.")
    if len(reference_b64) > _MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"Too many reference images ({len(reference_b64)}); max is {_MAX_REFERENCE_IMAGES}."
        )
    reference_images = [_decode_b64_image(b) for b in reference_b64] or None

    # Resolution. When editing, the pipeline derives output dims from the first
    # reference image, so height/width are optional. For pure text-to-image we
    # default to 1024x1024 (the model's training resolution).
    raw_height = data.get("height")
    raw_width = data.get("width")
    height = _snap_dim(raw_height, hi=_MAX_HEIGHT) if raw_height else None
    width = _snap_dim(raw_width, hi=_MAX_WIDTH) if raw_width else None
    if reference_images is None:
        height = height or 1024
        width = width or 1024

    # With cpu offload, latents are allocated on the active execution device
    # (cuda) so the generator must live there too.
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    call_kwargs = dict(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    if reference_images is not None:
        call_kwargs["image"] = reference_images
    if height is not None:
        call_kwargs["height"] = height
    if width is not None:
        call_kwargs["width"] = width

    t0 = time.time()
    result = model(**call_kwargs)
    logger.info("Generation completed in %.1fs", time.time() - t0)
    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def output_fn(prediction, accept):
    return prediction, "image/png"
