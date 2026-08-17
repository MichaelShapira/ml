"""SageMaker async inference handler for MiniMax Music 3 (text-to-music).

Give it lyrics and a music description; it returns a 32 kHz 16-bit stereo
WAV, up to 5 minutes long. The model is an 11B-parameter stack: an 8B Qwen3
global LLM (autoregressive, one audio frame per step at 25 fps), a 0.6B
local LLM for the residual RVQ codebooks, a 2.4B flow-matching DiT, and a
Flow-VAE vocoder.

Weights layout
--------------
This handler expects the model already on disk at `model_dir` (SageMaker
mounts the uncompressed S3 prefix at /opt/ml/model). diffusers modular
layout, staged by `prepare_weights.py` at the repo root:

    /opt/ml/model/
    ├── modular_model_index.json
    ├── condition_encoder/
    ├── language_model/       (8B Qwen3, bf16)
    ├── rvq_depth_decoder/    (0.6B local LLM)
    ├── scheduler/
    ├── tokenizer/
    ├── transformer/          (2.4B flow-matching DiT)
    └── vocoder/              (Flow-VAE decoder)

Memory strategy
---------------
bf16 weights are ~22 GB, fully resident on the 48 GB L40S (`pipe.to("cuda")`).
Deliberately NO `enable_model_cpu_offload()`: that trade is acceptable for a
4-step image diffusion model, but Music 3's global LLM runs THOUSANDS of
sequential decode steps per song (25 frames per second of audio, with
classifier-free guidance doubling every pass) — per-step PCIe swaps would
multiply an already minutes-long generation. 48 GB leaves ample headroom for
the CFG-doubled KV cache of a full 5-minute song (~3 GB).
"""

import gc
import io
import json
import logging
import os
import subprocess
import sys
import time

import torch


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _ensure_dependencies():
    """Guarantee a `transformers` compatible with Qwen3, the diffusers PR
    branch, AND the DLC's torch.

    Version triangle, NEW GEOMETRY vs the FLUX.2 endpoint (its 4.56.2 pin
    broke this endpoint's first deploy — do not restore it):
      - The pinned diffusers commit requires huggingface-hub>=1.23,<2.0.
      - transformers 4.x caps huggingface-hub<1.0 -> pip ResolutionImpossible
        at cold start, TorchServe never boots, ping health check fails.
      - transformers 5.15.0 accepts hub>=1.5,<2.0, ships Qwen3ForCausalLM,
        and its torch.float8_e8m0fnu references are hasattr-guarded
        (verified in the wheel), so it imports cleanly on the torch-2.5 DLC
        unlike early 5.x.

    The SageMaker toolkit's `pip install -r requirements.txt` runs without
    `--upgrade`, so a preinstalled DLC version can win over the pin. Re-enforce
    the bounded range here at runtime (belt-and-suspenders; requirements.txt
    already pins ==5.15.0). Runs once at container start (cold start only).

    NOTE: this function NEVER touches torch. torch is the DLC's CUDA-matched
    build; reinstalling it would break the GPU kernels.
    """
    from importlib.metadata import version as _pkg_version
    from packaging.version import parse as _parse

    _TARGET = "transformers==5.15.0"
    _LO, _HI = _parse("5.15.0"), _parse("6.0.0")

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
        # --no-deps so pinning transformers can NEVER pull a different torch
        # or disturb already-resolved deps.
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", _TARGET]
        )
        for mod in [m for m in list(sys.modules) if m == "transformers" or m.startswith("transformers.")]:
            del sys.modules[mod]
        logger.info("Installed %s and reloaded.", _TARGET)

    # Force the lazy loaders to resolve NOW, at cold start, where the log is
    # obvious — not on the first request. `from diffusers import X` succeeds
    # even with a broken transformers because diffusers imports lazily.
    from transformers import Qwen3ForCausalLM  # noqa: F401
    logger.info("Qwen3ForCausalLM import confirmed.")

    import diffusers
    from diffusers import ModularPipeline  # noqa: F401
    logger.info("diffusers %s with ModularPipeline confirmed.", diffusers.__version__)


# Module-level pipeline cache. SageMaker calls model_fn once per worker.
pipe = None

# Bounds for a single request. 300 s = the model's 5-minute native limit;
# the floor keeps degenerate requests from producing useless click-length
# clips while still allowing cheap caption auditions.
_MIN_DURATION = 5.0
_MAX_DURATION = 300.0
_DEFAULT_DURATION = 60.0


def _localize_model_index(model_dir):
    """Build a loadable local view of the model with a patched index.

    Returns the directory to pass to `ModularPipeline.from_pretrained`.

    TWO traps compound here (both hit on real deploys of this endpoint):
    1. The shipped modular_model_index.json stores
       `pretrained_model_name_or_path: "MiniMaxAI/MiniMax-Music3"` for every
       component — the Hub repo id, NOT a relative path. `load_components()`
       would resolve each component against the Hub, which this container
       must never touch (no token, HF_HUB_OFFLINE=1). The repo id must be
       rewritten to a local path.
    2. SageMaker mounts the model directory READ-ONLY (observed:
       `OSError: [Errno 30] Read-only file system` on the in-place write),
       so the index cannot be patched where it lives.

    Solution: stage a writable directory under /tmp containing SYMLINKS to
    every component folder in the read-only mount plus a patched copy of the
    index whose component paths point at the staging dir. Symlinks, not
    copies — ~29 GB of weights must not be duplicated onto the container's
    disk. HF's loaders follow symlinks fine. Idempotent across worker
    restarts (the staging dir is rebuilt).
    """
    src_index = os.path.join(model_dir, "modular_model_index.json")
    with open(src_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    staging = "/tmp/minimax_music3_local"
    os.makedirs(staging, exist_ok=True)

    # Symlink every entry of the mount except the index (patched below) and
    # the code/ folder (not a model component).
    linked = 0
    for entry in os.listdir(model_dir):
        if entry in ("modular_model_index.json", "code"):
            continue
        dst = os.path.join(staging, entry)
        if os.path.islink(dst) or os.path.exists(dst):
            continue
        os.symlink(os.path.join(model_dir, entry), dst)
        linked += 1

    changed = 0

    def _walk(node):
        nonlocal changed
        if isinstance(node, dict):
            if node.get("pretrained_model_name_or_path") not in (None, staging):
                node["pretrained_model_name_or_path"] = staging
                changed += 1
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(index)
    with open(os.path.join(staging, "modular_model_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    logger.info(
        "Staged local model view at %s (%d symlinks, %d component path(s) localized)",
        staging, linked, changed,
    )
    return staging


def model_fn(model_dir):
    """Load MiniMax Music 3 from the local mount, fully resident on CUDA."""
    global pipe
    if pipe is not None:
        return pipe

    _ensure_dependencies()

    # Weights are local. Any Hub call from here on is a bug — make it fail
    # fast and loudly instead of hanging on a network that has no token.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from diffusers import ModularPipeline

    load_dir = _localize_model_index(model_dir)

    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    logger.info("Loading pipeline graph from %s", load_dir)
    pipe = ModularPipeline.from_pretrained(load_dir)
    logger.info("Pipeline graph loaded (%.1fs)", time.time() - t0)

    t0 = time.time()
    pipe.load_components(dtype=torch.bfloat16)
    logger.info("Components loaded in bf16 (%.1fs)", time.time() - t0)

    t0 = time.time()
    pipe.to("cuda")
    logger.info("Pipeline fully resident on CUDA, no offload (%.1fs)", time.time() - t0)

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info(
            "GPU memory after load: %.1f GB used / %.1f GB total",
            (total - free) / 1e9, total / 1e9,
        )
    return pipe


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)


def predict_fn(data, model):
    # Two fields carry the request, mirroring the model's own contract:
    # lyrics (what gets sung) and prompt (genre / instrumentation / tempo /
    # mood — this is the strongest control the model has).
    lyrics = data.get("lyrics") or data.get("input")
    prompt = data.get("prompt") or data.get("instructions") or data.get("description")

    if not lyrics or not isinstance(lyrics, str):
        raise ValueError(
            "Field 'lyrics' (str) is required. For an instrumental, pass a "
            "minimal lyric like '[Intro]\\n(instrumental)' and describe the "
            "piece as instrumental in 'prompt'."
        )
    if not prompt or not isinstance(prompt, str):
        raise ValueError(
            "Field 'prompt' (str) is required: the music description "
            "(genre, BPM, key, instrumentation, vocal style, production)."
        )

    # Structure tags ([Verse], [Chorus], ...) must sit on their OWN lines:
    # the model's normalization keeps only the tag from a line that starts
    # with one and silently drops the rest of that line's lyric text.
    for line in lyrics.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped and stripped.index("]") < len(stripped) - 1:
            logger.warning(
                "Lyric line %r mixes a section tag with lyric text on one "
                "line; the text after the tag will be silently dropped by "
                "the model. Put tags on their own lines.", stripped[:60],
            )

    duration = float(data.get("audio_duration", _DEFAULT_DURATION))
    duration = max(_MIN_DURATION, min(_MAX_DURATION, duration))

    seed = data.get("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(int(seed))

    t0 = time.time()
    logger.info("Generating %.0fs of audio (seed=%s)...", duration, seed)
    audio = model(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=duration,
        generator=generator,
        output="audios",
    )[0]
    logger.info("Generation completed in %.1fs", time.time() - t0)

    # The model card's snippet does `audio.T.float().cpu().numpy()`, but the
    # pipeline actually returns a NUMPY array with output="audios" (observed
    # on a live request: AttributeError: 'numpy.ndarray' object has no
    # attribute 'float'). Handle both, in case a future diffusers build
    # returns a tensor again.
    import numpy as np

    sample_rate = int(getattr(model, "sampling_rate", 32000) or 32000)
    if hasattr(audio, "cpu"):  # torch tensor
        pcm = audio.detach().float().cpu().numpy()
    else:
        pcm = np.asarray(audio, dtype=np.float32)

    # Layout is (channels, samples); soundfile wants (samples, channels).
    # Decide by shape rather than assumption so a (samples, channels) return
    # from a future build doesn't get double-transposed: the channel axis is
    # the small one (stereo = 2).
    if pcm.ndim == 2 and pcm.shape[0] < pcm.shape[1]:
        pcm = pcm.T

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    logger.info(
        "Encoded %.1fs %d Hz WAV (%.1f MB)",
        pcm.shape[0] / sample_rate, sample_rate, len(wav_bytes) / 1e6,
    )
    # NOTE: a 5-minute stereo 16-bit WAV is ~38 MB — far above TorchServe's
    # ~6.5 MB default response cap. The deploy notebook sets
    # TS_MAX_RESPONSE_SIZE on the endpoint; without it long songs fail with
    # an opaque 507/"response size exceeded" only visible in CloudWatch.
    return wav_bytes


def output_fn(prediction, accept):
    return prediction, "audio/wav"
