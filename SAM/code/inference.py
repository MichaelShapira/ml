"""SageMaker async inference handler for SAM 3 (Segment Anything Model 3).

What this endpoint does
-----------------------
Promptable Concept Segmentation (PCS): given an image and a short noun-phrase
text prompt (e.g. "chair"), SAM 3 returns instance masks for EVERY object in
the image that matches the concept. No clicks, no boxes — just text. This is
exactly the "type a word, mask the thing" capability.

The endpoint returns the model's raw output as JSON so the caller can render
however it likes:

    {
      "prompt":        "chair",
      "image_size":    [H, W],
      "num_instances": N,
      "scores":        [float, ...],
      "boxes":         [[x1, y1, x2, y2], ...],   # absolute pixels, xyxy
      "masks":         ["<base64 PNG>", ...]       # 1 binary mask per instance
    }

Each mask is a single-channel PNG (0 / 255), same H x W as the input image, so
the client can overlay it on the original image directly.

Weights layout
--------------
SAM 3 is loaded from disk at `model_dir` (SageMaker mounts the uncompressed S3
prefix at /opt/ml/model). The transformers-native layout is:

    /opt/ml/model/
    ├── config.json
    ├── model.safetensors
    ├── processor_config.json
    ├── tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
    └── special_tokens_map.json

See `prepare_weights.py` at the repo root for how to stage that to S3. The
container never talks to the Hugging Face Hub at runtime and holds no HF token.

Memory / dtype
--------------
SAM 3 is ~860M params (~3.4 GB in fp32), trivially resident on any of the
recommended GPUs (T4/A10G/L4, 16-24 GB). We default to fp32 for numerical
robustness; set SAM3_DTYPE=bfloat16 (or float16) on the endpoint env to trade a
little accuracy for speed/VRAM. The model is meant to run at 1008px — don't
change the processor resolution unless you measured the accuracy hit.
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

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


_MIN_TRANSFORMERS = "4.57.6"  # first release line that ships Sam3Model / Sam3Processor
_MAX_TRANSFORMERS = "5.0.0"   # exclusive: 5.x needs torch>=2.7 (float8); DLC is torch 2.6


def _ensure_dependencies():
    """Guarantee a transformers in [4.57.6, 5.0) so the SAM 3 classes exist AND
    it stays importable on the torch 2.6 DLC.

    The SageMaker toolkit installs requirements.txt without --upgrade, and an
    older transformers may already be baked into the DLC. We re-check at cold
    start and re-pin if it's outside the window. We install with --no-deps so we
    can NEVER disturb the DLC's CUDA-matched torch build.

    Version window rationale (same trap as the FLUX2 notebook in this repo):
      - < 4.57.6: no Sam3Model / Sam3Processor.
      - >= 5.0:   imports torch.float8_e8m0fnu, absent on torch < 2.7, so it
                  crashes at import on the PyTorch 2.6 DLC.
    """
    from importlib.metadata import version as _pkg_version
    from packaging.version import parse as _parse

    _LO, _HI = _parse(_MIN_TRANSFORMERS), _parse(_MAX_TRANSFORMERS)
    _SPEC = f"transformers>={_MIN_TRANSFORMERS},<{_MAX_TRANSFORMERS}"

    needs_fix = False
    try:
        import transformers  # noqa: F401

        current = _parse(_pkg_version("transformers"))
        if current < _LO or current >= _HI:
            logger.warning("transformers %s outside [%s, %s); re-pinning.", current, _MIN_TRANSFORMERS, _MAX_TRANSFORMERS)
            needs_fix = True
        else:
            logger.info("transformers %s is in the supported range for SAM 3.", current)
    except Exception as exc:  # transformers missing entirely
        logger.warning("transformers not importable (%s); installing.", exc)
        needs_fix = True

    if needs_fix:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", _SPEC]
        )
        for mod in [m for m in list(sys.modules) if m == "transformers" or m.startswith("transformers.")]:
            del sys.modules[mod]
        logger.info("Pinned %s and reloaded.", _SPEC)

    # Force the lazy loader to resolve the SAM 3 symbols so a bad install fails
    # loudly at cold start rather than on the first request.
    from transformers import Sam3Model, Sam3Processor  # noqa: F401
    logger.info("Sam3Model / Sam3Processor import confirmed.")


# Module-level cache. SageMaker calls model_fn once per worker.
_STATE = {"model": None, "processor": None}

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_MAX_INSTANCES = 100  # cap returned instances to bound payload size


def model_fn(model_dir):
    if _STATE["model"] is not None:
        return _STATE

    _ensure_dependencies()
    from transformers import Sam3Model, Sam3Processor

    import os

    dtype = _DTYPES.get(os.environ.get("SAM3_DTYPE", "float32").strip().lower(), torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    t0 = time.time()
    logger.info("Loading SAM 3 from %s (dtype=%s, device=%s)", model_dir, dtype, device)
    model = Sam3Model.from_pretrained(model_dir, torch_dtype=dtype, local_files_only=True)
    model.to(device)
    model.eval()
    processor = Sam3Processor.from_pretrained(model_dir, local_files_only=True)
    logger.info("SAM 3 loaded (%.1fs)", time.time() - t0)

    _STATE["model"] = model
    _STATE["processor"] = processor
    return _STATE


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)


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


def _mask_to_b64_png(mask: np.ndarray) -> str:
    """Encode an HxW boolean/0-1 mask as a base64 single-channel PNG (0/255)."""
    arr = (np.asarray(mask) > 0).astype("uint8") * 255
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def predict_fn(data, state):
    model = state["model"]
    processor = state["processor"]

    image_b64 = data.get("image") or data.get("image_b64")
    if not image_b64 or not isinstance(image_b64, str):
        raise ValueError("Field 'image' (base64-encoded PNG/JPEG string) is required.")
    prompt = data.get("text") or data.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Field 'text' (str concept prompt, e.g. 'chair') is required.")

    # Confidence + mask binarization thresholds (overridable per request).
    threshold = float(data.get("threshold", 0.5))
    mask_threshold = float(data.get("mask_threshold", 0.5))
    max_instances = int(data.get("max_instances", _MAX_INSTANCES))

    image = _decode_b64_image(image_b64)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    # Match the image tensor to the model dtype (fp16/bf16 runs), leave token ids as-is.
    if "pixel_values" in inputs and inputs["pixel_values"].dtype.is_floating_point:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    logger.info("SAM 3 forward done in %.2fs", time.time() - t0)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results.get("masks")
    boxes = results.get("boxes")
    scores = results.get("scores")

    def _to_list(x):
        if x is None:
            return []
        if hasattr(x, "detach"):
            return x.detach().cpu().tolist()
        return list(x)

    scores_list = _to_list(scores)
    boxes_list = _to_list(boxes)

    # Sort instances by score (desc) and cap to bound the JSON payload.
    order = list(range(len(scores_list)))
    order.sort(key=lambda i: scores_list[i] if i < len(scores_list) else 0.0, reverse=True)
    order = order[:max_instances]

    masks_np = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)

    out_masks, out_boxes, out_scores = [], [], []
    for i in order:
        out_masks.append(_mask_to_b64_png(masks_np[i]))
        out_boxes.append([float(v) for v in boxes_list[i]] if i < len(boxes_list) else None)
        out_scores.append(float(scores_list[i]) if i < len(scores_list) else None)

    response = {
        "prompt": prompt,
        "image_size": [image.height, image.width],
        "num_instances": len(out_masks),
        "scores": out_scores,
        "boxes": out_boxes,
        "masks": out_masks,
    }
    return response


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
