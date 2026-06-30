"""SageMaker async inference handler for NAFNet image modification.

What this endpoint does
-----------------------
Restoration / enhancement of an image (or, for the editor, the masked region
of an image) using Megvii's NAFNet family:

    task="denoise"     -> NAFNet-SIDD-width64   (removes sensor/ISO noise)
    task="deblur"      -> NAFNet-GoPro-width64  (motion deblur; variant="reds"
                          uses NAFNet-REDS-width64 for video-style blur)
    task="stereo_sr"   -> NAFSSR-L_4x           (4x stereo super-resolution;
                          needs a left+right image pair)

The endpoint mirrors the SAM 3 endpoint's async S3 contract used by the raw
editor: the SPA PUTs a request JSON (base64 image[s] + task), calls
InvokeEndpointAsync, then polls the OutputLocation for the result JSON.

Non-destructive masked editing
------------------------------
denoise/deblur return a FULL-resolution processed image the same size as the
input, so the editor composites it over the working buffer using the existing
mask alpha (exactly like an AI/brush mask) — the original RAW is never touched
and the op becomes one undoable History step. If a `mask` (base64 PNG, white =
apply) is supplied the endpoint ALSO composites server-side so the notebook can
demonstrate "masked region only"; the SPA normally composites client-side and
omits `mask`.

Request JSON (staged to S3, referenced by InputLocation)
    {
      "task":        "denoise" | "deblur" | "stereo_sr",
      "image":       "<base64 PNG/JPEG>",     # main / left image
      "image_right": "<base64>",              # required for stereo_sr only
      "variant":     "gopro" | "reds",        # deblur only (default gopro)
      "mask":        "<base64 PNG, 0/255>",   # optional, denoise/deblur only
      "tile":         0,                       # optional tile size (px), 0=off
      "tile_overlap": 32                       # optional tile overlap (px)
    }

Response JSON (written to OutputLocation)
    {
      "task":        "denoise",
      "image_size":  [H, W],
      "output_size": [H2, W2],
      "image":       "<base64 PNG result>",
      "image_right": "<base64>"               # stereo_sr only
    }

Weights layout (uncompressed S3 prefix mounted at /opt/ml/model)
    /opt/ml/model/
    ├── NAFNet-SIDD-width64.pth
    ├── NAFNet-GoPro-width64.pth
    ├── NAFNet-REDS-width64.pth
    ├── NAFSSR-L_4x.pth
    └── code/{inference.py, requirements.txt, nafnet_archs.py}

Models are loaded lazily per task on first use and cached, so cold start stays
cheap and a g4dn.xlarge (T4, 16 GB) never holds more than it needs.
"""

import base64
import binascii
import gc
import io
import json
import logging
import os
import time

import numpy as np
import torch
from PIL import Image

import nafnet_archs as A

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# task -> (builder, weight filename, needs dual input)
_REGISTRY = {
    "denoise":   (A.build_denoise, "NAFNet-SIDD-width64.pth", False),
    "deblur":    (A.build_deblur,  "NAFNet-GoPro-width64.pth", False),
    "stereo_sr": (lambda: A.build_stereo_sr(4), "NAFSSR-L_4x.pth", True),
}
# deblur "reds" variant swaps only the weight file (same architecture).
_DEBLUR_VARIANTS = {"gopro": "NAFNet-GoPro-width64.pth", "reds": "NAFNet-REDS-width64.pth"}

_STATE = {"device": None, "model_dir": None, "models": {}}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def model_fn(model_dir):
    # Don't load any network here — just record where weights live and warm the
    # GPU. Networks are built lazily in predict_fn keyed by (task, weight file).
    _STATE["model_dir"] = model_dir
    _STATE["device"] = _device()
    gc.collect()
    if _STATE["device"] == "cuda":
        torch.cuda.empty_cache()
    logger.info("NAFNet handler ready (device=%s, model_dir=%s)", _STATE["device"], model_dir)
    logger.info("Weights present: %s", sorted(
        f for f in os.listdir(model_dir) if f.endswith(".pth")))
    return _STATE


def _load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu")
    # NAFNet/NAFSSR release checkpoints store weights under "params".
    if isinstance(ckpt, dict) and "params" in ckpt:
        ckpt = ckpt["params"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in ckpt.items()}


def _get_model(task, weight_file):
    cache_key = weight_file
    if cache_key in _STATE["models"]:
        return _STATE["models"][cache_key]

    builder = _REGISTRY[task][0]
    path = os.path.join(_STATE["model_dir"], weight_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Weight file '{weight_file}' for task '{task}' not found in model dir. "
            f"Run prepare_weights.py + re-sync the prefix.")

    t0 = time.time()
    model = builder()
    model.load_state_dict(_load_state_dict(path), strict=True)
    model.eval().to(_STATE["device"])
    _STATE["models"][cache_key] = model
    logger.info("Loaded %s for task=%s (%.1fs)", weight_file, task, time.time() - t0)
    return model


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)


def _decode_b64_image(b64, mode="RGB"):
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 image payload: {exc}") from exc
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img.convert(mode)


def _to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0   # H,W,3 in [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return t


def _to_image(t):
    t = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((t * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@torch.no_grad()
def _run_single(model, t, tile=0, tile_overlap=32):
    """Run a same-size restoration model, optionally tiled to bound VRAM."""
    device = _STATE["device"]
    t = t.to(device)
    if not tile or (t.shape[-2] <= tile and t.shape[-1] <= tile):
        return model(t).cpu()

    # Simple overlap-add tiling for large images on a 16 GB T4.
    b, c, h, w = t.shape
    stride = tile - tile_overlap
    out = torch.zeros((b, c, h, w), dtype=torch.float32)
    weight = torch.zeros((1, 1, h, w), dtype=torch.float32)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            y1, x1 = max(y2 - tile, 0), max(x2 - tile, 0)
            patch = model(t[:, :, y1:y2, x1:x2]).cpu()
            out[:, :, y1:y2, x1:x2] += patch
            weight[:, :, y1:y2, x1:x2] += 1.0
    return out / weight.clamp(min=1.0)


@torch.no_grad()
def _run_stereo(model, tl, tr):
    device = _STATE["device"]
    inp = torch.cat([tl, tr], dim=1).to(device)   # 1,6,H,W
    out = model(inp).cpu()                          # 1,6,H2,W2
    left, right = out[:, :3], out[:, 3:]
    return left, right


def _composite(original, processed, mask_img):
    """original/processed: PIL RGB same size. mask_img: PIL L (white=apply).
    Returns PIL RGB = original outside mask, processed inside (feathered)."""
    if mask_img.size != original.size:
        mask_img = mask_img.resize(original.size, Image.BILINEAR)
    a = (np.asarray(mask_img, dtype=np.float32) / 255.0)[..., None]
    o = np.asarray(original, dtype=np.float32)
    p = np.asarray(processed, dtype=np.float32)
    blended = o * (1.0 - a) + p * a
    return Image.fromarray((blended + 0.5).astype(np.uint8), mode="RGB")


def predict_fn(data, state):
    task = (data.get("task") or "").strip().lower()
    if task not in _REGISTRY:
        raise ValueError(f"Field 'task' must be one of {sorted(_REGISTRY)}; got {task!r}.")

    image_b64 = data.get("image")
    if not image_b64 or not isinstance(image_b64, str):
        raise ValueError("Field 'image' (base64-encoded image string) is required.")
    image = _decode_b64_image(image_b64)
    tile = int(data.get("tile", 0))
    tile_overlap = int(data.get("tile_overlap", 32))

    if task == "stereo_sr":
        right_b64 = data.get("image_right")
        if not right_b64:
            raise ValueError("task 'stereo_sr' requires 'image_right' (the right-view image).")
        right = _decode_b64_image(right_b64)
        if right.size != image.size:
            right = right.resize(image.size, Image.BILINEAR)
        model = _get_model(task, _REGISTRY[task][1])
        left_out, right_out = _run_stereo(model, _to_tensor(image), _to_tensor(right))
        out_l, out_r = _to_image(left_out), _to_image(right_out)
        return {
            "task": task,
            "image_size": [image.height, image.width],
            "output_size": [out_l.height, out_l.width],
            "image": _png_b64(out_l),
            "image_right": _png_b64(out_r),
        }

    # denoise / deblur (same-size restoration)
    weight_file = _REGISTRY[task][1]
    if task == "deblur":
        variant = (data.get("variant") or "gopro").strip().lower()
        weight_file = _DEBLUR_VARIANTS.get(variant, weight_file)

    model = _get_model(task, weight_file)
    out_t = _run_single(model, _to_tensor(image), tile=tile, tile_overlap=tile_overlap)
    processed = _to_image(out_t)

    mask_b64 = data.get("mask")
    if mask_b64:
        processed = _composite(image, processed, _decode_b64_image(mask_b64, mode="L"))

    return {
        "task": task,
        "image_size": [image.height, image.width],
        "output_size": [processed.height, processed.width],
        "image": _png_b64(processed),
    }


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
