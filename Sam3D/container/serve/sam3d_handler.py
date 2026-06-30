"""SAM 3D Objects reconstruction logic for the SageMaker endpoint.

NOTE on the module name: this file is intentionally NOT called `inference.py`.
The upstream repo ships `notebook/inference.py` on PYTHONPATH, so a handler named
`inference` collides with it — `import inference` would resolve to the repo's
module (which has no `reconstruct`) instead of ours. Keeping a unique name lets
`from inference import ...` below unambiguously reach the repo's pipeline.

Thin wrapper over the upstream repo's `notebook/inference.py`. Given an RGB image
and a binary mask marking ONE object, it:

  1. lifts the masked object to a 3D Gaussian Splat  (Inference.__call__)
  2. serializes it to a .ply
  3. optionally renders a turntable GIF preview     (make_scene + render_video)

This mirrors notebook/demo_single_object.ipynb exactly; we only swap file I/O
for in-memory bytes so the result can travel back through the async endpoint.

Weights: the S3 prefix is mounted at /opt/ml/model, so pipeline.yaml +
all *.ckpt live flat there (see prepare_weights.py). SAM3D_CONFIG overrides.
"""

import base64
import binascii
import io
import logging
import os
import tempfile
import time

import numpy as np
from PIL import Image

logger = logging.getLogger("sam3d")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Default location of the mounted weights prefix (flattened checkpoints/ tree).
_CONFIG_PATH = os.environ.get("SAM3D_CONFIG", "/opt/ml/model/pipeline.yaml")

_STATE = {"inference": None, "helpers": None}


def _load_helpers():
    """Import the upstream pipeline lazily so import errors surface at cold start
    in CloudWatch rather than at module import time on the gunicorn master."""
    # These come from /opt/sam-3d-objects/notebook (on PYTHONPATH). Safe now that
    # this handler module is NOT named `inference`.
    from inference import (  # noqa: E402
        Inference,
        make_scene,
        ready_gaussian_for_video_rendering,
        render_video,
    )

    return {
        "Inference": Inference,
        "make_scene": make_scene,
        "ready_gaussian_for_video_rendering": ready_gaussian_for_video_rendering,
        "render_video": render_video,
    }


def load_model():
    """Build the inference pipeline once per worker. Heavy (~13 GB ckpts + GPU)."""
    if _STATE["inference"] is not None:
        return _STATE

    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(
            f"pipeline config not found at {_CONFIG_PATH}. Did the S3 weights prefix "
            f"mount at /opt/ml/model? Expected a flattened checkpoints/ tree "
            f"(pipeline.yaml + *.ckpt). See prepare_weights.py."
        )

    helpers = _load_helpers()
    t0 = time.time()
    logger.info("Loading SAM 3D Objects pipeline from %s ...", _CONFIG_PATH)
    inference = helpers["Inference"](_CONFIG_PATH, compile=False)
    logger.info("Pipeline loaded (%.1fs)", time.time() - t0)

    _STATE["inference"] = inference
    _STATE["helpers"] = helpers
    return _STATE


def _decode_b64_image(b64: str, mode: str) -> Image.Image:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 payload: {exc}") from exc
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise ValueError(f"Could not decode bytes as an image: {exc}") from exc
    return img.convert(mode)


def reconstruct(data: dict) -> dict:
    """Run the full image+mask -> 3D pipeline. `data` is the parsed request JSON."""
    state = load_model()
    inference = state["inference"]
    helpers = state["helpers"]

    image_b64 = data.get("image")
    mask_b64 = data.get("mask")
    if not isinstance(image_b64, str) or not image_b64:
        raise ValueError("Field 'image' (base64 PNG/JPEG, RGB) is required.")
    if not isinstance(mask_b64, str) or not mask_b64:
        raise ValueError("Field 'mask' (base64 single-channel PNG, 0/255, same H*W as image) is required.")

    seed = int(data.get("seed", 42))
    render_preview = bool(data.get("render_preview", True))
    preview_frames = int(data.get("preview_frames", 48))
    preview_resolution = int(data.get("preview_resolution", 384))

    image = np.asarray(_decode_b64_image(image_b64, "RGB"), dtype=np.uint8)
    mask_img = _decode_b64_image(mask_b64, "L")
    if mask_img.size != (image.shape[1], image.shape[0]):
        raise ValueError(
            f"mask size {mask_img.size} (WxH) does not match image "
            f"{(image.shape[1], image.shape[0])}. They must align pixel-for-pixel."
        )
    mask = np.asarray(mask_img) > 0  # boolean HxW, as the pipeline expects

    timing = {}

    t0 = time.time()
    output = inference(image, mask, seed=seed)
    timing["reconstruct"] = round(time.time() - t0, 2)
    logger.info("Reconstruction done in %.2fs", timing["reconstruct"])

    # Build a CENTERED, unit-normalized scene from the object before saving. Raw
    # object-local gaussians (output["gs"]) are off-center and arbitrarily scaled,
    # which makes web viewers (gr.Model3D etc.) show black or only tilt vertically.
    # make_scene + ready_gaussian_for_video_rendering centers + normalizes it, so
    # any orbit-camera viewer frames it and rotates left/right correctly. This is
    # exactly the repo's demo_single_object.ipynb visualization path.
    scene_gs = helpers["make_scene"](output)
    scene_gs = helpers["ready_gaussian_for_video_rendering"](scene_gs)

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tf:
        ply_path = tf.name
    try:
        scene_gs.save_ply(ply_path)
        with open(ply_path, "rb") as fh:
            ply_bytes = fh.read()
    finally:
        try:
            os.remove(ply_path)
        except OSError:
            pass

    num_gaussians = None
    try:
        num_gaussians = int(scene_gs.get_xyz.shape[0])
    except Exception:
        pass

    preview_gif_b64 = None
    if render_preview:
        try:
            t1 = time.time()
            video = helpers["render_video"](
                scene_gs,
                r=1,
                fov=60,
                pitch_deg=15,
                yaw_start_deg=-45,
                resolution=preview_resolution,
                num_frames=preview_frames,
            )["color"]
            import imageio
            buf = io.BytesIO()
            imageio.mimsave(buf, video, format="GIF", duration=1000 / 30, loop=0)
            preview_gif_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            timing["render"] = round(time.time() - t1, 2)
            logger.info("Turntable render done in %.2fs", timing["render"])
        except Exception as exc:  # preview is best-effort; never fail the 3D result on it
            logger.warning("Preview render failed (returning .ply only): %s", exc)

    return {
        "ply_b64": base64.b64encode(ply_bytes).decode("ascii"),
        "num_gaussians": num_gaussians,
        "preview_gif_b64": preview_gif_b64,
        "timing_s": timing,
    }
