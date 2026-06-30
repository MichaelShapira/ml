"""Interactive web app for SAM 3D Objects — click to mask, then orbit the 3D result.

Reproduces the Meta `convert-image-to-3d` UX:
  * click an object in the photo -> SAM (local) segments it; the mask overlays live
  * click again to refine (Add = foreground, Remove = background) until it's right
  * "Generate 3D" -> the SageMaker async endpoint lifts it to a 3D Gaussian Splat
  * the result loads in a viewer you rotate/zoom with the mouse (gr.Model3D)

Two ways to run it:

  A) From the notebook (reuses the configured endpoint + invoke helper):
         import app
         app.build_demo(invoke_fn=invoke_async).launch(share=True)

  B) Standalone web app (uses your AWS creds + a few env vars):
         export SAM3D_ENDPOINT=sam3d-objects-g6e        # endpoint name
         export AWS_REGION=us-east-1
         python app.py
     The S3 I/O prefixes default to the SageMaker default bucket
     (sagemaker-<region>-<account>/sam3d-inputs|outputs|failures).

The masking model is a small `facebook/sam-vit-base` that runs locally (CPU is
fine). Only the heavy 3D lift goes to the GPU endpoint.
"""

import base64
import io
import json
import os
import tempfile
import time
import uuid
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

# ---- lazy singletons -------------------------------------------------------
_SAM = {"model": None, "proc": None, "dev": None}


def _ensure_compatible_hub():
    """The base conda env often ships huggingface_hub>=1.0, but the installed
    transformers pins huggingface_hub<1.0 and refuses to import otherwise. Pin a
    compatible hub BEFORE transformers is first imported (it's imported lazily in
    _load_sam, so this runs before any transformers import)."""
    import subprocess
    import sys
    try:
        from importlib.metadata import version
        from packaging.version import parse

        if parse(version("huggingface_hub")) >= parse("1.0.0"):
            print("Pinning huggingface_hub<1.0 for transformers compatibility...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.34,<1.0"]
            )
            for m in [m for m in list(sys.modules) if m.startswith("huggingface_hub")]:
                del sys.modules[m]
    except Exception as exc:
        print(f"(hub version check skipped: {exc})")


def _load_sam():
    if _SAM["model"] is not None:
        return _SAM
    _ensure_compatible_hub()
    import torch
    from transformers import SamModel, SamProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _SAM["model"] = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()
    _SAM["proc"] = SamProcessor.from_pretrained("facebook/sam-vit-base")
    _SAM["dev"] = dev
    return _SAM


def _image_key(image):
    return (image.shape, hash(image.tobytes()))


def _get_embedding(sam, image):
    """Compute SAM's image embedding once per image and cache it. The ViT encoder
    is the expensive part; caching it makes every refine-click after the first
    nearly instant (only the lightweight prompt decoder runs per click)."""
    import torch

    key = _image_key(image)
    if _SAM.get("emb_key") == key and _SAM.get("emb") is not None:
        return _SAM["emb"]
    inp = sam["proc"](image, return_tensors="pt").to(sam["dev"])
    with torch.no_grad():
        emb = sam["model"].get_image_embeddings(inp["pixel_values"])
    _SAM["emb_key"] = key
    _SAM["emb"] = emb
    return emb


def mask_from_points(image, points, labels):
    """Run SAM with all accumulated click points (labels: 1=foreground, 0=background).
    Reuses a cached image embedding so refine-clicks don't re-run the ViT encoder."""
    if image is None or not points:
        return None
    import torch

    sam = _load_sam()
    emb = _get_embedding(sam, image)
    inp = sam["proc"](
        image,
        input_points=[[[float(x), float(y)] for x, y in points]],
        input_labels=[[int(l) for l in labels]],
        return_tensors="pt",
    ).to(sam["dev"])
    with torch.no_grad():
        out = sam["model"](
            input_points=inp["input_points"],
            input_labels=inp["input_labels"],
            image_embeddings=emb,           # skip the heavy vision encoder
            multimask_output=True,
        )
    masks = sam["proc"].image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu()
    )[0][0]
    best = int(out.iou_scores[0, 0].argmax())
    return masks[best].numpy().astype(bool)


def render_overlay(image, mask, points, labels):
    """Cyan translucent mask + click dots (green=add, red=remove)."""
    ov = image.copy()
    if mask is not None:
        ov[mask] = (0.5 * ov[mask] + 0.5 * np.array([0, 200, 255])).astype(np.uint8)
    pim = Image.fromarray(ov)
    draw = ImageDraw.Draw(pim)
    for (x, y), l in zip(points, labels):
        c = (0, 230, 0) if int(l) == 1 else (255, 40, 40)
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=c, outline=(255, 255, 255), width=2)
    return np.array(pim)


def _b64_png(arr):
    img = arr if isinstance(arr, Image.Image) else Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---- default (standalone) async invoke via boto3 ---------------------------

def _default_invoke_factory():
    import boto3

    region = os.environ.get("AWS_REGION") or boto3.Session().region_name or "us-east-1"
    endpoint = os.environ.get("SAM3D_ENDPOINT", "sam3d-objects-g6e")
    acct = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    bucket = os.environ.get("SAM3D_BUCKET", f"sagemaker-{region}-{acct}")
    in_pref = os.environ.get("SAM3D_INPUT_PREFIX", "sam3d-inputs/")
    out_pref = os.environ.get("SAM3D_OUTPUT_PREFIX", "sam3d-outputs/")

    s3 = boto3.client("s3", region_name=region)
    smr = boto3.client("sagemaker-runtime", region_name=region)

    def _split(uri):
        p = urlparse(uri)
        return p.netloc, p.path.lstrip("/")

    def invoke(payload, timeout_s=1800, poll_s=2.0):
        key = f"{in_pref}{uuid.uuid4().hex}.json"
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload).encode(),
                      ContentType="application/json")
        resp = smr.invoke_endpoint_async(
            EndpointName=endpoint, InputLocation=f"s3://{bucket}/{key}",
            ContentType="application/json", InvocationTimeoutSeconds=3600)
        ob, ok = _split(resp["OutputLocation"])
        fb, fk = _split(resp["FailureLocation"]) if resp.get("FailureLocation") else (None, None)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                return json.loads(s3.get_object(Bucket=ob, Key=ok)["Body"].read())
            except s3.exceptions.NoSuchKey:
                pass
            if fb:
                try:
                    raise RuntimeError(s3.get_object(Bucket=fb, Key=fk)["Body"].read().decode())
                except s3.exceptions.NoSuchKey:
                    pass
            time.sleep(poll_s)
        raise TimeoutError(f"no result after {timeout_s}s")

    return invoke


def build_demo(invoke_fn=None, sample_url=None):
    """Build the Gradio UI. `invoke_fn(payload)->result dict` lifts image+mask to 3D.

    If invoke_fn is None, a boto3-based default is created from env vars.
    """
    import gradio as gr

    _ensure_compatible_hub()  # heal the hub/transformers version conflict before the UI is used
    if invoke_fn is None:
        invoke_fn = _default_invoke_factory()

    # Optional starter image so the app is usable immediately.
    sample = None
    sample_url = sample_url or os.environ.get(
        "SAM3D_SAMPLE_URL",
        "https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/"
        "notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png",
    )
    try:
        import urllib.request
        p = os.path.join(tempfile.gettempdir(), "sam3d_sample.png")
        urllib.request.urlretrieve(sample_url, p)
        sample = np.array(Image.open(p).convert("RGB"))
    except Exception:
        sample = None

    def on_select(orig, pts, lbls, mode, evt: gr.SelectData):
        if orig is None:
            return gr.update(), pts, lbls, None
        x, y = int(evt.index[0]), int(evt.index[1])
        pts = pts + [[x, y]]
        lbls = lbls + [1 if mode == "Add (foreground)" else 0]
        mask = mask_from_points(orig, pts, lbls)
        return render_overlay(orig, mask, pts, lbls), pts, lbls, mask

    def on_upload(image):
        # new image -> reset everything; image is the source of truth
        return image, image, [], [], None

    def on_clear(orig):
        return orig, [], [], None

    def generate(orig, mask, seed, progress=gr.Progress()):
        if orig is None:
            raise gr.Error("Upload or pick an image first.")
        if mask is None or not mask.any():
            raise gr.Error("Click the object in the image to create a mask first.")
        progress(0.15, desc="Lifting to 3D on the SageMaker endpoint (first run also loads the model)...")
        payload = {
            "image": _b64_png(orig),
            "mask": _b64_png((mask.astype("uint8") * 255)),
            "seed": int(seed),
            "render_preview": False,  # we show an interactive viewer instead of a GIF
        }
        t0 = time.time()
        result = invoke_fn(payload)
        total = time.time() - t0
        if "ply_b64" not in result:
            raise gr.Error(f"Endpoint returned no model: {result}")
        ply = base64.b64decode(result["ply_b64"])
        path = os.path.join(tempfile.gettempdir(), f"recon_{uuid.uuid4().hex}.ply")
        with open(path, "wb") as f:
            f.write(ply)
        n = result.get("num_gaussians")
        t = result.get("timing_s", {}) or {}
        recon = t.get("reconstruct")
        bits = []
        if n:
            bits.append(f"{n:,} gaussians")
        if recon is not None:
            bits.append(f"GPU reconstruct {recon:.1f}s")
        bits.append(f"total {total:.0f}s")
        if recon is not None and total - recon > 30:
            bits.append("(first run includes a one-time model load; next runs are much faster)")
        progress(1.0, desc="Done")
        return path, " · ".join(bits)

    with gr.Blocks(title="SAM 3D Objects — image to 3D") as demo:
        gr.Markdown(
            "# SAM 3D Objects — click an object, get a 3D model\n"
            "1. **Click** the object in the image. The mask appears in cyan. "
            "**Click again** to refine (switch to *Remove* to subtract areas).\n"
            "2. Press **Generate 3D**. 3. **Drag to rotate**, scroll to zoom the result."
        )
        orig_state = gr.State(sample)
        pts_state = gr.State([])
        lbls_state = gr.State([])
        mask_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(value=sample, type="numpy", label="Click the object (add points to refine)",
                               interactive=True, height=460)
                mode = gr.Radio(["Add (foreground)", "Remove (background)"],
                                value="Add (foreground)", label="Click mode")
                with gr.Row():
                    clear_btn = gr.Button("Clear points")
                    gen_btn = gr.Button("Generate 3D", variant="primary")
                seed = gr.Slider(0, 9999, value=42, step=1, label="Seed")
                status = gr.Markdown("")
            with gr.Column(scale=1):
                model3d = gr.Model3D(label="3D result — drag to rotate, scroll to zoom",
                                     clear_color=[0.05, 0.05, 0.07, 1.0], height=460)

        img.upload(on_upload, [img], [orig_state, img, pts_state, lbls_state, mask_state])
        img.select(on_select, [orig_state, pts_state, lbls_state, mode],
                   [img, pts_state, lbls_state, mask_state])
        clear_btn.click(on_clear, [orig_state], [img, pts_state, lbls_state, mask_state])
        gen_btn.click(generate, [orig_state, mask_state, seed], [model3d, status])

    return demo


if __name__ == "__main__":
    build_demo().launch(server_name="0.0.0.0", share=True)
