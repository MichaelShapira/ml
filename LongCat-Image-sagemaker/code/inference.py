import io
import gc
import json
import torch
from diffusers import LongCatImagePipeline

pipe = None

def model_fn(model_dir):
    global pipe
    if pipe is None:
        gc.collect()
        torch.cuda.empty_cache()

        pipe = LongCatImagePipeline.from_pretrained(
            "meituan-longcat/LongCat-Image",   # HuggingFace repo ID, not model_dir
            torch_dtype=torch.bfloat16,
            # no local_files_only — let it download and cache
        )
        pipe.to("cuda")
    return pipe

def input_fn(request_body, content_type):
    if content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(data, model):
    prompt = data["inputs"]
    negative_prompt = data.get("negative_prompt", "")
    guidance_scale = max(1.0, min(10.0, float(data.get("guidance_scale", 4.0))))
    num_inference_steps = max(10, min(60, int(data.get("num_inference_steps", 50))))
    height = int(data.get("height", 768))
    width = int(data.get("width", 1344))

    # Snap to nearest multiple of 32 (pipeline requirement)
    height = (height // 32) * 32
    width = (width // 32) * 32

    image = model(
        prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        enable_cfg_renorm=True,
        enable_prompt_rewrite=True,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

def output_fn(prediction, accept):
    return prediction, "image/png"