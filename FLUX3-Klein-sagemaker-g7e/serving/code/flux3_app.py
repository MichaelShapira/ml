"""Ray Serve application: FLUX 3 Image Klein (step-distilled) + its prompt upsampler on one GPU.

Runs inside the AWS Ray Serve SageMaker DLC
(763104351884.dkr.ecr.<region>.amazonaws.com/ray:serve-ml-sagemaker-cuda-v1.5.7: PyTorch 2.10 / CUDA 12.9,
Transformers 5.19, Python 3.13). The DLC entrypoint installs `code/requirements.txt`, runs
`serve run /opt/ml/model/config.yaml`, and proxies SageMaker's /invocations to this deployment's "/" route.

Model layout under /opt/ml/model/weights (written by scripts/stage_weights.py):

    modular_model_index.json   <- the Hub repo's pipeline index (Flux3DistilledModularPipeline)
    transformer/ vae/ text_encoder/
    prompt_upsampler/          <- black-forest-labs/flux-3-image-klein-pup (Qwen3.5-9B fine-tune)
    _flux3_stage.json          <- staging manifest: revisions + where each external component was put

Everything loads in BF16 except the VAE (FP32), fully resident on the GPU: about 39 GB of weights on a 96 GB
RTX PRO 6000 Blackwell (ml.g7e.2xlarge). No CPU offload, no quantization.

Request (JSON):
    prompt | inputs      str, required. A short request; the upsampler expands it.
    images               list of base64 PNG/JPEG/WebP strings, optional, at most 10 (editing / multi-reference).
    upsample             bool, default true. false = send `prompt` to the image model as written (a full caption).
    boxes                list of layout boxes, optional (see the Diffusers FLUX 3 docs); needs upsample=true.
    num_inference_steps  int, default 16, clamped to 4..16 (the distilled model is trained for 4 to 16).
    height, width        int, optional. Snapped down to a multiple of 16, clamped to 256..2048. An edit defaults to
                         the first reference's size; text-to-image defaults to 1024x1024.
    seed                 int, optional. Random when absent; always reported in JSON responses.
    max_new_tokens       int, optional, upsampler caption budget, default 3584.
    response_format      "png" (default, raw image/png bytes) or "json" (base64 image + caption + timings).
    guidance_scale       accepted and ignored: the distilled checkpoint has no classifier-free guidance.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import os
import secrets
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The endpoint never talks to the Hub: weights are local. Set before diffusers/transformers import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


from ray import serve  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse, Response  # noqa: E402

WEIGHTS_DIR = Path(os.environ.get("FLUX3_WEIGHTS_DIR", "/opt/ml/model/weights"))
STAGE_MANIFEST = "_flux3_stage.json"
DEVICE = os.environ.get("FLUX3_DEVICE", "cuda")

MAX_REFERENCES = 10
DEFAULT_STEPS = 16
MIN_STEPS, MAX_STEPS = 4, 16
MIN_SIDE, MAX_SIDE, DIM_MULTIPLE = 256, 2048, 16
MAX_NEW_TOKENS = 3584
MAX_PROMPT_CHARS = 8000
# BFL's reference decoding for the upsampler (flux3/pup.py, and the pup model card): greedy, with a presence
# penalty of 1.5 for text-to-image requests and none for edits. Diffusers' own block decodes greedily without it.
PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}
# Test hook, unset in production: extra image-pipeline inputs as JSON. Hugging Face's tiny FLUX 3 test pipeline
# (used to smoke-test the container without the gated weights) needs {"text_encoder_out_layers": [1]}.
PIPELINE_EXTRA = {
    key: tuple(value) if isinstance(value, list) else value
    for key, value in json.loads(os.environ.get("FLUX3_PIPELINE_EXTRA") or "{}").items()
}


def _marker(event: str, **fields: Any) -> None:
    """One greppable milestone line in CloudWatch: `[flux3] EVENT {...}`."""
    # stderr: Ray forwards a replica's stderr to `serve run` (the container log -> CloudWatch) reliably; its stdout
    # lines were observed to go missing on the SageMaker Ray Serve DLC.
    import sys

    line = f"[flux3] {event} {json.dumps(fields, default=str)}" if fields else f"[flux3] {event}"
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def prepare_linear_attention() -> str:
    """Pick the backend for the prompt upsampler's (Qwen3.5) gated-delta-rule layers. Must run before transformers
    imports the Qwen3.5 model code, which binds the implementation at import time.

    flash-linear-attention runs them as Triton kernels; Transformers otherwise uses its reference torch code, which it
    warns is much slower. Triton builds a small CUDA launcher with the system C compiler on first use, and the Ray
    Serve DLC ships none, so gcc is installed from the Amazon Linux repos (about 15 s, once per container). If that
    is impossible, or FLUX3_LINEAR_ATTENTION=torch, fla is hidden and the torch path is used.
    """
    import importlib.util
    import shutil
    import subprocess
    import sys

    def use_torch(reason: str) -> str:
        sys.modules["fla"] = None  # `import fla` now raises ImportError -> Transformers' torch fallback
        return f"torch-fallback ({reason})"

    if os.environ.get("FLUX3_LINEAR_ATTENTION", "fla").lower() == "torch":
        return use_torch("FLUX3_LINEAR_ATTENTION=torch")
    if sys.modules.get("fla", False) is None:
        return "torch-fallback (fla hidden)"
    if importlib.util.find_spec("fla") is None:
        return "torch-fallback (flash-linear-attention not installed)"
    if os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc"):
        return "flash-linear-attention"
    if shutil.which("dnf") and os.geteuid() == 0:
        t0 = time.time()
        proc = subprocess.run(["dnf", "install", "-y", "-q", "gcc"], capture_output=True, text=True, timeout=900)
        if proc.returncode == 0 and shutil.which("gcc"):
            _marker("GCC_INSTALLED", seconds=round(time.time() - t0, 1), reason="Triton needs a C compiler")
            return "flash-linear-attention"
        _marker("GCC_INSTALL_FAILED", returncode=proc.returncode, stderr=proc.stderr[-800:])
    return use_torch("no C compiler for Triton")


def _disable_progress_bars(block) -> None:
    """ModularPipeline.set_progress_bar_config only reaches top-level blocks; the denoise loop is nested."""
    if hasattr(block, "set_progress_bar_config"):
        block.set_progress_bar_config(disable=True)
    for child in getattr(block, "sub_blocks", {}).values():
        _disable_progress_bars(child)


class RequestError(ValueError):
    """A problem with the request itself (HTTP 400), not with the server."""


@dataclass
class GenerationRequest:
    prompt: str
    images: list = field(default_factory=list)  # PIL images
    upsample: bool = True
    boxes: list | None = None
    steps: int = DEFAULT_STEPS
    height: int | None = None
    width: int | None = None
    seed: int | None = None
    max_new_tokens: int = MAX_NEW_TOKENS
    response_format: str = "png"


def _decode_image(data: str, index: int):
    from PIL import Image, ImageOps

    if not isinstance(data, str) or not data:
        raise RequestError(f"images[{index}] must be a non-empty base64 string")
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestError(f"images[{index}] is not valid base64: {exc}") from exc
    try:
        image = Image.open(io.BytesIO(raw))
        if getattr(image, "is_animated", False):
            image.seek(0)
        image = ImageOps.exif_transpose(image)
        image.load()
    except Exception as exc:  # noqa: BLE001 - any decode failure is the caller's problem
        raise RequestError(f"images[{index}] could not be decoded as an image: {exc}") from exc
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return image.convert("RGB")


def _int_field(data: dict, name: str, default: int | None) -> int | None:
    value = data.get(name, default)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"'{name}' must be an integer") from exc


def _snap(value: int | None) -> int | None:
    if value is None:
        return None
    value = max(MIN_SIDE, min(MAX_SIDE, value))
    return (value // DIM_MULTIPLE) * DIM_MULTIPLE


def parse_request(body: bytes) -> GenerationRequest:
    try:
        data = json.loads(body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequestError(f"body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RequestError("body must be a JSON object")

    prompt = data.get("prompt", data.get("inputs"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestError("'prompt' (or 'inputs') must be a non-empty string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise RequestError(f"'prompt' is longer than {MAX_PROMPT_CHARS} characters")

    raw_images = data.get("images")
    if raw_images is None and data.get("image") is not None:
        raw_images = [data["image"]]
    raw_images = raw_images or []
    if not isinstance(raw_images, list):
        raise RequestError("'images' must be a list of base64 strings")
    if len(raw_images) > MAX_REFERENCES:
        raise RequestError(f"at most {MAX_REFERENCES} reference images, got {len(raw_images)}")
    images = [_decode_image(item, i) for i, item in enumerate(raw_images)]

    upsample = data.get("upsample", True)
    if not isinstance(upsample, bool):
        raise RequestError("'upsample' must be true or false")
    boxes = data.get("boxes")
    if boxes is not None:
        if not isinstance(boxes, list) or not all(isinstance(b, dict) for b in boxes):
            raise RequestError("'boxes' must be a list of objects")
        if not upsample:
            raise RequestError("'boxes' needs upsample=true (the upsampler writes the box table)")

    steps = _int_field(data, "num_inference_steps", DEFAULT_STEPS)
    steps = max(MIN_STEPS, min(MAX_STEPS, steps))
    max_new_tokens = _int_field(data, "max_new_tokens", MAX_NEW_TOKENS)
    max_new_tokens = max(64, min(MAX_NEW_TOKENS, max_new_tokens))
    seed = _int_field(data, "seed", None)
    if seed is not None and not 0 <= seed < 2**63:
        raise RequestError("'seed' must be between 0 and 2**63 - 1")

    response_format = str(data.get("response_format", "png")).lower()
    if response_format not in ("png", "json"):
        raise RequestError("'response_format' must be 'png' or 'json'")

    return GenerationRequest(
        prompt=prompt.strip(),
        images=images,
        upsample=upsample,
        boxes=boxes or None,
        steps=steps,
        height=_snap(_int_field(data, "height", None)),
        width=_snap(_int_field(data, "width", None)),
        seed=seed,
        max_new_tokens=max_new_tokens,
        response_format=response_format,
    )


class Flux3Engine:
    """Loads the image pipeline and the prompt upsampler once and runs one request at a time."""

    def __init__(self, weights_dir: Path = WEIGHTS_DIR, device: str = DEVICE):
        self.linear_attention_kernels = prepare_linear_attention()  # before transformers loads Qwen3.5
        import torch
        from diffusers import ModularPipeline
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.torch = torch
        self.device = device
        self.lock = threading.Lock()
        self.weights_dir = Path(weights_dir)
        started = time.time()

        manifest_path = self.weights_dir / STAGE_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} is missing: the weights were not staged by stage_weights.py")
        self.manifest = json.loads(manifest_path.read_text())
        overrides = self.manifest.get("components", {})
        self._verify_files()
        _marker(
            "LOADING",
            weights=str(self.weights_dir),
            repo=self.manifest.get("repo"),
            revision=self.manifest.get("revision"),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )

        # The image pipeline = the repo's default blocks minus the upsampler step; the upsampler runs separately so
        # we can apply BFL's decoding settings and report the caption and its timing.
        blocks = ModularPipeline.from_pretrained(str(self.weights_dir)).blocks
        blocks.sub_blocks.pop("prompt_upsampler")
        self.pipe = blocks.init_pipeline(str(self.weights_dir))
        # One GENERATED line per request instead of tqdm bars. `pipe.blocks` is a deep copy: walk the live `_blocks`.
        _disable_progress_bars(self.pipe._blocks)
        self.pipeline_class = type(self.pipe).__name__

        for name in self.pipe.pretrained_component_names:
            t0 = time.time()
            # Always load from the staged copy, never the Hub: the index's subfolder for components from the main
            # repo, the manifest's subfolder for the ones staged from other repos (the prompt upsampler).
            subfolder = overrides[name]["subfolder"] if name in overrides else self.pipe.get_component_spec(name).subfolder
            self.pipe.load_components(
                names=[name],
                dtype=torch.float32 if name == "vae" else torch.bfloat16,
                pretrained_model_name_or_path=str(self.weights_dir),
                subfolder=subfolder or "",
                revision=None,
            )
            component = getattr(self.pipe, name, None)
            if component is None:  # load_components only logs a warning on failure
                raise RuntimeError(f"component '{name}' failed to load; see the warning above")
            if isinstance(component, torch.nn.Module):
                component.to(device)
            _marker("LOADED", component=name, cls=type(component).__name__, seconds=round(time.time() - t0, 1))
        self.pipe.to(device)

        pup_info = overrides.get("prompt_upsampler", {"subfolder": "prompt_upsampler"})
        pup_dir = self.weights_dir / pup_info["subfolder"]
        t0 = time.time()
        self.pup_processor = AutoProcessor.from_pretrained(str(pup_dir))
        self.pup = Qwen3_5ForConditionalGeneration.from_pretrained(str(pup_dir), dtype=torch.bfloat16)
        self.pup.to(device).eval()
        prompts_file = pup_dir / "system_prompts.json"
        if prompts_file.exists():
            self.system_prompts = json.loads(prompts_file.read_text())
        else:  # the copy Diffusers ships (same text at the commit it pins)
            from diffusers.modular_pipelines.flux3.prompt_upsampler import SYSTEM_PROMPTS

            self.system_prompts = dict(SYSTEM_PROMPTS)
        _marker("LOADED", component="prompt_upsampler", cls=type(self.pup).__name__, seconds=round(time.time() - t0, 1))

        _marker(
            "MODEL_LOADED",
            pipeline=self.pipeline_class,
            seconds=round(time.time() - started, 1),
            linear_attention=self.linear_attention_kernels,
            **self.gpu_memory(),
        )

    # ------------------------------------------------------------------ helpers
    def _verify_files(self) -> None:
        """Every staged file is on disk with its staged size; otherwise fail now with the list, not later with a
        confusing "couldn't connect to huggingface.co" from a half-copied model prefix."""
        files = self.manifest.get("files") or []
        if not files:
            return
        keys = [f["key"] for f in files]
        root = os.path.commonpath(keys) if len(keys) > 1 else os.path.dirname(keys[0])
        problems = []
        for f in files:
            relative = f.get("path") or os.path.relpath(f["key"], root)
            local = self.weights_dir / relative
            size = local.stat().st_size if local.exists() else None
            if size != f["size"]:
                problems.append(f"{relative}: expected {f['size']} bytes, found {size}")
        if problems:
            raise FileNotFoundError(f"{len(problems)} staged files missing or incomplete under {self.weights_dir} "
                                    f"(was the model prefix still being copied?): {problems[:8]}")
        _marker("FILES_VERIFIED", files=len(files), gb=round(sum(f["size"] for f in files) / 1e9, 2))

    def gpu_memory(self) -> dict[str, float]:
        torch = self.torch
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {
            "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "gpu_free_gb": round(free / 1e9, 2),
            "gpu_total_gb": round(total / 1e9, 2),
        }

    def upsample(self, request: str, images: list, boxes: list | None, max_new_tokens: int) -> str:
        """The caption for `request`: diffusers' FLUX 3 upsampler step, with BFL's presence penalty for t2i."""
        torch = self.torch
        from diffusers.modular_pipelines.flux3 import prompt_upsampler as pupmod
        from transformers import LogitsProcessor, LogitsProcessorList

        references = pupmod.prepare_references(images) if images else []
        task = "i2i" if references else "t2i"
        mode = task + ("_bbox" if boxes else "")
        locked = pupmod.boxes_to_canvas(boxes) if boxes else None
        if locked:
            request = pupmod.format_layout_request(request, locked)
        if references:
            scaffold = pupmod.I2I_SCAFFOLD[bool(locked)].format(instruction=request)
            content = [{"type": "image", "image": image} for image in references]
            content.append({"type": "text", "text": scaffold})
        else:
            content = [{"type": "text", "text": request}]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompts[mode]}]},
            {"role": "user", "content": content},
        ]
        inputs = self.pup_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.pup.device)
        prompt_len = inputs["input_ids"].shape[1]

        class PresencePenalty(LogitsProcessor):
            """vLLM-style presence penalty: subtract `penalty` from every token already generated."""

            def __init__(self, penalty: float, start: int):
                self.penalty, self.start = penalty, start

            def __call__(self, input_ids, scores):
                generated = input_ids[:, self.start :]
                if generated.shape[1] == 0:
                    return scores
                seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, generated, True)
                return scores - self.penalty * seen.to(scores.dtype)

        penalty = PRESENCE_PENALTY[task]
        processors = LogitsProcessorList([PresencePenalty(penalty, prompt_len)]) if penalty else None
        with torch.inference_mode():
            generated = self.pup.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, logits_processor=processors
            )
        caption = self.pup_processor.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()
        if not caption:
            raise RuntimeError("the prompt upsampler returned an empty caption")
        return pupmod.reconcile_boxes(caption, locked) if locked else caption

    # ------------------------------------------------------------------ main entry
    def generate(self, req: GenerationRequest) -> dict[str, Any]:
        torch = self.torch
        with self.lock:
            started = time.time()
            seed = req.seed if req.seed is not None else secrets.randbelow(2**31)
            if self.torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            caption = req.prompt
            upsample_s = 0.0
            if req.upsample:
                t0 = time.time()
                caption = self.upsample(req.prompt, req.images, req.boxes, req.max_new_tokens)
                upsample_s = time.time() - t0

            kwargs: dict[str, Any] = {
                "prompt": caption,
                "num_inference_steps": req.steps,
                "generator": torch.Generator().manual_seed(seed),  # CPU noise: reproducible across GPUs
                "output": "images",
                **PIPELINE_EXTRA,
            }
            if req.images:
                kwargs["image"] = req.images[0] if len(req.images) == 1 else req.images
            elif req.height is None and req.width is None:
                kwargs["height"] = kwargs["width"] = 1024
            if req.height is not None:
                kwargs["height"] = req.height
            if req.width is not None:
                kwargs["width"] = req.width

            t0 = time.time()
            with torch.inference_mode():
                image = self.pipe(**kwargs)[0]
            generate_s = time.time() - t0

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            result = {
                "png": buffer.getvalue(),
                "caption": caption,
                "seed": seed,
                "width": image.width,
                "height": image.height,
                "steps": req.steps,
                "upsampled": req.upsample,
                "references": len(req.images),
                "timings_s": {
                    "upsample": round(upsample_s, 2),
                    "generate": round(generate_s, 2),
                    "total": round(time.time() - started, 2),
                },
                "gpu_peak_gb": self.gpu_memory().get("gpu_peak_gb"),
            }
            _marker(
                "GENERATED",
                seed=seed,
                size=f"{image.width}x{image.height}",
                steps=req.steps,
                references=len(req.images),
                upsampled=req.upsample,
                timings_s=result["timings_s"],
                caption_chars=len(caption),
                caption_head=caption[:300],
                **self.gpu_memory(),
            )
            return result

    def warmup(self) -> None:
        """One text-to-image and one edit, both through the upsampler: compiles kernels and proves the full path."""
        from PIL import Image

        t0 = time.time()
        t2i = self.generate(
            GenerationRequest(prompt="a red apple on a wooden table", steps=4, height=512, width=512, seed=0,
                              max_new_tokens=128)
        )
        gradient = Image.linear_gradient("L").resize((512, 512)).convert("RGB")
        self.generate(
            GenerationRequest(prompt="make it blue", images=[gradient], steps=4, seed=0, max_new_tokens=128)
        )
        _marker("WARMUP_DONE", seconds=round(time.time() - t0, 1), t2i_timings_s=t2i["timings_s"])


@serve.deployment
class Flux3Klein:
    """Ray Serve ingress. Options (GPUs, concurrency, health checks) live in /opt/ml/model/config.yaml."""

    def __init__(self, weights_dir: str | None = None, warmup: bool | None = None):
        try:
            self.engine = Flux3Engine(Path(weights_dir) if weights_dir else WEIGHTS_DIR)
            if warmup if warmup is not None else os.environ.get("FLUX3_WARMUP", "1") == "1":
                self.engine.warmup()
        except Exception:
            _marker("FATAL", error=traceback.format_exc()[-4000:])
            raise
        self.fatal: str | None = None
        _marker("READY", pipeline=self.engine.pipeline_class, **self.engine.gpu_memory())

    async def __call__(self, request: Request) -> Response:
        if request.method == "GET":  # local smoke checks; SageMaker only POSTs
            return JSONResponse({"status": "ready", "pipeline": self.engine.pipeline_class})
        try:
            req = parse_request(await request.body())
        except RequestError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            result = await asyncio.to_thread(self.engine.generate, req)
        except RequestError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # noqa: BLE001 - report, keep the replica alive unless CUDA is broken
            message = f"{type(exc).__name__}: {exc}"
            _marker("ERROR", error=traceback.format_exc()[-4000:])
            torch = self.engine.torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            elif "CUDA error" in str(exc) or "device-side assert" in str(exc):
                self.fatal = message  # the CUDA context is gone; check_health makes Ray replace the replica
            return JSONResponse({"error": message}, status_code=500)

        if req.response_format == "json":
            payload = {k: v for k, v in result.items() if k != "png"}
            payload["image"] = base64.b64encode(result["png"]).decode("ascii")
            payload["model"] = "black-forest-labs/flux-3-image-klein"
            return JSONResponse(payload)
        return Response(content=result["png"], media_type="image/png")

    async def check_health(self) -> None:
        if self.fatal:
            raise RuntimeError(f"unrecoverable CUDA failure: {self.fatal}")


app = Flux3Klein.bind()
