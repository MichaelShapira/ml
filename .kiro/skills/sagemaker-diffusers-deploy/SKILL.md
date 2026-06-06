---
name: sagemaker-diffusers-deploy
description: Deploy Hugging Face diffusers image models (FLUX.2, LongCat, SDXL, etc.) on Amazon SageMaker async endpoints. Activate whenever the user wants to host, deploy, or serve a diffusers / text-to-image / image-editing model on SageMaker, mentions a SageMaker PyTorch DLC, hits container startup / model_fn / TorchServe worker crashes, sees "Backend worker process died", "default_model_fn", "Exactly one .pth or .pt file is required", dependency/version errors like "Could not import module Qwen3ForCausalLM" or "module 'torch' has no attribute 'float8_e8m0fnu'", or asks about uncompressed S3 model artifacts, separating code from weights, or CPU offload vs quantization on a g6e/g5/g6 instance.
---

# Deploying diffusers image models on SageMaker async endpoints

Goal: stand up a SageMaker **asynchronous** endpoint serving a Hugging Face `diffusers` image model (text-to-image + image editing) without burning hours on the recurring traps. This skill encodes hard-won fixes — especially the **dependency version triangle** and the **code/weights packaging** that the SageMaker PyTorch DLC gets wrong by default.

Reference implementation: `ml/FLUX2-Klein-sagemaker/` (FLUX.2 [klein] 9B). Sister example: `ml/LongCat-Image-sagemaker/`.

## Mental model

Three moving parts have to agree, and the defaults do NOT make them agree:

1. **The container** (SageMaker PyTorch DLC) ships a fixed, often-old set of `torch` + `transformers` + `diffusers`.
2. **The model's pipeline class** lives in a specific `diffusers` version and imports specific `transformers` classes.
3. **The packaging** (how code and weights reach `/opt/ml/model`) decides whether your `inference.py` is even loaded.

Most failures are one of these three disagreeing silently. Diagnose by reading CloudWatch, not by guessing.

---

## TRAP 1 — The dependency version triangle (the big one)

The single most time-consuming class of failure. Image pipelines pull a chain: `diffusers` → a pipeline module → `transformers` classes → `torch` dtypes/ops. Each link has a version window, and the DLC's preinstalled versions usually sit outside it.

### Symptoms and their exact causes

| CloudWatch error | Root cause | Fix |
|---|---|---|
| `Could not import module 'Qwen3ForCausalLM'` (or any new model class) | `transformers` is **too old** — the class doesn't exist yet | Pin `transformers` to a version new enough to define the class |
| `module 'torch' has no attribute 'float8_e8m0fnu'` (raised from `transformers/integrations/finegrained_fp8.py`) | `transformers` is **too new** — it references a torch dtype the DLC's torch lacks | Add an **upper bound**: `transformers<5.0` (or a specific 4.x) |
| `ImportError: cannot import name 'XPipeline'` | `diffusers` is **too old** / wrong branch | Pin the `diffusers` release that ships the pipeline, or `git+https://...` if unreleased |
| `cannot import name ...` from a sub-dep | transitive pin drift | Pin the offending package explicitly |

### The rule: ALWAYS use bounded version ranges

An unbounded `transformers>=4.51.0` is a landmine. `pip install --upgrade` will happily pull `transformers==5.x`, which then crashes against the DLC's older `torch`. Every pin needs both ends:

```
transformers>=4.51.0,<5.0        # has the class, but not the torch-2.7-only code
```

For the FLUX.2 [klein] case specifically, the verified-good combination on the **PyTorch 2.5 DLC (torch 2.5, py311)** is:

```
diffusers==0.38.0          # first release shipping Flux2KleinPipeline
transformers==4.56.2       # has Qwen3ForCausalLM, does NOT reference torch.float8_e8m0fnu
accelerate>=0.33.0
sentencepiece
protobuf
safetensors
```

### How to verify a pin BEFORE deploying (saves 15-min deploy cycles)

Don't guess — inspect the wheel locally. Confirm the class exists and the toxic symbol doesn't:

```bash
# Does diffusers X.Y.Z ship the pipeline?
curl -sL https://pypi.org/pypi/diffusers/0.38.0/json | \
  python3 -c "import sys,json; print('flux2' in str(json.load(sys.stdin)))"

# Download a transformers wheel and grep it
python3 - <<'PY'
import zipfile, urllib.request
url = "https://files.pythonhosted.org/.../transformers-4.56.2-py3-none-any.whl"
urllib.request.urlretrieve(url, "/tmp/tf.whl")
z = zipfile.ZipFile("/tmp/tf.whl")
src = "".join(z.read(n).decode("utf-8","ignore") for n in z.namelist() if n.endswith(".py"))
print("has class:", "class Qwen3ForCausalLM" in src)
print("references torch-2.7 dtype:", "float8_e8m0fnu" in src)  # must be False
PY
```

### Belt-and-suspenders: enforce the pin inside `model_fn`

The SageMaker toolkit's `pip install -r requirements.txt` runs **without `--upgrade`**, so a preinstalled DLC version can win even when requirements.txt asks for something else. Don't trust it. Add a guard at the top of `model_fn` that checks the actual installed version against a bounded range and re-pins if it's outside:

```python
def _ensure_dependencies():
    from importlib.metadata import version as v
    from packaging.version import parse
    import subprocess, sys
    TARGET, LO, HI = "transformers==4.56.2", parse("4.51.0"), parse("5.0.0")
    cur = parse(v("transformers"))
    if cur < LO or cur >= HI:
        subprocess.check_call([sys.executable, "-m", "pip", "install", TARGET])
        for m in [m for m in list(sys.modules) if m.startswith("transformers")]:
            del sys.modules[m]   # purge stale submodules so the new version resolves
    from transformers import Qwen3ForCausalLM  # noqa — force the lazy import to resolve NOW
```

Key subtlety: `from diffusers import XPipeline` **succeeds even with the wrong transformers** because diffusers imports the pipeline module lazily. The real failure only fires when the pipeline is constructed. So check the transformers **version directly**; don't wrap the diffusers import in try/except and call it done.

---

## TRAP 2 — Code vs weights packaging (the silent fallback)

### Symptom
```
default_pytorch_inference_handler.py ... default_model_fn
ValueError: Exactly one .pth or .pt file is required for PyTorch models: []
```
`default_model_fn` is the **built-in** handler. If you see it, your `inference.py` was never loaded — the container couldn't find it and fell back to the default, which looks for a `.pth` file and dies.

### Why it happens
Passing `entry_point` + `source_dir` together with a **dict `model_data`** (uncompressed S3 prefix) silently breaks. The SDK's repack path hits:
```python
if isinstance(self.model_data, dict):
    logging.warning("ModelDataSource currently doesn't support model repacking")
    return   # <-- code never gets wired up
```

### The fix: put code INSIDE the uncompressed weights prefix
For uncompressed-prefix model data, the container hardcodes (verified in `sagemaker_inference/environment.py`):
```python
code_dir = os.path.join(model_dir, "code")   # /opt/ml/model/code
```
It reads the handler from `/opt/ml/model/code/inference.py` and installs `/opt/ml/model/code/requirements.txt`. So:

- Upload `inference.py` + `requirements.txt` to `s3://<bucket>/<prefix>/code/`.
- Do **NOT** pass `entry_point` / `source_dir` to `PyTorchModel`.
- Set env `SAGEMAKER_PROGRAM=inference.py` and `SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code` (explicit intent; the container hardcodes the dir anyway).
- `model_data = {"S3DataSource": {"S3Uri": prefix, "S3DataType": "S3Prefix", "CompressionType": "None"}}`.

### Why this layout (not a tarball)
Weights as **uncompressed objects** under a prefix means SageMaker mirrors them to `/opt/ml/model/` verbatim — no tar/gz on either side. Editing `inference.py` is then a tiny `aws s3 cp code/inference.py s3://.../code/inference.py`; you never re-archive 30+ GB of weights. Redeploy = re-upload the small code file + recreate the endpoint.

---

## TRAP 3 — Cold-start timeouts and silent failures

### Symptoms
- Deploy sits in `Creating` 20-40 min then `Failed` with **no endpoint log group** at all.
- A log group missing = the container never booted; failure was in the managed bootstrap (download / pip / capacity), before TorchServe.

### Causes and fixes
- **From-source builds are slow + fragile.** `git+https://github.com/huggingface/diffusers` can take 6-8 min and hang on pip backtracking. **Pin a release wheel** the moment the pipeline lands in one (`diffusers==0.38.0` for FLUX.2 klein). This was the difference between a wedged deploy and a working one.
- **Timeouts too low.** Set both generously:
  ```python
  predictor = model.deploy(...,
      container_startup_health_check_timeout=3600,   # max
      model_data_download_timeout=3600,
  )
  ```
- **Stale endpoint.** Redeploying onto an existing (crash-looping) endpoint name does NOT reliably replace the model/config. Delete first:
  ```python
  sm.delete_endpoint(EndpointName=name); sm.delete_endpoint_config(EndpointConfigName=name)
  ```

### Always check status, not just logs
```bash
aws sagemaker describe-endpoint --endpoint-name <name> \
  --query '{Status:EndpointStatus,Reason:FailureReason}' --output text
aws logs tail /aws/sagemaker/Endpoints/<name> --since 30m --format short
```
`Creating` = still working. `Failed` + `FailureReason` = the real answer. Missing log group = container never booted.

### Read truncated tracebacks correctly
diffusers/transformers lazy-import wrappers print a generic `RuntimeError: Failed to import ...` and the **real cause is the line AFTER** `look up to see its traceback):`. Grep for the line after, not the wrapper:
```bash
aws logs tail /aws/sagemaker/Endpoints/<name> --since 30m --format short | \
  grep -A2 "look up to see its traceback" | grep -ivE "raise RuntimeError|look up|^--"
```

---

## TRAP 4 — Memory strategy (offload vs quantization vs FP8)

For a 9B-class model + large text encoder (~34 GB bf16) on an L40S (g6e.2xlarge, 48 GB, Ada sm_89):

- **bf16 + `enable_model_cpu_offload()`** — model-card default, zero extra deps, always works. BUT it's **per-request, not one-time**: ~30 GB of PCIe transfer per request. Realistic latency **~7-12 s/request** (not 4 — host↔device for many small unpinned tensors runs ~6 GB/s, and the offload hooks don't overlap transfers). Cold start is slightly faster (no initial GPU move). Right for async/batch; sluggish for interactive.
- **`enable_model_cpu_offload()` and `pipe.to("cuda")` are mutually exclusive** — calling `.to()` after enabling offload breaks the hooks.
- **bnb 8-bit (`bitsandbytes`)** — fully GPU-resident, ~1-2 s/request. BUT `bitsandbytes` is a CUDA C-extension; it must match the DLC's CUDA + torch. **Verify `import bitsandbytes` works on the DLC before committing** — it's the #1 silent dependency break. Pre-quantize in the prep script and `save_pretrained` to avoid runtime quant cost.
- **Vendor "FP8" single-file checkpoints (e.g. BFL FLUX.2 FP8)** are NOT a drop-in for stock diffusers. They target the vendor repo / ComfyUI. `from_single_file(..., torch_dtype=bf16)` upcasts them to bf16 (no size win), and stock diffusers has no FP8 compute path. Also check the GPU arch claim: the FLUX.2 `9b-kv-fp8` card says "RTX 5090+" (Blackwell sm_120) — won't run on L40S (Ada). Don't assume FP8 = smaller-and-faster on arbitrary hardware.

Decision: start with **bf16 + offload** (certain to work, measure real latency), upgrade to **bnb 8-bit** only if latency hurts and only after verifying the bnb import on the DLC.

---

## Inference handler contract (`inference.py`)

Four functions the SageMaker PyTorch toolkit calls:
- `model_fn(model_dir)` — load once. Call `_ensure_dependencies()` FIRST, then import the pipeline, then `from_pretrained(model_dir, torch_dtype=torch.bfloat16, local_files_only=True)` (weights are local; no Hub, no `HF_TOKEN` in the container), then `enable_model_cpu_offload()`.
- `input_fn(body, content_type)` — parse `application/json`.
- `predict_fn(data, model)` — validate/clamp params, decode base64 reference images, run the pipeline, return PNG bytes.
- `output_fn(prediction, accept)` — return `(bytes, "image/png")`.

Notes:
- Snap height/width to the model's real constraint (FLUX.2: multiple of **16** — vae_scale_factor 8 × 2×2 patch packing; not 32).
- Unified pipelines (`Flux2KleinPipeline`) do text-to-image with just `prompt`, and editing / multi-reference when called with `image=[PIL, ...]`.
- Distilled models (FLUX.2 klein): defaults are `num_inference_steps=4`, `guidance_scale=1.0`.
- Log timing at each stage (`Loading pipeline...`, `Pipeline loaded (Ns)`, `CPU offload enabled`) so a slow/stuck `model_fn` is visible in CloudWatch.

---

## Client-side image handling (notebook)

- Async endpoints don't take inline payloads: PUT the JSON to S3, `invoke_endpoint_async(InputLocation=...)`, poll the `OutputLocation` for the PNG.
- Re-encode any local image to **PNG for transport** (lossless) regardless of source format — server stays format-agnostic.
- **WebP input**: guard with `PIL.features.check("webp")` (clear message if Pillow lacks WebP); take frame 0 of animated WebP (`is_animated`); flatten alpha (RGBA/LA/palette-transparency) onto white before `convert("RGB")`.

---

## Security / hygiene

- **Never** put `HF_TOKEN` in the endpoint `env` once weights are local — it's both useless (no Hub access at runtime) and leakable via `DescribeEndpointConfig`. The token belongs only on the machine running the one-time weight prep.
- Scan notebooks for committed tokens before sharing; rotate any that leaked.
- Gated / non-commercial licenses (FLUX): respect them; wire in the vendor's NSFW/content filters before exposing an endpoint.

---

## The fast path (summary checklist)

1. **Prep weights once**: `snapshot_download` the bf16 repo locally → `aws s3 sync` to `s3://<bucket>/<prefix>/`. (No `HF_TOKEN` after this.)
2. **Pin a release wheel** for the pipeline (`diffusers==X`), and a **bounded** `transformers>=A,<B` verified to have the class but not the toxic torch symbol.
3. **Upload code into the prefix**: `aws s3 cp code/{inference.py,requirements.txt} s3://<bucket>/<prefix>/code/`.
4. **Deploy** with dict `model_data` (uncompressed prefix), NO `entry_point`/`source_dir`, 3600s timeouts, delete any existing endpoint first.
5. **Watch the right things**: `describe-endpoint` status; log group appearing = container booted; then `Qwen3ForCausalLM import confirmed` → `Pipeline loaded` → `CPU offload enabled` → `InService`.
6. Iterate on `inference.py` by re-uploading just the one S3 object + redeploying. Never re-archive weights.
