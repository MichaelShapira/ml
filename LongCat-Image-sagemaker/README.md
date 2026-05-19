# LongCat-Image on SageMaker Async Inference

End-to-end recipe for serving Meituan's [LongCat-Image](https://huggingface.co/meituan-longcat/LongCat-Image) text-to-image diffusion model behind an Amazon SageMaker **asynchronous** endpoint.

The notebook (`longcat-sagemaker.ipynb`) walks through packaging, deployment, invocation, and rendering. The `code/` folder holds the source bundle that SageMaker ships into the inference container.

---

## Why async inference?

LongCat-Image is a diffusion model. A single 1344×768 generation at 40 steps takes ~25–30 seconds on an L40S GPU. SageMaker real-time endpoints have a **60-second hard request timeout** and are billed for idle GPU time. Async endpoints fit this workload because:

- Requests are queued internally; clients don't hold a connection open.
- Inputs and outputs flow through S3, which is the natural staging layer for batches and downstream consumers.
- SNS notifications fan out completion events to Lambda, SQS, or Slack without polling.
- Auto-scaling can drop the fleet to zero instances when the queue is empty, which is impossible on real-time endpoints.

---

## Architecture

```
┌──────────────┐    PUT JSON     ┌──────┐
│ Client / NB  │ ───────────────▶│  S3  │  longcat-inputs/
└──────────────┘                 └──────┘
       │                              │
       │ invoke_endpoint_async        │ InputLocation = s3://...
       ▼                              ▼
┌────────────────────────────────────────────┐
│ SageMaker Async Endpoint (ml.g6e.2xlarge)  │
│  ├─ PyTorch DLC 2.5 / py311                │
│  ├─ source_dir=code/  entry=inference.py   │
│  └─ Hugging Face cache on container disk   │
└────────────────────────────────────────────┘
       │ writes PNG bytes
       ▼
┌──────┐    SNS    ┌────────────────┐
│  S3  │ ────────▶ │ Success/Error  │
└──────┘            │ subscribers    │
longcat-outputs/    └────────────────┘
```

---

## Repo Layout

```
LongCat-Image-sagemaker/
├── longcat-sagemaker.ipynb     # Driver notebook (deploy + invoke + render)
└── code/                       # Inference source bundle (uploaded by SageMaker)
    ├── inference.py            # SageMaker handler: model_fn / input_fn / predict_fn / output_fn
    ├── requirements.txt        # Extra Python deps installed at container start
    └── sitecustomize.py        # Startup-time monkey-patch for diffusers typing bug
```

The notebook deploys with:

```python
PyTorchModel(
    model_data=s3_path,           # empty model.tar.gz placeholder
    entry_point="inference.py",
    source_dir="code",
    framework_version="2.5",
    py_version="py311",
    env={"HF_TOKEN": "..."},
)
```

`source_dir="code"` is the key line. SageMaker tarballs the entire folder, uploads it alongside the (empty) `model.tar.gz`, extracts it inside the container, and prepends it to `sys.path`.

---

## Why the files under `code/` exist

### `inference.py` — the SageMaker handler (required)

SageMaker's PyTorch inference toolkit looks for four functions in the entry-point script:

| Function | Purpose |
|---|---|
| `model_fn(model_dir)` | Called once at container start. Loads `LongCatImagePipeline` from the `meituan-longcat/LongCat-Image` Hub repo using the `HF_TOKEN` env var, casts to `bfloat16`, and moves the pipeline to CUDA. |
| `input_fn(body, content_type)` | Parses the request. Only `application/json` is accepted; the body comes from the S3 object referenced by `InputLocation`. |
| `predict_fn(data, model)` | Validates and clamps `guidance_scale`, `num_inference_steps`, `height`, and `width`, snaps height/width to multiples of 32 (a hard constraint of the LongCat pipeline), runs inference with `enable_cfg_renorm=True` and `enable_prompt_rewrite=True`, and returns PNG bytes. |
| `output_fn(prediction, accept)` | Returns `(bytes, "image/png")`. SageMaker writes the bytes verbatim to the configured `output_path` in S3. |

Without this file SageMaker has nothing to call. This is the single most important file in the bundle.

### `requirements.txt` — extra Python dependencies (required)

The base PyTorch DLC ships with PyTorch and a stable `transformers`, but **not** with the LongCat pipeline class. The pipeline only exists on the `main` branch of `huggingface/diffusers`, so the file pulls it directly from GitHub:

```
git+https://github.com/huggingface/diffusers
transformers>=4.40.0
accelerate
sentencepiece
```

The SageMaker toolkit detects `requirements.txt` inside `source_dir` and runs `pip install -r requirements.txt` before invoking `model_fn`. Without it the import in `inference.py` fails with `ImportError: cannot import name 'LongCatImagePipeline'`.

### `sitecustomize.py` — startup-time compatibility patch (required for cold starts)

CPython imports `sitecustomize` automatically if it can be found on `sys.path`. SageMaker puts `code/` on `sys.path`, so this module runs **before** any user import.

The current `diffusers` commit pinned by `requirements.txt` uses PEP 604 string-form annotations (`'torch.Tensor | None'`) inside `models/attention_dispatch.py`. Those annotations resolve at decoration time and break on the Python build inside the DLC. The patch rewrites them to `Optional[...]` / `Tuple[...]` and inserts the matching `typing` import.

It guards itself with a `patched_by_sitecustomize` sentinel so re-runs are no-ops. Once upstream `diffusers` fixes the file, this module can be deleted.

> A running endpoint may *appear* to work without this file because the patch is already on disk inside the live container. The next cold start (auto-scale event, redeploy, or instance replacement) will fail without it.


---

## How a request flows

1. **Client uploads JSON to S3** (`longcat-inputs/request-<epoch>.json`) with `inputs`, `negative_prompt`, `guidance_scale`, `num_inference_steps`, `height`, `width`.
2. **`invoke_endpoint_async`** returns immediately with an `OutputLocation` pointing at `longcat-outputs/<id>.out`.
3. **Container reads the input** from S3, calls `predict_fn`, returns PNG bytes.
4. **SageMaker writes the bytes** to `OutputLocation` and publishes to the configured SNS `SuccessTopic`.
5. **Client `GetObject`s** the result (or, in production, reacts to the SNS event) and decodes it with PIL.

---

## Running the notebook

Prereqs:

- A SageMaker execution role with S3 read/write on the default bucket, SNS publish to the topics referenced in the notebook, and the standard `AmazonSageMakerFullAccess` policy.
- `ml.g6e.2xlarge` quota in the target region (request via Service Quotas if the default is 0).
- A Hugging Face token with read access to `meituan-longcat/LongCat-Image`. **Move this out of the notebook before sharing the repo** — see the security note below.

Steps:

1. Open `longcat-sagemaker.ipynb` in SageMaker Studio (or any environment with credentials for the role).
2. Execute cells 1–8 to deploy the endpoint. First deploy takes 8–12 minutes (image pull + dependency install + model download from the Hub).
3. Execute cells 9–16 to send a prompt and render the result inline.
4. When done, delete the endpoint:
   ```python
   predictor.delete_endpoint()
   ```

---

## Production: store model weights in S3, not on the Hugging Face Hub

The notebook ships an **empty** `model.tar.gz` and lets the container download `meituan-longcat/LongCat-Image` from the Hub on first invocation, authenticated by `HF_TOKEN`. That's fine for a demo. For production, mirror the weights into S3 and point `model_data` at the real archive. Reasons:

### 1. Cold-start latency and predictability
A first-time download from the Hub is ~30 GB over the public internet. That can be 5–10 minutes per cold start, gated by Hub bandwidth and rate limits. S3-to-EC2 in the same region is multi-GB/s and bounded by the instance's EBS throughput. Cold starts drop from minutes to seconds, which matters for auto-scaling, blue/green deploys, and SLA commitments.

### 2. Reproducibility and immutability
A Hub repo can change. A new commit, a removed file, a renamed weight key, or a license change will silently affect every new container that pulls `from_pretrained(...)`. An S3 object with versioning enabled is content-addressable: the same `model_data` URI yields the same bytes forever.

### 3. No external dependency on the inference path
Pulling weights from the Hub at startup couples your endpoint's availability to a third party. If the Hub is degraded, your endpoint cannot scale up or recover from instance replacement. Hosting the weights in S3 keeps the failure domain inside AWS.

### 4. Security and least privilege
Embedding `HF_TOKEN` in the endpoint environment means anyone who can describe the endpoint config can read the token. With weights in S3 you delete the token from the runtime, scope the execution role to `s3:GetObject` on a single prefix, and apply the bucket policy controls (KMS encryption, VPC endpoints, access logging) you already use for the rest of your data.

### 5. Cost
Hub egress is free today, but you're paying for the GPU instance to sit idle while it downloads. A 10-minute cold start on `ml.g6e.2xlarge` at on-demand pricing is real money per scale-out event. S3 GETs in the same region are essentially free in comparison.

### 6. Network controls and compliance
Many production accounts disallow outbound internet from inference subnets. With S3 + a VPC gateway endpoint the entire model pull stays on the AWS backbone, satisfying VPC-only and FedRAMP-style requirements without poking holes in egress controls.

### How to switch

1. Snapshot the Hub repo once into S3:
   ```bash
   pip install huggingface_hub
   python - <<'PY'
   from huggingface_hub import snapshot_download
   snapshot_download(
       "meituan-longcat/LongCat-Image",
       local_dir="model",
       local_dir_use_symlinks=False,
       token="hf_...",
   )
   PY
   tar -czf model.tar.gz -C model .
   aws s3 cp model.tar.gz s3://<bucket>/longcat/model.tar.gz
   ```
2. Update `inference.py` so `model_fn` loads from `model_dir` (which SageMaker extracts from `model.tar.gz`) instead of the Hub repo ID:
   ```python
   pipe = LongCatImagePipeline.from_pretrained(
       model_dir,
       torch_dtype=torch.bfloat16,
       local_files_only=True,
   )
   ```
3. Drop `HF_TOKEN` from the `env` dict and point `model_data` at the new S3 URI.

---

## Security note

The notebook currently embeds an `HF_TOKEN` literal in cell 8 and a hardcoded SNS topic ARN. Before committing or sharing:

- Rotate the token at https://huggingface.co/settings/tokens.
- Source it from Secrets Manager or SSM Parameter Store at deploy time, or remove it entirely once you switch to S3-hosted weights (see above).
- Move the SNS topic ARN to a parameter or environment variable so the notebook isn't account-specific.

---

## Cleanup

```python
predictor.delete_endpoint()             # stop GPU billing
sagemaker.Session().delete_model(model.name)
```

S3 objects under `longcat-inputs/`, `longcat-outputs/`, `longcat-failures/`, and `longcat-dummy/` are not deleted automatically. Add a lifecycle rule on the default SageMaker bucket if you want them aged out.
