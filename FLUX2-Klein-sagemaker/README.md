# FLUX.2 [klein] 9B on SageMaker Async Inference

End-to-end recipe for serving Black Forest Labs' [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) behind an Amazon SageMaker **asynchronous** endpoint that supports both image generation and image manipulation.

The notebook (`flux2-klein-sagemaker.ipynb`) drives the deployment. The `prepare_weights.py` script stages weights into S3 once. The `code/` directory holds the SageMaker source bundle.

---

## Why this layout

Weights and code travel separately:

- **Weights** live as uncompressed objects under an S3 prefix. SageMaker mounts them directly to `/opt/ml/model/`. No tarball, no re-archiving on redeploy.
- **Code** lives in `code/` and is uploaded as a tiny `sourcedir.tar.gz` on every deploy.

Editing the handler and redeploying does not touch the 33 GB of weights. The SDK uploads `sourcedir.tar.gz` (a few KB) and the deploy returns in seconds. Cold start is bounded by EBS throughput (multi-GB/s) instead of Hub bandwidth. The endpoint never holds an `HF_TOKEN` because it never talks to the Hub at runtime.

---

## Memory strategy: bf16 + CPU offload

Why not all-resident on GPU:

- FLUX.2 [klein] 9B is a 9 B flow transformer **plus** an 8 B Qwen3 text encoder. At bf16 that's ~34 GB of model weights.
- The L40S in `ml.g6e.2xlarge` has 48 GB. Adding activations and reference-image latents on top of 34 GB doesn't leave enough headroom to run reliably under load.
- The model card recommends `enable_model_cpu_offload()`. We follow that.

Offload trade-off: on every request, the text encoder, transformer, and VAE are swapped between CPU and GPU through PCIe. Realistic per-request latency on the L40S (PCIe Gen4) is **~7–12 seconds**, dominated by ~30 GB of host↔device transfer at ~6 GB/s sustained. The actual diffusion compute is ~1 second on a fully-resident GPU.

This shape fits async / batch workloads cleanly. If you need sub-second interactive latency, the next step is `bitsandbytes` 8-bit pre-quantized weights (fully GPU-resident), which adds a CUDA-extension dependency on the SageMaker DLC and should be validated on a notebook before committing.

---

## Architecture

```
                                  ┌─────────────────────┐
                                  │ prepare_weights.py  │
                                  │  (one-time)         │
                                  └──────────┬──────────┘
                                             │ snapshot_download
                                             ▼
                                       weights/  (~33 GB diffusers format)
                                             │
                                             │ aws s3 sync
                                             ▼
            ┌──────────────────────────────────────────────────────────┐
            │  s3://<bucket>/flux2-klein/weights/                      │
            │   ├─ model_index.json                                    │
            │   ├─ scheduler/                                          │
            │   ├─ text_encoder/   (8 B Qwen3, bf16)                   │
            │   ├─ tokenizer/                                          │
            │   ├─ transformer/    (9 B flow, bf16)                    │
            │   └─ vae/                                                │
            └──────────────────────────────────────────────────────────┘
                                             │
                                             │ ModelDataSource: S3Prefix, CompressionType=None
                                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│ SageMaker Async Endpoint (ml.g6e.2xlarge, L40S 48 GB Ada sm_89)        │
│  /opt/ml/model/   <- mirrored from S3 prefix verbatim                  │
│  /opt/ml/code/    <- sourcedir.tar.gz (uploaded per deploy)            │
│    └─ inference.py: Flux2KleinPipeline.from_pretrained(model_dir,      │
│                       local_files_only=True).enable_model_cpu_offload()│
└────────────────────────────────────────────────────────────────────────┘
```

---

## Repo layout

```
FLUX2-Klein-sagemaker/
├── flux2-klein-sagemaker.ipynb     # Driver notebook
├── prepare_weights.py              # One-time weight-staging script
├── code/                           # Inference source bundle (per-deploy upload)
│   ├── inference.py
│   └── requirements.txt
└── README.md
```

---

## Prereqs

- A SageMaker execution role with S3 read/write on the default bucket, SNS publish to the topic referenced in the notebook, and `AmazonSageMakerFullAccess`.
- `ml.g6e.2xlarge` quota in the target region. Don't downgrade to `ml.g6.*` (L4, 24 GB) — the offload swap requires enough VRAM for the largest single component (~18 GB transformer + activations).
- A Hugging Face token with the EULA accepted on `black-forest-labs/FLUX.2-klein-9B`. Set `HF_TOKEN` before running the prep script.
- An SNS topic for success / error notifications. Set its ARN as `FLUX2_SNS_TOPIC_ARN` before running the deploy cell.
- About 35 GB of free local disk to materialize the weights before syncing to S3.

---

## One-time setup

From a machine with `HF_TOKEN` exported and the AWS CLI configured (your laptop, the SageMaker notebook instance, anywhere with ~35 GB of free disk):

```bash
pip install hf_transfer

HF_HUB_ENABLE_HF_TRANSFER=1 python prepare_weights.py --out weights

aws s3 sync weights/ s3://<bucket>/flux2-klein/weights/
```

`huggingface_hub` and `torch` are pre-installed on SageMaker notebook images. `hf_transfer` is the only extra dep, and it cuts the ~33 GB Hub download from ~30 minutes to ~5–10. `aws s3 sync` is idempotent — re-running only uploads files that have changed, which makes recovery from a partial run cheap.

To verify the prefix is populated:
```bash
aws s3 ls --recursive s3://<bucket>/flux2-klein/weights/ --human-readable --summarize
```
Expected: ~33 GB across `model_index.json`, `scheduler/`, `text_encoder/`, `tokenizer/`, `transformer/`, `vae/`.

---

## Deploying

1. Open `flux2-klein-sagemaker.ipynb`. Sections 1–2 set up the SDK and configuration.
2. Skip section 3 if the S3 prefix is already populated.
3. Run section 4 to deploy. First deploy takes 8–12 minutes (image pull + `pip install diffusers` from source + ~33 GB EBS download).
4. Run sections 6–8 for text-to-image, single-reference editing, and multi-reference generation demos.

---

## Redeploying after a code change

This is the workflow that's now cheap. When you edit `code/inference.py`:

1. Skip section 3 (weights are unchanged).
2. Re-run section 4. The SDK uploads only `sourcedir.tar.gz` (a few KB).
3. The new endpoint mounts the same weights prefix and uses the new code. Cold start is ~2–3 minutes.

For zero-downtime swaps on a running endpoint, use `update_endpoint` with a new `EndpointConfig` instead of `delete_endpoint` + `deploy`.

---

## Request schema

The endpoint accepts a single JSON object:

| Field | Type | Default | Description |
|---|---|---|---|
| `inputs` | string | required | The prompt. Also accepted as `prompt`. |
| `images` | list of base64 strings | `[]` | Optional reference images for editing / multi-reference generation. PNG or JPEG. Max 4. |
| `num_inference_steps` | int | `4` | Distilled model — 4 is the sweet spot. Clamped to `[1, 20]`. |
| `guidance_scale` | float | `1.0` | Step-distilled — guidance is effectively ignored above 1.0. Clamped to `[0, 10]`. |
| `height` | int | `1024` (text-to-image) / from reference (editing) | Snapped to a multiple of 16 (FLUX.2 VAE constraint). |
| `width` | int | `1024` (text-to-image) / from reference (editing) | Snapped to a multiple of 16 (FLUX.2 VAE constraint). |
| `seed` | int | random | Optional. Pin for reproducible generations. |

Response is `image/png` bytes written to the configured `OutputLocation` in S3.

---

## What's in `code/`

### `inference.py`

| Function | Purpose |
|---|---|
| `model_fn(model_dir)` | `Flux2KleinPipeline.from_pretrained(model_dir, torch_dtype=bf16, local_files_only=True)`, then `enable_model_cpu_offload()`. No Hub access. |
| `input_fn(body, content_type)` | Parses `application/json` from the staged S3 object. |
| `predict_fn(data, model)` | Validates and clamps params, base64-decodes any reference images, snaps height/width to multiples of 16, runs the pipeline, returns PNG bytes. |
| `output_fn(prediction, accept)` | Returns `(bytes, "image/png")`. |

### `requirements.txt`

```
git+https://github.com/huggingface/diffusers
transformers>=4.51.0
accelerate>=0.33.0
sentencepiece
protobuf
safetensors
```

`Flux2KleinPipeline` is on the `main` branch of `diffusers` only. `transformers>=4.51` is mandatory — that's the minimum version that ships `Qwen3ForCausalLM`, and the PyTorch 2.5 DLC ships an older transformers. The container does not need `huggingface_hub` or `hf_transfer` because it never talks to the Hub at runtime.

---

## Troubleshooting

**Deploy hangs or fails silently.** Tail container logs and inspect failure reason:
```bash
aws logs tail /aws/sagemaker/Endpoints/flux2-klein-9b --follow --since 1h
aws sagemaker describe-endpoint --endpoint-name flux2-klein-9b --query FailureReason --output text
```

**`local_files_only=True` raises FileNotFoundError.** The S3 prefix is missing components. Run `aws s3 ls --recursive s3://<bucket>/flux2-klein/weights/` and confirm `model_index.json`, `scheduler/`, `text_encoder/`, `tokenizer/`, `transformer/`, and `vae/` are all present.

**OOM on first invocation.** Confirm the instance is `ml.g6e.2xlarge` (L40S, 48 GB) and not a smaller variant. The largest single offloaded component is the ~18 GB transformer plus activations.

**Per-request latency feels high.** Expected. Offload swaps ~30 GB of weights through PCIe per request. If you need sub-2-second response times, see "Memory strategy" above for the bnb 8-bit upgrade path.

---

## Security notes

- The model is gated under a non-commercial license. Do not deploy commercially without a separate license from BFL.
- BFL ships [inference filters for NSFW and protected content](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B); the FLUX Non-Commercial License requires using filters or manual review. Add those checks before exposing the endpoint to end users.
- The endpoint container holds no `HF_TOKEN`. Anyone with `sagemaker:DescribeEndpointConfig` cannot harvest a Hub credential from the env.

---

## AI Photo Booth app (`deploy.sh`)

The repo also contains the AI Photo Booth kiosk app (`ui/`, `backend/`, `cdk/`). `deploy.sh` builds and deploys the `AiPhotoBoothStack`, regenerates `ui/.env.local` from the stack outputs, and sets permanent passwords for the initial admin and visitor users.

### Initial user passwords are required parameters

The script has **no built-in password defaults** — no credentials are stored in this repo. You must supply both the admin and visitor passwords on every run, either as flags or environment variables:

```bash
# As flags
./deploy.sh --admin-password 'S3cret!Admin' --visitor-password 'S3cret!Visit'

# Or as environment variables
ADMIN_PASSWORD='S3cret!Admin' VISITOR_PASSWORD='S3cret!Visit' ./deploy.sh
```

The script exits with an error if either password is missing.

### Password policy

The Cognito user pool enforces a password policy on all users. Each password must:

- be at least **8 characters** long,
- contain at least one **uppercase** letter,
- contain at least one **lowercase** letter, and
- contain at least one **special character** (symbol).

Passwords that don't meet this policy are rejected when the script sets them, so choose values that satisfy all four rules.

### Other deploy options

Every other value has a default and can be overridden by the matching flag or environment variable (e.g. `--region`, `--sender-email`, `--endpoint-name`, `--admin-username`, `--scheduler-timezone`). Run `./deploy.sh --help` for the full list.

On a re-deploy where the users already have their passwords, pass `--skip-passwords` instead of the two password flags. The script then skips the password step and nobody's password changes.

### FLUX 3 Image Klein endpoint (ml.g7e.2xlarge)

`deploy.sh` also prepares a second endpoint config, `FluxAiBoot-flux3-g7e`, that serves [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) on an `ml.g7e.2xlarge` (RTX PRO 6000 Blackwell, 96 GB). The serving code lives in [`../FLUX3-Klein-sagemaker-g7e`](../FLUX3-Klein-sagemaker-g7e), which is also a standalone notebook for testing the model without the booth.

Step 5 of the script does the following, and every part is idempotent:

1. **Hugging Face token.** The FLUX 3 weights are gated, so the first run asks for a Hugging Face read token (hidden input) from an account that has accepted the license. The script checks that the token can read the repo, then stores it in Secrets Manager as `AmazonSageMaker-flux3-klein-hf-token`. Later runs reuse the stored token and don't ask again. Pass `--hf-token <token>` (or `HF_TOKEN=...`) to run non-interactively or to replace the stored token. The token is passed to the AWS CLI and curl through temp files, never on a command line.
2. **Serving code.** Uploads `serving/config.yaml` and `serving/code/` to `s3://<io-bucket>/flux3-klein-g7e/model/`.
3. **Weights.** If `flux3-klein-g7e/model/weights/_flux3_stage.json` is missing, a CPU SageMaker Processing job copies the ~39 GB of weights from the Hub to S3 (about 10 minutes). Only this job reads the token. The endpoint loads the weights from S3 and holds no Hub credential.
4. **Model and endpoint config.** Creates the SageMaker Model and EndpointConfig `FluxAiBoot-flux3-g7e` (AWS Ray Serve DLC `ray:serve-ml-sagemaker-cuda-v1.5.7`, inference AMI `al2023-ami-sagemaker-inference-gpu-4-1`, the booth's usual async I/O prefixes). The CDK stack creates the execution role, `Flux3SageMakerRole`. If an endpoint named `FluxAiBoot-flux3-g7e` is running, the model and config are left alone; stop the endpoint and re-run to apply changes.
5. **Booth registration.** Adds the config to the booth's managed endpoint configs. The current config is not changed.

The script does **not** start a g7e instance. To use FLUX 3, sign in as admin, open **Endpoint**, click **Make current** on `FluxAiBoot-flux3-g7e`, then **Start**. Startup takes about 5 minutes, and g7e capacity can be limited, so a start may need to be retried. Switch back by making the FLUX.2 config current again.

Related flags: `--hf-token`, `--flux3-config-name`, `--flux3-instance-type` and `--skip-flux3` (skips step 5 entirely).

### Endpoints in other regions

g7e capacity differs by region: in November 2026, us-east-1 failed every start with `InsufficientInstanceCapacity` while us-east-2 had an instance within 5 minutes. The booth itself (sign-in, schedule, prompts, UI) stays in its region, but the admin can run the endpoint in another one:

```bash
./deploy.sh --skip-passwords --endpoint-regions us-east-2
```

For each extra region, `deploy.sh`:

1. uses that region's default SageMaker bucket, `sagemaker-<region>-<account>`, and creates it (private) if it doesn't exist;
2. passes the region to the stack, which grants the booth roles, the invoke proxy, the scheduler and the FLUX 3 role access to SageMaker in that region and to the booth's prefixes in that bucket, and sets the bucket's CORS for the booth's CloudFront domain;
3. copies the staged FLUX 3 weights from the home bucket (S3 to S3, about 39 GB, once), uploads the serving code, and creates the same `FluxAiBoot-flux3-g7e` model and endpoint config there;
4. adds that config to the booth's managed list as a separate row.

In **Admin → Endpoint**, a region selector appears next to the config picker, and each row shows its region. Start, Stop and status act on the endpoint in the row's region. **Make current** records the region with the current config, so the booth sends requests to that region's bucket and the invoke proxy and the scheduler drive the endpoint there. A config with the same name in two regions is two rows; make the one with capacity current and start it.

A plain re-deploy keeps the regions the stack already has. `--endpoint-regions none` removes them; managed rows for a removed region then stop working until the region is added back. Each endpoint keeps its requests and results in its own region's bucket, and the browser uploads to and polls that bucket directly.

**How the booth picks FLUX 3.** The booth reads the current config name before each generation. A name that matches `flux3` or `flux-3` (case-insensitive) selects the FLUX 3 prompts and sampling (16 steps, guidance 1); any other name keeps the FLUX.2 behaviour. A custom `--flux3-config-name` must therefore contain `flux3`, and the script rejects one that doesn't. The Endpoint tab shows which model family each config runs.

**Prompts per model family.** Each effect has a FLUX.2 prompt and a FLUX 3 prompt. The FLUX 3 prompts follow Black Forest Labs' FLUX 3 editing guidance: one positive instruction that names the change, then what must stay (faces, pose, position, lighting), with the image roles stated explicitly in the two-image merge. The endpoint's prompt upsampler expands them into a full caption, so they are shorter than the FLUX.2 prompts. Prompt edits in the admin view are saved for the model family of the current config, so a FLUX.2 custom prompt and a FLUX 3 custom prompt for the same effect don't overwrite each other.

### Generated local config

`deploy.sh` writes `ui/.env.local` with account-specific resource ids for local development (`npm run dev`). This file is **gitignored** and must never be committed. Python bytecode caches (`__pycache__/`) and staged model `weights/` are gitignored as well.

---

## Cleanup

```python
predictor.delete_endpoint()
sagemaker.Session().delete_model(model.name)
```

S3 objects under `flux2-klein-inputs/`, `flux2-klein-outputs/`, `flux2-klein-failures/` are not deleted automatically. Add a lifecycle rule on the default SageMaker bucket if you want them aged out. The `flux2-klein/weights/` prefix should be kept for as long as you might redeploy.
