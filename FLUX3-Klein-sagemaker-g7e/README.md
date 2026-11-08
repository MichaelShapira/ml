# FLUX 3 Image Klein on SageMaker (ml.g7e.2xlarge)

Black Forest Labs' [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein), the
step-distilled 5.6B checkpoint, plus its [prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup),
served from one SageMaker **asynchronous** endpoint on an `ml.g7e.2xlarge` (1x NVIDIA RTX PRO 6000 Blackwell,
96 GB). The endpoint handles text-to-image, editing, multi-reference composition (up to 10 images) and bounding-box
layouts. Everything runs in BF16 (the VAE in FP32), fully resident on the GPU: about 39 GB of weights, with no CPU
offload and no quantization.

This folder has two uses:

- **Standalone test.** `flux3-klein-g7e-standalone-test.ipynb` stages the weights, deploys an endpoint, runs
  text-to-image, editing, composition and layout requests, and deletes the endpoint. `flux3_sm.py` holds the boto3
  helpers it calls. It needs nothing from the photo booth.
- **Serving code for the AI Photo Booth.** [`../FLUX2-Klein-sagemaker/deploy.sh`](../FLUX2-Klein-sagemaker/deploy.sh)
  uploads `serving/` and runs `scripts/stage_weights.py` to create the booth's `FluxAiBoot-flux3-g7e` endpoint
  config. See [Use in the AI Photo Booth](#use-in-the-ai-photo-booth).

## Why this stack

| Layer | Choice | Reason |
|---|---|---|
| Container | AWS Ray Serve DLC for SageMaker, `763104351884.dkr.ecr.<region>.amazonaws.com/ray:serve-ml-sagemaker-cuda-v1.5.7` | Blackwell (sm_120) needs CUDA 12.8+ kernels. The SageMaker PyTorch inference DLC stops at PyTorch 2.6 / CUDA 12.4, and AWS now points TorchServe users to the Ray Serve DLC. Hugging Face's SageMaker DLCs have no CUDA 12.8+ build either. The Ray DLC ships PyTorch 2.10 (CUDA 12.8 wheels), the CUDA 12.9.1 runtime, Transformers 5.19, huggingface-hub 1.33 and Python 3.13. Its entrypoint installs `code/requirements.txt`, runs `serve run /opt/ml/model/config.yaml` and adapts SageMaker's `/ping` and `/invocations`. No image to build. |
| Host AMI | `InferenceAmiVersion=al2023-ami-sagemaker-inference-gpu-4-1` | NVIDIA driver 580 / CUDA 13.0. The Ray DLC docs require an explicit inference AMI for GPU endpoints. |
| Pipeline | Diffusers 0.42.1 `ModularPipeline` (`Flux3DistilledModularPipeline`) | Hugging Face's FLUX 3 integration, written with BFL. 0.42.0 is the first release that has it. |
| Upsampler | Transformers `Qwen3_5ForConditionalGeneration` + flash-linear-attention 0.6.0 | Same prompts and reference preprocessing as Diffusers' upsampler block, plus BFL's reference decoding (greedy, presence penalty 1.5 for text-to-image). The flash-linear-attention Triton kernels replace Transformers' torch fallback for Qwen3.5's linear-attention layers. Triton compiles a small CUDA launcher with the system C compiler, which the Ray image doesn't ship, so `flux3_app.py` installs `gcc` with `dnf` at container start (about 15 s). If that fails, it falls back to the torch path and logs which one it used (`MODEL_LOADED ... linear_attention`). |

Options considered and not used: BFL's reference code (pins vLLM 0.21 and torch 2.11, which would mean building a
custom image); TensorRT-LLM VisualGen (fastest on Blackwell, but needs a custom container and doesn't run the
upsampler); SGLang / vLLM-Omni DLCs (no upsampler integration). NVFP4/FP8 builds are faster on Blackwell but change
the output slightly, so this deployment stays in BF16 for quality.

## Layout

```
FLUX3-Klein-sagemaker-g7e/
├── flux3-klein-g7e-standalone-test.ipynb  # standalone test: stage -> upload code -> deploy -> generate -> delete
├── flux3_sm.py                            # boto3 helpers used by the notebook
├── scripts/stage_weights.py               # Hub -> S3, runs as a SageMaker Processing job
└── serving/
    ├── config.yaml                        # Ray Serve app config  -> s3://.../model/config.yaml
    └── code/
        ├── flux3_app.py                   # Ray Serve deployment -> s3://.../model/code/
        └── requirements.txt               # installed by the DLC entrypoint at container start
```

S3 (default bucket `sagemaker-<region>-<account>`):

```
flux3-klein-g7e/
├── model/                  # ModelDataSource (S3Prefix, uncompressed) -> /opt/ml/model
│   ├── config.yaml
│   ├── code/flux3_app.py, code/requirements.txt
│   └── weights/            # modular_model_index.json, transformer/, vae/, text_encoder/, prompt_upsampler/,
│                           # LICENSE.md, NOTICE.md, _flux3_stage.json (staging manifest)
├── io/requests|outputs|failures/
└── staging/code/stage_weights.py
```

The weights are staged once. Editing `serving/` means re-uploading a few KB and redeploying.

## Request

`POST` JSON through `invoke_endpoint_async` (see `flux3_sm.invoke`):

| Field | Default | Notes |
|---|---|---|
| `prompt` (or `inputs`) | required | A short request. The upsampler expands it into a full scene caption. |
| `images` | `[]` | Base64 PNG/JPEG/WebP, at most 10. One image means an edit; several means a composition ("the first image", "the second image", ...). |
| `upsample` | `true` | `false` sends `prompt` to the image model as written. Use it for full captions. |
| `boxes` | none | Layout boxes, `bbox: [y_min, x_min, y_max, x_max]` in 0-1. Needs `upsample: true`. |
| `num_inference_steps` | 16 | Clamped to 4..16; the distilled model is trained for that range. |
| `height`, `width` | 1024 (t2i) or the reference's size (edit) | Snapped down to a multiple of 16, clamped to 256..2048. |
| `seed` | random | Reported back in JSON responses. |
| `response_format` | `png` | `png` returns raw `image/png`. `json` returns `{image (base64), caption, seed, width, height, timings_s, ...}`. |
| `guidance_scale` | ignored | The distilled checkpoint has no classifier-free guidance. The field is accepted for FLUX.2 booth compatibility. |

Errors: a bad request returns HTTP 400 with `{"error": ...}`, which an async endpoint writes to the failure location.

## Measured on ml.g7e.2xlarge (us-east-2, 2026-11-08)

From the notebook run, distilled checkpoint, 16 steps, BF16:

| Request | Output | Upsampler | Image model | End to end (incl. S3 + async queue) |
|---|---|---|---|---|
| Text-to-image | 1024x1024 | 7.0 s | 3.2 s | 12.4 s |
| Edit, 1 reference | 1024x1024 | 6.6 s | 6.6 s | 15.5 s |
| Composition, 2 references | 1360x768 | 5.9 s | 10.6 s | 19.4 s |
| Layout, 2 boxes | 1024x1024 | 11.3 s | 3.3 s | 17.6 s |
| Caption as written (`upsample: false`), PNG | 1024x1024 | none | 3.2 s | 5.7 s |

The weights take 39.1 GB of GPU memory and peak use was 43.2 GB of 96 GB. A cold deploy took 4.9 minutes from
CreateEndpoint to READY: the container starts, downloads 39 GB, installs gcc, loads in 7 s and warms up in 29 s.
The upsampler costs about as much as the image itself. For a fixed set of prompts, cache the captions and send them
with `upsample: false`.

## Operations

- **Capacity.** `flux3_sm.deploy` retries capacity failures at least 3 times, backing off 2, 5, 10 and 15 minutes.
  Any other failure stops with SageMaker's `FailureReason`. Each capacity failure takes about 30 minutes to surface.
  On 2026-11-07/08, us-east-1 failed 5 times in a row with `InsufficientInstanceCapacity`, while us-east-2 provided
  an instance within 4 minutes three times. The weights must be in the endpoint's region: copy the model prefix to
  that region's bucket (`aws s3 sync s3://<bucket-a>/flux3-klein-g7e/model/ s3://<bucket-b>/flux3-klein-g7e/model/
  --source-region <a> --region <b>`) and set `REGION` in the notebook.
- **Readiness.** The Ray DLC's `/ping` reports healthy once Ray Serve has a route, before the model has loaded, so
  `InService` comes early. `flux3_sm.wait_ready` waits for the app's `[flux3] READY` line, which is printed after the
  weights are loaded and a warmup text-to-image and edit have run. Requests sent earlier wait in the queue.
- **Logs.** `aws logs tail /aws/sagemaker/Endpoints/<name>`. App milestones are greppable: `[flux3] LOADING`,
  `LOADED`, `MODEL_LOADED`, `WARMUP_DONE`, `READY`, `GENERATED` (timings, sizes, peak GPU memory), `ERROR`, `FATAL`.
  No log group at all means the container never booted (image pull, weight download or capacity), not a code problem.
- **Cost.** Delete the endpoint when done (notebook section 7). `set_auto_stop` is a server-side safety net: at the
  scheduled time Application Auto Scaling sets the variant to 0 instances, even if the notebook is gone.
- **No Hub access at runtime.** The container runs with `HF_HUB_OFFLINE=1` and holds no token. The token is only
  used by the staging job. The notebook passes it through a temporary Secrets Manager secret
  (`AmazonSageMaker-flux3-hf-token-<timestamp>`) that it deletes as soon as the job ends. The booth's `deploy.sh`
  keeps it in `AmazonSageMaker-flux3-klein-hf-token` so later deploys can re-stage without asking again.

## Use in the AI Photo Booth

The booth in [`../FLUX2-Klein-sagemaker`](../FLUX2-Klein-sagemaker) can run on this model. Its `deploy.sh` asks for the
Hugging Face token once, stores it in Secrets Manager, stages the weights to
`s3://sagemaker-<region>-<account>/flux3-klein-g7e/model/weights/` if they are not there yet, and creates the
SageMaker Model and EndpointConfig `FluxAiBoot-flux3-g7e` from `serving/`. It reuses weights that this notebook
already staged in the same bucket. The booth endpoint uses the booth's async I/O prefixes (`flux2-klein-inputs/`,
`flux2-klein-outputs/`, `flux2-klein-failures/`) instead of `flux3-klein-g7e/io/`.

The deploy does not start an instance. An admin makes `FluxAiBoot-flux3-g7e` current in the booth's Endpoint tab and
starts it there. Because the config name contains `flux3`, the booth then sends FLUX 3 prompts (short edit
instructions that the upsampler expands) with 16 steps. The booth README describes the flags and the prompts.

The notebook's endpoint is `flux3-klein-g7e` and the booth's is `FluxAiBoot-flux3-g7e`, so neither one creates or
deletes the other's endpoint.

## License

FLUX 3 Image Klein is released under the FLUX Kommunity License v1.0. It covers non-commercial use, and commercial
use of outputs only for organizations under US$5M annual revenue (affiliates included). Testing and evaluation in a
non-production environment counts as non-commercial. Use with end users, such as a public photo booth, needs a
license from Black Forest Labs (https://bfl.ai/licensing). The upsampler is Apache 2.0, as is the Qwen3-VL text
encoder.
