# MiniMax-H3 on Amazon SageMaker

Deploy [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) behind a SageMaker
asynchronous inference endpoint on one `ml.p5.48xlarge` (8x H100 80 GB) and demo every capability
the open checkpoints expose: text-to-video-and-audio, keyframe conditioning, and omni-reference
generation including video-to-video editing and lip-sync.

H3 is a joint audio-video diffusion transformer. One denoising loop over a single packed multimodal
sequence produces the video **and** its synchronised 32 kHz stereo soundtrack — no vocoder, no
separate audio pass. Output is 24 fps, 4-15 s, 768-pixel short edge.

Start with **`MiniMax-H3-SageMaker.ipynb`**.

## 2026-09: rebased onto the AWS SageMaker SGLang DLC

The serving image originally had to build on `lmsysorg/sglang:dev` and pin a pre-release `main`
commit, because MiniMax-H3 was not in any tagged SGLang release at the time. That is no longer true:
SGLang **v0.5.17** is the first tagged release carrying the pipeline, and the
[2026/08/08 AWS DLC release](https://github.com/aws/deep-learning-containers/blob/main/README.md)
(`763104351884.dkr.ecr.<region>.amazonaws.com/sglang:0.5.17-gpu-py312-cu130-ubuntu24.04-sagemaker`)
ships it with the `[diffusion]` extra, CVE-patched by AWS. The project now builds on that DLC.

What changed, and why it is strictly better:

- **`container/Dockerfile`** — the whole checkout-and-reinstall dance is gone: no `git checkout` of
  a pinned commit inside the image, no editable `pip install -e "python[diffusion]"`, no flashinfer
  companion-wheel realignment (a tagged release ships matched wheels by construction). What remains
  is ffmpeg, boto3 and the serving shim on top of the DLC. The build-time assertion that
  `sglang.multimodal_gen...minimax_h3` imports is kept — a missing pipeline still fails the build,
  not the deploy. The DLC's own single-model entrypoint at `/usr/bin/serve` is overwritten with the
  H3 shim so `docker run <image> serve` always lands on it.
- **`container/build_and_push.sh` and `h3/codebuild.py`** — the base now comes from AWS's DLC
  registry (account `763104351884`), which needs its own `docker login`; both build paths do it, and
  the CodeBuild role gained a `PullDlcBaseImage` statement scoped to that registry's `sglang`
  repository. The `SGLANG_COMMIT` build argument is gone. The floating
  `0.5.17-...-sagemaker` tag tracks AWS's patch releases, so a rebuild picks up security fixes
  automatically.
- **Reproducibility note** — the previously pinned commit (`d257b58`) is an ancestor of the
  `v0.5.17` tag (verified: 0 commits behind), so nothing the endpoint ran before was lost in the
  move; the tag adds 24 commits on top.

Verified end to end: rebuilt in CodeBuild, redeployed, and re-ran the demos against the new image.

**Sample generated after the change** — the section 9d demo (`i2va`, a user photo as the first
frame, static camera, a dachshund squeezing toothpaste onto a brush and brushing his own teeth):
[samples/dog.mp4](samples/dog.mp4)

## Two properties of the notebook worth knowing up front

**Run cell 1 and cell 2, then jump to any section.** Every later cell depends on the bootstrap cell
and nothing else — verified statically, not just intended. Two mechanisms:

- **State lives in S3** (`h3/state.py`, at `s3://<bucket>/minimax-h3/io/_notebook_state.json`): the
  role ARN, image URI, endpoint name, and the S3 URI of every generated clip and extracted artifact.
  Restart the kernel, open a new Studio space, switch machines — the bootstrap picks it all back up.
- **Expensive steps are memoised.** Staging, the image build, the deploy and every demo check state
  first and say so when they skip. `force=True` (or `FORCE_REGEN = True`) redoes one deliberately.
  At $63.30/hour, re-running section 8 to reach section 10 is real money.

**Nothing is written to local disk.** Clips play from presigned S3 URLs, so no MP4s accumulate and the
`.ipynb` does not fill with base64. First frames, last frames and soundtracks are extracted **on the
endpoint** — it already has ffmpeg — and written straight to S3, which is what lets the keyframe and
reference demos chain off a generated clip with zero local bytes. Weight staging streams Hub → S3
through a bounded in-memory buffer and needs no scratch space at all. The single exception is the
optional Regenerate-2K call in section 12, which has to put the clip in an HTTPS request body; that
goes S3 → memory → API, never to disk.

## Layout

```
MiniMax-H3-SageMaker.ipynb   the notebook: stage -> build -> deploy -> 12 demos -> teardown
container/
  Dockerfile                 SageMaker SGLang 0.5.17 DLC + ffmpeg + the serving shim
  serve.py                   SageMaker shim: /ping, /invocations, two SGLang backends
  serve                      container entrypoint (SageMaker runs `docker run <image> serve`)
  build_and_push.sh          build for linux/amd64 and push to ECR
scripts/
  stage_weights.py           stream the 288 GB checkpoint from the Hub into S3
  notebook-lifecycle-on-start.sh   move Docker's data-root off the root volume
h3/
  prompts.py                 H3-Context-IR style prompt composers for both task families
  client.py                  async endpoint client, condition builders, S3-resident Result
  deploy.py                  model / endpoint-config / endpoint lifecycle
  iam.py                     dedicated least-privilege endpoint execution role
  codebuild.py               build the image in CodeBuild (no local Docker needed)
  state.py                   S3-backed cross-cell state (what makes cells resumable)
  media.py                   optional local ffmpeg helpers (the endpoint does this by default)
samples/
  dog.mp4                    section 9d output generated on the DLC-based image
```

### `HF_TOKEN`

Optional. MiniMax-H3 is **not gated**, so staging works without one, but an authenticated Hub download
gets higher rate limits and steadier CDN throughput on a 288 GB pull. There is a placeholder in the
notebook's bootstrap cell, or export `HF_TOKEN` before running `scripts/stage_weights.py`. It is used
only by the staging step and is never placed in the endpoint's environment — with weights in S3 the
container never talks to the Hub, so there is nothing to leak via `DescribeEndpointConfig`.

## What is open and what is not

MiniMax ships H3 as three modules; only the middle one is open:

| Module | Role | Open |
|---|---|---|
| H3-Context-IR | Hosted rewriter producing the structured representation H3-Base was trained on | No |
| **H3-Base** | The 768p generator, two checkpoint partitions (`FL2VA`, `Ref2VA`) | **Yes** |
| H3-Regenerate-2K | In-context regeneration of the 768p result at 2K | No |

The model card says Context-IR "is critical to the quality of the final output". `h3/prompts.py`
builds that structured prompt locally, following the two official writing guides, so the checkpoint
receives the shape it expects. Section 12 of the notebook shows the hosted alternative.

## Design decisions worth knowing before you change anything

**One endpoint, two SGLang backends.** `FL2VA` (serving `t2va` and `fl2va`) and `Ref2VA` are
separate 144.1 GB checkpoints. The endpoint quota here is one `ml.p5.48xlarge`, so `serve.py` runs
both backends in one container — GPUs 0-3 for `FL2VA`, GPUs 4-7 for `Ref2VA` — and routes
`/invocations` by `task`. Two endpoints would mean two p5s.

**Async, not real-time.** A 5-second clip at 50 steps takes minutes on 4x H100 and a real-time
endpoint caps a response at 60 s. Async reads the request from S3, allows up to
`InvocationTimeoutSeconds` (3600), and writes the response to S3.

**SGLang, not diffusers.** SGLang is MiniMax's first recommendation and the only path with
published, measured multi-GPU topologies for H3. The diffusers integration exists (Modular
Diffusers blocks) but is on an unmerged branch and has no context-parallel serving path — its
recipes target single- and dual-GPU offload. `serve.py` shells out to `sglang serve` and proxies.

**Built on the AWS SageMaker SGLang DLC.** SGLang `v0.5.17` is the first tagged release that
carries the MiniMax-H3 pipeline (verified absent from `v0.5.16`, present in `v0.5.17`), and the
[2026/08/08 DLC release](https://github.com/aws/deep-learning-containers/blob/main/README.md)
(`763104351884.dkr.ecr.<region>.amazonaws.com/sglang:0.5.17-gpu-py312-cu130-ubuntu24.04-sagemaker`)
ships it with the `[diffusion]` extra, CVE-patched by AWS. An earlier revision pinned a pre-release
`main` commit on top of `lmsysorg/sglang:dev`; that is gone. The Dockerfile still *asserts at build
time* that `sglang.multimodal_gen...minimax_h3` imports — a missing pipeline should fail a build,
not a deploy.

**`tp=4, ulysses=1` by default, not the documented speed recipe.** The cookbook's measured 4x H100
placements: `tp2+ulysses2` 13.25 s / 66.04 GB, FSDP+`ulysses4` 13.36 s / 57.01 GB, `tp4+ulysses1`
13.86 s / 49.80 GB. Pure `ulysses4` does not fit in 80 GB. Those numbers are for a 5-second clip;
the demos go to 15 s, which roughly triples the packed sequence and its activations. 49.8 GB
resident leaves ~30 GB of headroom for that instead of ~14 GB. Set `TP_SIZE=2, ULYSSES_DEGREE=2`
if you only care about short clips.

**Weights as an uncompressed S3 prefix, presented as a warm HF cache.** `CompressionType: None`
means SageMaker mirrors the prefix into `/opt/ml/model` verbatim, so iterating on the container
never means re-archiving 288 GB. The cookbook is explicit that SGLang owns the checkpoint-directory
mapping and you should not point `--model-path` at a hand-downloaded subdirectory — so at boot
`serve.py` builds a Hugging Face cache entry out of the staged tree using **symlinks** (no copy, no
duplicated disk) at the exact revision recorded in `_h3_stage.json`, sets `HF_HUB_OFFLINE=1`, and
passes the plain repo id. `H3_MODEL_PATH` overrides all of it if you need to.

**`/ping` gates on a real generation.** With `H3_WARMUP_ON_START=1` (the default) each backend runs
one tiny 4-step clip before readiness flips. `InService` therefore means "can actually generate"
rather than "a socket is open", which is the difference between a clean first request and a
mystifying failure.

**Backends start staggered with distinct `MASTER_PORT`s.** Two `torch.distributed` groups
initialising simultaneously on one host race for a rendezvous port, and the resulting failure does
not look like a port problem.

**The endpoint gets its own least-privilege role, not `AmazonSageMakerFullAccess`.** Following the
`whisper_yd` pattern in this workspace (whose steering says outright: *do not revert the endpoint to
the broad SageMaker notebook role*), `h3/iam.py` creates `minimax-h3-endpoint-role` scoped to our ECR
repo, read on the weights prefix, read+write on the `io/` prefix, `ListBucket` on the bucket, this
endpoint's own log group, and namespace-conditioned `PutMetricData`. No `s3:*`, no `DeleteObject`, no
access to any other bucket, repo or log group.

**Scope it by resource ARN, not by `Condition`.** The first version conditioned `s3:ListBucket` on
`s3:prefix` and the trust policy on `aws:SourceAccount`. Both read as tighter; both made the deploy
fail with `The provided role '...' is invalid`. SageMaker validates the execution role by evaluating
its policy with an **empty request context**, so any `Condition` referencing a request key evaluates
false — even though it would pass in real use. `diagnose_endpoint_role()` reproduces each check in that
error message, which names three things without saying which one failed.

The **deploying** principal needs `iam:PassRole` on that role — `create_model` passes it to
SageMaker, and the resulting `AccessDenied` never mentions PassRole. The preflight cell simulates it.

**SNS notification is supported but off by default.** The other async projects here (`wan_2_2`,
`whisper_yd`) wire `NotificationConfig`; set `H3_SNS_SUCCESS_TOPIC` / `H3_SNS_ERROR_TOPIC` to do the
same. `IncludeInferenceResponseIn` is deliberately never set: without `H3_OUTPUT_S3_PREFIX` the
response body is base64 video, which does not belong on a topic.

## Quick start

```bash
# 0. credentials + region
export AWS_DEFAULT_REGION=us-east-1
aws sts get-caller-identity

# 1. stage the checkpoint (288 GB; resumable; ~30-120 min).
#    Default --stream mode relays Hub -> S3 in memory: no local disk at all.
python scripts/stage_weights.py --bucket <bucket> --prefix minimax-h3/weights \
    --variants fl2va,ref2va

# 2. build the image (once). Needs ~80 GB of *Docker* disk - see the notebook
#    instance section below, because that is probably not the volume you think.
cd container && ./build_and_push.sh -r us-east-1 -n minimax-h3-sglang -t latest && cd ..

# 3. deploy and demo
jupyter lab MiniMax-H3-SageMaker.ipynb
```

To skip step 1, set `H3_WEIGHTS_SOURCE=hub` and the container downloads from the Hub on first boot.
That is faster on a p5's network but non-deterministic, and it repeats on every endpoint recreation.

To serve only text and keyframe tasks, set `VARIANTS="fl2va"`: 144 GB instead of 288 and roughly
half the cold start. `ref2va` requests then return a 400 naming the reason.

## Which notebook instance to run this on

**`ml.m5.4xlarge`, `VolumeSizeInGB=100`.** ~$0.92/hr. No GPU — every bit of model compute happens on
the p5 endpoint, and `ml.p5.48xlarge for notebook instance usage` is 0 in this account anyway.

```bash
aws sagemaker create-notebook-instance \
  --notebook-instance-name minimax-h3-dev \
  --instance-type ml.m5.4xlarge \
  --volume-size-in-gb 100 \
  --role-arn arn:aws:iam::<acct>:role/<SageMakerExecutionRole> \
  --platform-identifier notebook-al2-v3
```

Sizing reasoning, per step:

| Step | Bottleneck | What the instance needs |
|---|---|---|
| Stage weights (288 GB) | Hub bandwidth | 16 vCPU is plenty for 6 concurrent HTTPS→S3 relays. **No disk** — the default `--stream` mode never writes a file. |
| Build the image (~30 GB extracted) | Disk, and it is *not* the volume you think | See the trap below |
| Run the 16 demos | nothing | Polls S3, downloads a few hundred MB of MP4, runs ffmpeg on single frames |

`ml.m5.2xlarge` also works and saves $0.46/hr; `ml.m5.8xlarge` gets you sustained 10 Gbps instead of
burstable "up to 10 Gbps", which matters only if the Hub turns out to be faster than your burst
credits last. The staging step is the only one that cares, and it runs once.

### Trap: `VolumeSizeInGB` does not size the volume Docker uses

A notebook instance has two volumes:

| Path | Size | Persists across stop/start |
|---|---|---|
| `/` | **~100 GB, fixed, not configurable** — mostly consumed by the bundled conda envs, typically 10-20 GB free | no |
| `/home/ec2-user/SageMaker` | `VolumeSizeInGB` (5 GB default, up to 16 TB) | yes |

Docker's data-root is `/var/lib/docker`, on the **root** volume. The SGLang DLC base image is ~14 GB
compressed and **~30 GB extracted**, so `docker build` dies with `no space left on device` regardless
of how large you set `VolumeSizeInGB`.

### So build in CodeBuild

`h3/codebuild.py` is the path the notebook uses by default, and it works on every host:

| Host | Docker? | Problem |
|---|---|---|
| SageMaker **Studio** | no daemon unless the domain sets `EnableDockerAccess` | and a space's EBS defaults to a 100 GB maximum |
| SageMaker **notebook instance** | yes | data-root on the fixed ~100 GB root volume, usually 10-20 GB free |
| **CodeBuild** | yes | 128 GB of build disk, nothing to configure |

It zips `container/`, creates a least-privilege build project (`BUILD_GENERAL1_LARGE`,
`privilegedMode` for Docker-in-Docker, 120-minute timeout), runs it, streams the logs and pushes to
ECR. Idempotent — a second run updates the project rather than failing. If the build runs out of disk,
re-run with `compute_type="BUILD_GENERAL1_2XLARGE"` (824 GB).

Deliberately *not* `sagemaker-studio-image-build`: it wraps the same idea but leaves the compute type
at the default, and that default's 64 GB of disk is too tight for a 30 GB image.

The build role is scoped the same way as the endpoint role — our ECR repo only, the one S3 prefix
holding the source zip, this project's own log group, and `ecr:GetAuthorizationToken` as the single
`"*"` resource because it has no resource-level permissions. Your own principal additionally needs
`codebuild:CreateProject`/`UpdateProject`/`StartBuild`/`BatchGetBuilds` plus `iam:PassRole` on the
build role. The notebook's preflight simulates all of it.

### Building locally instead

Only worth it if Docker's data-root already has ~80 GB free. Two ways to get there:

1. **Relocate the data-root to the ML volume** with `scripts/notebook-lifecycle-on-start.sh` as an
   **on-start** lifecycle configuration (on-start, not on-create: the root volume is rebuilt from the
   AMI on every start). Then `VolumeSizeInGB=200` and `container/build_and_push.sh` works normally.
2. **A plain EC2 box** with a 200 GB gp3 root volume.

Check first, either way:

```bash
df -h / /home/ec2-user/SageMaker
docker info --format 'Docker Root Dir: {{.DockerRootDir}}'
df -h "$(docker info --format '{{.DockerRootDir}}')"
```

### Notebook size

Clips are **not** embedded. `Result.show()` renders a presigned S3 URL in a `<video>` tag, so the
`.ipynb` stays small and no MP4s accumulate locally. (An earlier version base64-inlined them because
JupyterLab will not serve a `<video src>` from an arbitrary local path; presigning solves the same
problem without the bytes.)

## Deviations from the other `ml/` async projects

Worth stating, since three sibling projects here (`LongCat-Image-sagemaker`, `FLUX2-Klein-sagemaker`,
`wan_2_2`) follow a common shape and this one departs from it in three places:

| Convention here | This project | Why |
|---|---|---|
| PyTorch/HF DLC + `source_dir` + `inference.py` | BYOC image on the SGLang DLC | The SGLang 0.5.17 DLC carries the pipeline, but its entrypoint serves one model the OpenAI way — H3 needs two SGLang backends on split GPUs behind `/ping` + `/invocations`. Multi-GPU Ulysses/TP is the whole point on a 33 B DiT + 32 B encoder. |
| `<model>-inputs/`, `<model>-outputs/`, `<model>-failures/` at bucket root | nested `minimax-h3/{weights,io/{requests,responses,failures,generated,demo-inputs}}` | The IAM policy conditions on `s3:prefix`, so one nested namespace per project keeps the statements short and makes the role trivially auditable. |
| Auto-scale to zero when idle | delete and redeploy | Scaling back up re-downloads 288 GB and reloads two models — a 20-40 minute cold start. For demo use the cost is identical and the failure modes are fewer. |

Everything else follows the established pattern: async inference for the same reasons, uncompressed
S3 weights so code iteration never re-archives them, no `HF_TOKEN` in the container (H3 is
ungated, so there is no token to leak in the first place), and `sagemaker.describe-endpoint` +
`logs tail` as the two diagnostics that answer different questions.

## Endpoint API

`POST /invocations`, `application/json`. Async, so the request body is an S3 object and the response
is written to another.

```jsonc
{
  "task": "t2va" | "fl2va" | "ref2va",
  "prompt": "integrated_multimodal_description: [Shot 1] ...",
  "conditions": [
    // fl2va: 1-2 images, role "keyframe". frame_index 0 = opening frame; omit = closing frame.
    // ref2va: <=9 image, <=3 video, <=3 audio, <=12 total. ORDER IS SEMANTIC.
    {"type": "image", "role": "keyframe", "frame_index": 0, "uri": "s3://... | data:... | https://..."},
    {"type": "video", "role": "reference", "uri": "s3://...", "start_time_seconds": 0.0},
    {"type": "audio", "role": "reference", "uri": "s3://..."}
  ],
  "target": {"short_edge": 768, "aspect_ratio": "16:9" | "auto" | "9:16" | ..., "duration_seconds": 5.0},
  "seed": 0,
  "num_inference_steps": 50,
  "flow_shift": 12.0,          // video schedule, as released
  "audio_flow_shift": 3.0,     // audio schedule, as released
  "num_outputs_per_prompt": 1, // 1-10; scalar seed expands as seed + index
  "output": {"s3_uri_prefix": "s3://bucket/prefix"},  // omit to get base64 inline
  "extract": true,             // or {"first_frame":true,"last_frame":true,"audio":true}
  "extra_body": {}             // forwarded verbatim to SGLang
}
```

`extract` asks the endpoint to also write the clip's first frame, last frame and 32 kHz stereo
soundtrack to S3 next to the MP4. ffmpeg is already in the image, so doing it there costs one PUT
instead of a download-and-re-upload round trip — and it is what makes the keyframe / reference demo
chain work without touching local disk. Failures are reported per artifact and never fail the
generation; a silent clip simply yields no `audio`.

Response:

```jsonc
{
  "status": "completed",
  "request_id": "...", "task": "t2va",
  "backend": {"variant": "fl2va", "port": 30010, "gpus": [0,1,2,3]},
  "job": { /* SGLang's job record, including its own metrics */ },
  "outputs": [{
    "variant": 0, "bytes": 5242880, "s3_uri": "s3://.../t2va-0.mp4",
    "artifacts": {"first_frame": "s3://.../first-0.png",
                  "last_frame":  "s3://.../last-0.png",
                  "audio":       "s3://.../audio-0.wav"}
  }],
  "timings": {"total_s": 214.7, "generate_s": 212.9, "download_s": 1.8},
  "request_echo": {"target": {...}, "seed": 0, "num_inference_steps": 50}
}
```

`uri` accepts `s3://` and `data:` (downloaded to container scratch and rewritten to `file://`), and
`http(s)://` / `file://` (passed straight through). Anything else is a 400 naming the problem.
Also useful: `GET /backends` returns each backend's GPUs, liveness and exact launch command.

## Container environment

| Variable | Default | Notes |
|---|---|---|
| `H3_VARIANTS` | `fl2va,ref2va` | Partitions to serve, in GPU order |
| `H3_GPUS_PER_BACKEND` | `4` | Must satisfy `len(variants) * this <= GPU count` |
| `H3_TP_SIZE` / `H3_ULYSSES_DEGREE` | `4` / `1` | See the topology note above |
| `H3_PERFORMANCE_MODE` | `speed` | `speed` keeps components resident; `memory` offloads |
| `H3_WARMUP_ON_START` | `1` | Gate `/ping` on a real generation |
| `H3_OUTPUT_S3_PREFIX` | *(unset)* | Default output location; unset means inline base64 |
| `H3_MODEL_PATH` | *(auto)* | Override the resolved model path or Hub id |
| `H3_JOB_TIMEOUT_S` | `3300` | Under SageMaker's 3600 s invocation ceiling on purpose |
| `H3_EXTRA_ARGS` | *(empty)* | Extra `sglang serve` flags, space separated |

Notebook-side environment (all optional): `H3_BUCKET`, `H3_ENDPOINT`, `H3_WEIGHTS_SOURCE`
(`s3` | `hub`), `H3_SNS_SUCCESS_TOPIC`, `H3_SNS_ERROR_TOPIC`, `MINIMAX_API_TOKEN`.

## Cost

`ml.p5.48xlarge` async inference in us-east-1 is **$63.296/hour** (~$1.06/minute), billed the whole
time the endpoint exists. Cold start is 20-40 minutes. Deploying and running every demo is roughly
**$45-60**. The staged weights are ~$6.60/month in S3 Standard.

Async endpoints *can* scale to zero, but scaling back up re-downloads 288 GB and reloads two
models, so for demo use deleting and redeploying costs the same and is simpler. The notebook's last
section deletes the endpoint and leaves the weights.

## Failure modes and their causes

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: no Docker daemon here` | SageMaker Studio has none unless the domain sets `EnableDockerAccess` | Use the CodeBuild cell in section 6 — it is the default path |
| `docker build`: `no space left on device` | Docker's data-root is on the notebook instance's fixed ~100 GB root volume, not the volume `VolumeSizeInGB` sizes | Use CodeBuild, or `scripts/notebook-lifecycle-on-start.sh` |
| CodeBuild fails in the `build` phase on disk | 128 GB was not enough | Re-run with `compute_type="BUILD_GENERAL1_2XLARGE"` (824 GB) |
| `Client.get() got an unexpected keyword argument 'stream'` | An earlier `stage_weights.py` used `huggingface_hub.utils.get_session()`, which returns `requests.Session` on 0.x and `httpx.Client` on 1.x | Fixed — that path is stdlib `urllib` now and has no opinion about which HTTP library is installed |
| `'...' is not in the notebook state yet. Run: section N` | You jumped to a cell whose input has not been produced | Run the named section; `state.keys()` shows what is recorded |
| Endpoint `Failed`, no log group | Container never booted: image pull, model download or capacity | `FailureReason`; check the image exists and the role can read the weights prefix |
| `AccessDenied` on `create_model` | Deploying principal lacks `iam:PassRole` on the endpoint role | Grant it scoped to `minimax-h3-endpoint-role`; the message never names PassRole |
| `The provided role '...' is invalid` on deploy | A `Condition` in the role policy. SageMaker validates the role with an **empty request context**, so a Condition on `s3:prefix` or `aws:SourceAccount` evaluates false | Run the role repair cell in section 4, then `diagnose_endpoint_role()`. Scope service roles by **resource ARN**, never by `Condition` |
| `/ping` 503 forever | A backend never became healthy | Read the `[fl2va]` / `[ref2va]` log lines — the SGLang traceback is there |
| First backend spams `Error in ZMQ Broker: Resource temporarily unavailable` from the moment the second one launches; second backend never healthy | SGLang binds a ZMQ job broker on `http port + 1`, so consecutive backend ports collide: backend 0's broker takes backend 1's HTTP port and SGLang silently relocates backend 1's server (+42) where the shim never probes | Fixed — ports are now spaced by `H3_PORT_STRIDE` (default 10). Rebuild the image if yours predates that |
| CUDA OOM on a long clip | 15 s at 768p roughly triples the packed sequence vs the benchmarked 5 s | `PERFORMANCE_MODE="memory"`, or `VARIANTS="fl2va"`, or a shorter clip |
| A demo has no `first_frame` artifact | It was generated before `extract=True` | `demo("t2va", force=True, ..., extract=True)` |

## Known constraints

- **`quality: "high"`** (SGLang's audited Cache-DiT path, ~1.40x faster, SSIM 0.931) is fail-closed
  to one exact workload on 4x H200. It is rejected before denoising on this topology.
- **No negative prompt, no guidance scale.** The released checkpoints are CFG-distilled; guidance is
  in the weights and each step is one forward pass. CFG parallelism is rejected, not silently
  ignored.
- **Ring attention is incompatible** with H3's packed multi-segment attention. Ulysses only.
- **Sparse attention is not in this release**, so long clips pay full attention cost.
- **`torch.compile` changes numerical output** and SGLang keeps the DiT eager under
  `--performance-mode speed`. Do not use a compiled run as ground truth.
- **`aspect_ratio: "auto"`** takes the canvas from the first keyframe for `fl2va`, but resolves to
  H3's 16:9 fallback for `ref2va` — it does not inherit a reference's geometry.
- **Audio can never be the only reference.** It must accompany an image or video.
- **2K output and Context-IR are hosted-only.** Not available on this endpoint.

## Security notes

- The endpoint runs as a dedicated least-privilege role (above). Add new capability as a new
  narrowly-scoped statement, never a wildcard.
- Each SGLang backend binds to `127.0.0.1` inside the container. Nothing but the shim's
  `/invocations` can reach them, and the shim is only reachable through SageMaker's IAM/SigV4 path.
- No `HF_TOKEN` anywhere. H3 is ungated, and with S3-staged weights the container never talks to the
  Hub at runtime.
- If the notebook has to create the bucket it enables Block Public Access on all four settings plus
  default SSE-S3, and it warns if a pre-existing bucket is not fully blocked. Generated media lands
  there.
- **Rotate any credentials that were pasted into a chat, ticket or other shared context.** This
  workspace has been bitten by that (`whisper_yd/SKILL.md`, pitfall 16). Use the execution role or a
  scoped profile and do not persist session tokens to disk.
- MiniMax moderates their hosted path. A self-deployed endpoint has no filter, so add one before
  exposing it to anyone.

## License

H3 is released under the MiniMax H3 Community License Agreement, not Apache/MIT. Read
[the license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) and its
[Q&A](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md) before any
use beyond evaluation.

## References

- [Model card](https://huggingface.co/MiniMaxAI/MiniMax-H3) — architecture, specs, reproducible cases
- [SGLang H3 cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3) — topologies, benchmarks, request schema
- [diffusers pipeline docs](https://github.com/huggingface/diffusers/blob/minimax-h3/docs/source/en/api/pipelines/minimax_h3.md) — Modular Diffusers integration, single-GPU recipes
- `docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` and `..._ref_en.md` in the model repo
