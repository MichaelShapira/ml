# SAM 3D Objects on SageMaker — click an object, get a 3D model

Deploy Meta's [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) behind a SageMaker **asynchronous** endpoint and reconstruct a full 3D model (shape + texture, as a 3D Gaussian Splat) **from a single image plus a mask**. This reproduces the open-source backend of Meta's [`convert-image-to-3d`](https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d) web demo: click an object → it gets masked → a 3D model is generated.

The driver is `sam3d-sagemaker.ipynb`. The masking ("click the object") is done client-side in the notebook with a lightweight SAM, and the 3D lift runs on the GPU endpoint. A turntable GIF is rendered server-side so you can preview the result inline.

```
Sam3D/
├── sam3d-sagemaker.ipynb     # deploy + click-to-mask + 3D reconstruct + preview
├── app.py                    # Gradio web app (server-side SAM masking + orbit 3D)
├── webapp.py                 # WebGPU app server (Flask proxy to the endpoint)
├── web/index.html            # in-browser WebGPU SAM masking + interactive splat viewer
├── monitor.py                # terminal progress/log monitor (build, endpoint, failures)
├── prepare_weights.py        # one-time gated-checkpoint staging to S3
├── container/
│   ├── Dockerfile            # Bring-Your-Own-Container (conda + CUDA + sam3d_objects)
│   ├── build_and_push.sh     # build linux/amd64 image, push to ECR
│   └── serve/
│       ├── predictor.py      # Flask app: /ping + /invocations (loads pipeline once)
│       ├── sam3d_handler.py  # image+mask -> .ply (+ turntable GIF). NOT named inference.py (collision)
│       └── serve             # gunicorn launcher (SageMaker entrypoint)
├── .gitignore
└── README.md
```

---

## Why this is a Bring-Your-Own-Container deploy (and SAM 3 was not)

The workspace's `SAM/` (SAM 3) and `FLUX2-Klein-sagemaker/` projects ride a stock SageMaker **PyTorch DLC** and only ship a small `code/inference.py` + `requirements.txt`. That works because those models are pure `transformers`/`diffusers` and need no compiled native code.

SAM 3D Objects is different. Its [setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) requires a full conda environment with **CUDA-toolkit-compiled** native packages:

- `torch==2.5.1+cu121`, `torchaudio`, `xformers==0.0.28.post3`
- `pytorch3d` (built from source against the CUDA toolkit)
- `flash_attn==2.8.3` (compiled)
- `kaolin==0.17.0` (NVIDIA wheels pinned to torch 2.5.1 / cu121)
- `gsplat` (built from source) for Gaussian-splat rendering
- `spconv-cu121`, `MoGe`, `open3d`, `bpy`, plus a `hydra` source patch

You cannot reliably install that chain at endpoint cold-start (compiling `pytorch3d` + `flash_attn` takes many minutes and frequently fails on dependency backtracking — exactly the trap the workspace's `sagemaker-diffusers-deploy` skill warns about). So we bake the whole environment into a Docker image once, push it to ECR, and serve with a minimal Flask app that satisfies the SageMaker `/ping` + `/invocations` contract.

**Weights stay out of the image.** The ~13 GB of gated checkpoints are staged to S3 once and mounted (uncompressed prefix) at `/opt/ml/model`, so iterating on the image never re-ships the weights, and iterating on weights never rebuilds the image.

---

## Request / response

The endpoint is async: the notebook PUTs a request JSON to S3, calls `invoke_endpoint_async(InputLocation=...)`, and polls the `OutputLocation`.

Request JSON:

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | base64 string | required | RGB PNG/JPEG of the full scene |
| `mask` | base64 string | required | Single-channel PNG (0/255), **same H×W as `image`**, marking the one object to lift |
| `seed` | int | `42` | Reproducibility seed |
| `render_preview` | bool | `true` | Render a turntable GIF of the result server-side |
| `preview_frames` | int | `48` | Frames in the turntable (more = smoother, slower) |
| `preview_resolution` | int | `384` | Turntable frame size in px |

Response JSON (written to `OutputLocation`):

```json
{
  "ply_b64":       "<base64 of the Gaussian-splat .ply>",
  "num_gaussians": 123456,
  "preview_gif_b64": "<base64 GIF turntable, or null>",
  "timing_s":      {"load": 0.0, "reconstruct": 31.2, "render": 6.8}
}
```

Returning the raw `.ply` keeps the endpoint generic — open it in any 3DGS viewer (e.g. [PlayCanvas SuperSplat](https://superspl.at/editor), [antimatter15/splat](https://antimatter15.com/splat/)), Blender, or the repo's own kaolin/gradio viewer. The turntable GIF is just a convenience preview so you see something the moment the call returns.

### Where the "click" happens

The web demo's click-to-segment is the SAM family. To stay faithful but keep the 3D endpoint single-purpose, the notebook generates the mask **client-side**: it runs a small `facebook/sam-vit-base` with your click point, shows the marked overlay, then sends `image` + `mask` to the endpoint. Swap in the workspace's existing SAM 3 text-prompt endpoint (`../SAM`) if you'd rather type a word than click.

---

## Cost / performance: which instance?

SAM 3D Objects' [setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md) states a hard floor of **≥ 32 GB VRAM**. That immediately rules out the 24 GB cards (`g5`/`g6` xlarge: A10G/L4) that the rest of this workspace uses. The model + its renderer is the dominant cost, so you want the cheapest single GPU that clears 32 GB.

NVIDIA **L40S (48 GB)** — the `g6e` family — is, in AWS's own words, "the most cost-efficient GPU instance for deploying generative AI models." It clears the 32 GB floor on a single GPU, so you don't pay for multi-GPU you can't use (the pipeline is single-GPU).

| Instance | GPU | VRAM | vCPU / RAM | ~$/hr (us-east-1)\* | Verdict |
|---|---|---|---|---|---|
| **`ml.g6e.2xlarge`** | **1× L40S** | **48 GB** | 8 / 64 GiB | **~$2.4** | **Recommended.** Clears 32 GB on one GPU; 64 GiB RAM + 8 vCPU comfortably absorb the heavy CPU-side mesh/pointmap post-processing and the ~13 GB checkpoint load. |
| `ml.g6e.xlarge` | 1× L40S | 48 GB | 4 / 32 GiB | ~$1.9 | Cost floor. Same GPU; trim if you confirm 32 GiB host RAM + 4 vCPU are enough for your images. Try this first to save ~20%. |
| `ml.g5.2xlarge` | 1× A10G | 24 GB | — | ~$1.5 | **Below the 32 GB floor — do not use.** |
| `ml.g6.2xlarge` | 1× L4 | 24 GB | — | ~$1.3 | **Below the 32 GB floor — do not use.** |
| `ml.p4d.24xlarge` | 8× A100 40 GB | 320 GB | — | ~$37 | Massive overkill; single-GPU model can't use 8 GPUs. |

\* On-demand SageMaker hosting, approximate, region-dependent — confirm on the [pricing page](https://aws.amazon.com/sagemaker/pricing/) before committing.

**Recommendation: start on `ml.g6e.2xlarge`.** It's the cheapest instance that meets the VRAM floor with headroom for the heavy CPU post-processing, and it's the same L40S card the FLUX.2 booth uses. Drop to `ml.g6e.xlarge` to shave ~20% once you've confirmed your workload fits 32 GiB host RAM. Larger `g6e` (4×/8× L40S) and A100/H100 boxes are wasted spend here.

For lower latency, set `render_preview=false` (skip the turntable) and reduce `preview_frames` — rendering is a meaningful slice of wall-clock per request.

---

## Quick start

**The notebook does the setup for you** — open `sam3d-sagemaker.ipynb` and run top to bottom. Section 2 configures the IAM permissions (CodeBuild + ECR + S3 + `PassRole`) and CodeBuild trust on your execution role; section 3 checks the `ml.g6e.2xlarge` endpoint quota; section 5 builds the image on CodeBuild; section 6 deploys. If your role can't self-grant IAM, section 2 prints the exact role name, JSON, and CLI for an admin.

Before you start: accept the licenses on the two gated HF repos —
[`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects) (the 3D model) and you'll need an access-granted `HF_TOKEN` for the weight-staging step.

Manual / terminal equivalent of the one-time steps:

```bash
# 0. One-time: accept the license on the gated HF repo
#    https://huggingface.co/facebook/sam-3d-objects  (Submit the access form)

# 1. Stage the ~13 GB gated checkpoints to S3 (needs an access-granted token)
export HF_TOKEN=hf_...
pip install "huggingface_hub[cli]>=0.26,<1.0"
python prepare_weights.py --out checkpoints
aws s3 sync checkpoints/ s3://<bucket>/sam3d/weights/

# 2. Build the BYOC image and push to ECR (linux/amd64).
#    On a SageMaker notebook (no Docker daemon), the reliable path is CodeBuild:
        pip install sagemaker-studio-image-build
        cd container && sm-docker build . --repository sam3d-objects:latest \
            --compute-type BUILD_GENERAL1_2XLARGE --build-arg SAM3D_COMMIT=main
#    (Studio exec role needs CodeBuild+ECR+S3 access and must trust codebuild.amazonaws.com.)
#    If you instead have Finch or Docker locally, build_and_push.sh auto-detects them:
        cd container && ./build_and_push.sh sam3d-objects us-east-1

# 3. Open sam3d-sagemaker.ipynb and run it: deploy -> click an object -> 3D + preview
```

Then iterate from the notebook.

---

## Interactive web app (click-to-mask + rotatable 3D)

`app.py` is a Gradio UI that reproduces the Meta `convert-image-to-3d` experience against your deployed endpoint:

- **Click** the object in the photo → a local `facebook/sam-vit-base` segments it and the mask overlays live.
- **Click again to refine** — *Add (foreground)* grows the selection, *Remove (background)* subtracts. SAM re-runs with all your points each click.
- **Generate 3D** → the masked object goes to the SageMaker endpoint and comes back as a Gaussian-splat `.ply`.
- The result loads in `gr.Model3D` — **drag to rotate, scroll to zoom** (no spinning GIF).

Run it two ways:

```bash
# Standalone web app (uses your AWS creds)
export SAM3D_ENDPOINT=sam3d-objects-g6e
export AWS_REGION=us-east-1
pip install gradio transformers torch
python app.py            # opens a local URL + a public gradio.live share link
```

or from the notebook (**section 11**), which reuses the configured endpoint and `invoke_async`:

```python
import app; app.build_demo(invoke_fn=invoke_async).launch(share=True)
```

Notes: the heavy 3D lift runs on the GPU endpoint; only the lightweight click-segmentation runs locally (CPU is fine). The app sets `render_preview=false` (no server GIF) since the viewer is interactive. Gradio's `Model3D` renders Gaussian-splat `.ply` directly; if your environment blocks the `share=True` tunnel, open the printed local URL via the Jupyter proxy or run `app.py` from a machine with a browser.

### WebGPU variant (fastest clicking — segmentation in the browser)

`webapp.py` + `web/index.html` move the segmentation **into the browser** via transformers.js + WebGPU (the same approach as Meta's demo). The image encoder runs once on *your* GPU and each click decodes in milliseconds with no server round-trip — much snappier than server-side SAM on a CPU-only notebook. The browser also renders the splat interactively (orbit/zoom). The Flask server only proxies the "Generate 3D" call to the SageMaker endpoint.

```bash
export SAM3D_ENDPOINT=sam3d-objects-g6e
export AWS_REGION=us-east-1
pip install flask boto3
python webapp.py        # serves on :7860
```

Opening it (Flask has no public tunnel like Gradio's `share=True`, and **WebGPU needs a secure context — https or localhost**):
- **Easiest:** run `webapp.py` on your **laptop** (with AWS creds + an endpoint-invoke IAM permission). Open `http://localhost:7860` — WebGPU uses your laptop GPU; the server signs the endpoint call.
- **On a SageMaker notebook:** reach it through the Jupyter proxy over https, e.g. `https://<your-studio-or-notebook-url>/proxy/7860/` (the proxy provides the secure context WebGPU requires; a plain `http://<host>:7860` will not enable WebGPU).
- Requires a WebGPU-capable browser (recent Chrome/Edge). Without WebGPU it falls back to slower WASM. SlimSAM weights (~tens of MB) download once into the browser cache.


---

## Monitoring progress (run in a separate terminal)

The image build and the endpoint cold start each take minutes. `monitor.py` (boto3 only) gives you bounded, out-of-band visibility — a one-shot snapshot by default, or `--follow` which polls until a terminal state or `--timeout` (never an endless loop):

```bash
python monitor.py build --follow       # latest CodeBuild image build: status, phases, logs
python monitor.py endpoint --follow     # endpoint status + FailureReason + CloudWatch logs
python monitor.py endpoint              # single snapshot + last 20 min of logs
python monitor.py failures              # dump recent async failure objects from S3
```

`build` finds the `sagemaker-studio*` CodeBuild project sm-docker creates; `endpoint` defaults to `sam3d-objects-g6e` (override with `--name`); `failures` reads `s3://sagemaker-<region>-<account>/sam3d-failures/`. It exits on its own at `SUCCEEDED`/`Failed`/`InService`, with a hint about the likely cause on failure.

---

## Caveats (some learned from the other SageMaker projects in this workspace)

- **No Docker daemon on SageMaker — use CodeBuild (`sm-docker`), or Finch.** SageMaker notebooks / Studio JupyterLab ship without `dockerd`, and installing a container runtime inside the app container is painful. The reliable path is the [SageMaker Studio Image Build CLI](https://aws.amazon.com/blogs/machine-learning/using-the-amazon-sagemaker-studio-image-build-cli-to-build-container-images-from-your-studio-notebooks/), which runs the build on **CodeBuild** (larger compute, no daemon) and pushes to ECR: `pip install sagemaker-studio-image-build`, then from `container/` run `sm-docker build . --repository sam3d-objects:latest --compute-type BUILD_GENERAL1_2XLARGE --build-arg SAM3D_COMMIT=<sha>`. The Studio execution role must have CodeBuild + ECR + S3 access (e.g. `AmazonSageMakerFullAccess`) **and** trust `codebuild.amazonaws.com` in its assume-role policy. If you have [Finch](https://runfinch.com) (AWS's Docker-compatible client) or Docker available instead, `build_and_push.sh` auto-detects `finch` then `docker`; override with `BUILDER=docker`.
- **Build for linux/amd64.** SageMaker runs x86-64. `sm-docker`/CodeBuild builds x86-64 natively; `build_and_push.sh` passes `--platform linux/amd64` (also matters on Apple-silicon Macs). The build compiles `pytorch3d`, `flash_attn`, and `gsplat` from source — slow (tens of minutes), memory-hungry, and needs ample disk for the CUDA + conda layers. Don't interrupt it.
- **`flash_attn` compile can exceed CodeBuild's default timeout.** If the job times out, raise the timeout on the auto-created `sagemaker-studio-*` CodeBuild project (no GPU is needed to build — `nvcc` is in the image — just time + RAM), or swap in a prebuilt flash-attn wheel (cp311 / torch 2.5 / cu121) to skip the compile.
- **Use the async, not real-time, endpoint.** A base64 image **and** the returned `.ply` + GIF blow past the real-time endpoint's ~6 MB payload cap and 60 s timeout. Async reads/writes via S3.
- **`facebook/sam-3d-objects` is gated.** Accept Meta's license and use an access-granted `HF_TOKEN` for `prepare_weights.py` only. The token is never placed on the endpoint — weights load from S3 with no Hub access at runtime.
- **Editing `serve/` does nothing until you rebuild + repush the image and recreate the endpoint.** The handler is frozen into the container. (Weights, by contrast, are an S3 prefix you can re-sync without rebuilding.)
- **VRAM floor is real.** Below 32 GB the pipeline OOMs or fails in PyTorch3D/kaolin. Stay on L40S (48 GB) or larger.
- **Delete the endpoint when idle** — the GPU instance is the dominant cost. Keep the `sam3d/weights/` prefix for cheap redeploys, and add an S3 lifecycle rule on the `sam3d-inputs/` / `sam3d-outputs/` / `sam3d-failures/` prefixes.
- **`iam:PassRole`.** The deploying principal needs `iam:PassRole` on the endpoint execution role, plus S3 read/write on the I/O bucket and ECR pull. `AccessDenied` on deploy is usually one of these.
- **The mask must match the image dimensions** exactly (same H×W). The endpoint embeds the mask in the image's alpha channel before running the pipeline, mirroring `demo.py`.

## Security notes

- The endpoint is IAM/SigV4-authenticated only (no public access). Keep Block Public Access + TLS-only + SSE on the bucket.
- No `HF_TOKEN` in the endpoint env — weights are S3-hosted, so `DescribeEndpointConfig` can't harvest a credential.
- The repo's `notebook/inference.py` enforces a hydra instantiate allow/deny list (`check_hydra_safety`) so a tampered `pipeline.yaml` can't instantiate arbitrary callables. We keep that check intact.

## License

This notebook/code is provided as-is. SAM 3D Objects weights and code are governed by Meta's [SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE) — review the terms before any commercial use.
