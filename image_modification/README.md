# NAFNet Image Modification on SageMaker

Deploy Megvii's [NAFNet](https://github.com/megvii-research/NAFNet) family behind a single SageMaker **asynchronous** endpoint and apply **denoise**, **deblur**, and **stereo super-resolution** to an image — or, in the raw editor, to the **masked region only**, non-destructively and with full undo history.

This is the deployment half of the "Image modification" feature for the raw editor's masked sections. Once a mask exists (from the SAM 3 semantic-segmentation endpoint, a brush stroke, or any other mask kind), the editor can run NAFNet on the image and composite the result back **inside the mask only**, exactly the way it already composites SAM masks.

```
image_modification/
├── image-modification-sagemaker.ipynb   # deploy + invoke + test all tasks
├── prepare_weights.py                    # one-time weight staging to S3
├── code/
│   ├── inference.py                      # SageMaker handler (model/input/predict/output_fn)
│   ├── nafnet_archs.py                   # vendored NAFNet + NAFSSR (no BasicSR dependency)
│   └── requirements.txt
└── README.md
```

---

## What the endpoint does

| `task` | model | input | output | use |
|--------|-------|-------|--------|-----|
| `denoise` | NAFNet-SIDD-width64 | 1 image | same size | remove sensor / high-ISO noise |
| `deblur` | NAFNet-GoPro-width64 (`variant:"reds"` → NAFNet-REDS-width64) | 1 image | same size | motion deblur |
| `stereo_sr` | NAFSSR-L (4×) | left + right pair | 4× pair | stereo super-resolution |

`denoise` and `deblur` return an image the **same size** as the input, so the editor blends it over the working buffer through the mask alpha. `stereo_sr` follows the [NAFSSR](https://arxiv.org/abs/2204.08714) design — it consumes a rectified left/right pair (6 channels) and returns both views upscaled 4×.

All three are NAFNet variants and fit comfortably on one `ml.g4dn.xlarge` (NVIDIA T4, 16 GB). The handler builds each network **lazily on first use** and caches it, so cold start is cheap and memory tracks only the tasks you actually call.

### Request / response (async, via S3)

Request JSON (PUT to S3, referenced by `InputLocation`):

| field | type | default | notes |
|-------|------|---------|-------|
| `task` | string | required | `denoise` \| `deblur` \| `stereo_sr` |
| `image` | base64 | required | PNG/JPEG; the main / left image |
| `image_right` | base64 | — | required for `stereo_sr` only |
| `variant` | string | `gopro` | `deblur` only: `gopro` \| `reds` |
| `mask` | base64 PNG | — | optional (denoise/deblur): white = apply; result is composited so only masked pixels change |
| `tile` | int | `0` | optional tile size in px for large inputs (0 = off) |
| `tile_overlap` | int | `32` | overlap-add seam width when tiling |

Response JSON (written to `OutputLocation`):

```json
{
  "task": "denoise",
  "image_size": [H, W],
  "output_size": [H2, W2],
  "image": "<base64 PNG>",
  "image_right": "<base64 PNG>"   // stereo_sr only
}
```

Returning a finished image (not raw tensors) keeps the client trivial. The async contract — PUT request JSON, `InvokeEndpointAsync`, poll `OutputLocation` — is **identical** to the SAM 3 endpoint the editor already drives, so the client code is a near copy.

---

## Quick start

1. **Stage weights once** (notebook section 2, or from any machine with AWS creds):
   ```bash
   pip install "huggingface_hub>=0.26" gdown
   python prepare_weights.py --out weights      # add --skip-stereo to omit NAFSSR
   aws s3 sync weights/ s3://<bucket>/nafnet/weights/
   ```
   The three NAFNet checkpoints come from the community HF mirror `mikestealth/nafnet-models` (no token); NAFSSR-L 4× comes from Megvii's official Google Drive via `gdown`.

2. **Upload the `code/` bundle** into the same prefix (notebook section 3).

3. **Deploy** the async endpoint on `ml.g4dn.xlarge` (notebook section 4).

4. **Test** every task and the masked-region path (notebook sections 6–9).

5. **Delete the endpoint** when idle (notebook section 10) — keep the endpoint *config* for a fast start/stop, the same pattern the editor's SAM tab uses.

---

## Design (mirrors the workspace's hardened SAM / FLUX2 pattern)

- **Weights as an uncompressed S3 prefix.** Every object under the prefix mounts to `/opt/ml/model/` verbatim — no tar round-trip on deploy.
- **Code in a `code/` subfolder inside the same prefix.** With dict `model_data` you must **not** pass `entry_point`/`source_dir` (that triggers the SDK repack path, which silently drops dict `model_data`). The toolkit auto-discovers `/opt/ml/model/code/inference.py`. Editing the handler = re-upload a few KB.
- **No BasicSR install.** The NAFNet / NAFSSR architectures are vendored into `nafnet_archs.py` (trimmed from the upstream Apache-2.0 source). The container needs only `pillow` + `numpy` on top of the DLC's torch.
- **Async, not real-time.** A base64 image in JSON plus model load easily exceeds the real-time endpoint's ~6 MB / 60 s limits; async reads/writes via S3.

### Why these checkpoints

The official `NAFNet-width64` configs are reproduced exactly in `nafnet_archs.py`:

| model | width | enc blocks | middle | dec blocks |
|-------|-------|-----------|--------|-----------|
| SIDD (denoise) | 64 | [2, 2, 4, 8] | 12 | [2, 2, 2, 2] |
| GoPro / REDS (deblur) | 64 | [1, 1, 1, 28] | 1 | [1, 1, 1, 1] |
| NAFSSR-L (stereo SR) | 128 | 128 NAFBlockSR, 4× | — | — |

Checkpoints are loaded with `strict=True`, so a config mismatch fails loudly at first use rather than producing silently-wrong output. The TLC (test-time local converter) `AvgPool2d` carries no learnable parameters, so these plain (non-`Local`) builders load the released weights identically.

---

## Integrating into the raw editor (next step)

The editor already has everything needed; this endpoint slots in beside SAM 3:

1. **New `imageMod` service** under `ui/src/features/raw-editor/services/`, modeled on `sam3Client.ts` / `sam3Constants.ts` / `sam3EndpointManager.ts`:
   - `imageModConstants.ts` — `ENDPOINT_NAME = "nafnet-imgmod"`, S3 prefixes `nafnet-inputs/` / `nafnet-outputs/`, reuse the same bucket/region and the Cognito-authorized `InvokeEndpointAsync` proxy URL.
   - `imageModClient.ts` — copy the `segment()` flow: downscale the current canvas → PUT request JSON (`task`, `image`) → invoke via the proxy → poll `OutputLocation` → decode the returned PNG. Keep the same silent STS/Cognito refresh-once behaviour.
   - `imageModEndpointManager.ts` — start/stop/status, identical to `sam3EndpointManager.ts`.

2. **New mask-panel action "Image modification"** alongside the existing Generative-AI mask UI, offering **Denoise** and **Deblur** (GoPro / REDS variants). It runs against the **current mask's alpha** (any mask kind — SAM, brush, linear, radial). **Stereo SR is intentionally not exposed in the UI** — it requires a rectified left/right stereo pair, which the single-image editor doesn't have; it stays available on the endpoint for notebook / API use only.

3. **Non-destructive apply + history.** The editor sends the working-buffer image (no `mask` field — composite client-side), receives the processed image, then writes processed pixels into the working buffer **only where the mask alpha > 0** and records it as one entry via `useEditHistory` (`hintRef` label e.g. `"Denoise · <mask name>"`). The original RAW and the linear source buffer are never touched, so it's reversible from the History panel exactly like every other edit. The server-side `mask` option exists for parity and notebook demos but the SPA prefers client-side compositing (lets the user re-blend without re-invoking).

This keeps the new feature a drop-in sibling of the SAM masking you already shipped, reusing the auth, async-S3, and history plumbing wholesale.

---

## Cost / performance

NAFNet denoise/deblur are ~68–116 M params and run in well under a second per megapixel on a T4. NAFSSR 4× is the heaviest (memory grows with the *output* resolution); keep stereo inputs modest (≈ ≤ 256×256 per view) or use the `tile` option for large single-image restoration. The async GPU endpoint is the dominant cost — **delete it when idle** and re-create from the kept config when needed.

---

## Caveats (learned from the other SageMaker projects here)

- **Editing `inference.py` / `nafnet_archs.py` does nothing until you re-upload + redeploy.** The handler is frozen into the container at deploy time. After any edit, re-run notebook section 3 (upload `code/`) and section 4 (redeploy).
- **SDK ↔ DLC coupling.** `sagemaker<3.0` resolves PyTorch image URIs up to 2.6 only; this notebook targets the 2.6 / py312 / CUDA 12.4 inference DLC. `pip install -U sagemaker` first if you want a newer DLC.
- **`iam:PassRole`.** The deploying principal needs `iam:PassRole` on the endpoint execution role plus S3 read/write on the bucket. `AccessDenied` on deploy is usually this.
- **NAFSSR weights are on Google Drive.** If `gdown` is rate-limited, download `NAFSSR-L_4x.pth` manually (see `docs/StereoSR.md` in the NAFNet repo) and drop it in `weights/` before the sync; `--skip-stereo` deploys denoise+deblur only.
- **Stereo SR is a true stereo model.** It expects a rectified left/right pair, not a single image upscaled twice — feeding unrelated images produces artifacts.

---

## Licenses

- NAFNet / NAFSSR code and weights: Apache-2.0, © 2022 Megvii. The vendored architectures in `nafnet_archs.py` retain that attribution.
- This notebook/handler glue is provided as-is for deployment in this workspace.
