# SAM 3 on SageMaker — text-prompt masking

Deploy Meta's [SAM 3](https://huggingface.co/facebook/sam3) behind a SageMaker **asynchronous** endpoint and segment objects by **text prompt**: send an image and a noun phrase like `"chair"`, get back a mask for every chair in the image.

The driver is `sam3-sagemaker.ipynb`. `prepare_weights.py` stages weights to S3 once. `code/` is the inference source bundle.

```
SAM/
├── sam3-sagemaker.ipynb   # deploy + invoke + test (marks masks on sample.png)
├── prepare_weights.py     # one-time weight staging to S3
├── code/
│   ├── inference.py       # SageMaker handler: model_fn / input_fn / predict_fn / output_fn
│   └── requirements.txt
└── README.md
```

---

## What SAM 3 does (and why it fits the "type a word, mask the thing" goal)

SAM 3 performs **Promptable Concept Segmentation (PCS)**: it takes a short noun-phrase prompt (e.g. `"yellow school bus"`, `"chair"`) and predicts instance + semantic masks for **every** object matching the concept. Unlike SAM 1/2 — which need a point, box, or mask prompt — SAM 3 has a built-in text encoder, so no Grounding-DINO-style box generator is needed. One model, text in, masks out.

## Why `facebook/sam3`, not `facebook/sam3.1`

| | `facebook/sam3` | `facebook/sam3.1` |
|---|---|---|
| `transformers` integration | **Native** (`Sam3Model` / `Sam3Processor`, `model.safetensors`) | None — `library_name: checkpoint`, ships only `sam3.1_multiplex.pt` |
| How to load | `Sam3Model.from_pretrained(...)` on a stock PyTorch DLC | Needs Meta's `facebookresearch/sam3` package + custom code |
| Text-prompt PCS on a single image | Yes | Yes (same capability) |
| Headline 3.1 gain | — | "Object Multiplex": ~faster multi-object **video** tracking |
| Gated | Yes | Yes |

For single-image text-prompt masking the two are equivalent in capability, and `sam3` is a far cleaner managed deploy. The notebook is structured so moving to 3.1 later is just a weights-prefix + loader change (you'd vendor Meta's `sam3` package into `code/` and load the `.pt`).

---

## Design (mirrors the workspace's hardened FLUX2 pattern)

- **Weights as an uncompressed S3 prefix.** SageMaker mounts every object under the prefix to `/opt/ml/model/` verbatim — no tar round-trip on deploy.
- **Code in a `code/` subfolder inside the same prefix.** With dict `model_data` you must **not** pass `entry_point`/`source_dir` (that triggers the SDK repack path, which silently drops dict `model_data`). The inference toolkit auto-discovers `/opt/ml/model/code/inference.py`. Editing the handler = re-upload a few KB.
- **No Hub at runtime.** Weights are staged once with an `HF_TOKEN`; the endpoint loads with `local_files_only=True` and holds no token.
- **Async, not real-time.** Input image (base64 in JSON) is read from S3 and the JSON result is written to S3, sidestepping the real-time endpoint's ~6 MB payload cap and 60 s timeout.

### Request / response

Request JSON (staged to S3, referenced by `InputLocation`):

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | base64 string | required | PNG/JPEG image bytes |
| `text` | string | required | Concept prompt, e.g. `"chair"` |
| `threshold` | float | `0.5` | Min instance confidence |
| `mask_threshold` | float | `0.5` | Mask binarization threshold |
| `max_instances` | int | `100` | Cap on returned instances (by score) |

Response JSON (written to `OutputLocation`):

```json
{
  "prompt": "chair",
  "image_size": [H, W],
  "num_instances": 3,
  "scores": [0.94, 0.88, 0.71],
  "boxes":  [[x1,y1,x2,y2], ...],
  "masks":  ["<base64 PNG, 0/255, H x W>", ...]
}
```

Returning raw masks (not a pre-rendered overlay) keeps the endpoint generic; the notebook does the "mark on the original image" step client-side (colored mask overlay + boxes → `marked_output.png`).

---

## Cost / performance: which instance?

SAM 3 is ~860M params (~3.4 GB fp32), runs at 1008px, and fits comfortably on any single modern inference GPU. It is **not** memory-bound like the diffusion models in this repo, so the cheapest capable GPU wins. Approximate SageMaker real-time/async on-demand pricing (us-east-1; check your region):

| Instance | GPU | VRAM | Arch | ~$/hr | Notes |
|---|---|---|---|---|---|
| **`ml.g6.xlarge`** | **L4** | 24 GB | Ada (2023) | **~$0.80–1.00** | **Best $/perf.** Newer, more efficient than A10G, cheaper than g5. Recommended default. |
| `ml.g5.xlarge` | A10G | 24 GB | Ampere | ~$1.4 | Proven in this repo (whisper). Solid fallback if g6 quota is 0. |
| `ml.g4dn.xlarge` | T4 | 16 GB | Turing (2018) | ~$0.74 | Cheapest. Still fits SAM 3; slower forward, no bf16 perf win. Budget option. |

**Recommendation: `ml.g6.xlarge`.** L4 gives the best price/performance for this size of model and leaves plenty of headroom. Drop to `ml.g4dn.xlarge` to minimize cost if latency isn't critical; step to `ml.g5.xlarge` only if L4 quota is unavailable. Larger `g6e`/A100 instances are overkill — they help diffusion/video models that are VRAM-bound, not SAM 3.

For lower latency / VRAM, set `SAM3_DTYPE=bfloat16` (or `float16`) on the endpoint env. Default is `float32` for numerical robustness; the handler casts `pixel_values` to the model dtype automatically.

---

## Caveats (learned from the other SageMaker projects in this workspace)

- **Use the async, not real-time, endpoint.** Real-time caps payload at ~6 MB and hard-times-out at 60 s; a base64 image in JSON plus model load easily exceeds that. Async reads/writes via S3.
- **Editing `inference.py` does nothing until you re-upload + redeploy.** The handler is frozen into the container at deploy time. After any edit, re-run section 3 (upload `code/`) and section 4 (redeploy).
- **Transformers version matters.** SAM 3 classes land in `transformers>=4.57.6`, and on the PyTorch 2.6 DLC (torch 2.6) they must stay `<5.0` — transformers 5.x imports `torch.float8_e8m0fnu`, which only exists on torch ≥ 2.7, so a 5.x install crashes at import. `requirements.txt` pins `[4.57.6, 5.0)` and `inference.py` re-checks at cold start (installing with `--no-deps` so it never disturbs the DLC's CUDA-matched torch).
- **SDK ↔ DLC version coupling.** The image URI is resolved client-side by the SageMaker SDK. `sagemaker<3.0.0` only knows PyTorch image URIs up to **2.6.0**, so `framework_version="2.7"` raises `Unsupported pytorch version: 2.7`. This notebook targets the **2.6 / py312 / CUDA 12.4** inference DLC. To use a newer DLC you'd first `pip install -U sagemaker`.
- **`facebook/sam3` is gated.** Accept the license and use an access-granted `HF_TOKEN` for `prepare_weights.py`. The token is needed only at staging time, never at runtime.
- **`iam:PassRole`.** The deploying principal needs `iam:PassRole` on the endpoint execution role, plus S3 read/write on the default bucket. `AccessDenied` on deploy is usually this.
- **Cold start = Hub vs S3.** Staging weights to S3 (vs downloading from the Hub at runtime) keeps cold start bounded by EBS throughput and removes a third-party dependency from the inference path.
- **Delete the endpoint when idle** — it's the dominant cost. Keep the `sam3/weights/` prefix for cheap redeploys; add an S3 lifecycle rule on the `sam3-inputs/` / `sam3-outputs/` / `sam3-failures/` prefixes.
- **Custom resolution degrades accuracy.** SAM 3 is meant to run at 1008px; don't shrink the processor size unless you've measured the hit.

---

## Security notes

- The endpoint is IAM/SigV4-authenticated only (no public access). Keep Block Public Access + TLS-only + SSE on the bucket.
- No `HF_TOKEN` in the endpoint env — weights are S3-hosted, so anyone with `DescribeEndpointConfig` can't harvest a credential.
- SAM 3 is under Meta's license (gated). Review the license terms before any commercial use.

## License

This notebook/code is provided as-is. The SAM 3 model weights are governed by Meta's SAM 3 license — see the [model page](https://huggingface.co/facebook/sam3).
