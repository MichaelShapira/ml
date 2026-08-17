# MiniMax Music 3 on Amazon SageMaker Async Inference

Deploy [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) — an open-weights
text-to-music model that turns **lyrics + a music description** into a complete, structured song
(32 kHz 16-bit stereo WAV, up to 5 minutes) — as a SageMaker asynchronous endpoint, and drive it
from a Jupyter notebook all the way to an inline-playable MP3.

```
lyrics + caption ──> S3 ──> SageMaker async endpoint (ml.g6e.2xlarge, L40S 48 GB)
                                   │  8B Qwen3 global LLM  (autoregressive, 25 frames/s)
                                   │  0.6B local LLM       (residual RVQ codebooks)
                                   │  2.4B flow-matching DiT + Flow-VAE vocoder
                                   ▼
                             WAV in S3 ──> notebook polls ──> MP3 + inline player
```

## Why these choices

| Decision | Reason |
|---|---|
| `ml.g6e.2xlarge` (L40S 48 GB) | bf16 weights are ~22 GB and classifier-free guidance is baked in (no off switch) — every request runs two decode rows, so a 5-minute song adds ~3 GB of KV cache. 24 GB cards don't fit. |
| Fully GPU-resident, **no CPU offload** | The global LLM runs thousands of *sequential* decode steps per song. Per-step PCIe swaps (fine for 4-step image models) would multiply an already minutes-long generation. |
| **Async** endpoint | A full song takes 3–8 minutes to generate; a real-time endpoint's 60 s invocation limit can't survive it. |
| Uncompressed S3 prefix + `code/` inside it | Edit the handler, re-upload two small files, redeploy — never re-archive ~29 GB of weights. |
| Weights filtered at download | The Hub repo is ~58 GB, but half is a duplicate checkpoint for the SGLang-Omni serving path. `prepare_weights.py` downloads only the diffusers components (~29 GB). |

## Repository layout

```
├── minimax-music3-sagemaker.ipynb   # the whole workflow: stage → deploy → generate → MP3
├── prepare_weights.py               # one-time filtered weight download from the Hub
├── code/
│   ├── inference.py                 # SageMaker handler (model_fn/input_fn/predict_fn/output_fn)
│   └── requirements.txt             # pinned container deps (see "Dependency notes")
└── README.md
```

## Prerequisites

- AWS account with SageMaker access and quota for **1× ml.g6e.2xlarge for endpoint usage**
- An execution role with S3 + SageMaker permissions (a SageMaker notebook/Studio role works as-is)
- ~29 GB of local disk for the one-time weight staging (no `HF_TOKEN` needed — the repo is public)
- SageMaker Python SDK **v2** (`pip install "sagemaker<3.0.0"` — the notebook does this)

## Quick start

1. **Stage weights (once):**
   ```bash
   pip install "huggingface_hub>=0.27" hf_transfer
   HF_HUB_ENABLE_HF_TRANSFER=1 python prepare_weights.py --out weights
   aws s3 sync weights/ s3://<your-bucket>/minimax-music3/weights/
   ```
2. **Upload the handler into the same prefix:**
   ```bash
   aws s3 cp code/inference.py      s3://<your-bucket>/minimax-music3/weights/code/inference.py
   aws s3 cp code/requirements.txt  s3://<your-bucket>/minimax-music3/weights/code/requirements.txt
   ```
3. **Open the notebook** and run sections 1–4 to deploy (cold start ≈ 8–12 min), then section 6 for
   a 15-second audition clip. Section 8c is a fully self-contained "zero to hero" cell that goes
   from empty kernel to an inline-playable MP3.

## Request format

```json
{
  "lyrics":         "[Verse]\nMorning light is spilling through the curtain\n[Chorus]\nSoftly the world begins to breathe",
  "prompt":         "Genre: acoustic pop. BPM: 96. Key: C major. Vocals: soft female lead... Arrangement: fingerpicked guitar...",
  "audio_duration": 60.0,
  "seed":           42
}
```

Rules that silently ruin output if broken:

- **Both `lyrics` and `prompt` are required.** For an instrumental, pass a minimal lyric like
  `[Intro]\n(instrumental)` and describe the piece as instrumental in the prompt.
- **Section tags (`[Verse]`, `[Chorus]`, `[Bridge]`, `[Solo]`, `[Outro]`…) go on their own lines.**
  A tag sharing a line with lyric text keeps the tag and silently drops the text.
- The **caption is the strongest control** — use the model's Structured Caption format
  (Global Metadata / Vocal Details / Arrangement, ~250–450 words) for full-song control.
- Requests are **deterministic in the seed**: identical inputs → identical audio. Change only the
  seed for another take of the same song.
- Reference audio is **not supported** — this model conditions on text only.

Response: `audio/wav` bytes in the async output location. The notebook converts to MP3 client-side
(`soundfile` bundles the codec; no ffmpeg).

## Latency & cost expectations

- Cold start: ~8–12 min (pip install + mirror 29 GB from S3 + load 22 GB to GPU).
- Generation on the L40S: a 15 s audition ≈ 30 s; a full 5-minute song ≈ 3–8 min.
- `ml.g6e.2xlarge` bills per hour while the endpoint exists — **delete the endpoint when done**
  (Cleanup section in the notebook).

## Dependency notes (the traps this repo already handles)

Every one of these failed a real deploy before being fixed — don't undo them:

1. **The MiniMax pipeline is not in any diffusers release** (verified against the 0.39.0 wheel).
   `requirements.txt` pins the exact PR commit from the model card as an immutable GitHub tarball.
   When [huggingface/diffusers#14456](https://github.com/huggingface/diffusers/pull/14456) ships in
   a release, switch to `diffusers==<release>` — after grepping the wheel to confirm the pipeline
   is actually in it.
2. **`transformers` must be 5.15+, not 4.x.** The diffusers PR branch requires
   `huggingface-hub>=1.23`; transformers 4.x caps hub `<1.0` → pip `ResolutionImpossible` →
   TorchServe never boots → ping health check fails. 5.15.0's `torch.float8_e8m0fnu` references are
   hasattr-guarded, so it imports cleanly on the torch-2.5 DLC (early 5.x did not).
3. **Never add `torch` to requirements.txt** — the DLC owns the CUDA-matched build.
4. **The model mount is read-only.** The shipped `modular_model_index.json` points components at
   the Hub repo id, so the handler stages a writable view (`/tmp/minimax_music3_local`) with
   symlinks to the weights and a patched index. `HF_HUB_OFFLINE=1` makes any residual Hub call fail
   fast instead of hanging.
5. **The pipeline returns a numpy array**, not the torch tensor the model card snippet implies —
   the handler handles both, and decides the WAV transpose by shape.
6. **`TS_MAX_RESPONSE_SIZE` is raised to 100 MB** — a 5-minute stereo WAV (~38 MB) exceeds
   TorchServe's ~6.5 MB default response cap.
7. Do **not** pass `entry_point`/`source_dir` with dict `model_data` — the SDK's repack path
   silently skips wiring the code and you get the default handler's
   `Exactly one .pth or .pt file is required` crash.

## Troubleshooting

| Symptom | Meaning | Look at |
|---|---|---|
| Endpoint `Failed`, "did not pass the ping health check" | Worker died during startup | `aws logs tail /aws/sagemaker/Endpoints/<name> --since 1h` |
| No CloudWatch log group at all | Container never booted (download/capacity) | `describe-endpoint` → `FailureReason` |
| `ResolutionImpossible` in logs | requirements.txt conflict | Dependency notes above |
| Generation succeeds, request 500s | Handler bug after the pipeline call | The traceback lines after `Transform failed` |
| Healthy cold start | — | `transformers 5.15.0 is in the supported range.` → `Qwen3ForCausalLM import confirmed.` → `Staged local model view` → `Pipeline fully resident on CUDA` |

## License & responsible use

- **This repo's code/notebook**: provided as-is for demonstration and educational use.
- **The model** is governed by the
  [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE):
  attribution ("MiniMax-Music3") on commercial UIs, written authorization above US $20M/year
  revenue, safeguards required if you expose generation to third parties, and an Acceptable Use
  Policy — notably: **no copyrighted lyrics as input** (all lyrics in this repo are original) and
  **publicly shared outputs must be disclosed as AI-generated**.
- Model lineage: Qwen3-8B (Apache 2.0), Stability stable-audio-tools (MIT), Descript DAC (MIT).

Not legal advice — read the full license before any use beyond demos and blogs.

## Acknowledgements

- [MiniMax](https://huggingface.co/MiniMaxAI) for the model and the
  [SGLang-Omni cookbook](https://sgl-project.github.io/sglang-omni/cookbook/minimax_music3.html)
- The [diffusers](https://github.com/huggingface/diffusers) team for the modular pipeline
  integration (PR #14456)
