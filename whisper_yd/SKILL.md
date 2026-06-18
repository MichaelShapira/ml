---
name: Yiddish Video → English Subtitle Pipeline
description: >
  Context and hard-won lessons for the SageMaker + Lambda + Bedrock pipeline in
  whisper.ipynb that turns an uploaded Yiddish video into timestamped English
  subtitles (.srt). Read this before modifying the notebook, the endpoint, the
  Lambdas, or code_whisper/inference.py.
---

# Yiddish Video → English Subtitle Pipeline

## What this is
An event-driven AWS pipeline (defined entirely in `whisper.ipynb`, with the model
server code in `code_whisper/inference.py`) that:

1. Takes a **video** uploaded to S3.
2. Extracts a 16 kHz mono **WAV** with ffmpeg (Lambda).
3. **Transcribes** it to Yiddish with **word-level timestamps** on a SageMaker
   async endpoint running **faster-whisper (CTranslate2)** with the
   `ivrit-ai/yi-whisper-large-v3` model.
4. **Translates** the transcript to English and emits **subtitles** (`.en.srt`
   and `.en.json`) using **Amazon Bedrock (Claude Opus 4.8)** in a Lambda.

The end goal is **subtitle-grade output**: accurate word timestamps grouped into
readable, sentence-level cues, translated with surrounding context.

## Architecture / data flow
```
video → s3://<pipeline-bucket>/videos/            (S3 ObjectCreated:* )
      → ffmpeg Lambda (whisper-ffmpeg-extract-audio)
      → s3://<pipeline-bucket>/audio/<name>.wav    (S3 ObjectCreated:*.wav)
      → transcribe Lambda (whisper-invoke-transcribe)  invoke_endpoint_async
      → whisper-yi-word endpoint (faster-whisper, async)
      → s3://<sagemaker-default-bucket>/whisper/word-output/<id>.out  (word JSON)
      → translate Lambda (whisper-translate-subtitles)  Bedrock Claude
      → s3://<pipeline-bucket>/transcripts-en/<id>.en.json + .en.srt
```

### Key resources / names
- Endpoint: `whisper-yi-word` (async, `ml.g5.xlarge`, faster-whisper).
- Endpoint role: `whisper-yi-word-endpoint-role` (dedicated, least-privilege).
- Lambdas: `whisper-ffmpeg-extract-audio`, `whisper-invoke-transcribe`,
  `whisper-translate-subtitles` (each with its own least-privilege role).
- Buckets: pipeline bucket `whisper-video-pipeline-<acct>-<region>` (videos/,
  audio/, transcripts-en/, layers/); SageMaker default bucket holds the model
  artifact and `whisper/word-output|word-failure/`.
- Model: `ivrit-ai/yi-whisper-large-v3`, converted to CTranslate2 (fp16) and
  stored as `CT2_MODEL_DATA`.

## Why these choices
- **Async inference, not real-time**: real-time SageMaker endpoints cap the request
  payload at ~6 MB; audio from real videos is much larger. Async reads input from S3
  (up to 1 GB) and writes results to S3.
- **faster-whisper (CTranslate2), not the raw HF Transformers pipeline**: the HF
  pipeline's word-level timestamps are unreliable on long audio (see pitfalls).
  faster-whisper gives robust, memory-efficient word timestamps + built-in VAD.
- **Bedrock Claude for translation, not NLLB**: started with NLLB-200
  (`ydd_Hebr`→`eng_Latn`) but switched to Claude on Bedrock — serverless, no GPU
  endpoint to manage, and better contextual subtitle phrasing. Translation is done
  per sentence-cue, batched with surrounding context for quality + timestamp alignment.
- **Lambda for ffmpeg, not a SageMaker endpoint**: audio extraction is CPU/IO-bound
  and short; a serverless function is cheaper and needs no babysitting.

## Files
- `whisper.ipynb` — single source of truth. Sections: config → endpoint role →
  CT2 conversion → deploy endpoint → ingestion (bucket/layer/IAM/Lambdas/triggers)
  → translate Lambda + trigger → run end-to-end → fetch → decommission → teardown.
- `code_whisper/inference.py` — SageMaker model server: `model_fn` loads
  faster-whisper; `transform_fn` transcribes with `word_timestamps=True` and returns
  `{"text","segments","chunks"}` where `chunks` are word tokens.
- `code_whisper/requirements.txt` — `faster-whisper`.

---

## PITFALLS — read before changing anything

These are real failures hit during development. Avoid repeating them.

### Transcription / model
1. **HF Transformers word-level timestamps are broken for long audio here.**
   - Chunked algorithm (`chunk_length_s` set) + `return_timestamps="word"` →
     `_find_longest_common_sequence` crashes: `'<=' not supported between NoneType
     and float`. It's a transformers chunk-merge bug.
   - Sequential algorithm (no `chunk_length_s`) + word → worker is OOM-killed
     ("Worker died", HTTP 500, no Python traceback) because word timestamps force
     eager attention (`output_attentions=True`) whose memory grows with audio length.
   - **Resolution: use faster-whisper.** Do NOT try to "fix" the HF pipeline path.
2. **GPU OOM scales with `batch_size` for word-level** in the old HF path
   (encoder self-attention 1500×1500 × heads × layers × batch). Irrelevant now with
   faster-whisper (memory-light), but remember the principle.
3. **"Worker died" (500, no traceback) = process killed (OOM/SIGKILL)**, not a code
   bug. A clean Python exception surfaces as a 400 PredictionException instead. Use
   this to distinguish OOM from logic errors.
4. **Instance sizing**: faster-whisper large-v3 fp16 uses only ~3-4 GB GPU → `g5.xlarge`
   (24 GB) is plenty. Bigger instances in the same family share the **same GPU VRAM**
   (e.g., g6e.xlarge and g6e.2xlarge are both 1× L40S 48 GB), so "go up a size" does
   NOT add GPU memory — only CPU/vCPU. Don't throw hardware at a GPU-OOM problem
   unless you change the GPU itself.

### Inference contract / output shape
5. **`transform_fn` returning `(json_string, content_type)` gets serialized by the
   model server as a JSON list**: the `.out` file looks like
   `["{...transcript...}", "application/json"]`, NOT a plain dict. Every consumer must
   unwrap this. Use the `parse_transcript()` helper (tries each list element, json-
   loads strings, returns the first dict with `text`/`chunks`). This bit us repeatedly
   (`'str' object has no attribute 'get'`, list-vs-dict errors).
6. **Editing `inference.py` does nothing until you REDEPLOY.** The endpoint freezes
   the code into `model.tar.gz` at deploy time. After any change to inference.py or
   requirements.txt, re-run the deploy cell (it deletes + repacks + recreates).

### Bedrock
7. **Claude Opus 4.8 rejects `temperature`** in `inferenceConfig`
   (`temperature is deprecated for this model`). Keep `inferenceConfig` minimal
   (`maxTokens` only). Re-add params only for models that accept them.
8. **Cross-region inference profile (`us.anthropic...`)** needs IAM on both the
   `inference-profile` ARN AND the `foundation-model` ARNs in the routed US regions
   (us-east-1/2, us-west-2). A single-region grant fails.

### S3 / events
9. **New-object detection must use `(key, LastModified)`, not key name.** Re-running
   with the same filename overwrites the same S3 key; a "have I seen this key?" check
   never fires and you get a false "stage failed" (this produced a misleading
   "ffmpeg failed" message when ffmpeg had actually succeeded).
10. **Large uploads are multipart** → they fire `s3:ObjectCreated:CompleteMultipartUpload`,
    not `:Put`. Subscribe to `s3:ObjectCreated:*` to catch every path.
11. **Recursion safety**: triggers use distinct, non-overlapping prefixes
    (`videos/` → ffmpeg, `audio/` → transcribe). The ffmpeg Lambda writes only under
    `audio/`. Never let a Lambda write into the prefix that triggers it.
12. **The SageMaker default bucket notification is shared.** When wiring the
    `whisper/word-output/` → translate trigger, READ the existing notification config,
    append our config, and write it back (non-destructive merge). Never overwrite it
    wholesale — you'd wipe other notifications.

### GPU libraries
13. **CTranslate2 on GPU needs cuDNN/cuBLAS that aren't auto-discovered** in the DLC.
    `inference.py` preloads them via `ctypes.CDLL` from the torch-bundled
    `nvidia/cudnn/lib` + `nvidia/cublas/lib` before importing faster_whisper. If you
    see `libcudnn_ops.so.9 not found`, this preload (or a pinned `nvidia-cudnn-cu12`)
    is the fix.

### Model packaging
14. **CTranslate2 conversion is a one-time step.** `ct2-transformers-converter`
    re-lays-out weights, quantizes to fp16, and copies `tokenizer.json` +
    `preprocessor_config.json`. It only changes the storage FORMAT, not the model's
    weights. Re-run only to rebuild the artifact. Tar files at the archive ROOT.

### Workflow / process
15. **Don't edit the notebook while it's open in JupyterLab** and being changed
    externally — you'll get a save conflict and may clobber on-disk edits. Reload from
    disk ("Revert File") rather than saving the stale editor buffer.
16. **Never paste long-lived or temp credentials into shared contexts.** During dev,
    temp creds were pasted and had to be rotated. Use the notebook's execution role or
    a scoped profile. Never persist secrets to disk.
17. **`iam:PassRole`**: deploying the endpoint passes `whisper-yi-word-endpoint-role`
    to SageMaker, so the deploying principal needs `iam:PassRole` on it. AccessDenied
    on deploy usually means this is missing.

---

## Security model (keep it this way)
- **No public/unauthenticated access anywhere.** SageMaker endpoint = IAM/SigV4 only.
  S3 = Block Public Access (all 4) + TLS-only Deny + explicit SSE-S3 + no public Allow.
  Lambdas = no Function URL/API Gateway; invoked only by S3 events scoped with
  `SourceArn` + `SourceAccount`.
- **Least privilege per role.** Each Lambda role gets only its own log group, the exact
  S3 prefix it touches, and the one action it performs. The endpoint runs as a
  dedicated role (`whisper-yi-word-endpoint-role`) limited to: pull the HF DLC image,
  read `*model.tar.gz`, read `audio/*`, write `whisper/word-output|word-failure/*`,
  its log group, and namespaced metrics. Do NOT revert the endpoint to the broad
  SageMaker notebook role.
- When adding capability, add a **new narrowly-scoped statement**, never a wildcard
  action/resource.

## Operational runbook
- **Deploy order** (run cells top to bottom): config → endpoint role → CT2 conversion
  (once) → deploy endpoint (wait `InService`, ~10 min) → bucket/layer/IAM/Lambdas →
  ingestion triggers → translate Lambda → translate trigger.
- **Run**: set `LOCAL_VIDEO`, run the end-to-end cell. It prints per-stage progress and,
  on failure, the exact CloudWatch log group + recent log lines + any failure object.
- **Diagnose a stuck/failed run** by following the artifacts:
  - no WAV in `audio/` → ffmpeg Lambda logs.
  - WAV but no `.out` in `whisper/word-output/` → check `whisper/word-failure/` and the
    endpoint log group `/aws/sagemaker/Endpoints/whisper-yi-word` ("Worker died" = OOM).
  - `.out` but no `.en.srt` → translate Lambda logs / Bedrock access.
- **Subtitle tuning** (translate Lambda env): `MAX_CUE_SECONDS`, `MAX_CUE_CHARS`,
  `PAUSE_GAP`, `TRANSLATE_BATCH`, `SRT_LINE_LEN`.
- **Language codes**: Yiddish source = `yi` (faster-whisper) / `ydd_Hebr` (FLORES/NLLB);
  English = `eng_Latn`.

## When extending
- Keep the notebook the single source of truth; Lambda code is embedded there as
  strings and deployed from it.
- Any consumer of endpoint output MUST use the robust `parse_transcript` unwrap.
- After changing `inference.py`/`requirements.txt`, redeploy the endpoint.
- Prefer reusing existing resource names so re-running updates in place rather than
  creating orphans.
