# Yiddish Video → English Subtitles

An event-driven AWS pipeline that turns an uploaded **Yiddish video** into
**timestamped English subtitles** (`.srt`). It transcribes with word-level
timestamps using Whisper (`ivrit-ai/yi-whisper-large-v3` via faster-whisper) and
translates with Amazon Bedrock (Claude), producing subtitle-ready output.

```
video ─▶ ffmpeg (Lambda) ─▶ WAV ─▶ faster-whisper (SageMaker async) ─▶ word timestamps
      ─▶ Bedrock/Claude translate + cue formatting (Lambda) ─▶ English .srt / .json
```

## Features

- **Hands-off**: drop a video in S3, get subtitles in S3 — every stage is triggered automatically.
- **Word-level timestamps** via faster-whisper (CTranslate2), grouped into readable, sentence-level cues.
- **Context-aware translation** with Claude on Bedrock (batched with surrounding lines for quality).
- **Large files supported** through SageMaker **asynchronous inference** (no 6 MB real-time payload limit).
- **Subtitle conventions** applied: line length, max 2 lines, minimum duration, reading-speed-friendly cues.
- **Secure by default**: no public access, encryption at rest + in transit, least-privilege IAM throughout.

## Architecture

```
        upload video
             │  (S3: videos/)
             ▼
  ┌────────────────────────┐
  │ whisper-ffmpeg-extract  │  extract 16 kHz mono WAV
  │ -audio (Lambda)         │
  └────────────────────────┘
             │  (S3: audio/*.wav)
             ▼
  ┌────────────────────────┐
  │ whisper-invoke-          │  invoke_endpoint_async
  │ transcribe (Lambda)      │
  └────────────────────────┘
             │
             ▼
  ┌────────────────────────┐
  │ whisper-yi-word          │  faster-whisper, word-level
  │ (SageMaker async, GPU)   │  timestamps + VAD
  └────────────────────────┘
             │  (S3: whisper/word-output/<id>.out)
             ▼
  ┌────────────────────────┐
  │ whisper-translate-       │  sentence cues → Bedrock/Claude
  │ subtitles (Lambda)       │  → .en.srt + .en.json
  └────────────────────────┘
             │
             ▼
   s3://<pipeline-bucket>/transcripts-en/<id>.en.srt
```

| Stage | Component | Output |
|-------|-----------|--------|
| Audio extraction | `whisper-ffmpeg-extract-audio` (Lambda + ffmpeg layer) | `audio/<name>.wav` |
| Transcription | `whisper-yi-word` (SageMaker async, faster-whisper) | `whisper/word-output/<id>.out` |
| Translation + subtitles | `whisper-translate-subtitles` (Lambda + Bedrock) | `transcripts-en/<id>.en.{srt,json}` |

## Repository layout

```
.
├── whisper.ipynb                  # Single source of truth: deploys & runs everything
├── code_whisper/
│   ├── inference.py               # SageMaker model server (faster-whisper)
│   └── requirements.txt           # faster-whisper
├── video_transcription_pipeline_design.md   # Design document
├── .kiro/skills/                  # AI-assistant context & lessons learned
└── README.md
```

## Prerequisites

- An AWS account with access to: **SageMaker**, **Lambda**, **S3**, **IAM**, **CloudWatch**, and **Amazon Bedrock** (Claude model access enabled).
- A SageMaker notebook / Studio environment, or a role with permission to create the resources above (including `iam:PassRole` for the endpoint execution role).
- A GPU quota for `ml.g5.xlarge` (async inference) in your region.
- The model artifact: `ivrit-ai/yi-whisper-large-v3` (converted to CTranslate2 by the notebook).

## Quick start

Open `whisper.ipynb` and run the cells top to bottom:

1. **Config** — sets region, account, names, and clients.
2. **Endpoint role** — creates the least-privilege execution role.
3. **Convert to CTranslate2** *(one-time)* — downloads and converts the model, uploads `CT2_MODEL_DATA`.
4. **Deploy endpoint** — deploys `whisper-yi-word` on `ml.g5.xlarge` (wait for `InService`, ~10 min).
5. **Ingestion** — bucket, ffmpeg layer, IAM roles, ffmpeg + transcribe Lambdas, S3 triggers.
6. **Translation** — deploys the Bedrock translate Lambda and wires its trigger.
7. **Run** — set `LOCAL_VIDEO` and run the end-to-end cell.

Then upload any video:

```python
s3.upload_file("my_video.mp4", PIPELINE_BUCKET, VIDEO_PREFIX + "my_video.mp4")
```

The English `.srt` lands in `s3://<pipeline-bucket>/transcripts-en/`.

## Configuration

Tune subtitles via the translate Lambda environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MAX_CUE_SECONDS` | `6` | Max duration of a subtitle cue |
| `MAX_CUE_CHARS` | `84` | Max characters before splitting a cue |
| `PAUSE_GAP` | `0.7` | Silence (s) that forces a new cue |
| `TRANSLATE_BATCH` | `30` | Cues per Bedrock call (context window) |
| `SRT_LINE_LEN` | `42` | Max characters per subtitle line |
| `MODEL_ID` | `us.anthropic.claude-opus-4-8` | Bedrock model |

Transcription settings live in the endpoint env (`language=yi`, `beam_size`, `vad_filter`, `compute_type`).

## Output format

`transcripts-en/<id>.en.json`:

```json
{
  "text": "full English transcript ...",
  "cues": [
    { "start": 2.72, "end": 5.04, "he": "...", "en": "Genye Shapiro, cassette no. 6." }
  ]
}
```

`transcripts-en/<id>.en.srt` — standard SubRip subtitles, ready for any video player.

## Troubleshooting

The end-to-end cell reports per-stage progress and, on failure, prints the relevant
CloudWatch log group and recent lines. Follow the artifacts to locate the failing stage:

| Symptom | Where to look |
|---------|---------------|
| No `audio/*.wav` | `whisper-ffmpeg-extract-audio` logs |
| WAV but no `whisper/word-output/*.out` | `whisper/word-failure/` + `/aws/sagemaker/Endpoints/whisper-yi-word` |
| `.out` but no `.en.srt` | `whisper-translate-subtitles` logs / Bedrock access |

"Worker died" (HTTP 500, no traceback) in endpoint logs means the worker was killed (out of memory).

## Security

- **No public access**: SageMaker endpoint is IAM-authenticated only; S3 buckets have Block Public Access on, TLS-only policies, and SSE-S3 encryption; Lambdas are invoked only by scoped S3 events.
- **Least privilege**: every Lambda and the endpoint run as dedicated roles scoped to the exact actions, resources, and log groups they need.
- Treat any credentials as sensitive — use IAM roles, never commit secrets.

## Cost notes

- The SageMaker GPU endpoint is the main cost; delete it when idle (see the teardown cell).
- S3 lifecycle rules expire intermediate videos/audio after 7 days.
- Bedrock is billed per translation request; Lambdas are pay-per-invocation.

## Cleanup

The notebook's **Teardown** cell (commented) removes the endpoint, Lambdas, layer, IAM roles,
triggers, and bucket. Uncomment and run to tear everything down.

## License

Inference code under `code_whisper/` is MIT-0. The `ivrit-ai/yi-whisper-large-v3` model
and `NLLB-200` (if used) carry their own licenses — review them before production use.
NLLB-200 is CC-BY-NC (non-commercial).
