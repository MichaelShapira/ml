# Wan2.2 TI2V-5B — SageMaker Async Endpoint

This notebook deploys the [Wan-AI/Wan2.2-TI2V-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) text+image-to-video model as an asynchronous SageMaker endpoint.

## What it does

1. Downloads the Wan2.2 TI2V-5B model from Hugging Face and uploads it to S3
2. Generates `inference.py` and `requirements.txt` for SageMaker serving
3. Creates a SageMaker async endpoint on `ml.g6e.2xlarge` (NVIDIA L40S, 44 GB VRAM)
4. Invokes the endpoint with a sample image-to-video request
5. Cleans up the endpoint, config, and model

## Inference payload

The endpoint accepts JSON with these fields:

| Parameter | Required | Default | Description |
|---|---|---|---|
| `bucket` | Yes | — | S3 bucket for output video |
| `file_name` | No | `i2v_output.mp4` | Output S3 key |
| `image_s3_uri` / `image_url` / `image_bytes` | Yes (one of) | — | Input image |
| `prompt` | No | Generic cinematic prompt | Text prompt |
| `negative_prompt` | No | Chinese/English quality filters | Negative prompt |
| `height` / `width` | No | Auto from image aspect ratio | Output resolution |
| `num_frames` | No | 81 | Number of frames (49 recommended for g6e.2xlarge) |
| `guidance_scale` | No | 7.5 | Prompt adherence (7.5–8.5 recommended) |
| `num_inference_steps` | No | 50 | Denoising steps |
| `fps` | No | 16 | Framerate (16 = model native) |
| `seed` | No | 0 | RNG seed for reproducibility |

## Hardware constraints (ml.g6e.2xlarge)

The `ml.g6e.2xlarge` was selected strictly for cost considerations. The model uses ~28–32 GB VRAM in bfloat16, leaving only ~12–16 GB headroom, which limits resolution and frame count. Tested limits:

- 49 frames @ 480×480 — works (~38 GB)
- 65+ frames @ 480×480 — OOM

For best performance (higher resolution, longer videos), use a larger instance such as `ml.p4de.24xlarge` (A100 80 GB), which has ~4× the effective headroom and can run 81 frames at 720×480 comfortably.

## Licenses

- This notebook is licensed under **MIT**.
- The Wan2.2 model itself is licensed under **Apache 2.0**. You must accept the license on the [Hugging Face model page](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) before downloading. Source: [Wan-AI on Hugging Face](https://huggingface.co/Wan-AI).

## Credits

Much of this notebook is based on the original Wan 2.1 SageMaker example: [Wan2.1-T2V-1.3B-Diffusers on aws-samples](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models/Wan/Wan2.1-T2V-1.3B-Diffusers).

## Sample files

Located in the same directory as this notebook:
- `astro.jpg` — sample input image
- `demo_astronaut.mp4` — sample output video

## Prerequisites

- SageMaker execution role with S3 and ECR access
- `sagemaker>=2.254.1`, `boto3`, `huggingface_hub`
- SNS topic ARN (for async completion notifications)
- Accepted Wan2.2 license on Hugging Face
