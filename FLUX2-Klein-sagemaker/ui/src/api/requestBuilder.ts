/**
 * Async_Request builder for the SPA (Requirements 7.2, 7.3, 7.4).
 *
 * Mirrors `backend/src/lib/request-builder.ts`. The UI package is type-checked
 * in isolation and cannot import `backend/src` (TS6307), so the same pure,
 * property-tested logic is duplicated here. Keeping the booth ranges and
 * defaults identical guarantees the browser builds the same request a
 * server-side caller would.
 */

import { getPromptForEffect, getRequestOverridesForEffect } from "../booth/effects";
import type { ModelFamily } from "./modelFamily";

/**
 * A single JSON inference request submitted to the booth's current endpoint.
 * Both endpoints (FLUX.2 `code/inference.py` and FLUX 3 `flux3_app.py`) accept
 * this shape and return PNG bytes.
 */
export interface AsyncRequest {
  /** Prompt text mapped from the selected effect. */
  inputs: string;
  /** Base64-encoded reference image(s); the booth always sends `[photo]`. */
  images: string[];
  /** Denoising steps, clamped to the family's range (see {@link SAMPLING}). */
  num_inference_steps: number;
  /**
   * Guidance scale, clamped to the family's range. FLUX 3 Image Klein is
   * guidance-distilled (CFG 1); its endpoint accepts and ignores the field.
   */
  guidance_scale: number;
  /**
   * Optional output width in px. Omitted for most effects (the endpoint then
   * derives the size from the reference image); set by effects that need a
   * specific output shape, e.g. a tall portrait for a half-body crop.
   */
  width?: number;
  /** Optional output height in px (see {@link AsyncRequest.width}). */
  height?: number;
  /** Optional fixed seed for reproducible generation. */
  seed?: number;
}

/** Inclusive booth bounds for `num_inference_steps` (FLUX.2). */
export const STEPS_RANGE = { min: 4, max: 20 } as const;

/** Inclusive booth bounds for `guidance_scale` (FLUX.2). */
export const GUIDANCE_RANGE = { min: 1, max: 10 } as const;

/** Booth defaults applied when a caller omits a parameter (FLUX.2). */
export const DEFAULTS = { num_inference_steps: 6, guidance_scale: 2.5 } as const;

/** Sampling defaults and bounds for one model family. */
export interface SamplingPolicy {
  defaults: { num_inference_steps: number; guidance_scale: number };
  steps: { min: number; max: number };
  guidance: { min: number; max: number };
}

/**
 * Per-family sampling. FLUX.2 keeps the booth's tuned 6 steps / guidance 2.5.
 * FLUX 3 Image Klein (distilled) is trained for 4 to 16 steps with 16 as the
 * default, and has no classifier-free guidance (CFG 1), per Black Forest Labs'
 * model card and docs.bfl.ai.
 */
export const SAMPLING: Readonly<Record<ModelFamily, SamplingPolicy>> = {
  flux2: { defaults: DEFAULTS, steps: STEPS_RANGE, guidance: GUIDANCE_RANGE },
  flux3: {
    defaults: { num_inference_steps: 16, guidance_scale: 1 },
    steps: { min: 4, max: 16 },
    guidance: { min: 1, max: 1 },
  },
};

/** Parameters for {@link buildAsyncRequest}. */
export interface BuildAsyncRequestParams {
  /** Effect id selected by the visitor; resolved to a prompt via the catalog. */
  effectId: string;
  /** The captured photo as a base64-encoded PNG/JPEG string. */
  photo: string;
  /** Model family of the current endpoint (default `flux2`). */
  family?: ModelFamily;
  /** Optional override for `num_inference_steps`; defaults to {@link DEFAULTS}. */
  steps?: number;
  /** Optional override for `guidance_scale`; defaults to {@link DEFAULTS}. */
  guidance?: number;
  /** Optional fixed seed; passed through unchanged when provided. */
  seed?: number;
}

/**
 * Strip a `data:` URI prefix from a base64 image string, returning only the raw
 * base64 payload. The browser captures the webcam still as a full data URL
 * (`data:image/png;base64,iVBOR...`), but the endpoint's `inference.py` decodes
 * with `base64.b64decode(..., validate=True)`, which rejects the prefix. Passing
 * a string without a prefix is returned unchanged.
 */
export function stripDataUriPrefix(value: string): string {
  const match = /^data:[^;,]*(?:;[^,]*)?,(.*)$/s.exec(value);
  return match ? match[1] : value;
}

/**
 * Resolve a parameter to an in-range value: fall back to `fallback` when the
 * input is omitted or non-finite, then clamp into `[min, max]`.
 */
function resolveInRange(
  value: number | undefined,
  fallback: number,
  min: number,
  max: number,
): number {
  const base = value === undefined || !Number.isFinite(value) ? fallback : value;
  return Math.min(max, Math.max(min, base));
}

/**
 * Build the {@link AsyncRequest} for a selected effect and captured photo.
 * Throws {@link UnknownEffectError} (from the catalog) for an unknown effect id.
 */
export function buildAsyncRequest({
  effectId,
  photo,
  family = "flux2",
  steps,
  guidance,
  seed,
}: BuildAsyncRequestParams): AsyncRequest {
  // Per-effect overrides (output size / sampling). Precedence for steps and
  // guidance: explicit caller argument > effect override > family default.
  // Effect sampling overrides were tuned for FLUX.2, so FLUX 3 ignores them.
  const overrides = getRequestOverridesForEffect(effectId);
  const sampling = SAMPLING[family];
  const tuned = family === "flux2" ? overrides : {};

  const request: AsyncRequest = {
    inputs: getPromptForEffect(effectId, family),
    // Strip any data-URI prefix so the endpoint receives raw base64 only.
    images: [stripDataUriPrefix(photo)],
    num_inference_steps: resolveInRange(
      steps,
      tuned.steps ?? sampling.defaults.num_inference_steps,
      sampling.steps.min,
      sampling.steps.max,
    ),
    guidance_scale: resolveInRange(
      guidance,
      tuned.guidance ?? sampling.defaults.guidance_scale,
      sampling.guidance.min,
      sampling.guidance.max,
    ),
  };
  // Pass an explicit output size only when the effect requests one; the
  // endpoint clamps to [256,1536] and snaps to a multiple of 16.
  if (overrides.width !== undefined) {
    request.width = overrides.width;
  }
  if (overrides.height !== undefined) {
    request.height = overrides.height;
  }
  if (seed !== undefined) {
    request.seed = seed;
  }
  return request;
}

/** Parameters for {@link buildMergeRequest}. */
export interface BuildMergeRequestParams {
  /** The fully-formed multi-reference merge prompt (see `buildMergePrompt`). */
  prompt: string;
  /**
   * Reference images, in prompt order (image 1, image 2, …), as base64 strings.
   * Any `data:` URI prefix is stripped so the endpoint receives raw base64. The
   * endpoint accepts up to 4 references; the booth merge uses exactly 2.
   */
  images: string[];
  /** Model family of the current endpoint (default `flux2`). */
  family?: ModelFamily;
  /** Optional override for `num_inference_steps`; defaults to the family default. */
  steps?: number;
  /** Optional override for `guidance_scale`; defaults to the family default. */
  guidance?: number;
  /** Optional fixed seed; passed through unchanged when provided. */
  seed?: number;
}

/**
 * Build an {@link AsyncRequest} for a multi-reference MERGE: an explicit prompt
 * plus two or more reference images. Unlike {@link buildAsyncRequest} (which
 * resolves a single effect prompt and sends `[photo]`), this passes the prompt
 * through verbatim and forwards every reference image, so the endpoint composes
 * across all of them.
 */
export function buildMergeRequest({
  prompt,
  images,
  family = "flux2",
  steps,
  guidance,
  seed,
}: BuildMergeRequestParams): AsyncRequest {
  const sampling = SAMPLING[family];
  const request: AsyncRequest = {
    inputs: prompt,
    images: images.map(stripDataUriPrefix),
    num_inference_steps: resolveInRange(
      steps,
      sampling.defaults.num_inference_steps,
      sampling.steps.min,
      sampling.steps.max,
    ),
    guidance_scale: resolveInRange(
      guidance,
      sampling.defaults.guidance_scale,
      sampling.guidance.min,
      sampling.guidance.max,
    ),
  };
  if (seed !== undefined) {
    request.seed = seed;
  }
  return request;
}
