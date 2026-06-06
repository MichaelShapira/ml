/**
 * Async_Request builder (pure logic — no AWS SDK, no I/O).
 *
 * The Generation_Service resolves the visitor's selected effect to a prompt and
 * assembles the single JSON inference request submitted to the FLUX2_Endpoint
 * via the S3 input location. This module owns that assembly: it maps the
 * selected effect to its prompt (via `effects.ts`), embeds the captured photo as
 * the sole reference image, and clamps the inference parameters into the booth's
 * conservative ranges so low values that degrade output quality are avoided.
 *
 * The booth ranges — `num_inference_steps` ∈ [4, 20] and `guidance_scale` ∈
 * [1, 10] — are a strict subset of the ranges the endpoint itself accepts
 * (`[1, 20]` / `[0, 10]`), so a clamped request is always valid server-side.
 *
 * This module is imported by both the scheduler-side code and the browser's
 * `ui/src/api/generation.ts`, so the SPA and any server-side caller build the
 * request with the same shared, property-tested logic.
 *
 * Requirements: 7.2, 7.3, 7.4 — design Properties 2 and 3.
 */

import { getPromptForEffect } from "./effects";

/**
 * Strip a `data:` URI prefix from a base64 image string, returning only the raw
 * base64 payload. Browser captures arrive as `data:image/png;base64,iVBOR...`,
 * but the endpoint's `inference.py` decodes with `b64decode(..., validate=True)`
 * which rejects the prefix. A string without a prefix is returned unchanged.
 */
export function stripDataUriPrefix(value: string): string {
  const match = /^data:[^;,]*(?:;[^,]*)?,(.*)$/s.exec(value);
  return match ? match[1] : value;
}

/**
 * A single JSON inference request submitted to the FLUX2_Endpoint.
 *
 * Mirrors the subset of the endpoint's request schema the photo booth uses. The
 * endpoint derives output dimensions from the first reference image when
 * editing, so the booth never sends `height`/`width`.
 */
export interface AsyncRequest {
  /** Prompt text mapped from the selected effect (Requirement 7.2). */
  inputs: string;
  /**
   * Base64-encoded reference image(s). The booth always sends exactly the
   * captured photo, so this is `[photo]` (length 1; the endpoint allows ≤ 4).
   * (Requirement 7.3.)
   */
  images: string[];
  /** Denoising steps, clamped to the booth range [4, 20] (Requirement 7.4). */
  num_inference_steps: number;
  /** Guidance scale, clamped to the booth range [1, 10] (Requirement 7.4). */
  guidance_scale: number;
  /** Optional fixed seed for reproducible generation. */
  seed?: number;
}

/** Inclusive booth bounds for `num_inference_steps`. */
export const STEPS_RANGE = { min: 4, max: 20 } as const;

/** Inclusive booth bounds for `guidance_scale`. */
export const GUIDANCE_RANGE = { min: 1, max: 10 } as const;

/**
 * Booth defaults applied when a caller omits a parameter. Both sit inside the
 * booth ranges, so an omitted (or non-finite) input always yields an in-range
 * value.
 */
export const DEFAULTS = { num_inference_steps: 6, guidance_scale: 2.5 } as const;

/** Parameters for {@link buildAsyncRequest}. */
export interface BuildAsyncRequestParams {
  /** Effect id selected by the visitor; resolved to a prompt via `effects.ts`. */
  effectId: string;
  /** The captured photo as a base64-encoded PNG/JPEG string. */
  photo: string;
  /** Optional override for `num_inference_steps`; defaults to {@link DEFAULTS}. */
  steps?: number;
  /** Optional override for `guidance_scale`; defaults to {@link DEFAULTS}. */
  guidance?: number;
  /** Optional fixed seed; passed through unchanged when provided. */
  seed?: number;
}

/**
 * Resolve a parameter to an in-range value: fall back to `fallback` when the
 * input is omitted or non-finite (`NaN`, `±Infinity`), then clamp into
 * `[min, max]`. Because `fallback` is itself in range, the result is always
 * within `[min, max]`, and any finite in-range input is returned unchanged.
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
 *
 * - `inputs` is the prompt mapped from `effectId` (Requirement 7.2). Unknown
 *   effect ids are rejected: the {@link UnknownEffectError} thrown by
 *   `getPromptForEffect` propagates to the caller.
 * - `images` is exactly `[photo]` (length 1) (Requirement 7.3).
 * - `num_inference_steps` and `guidance_scale` are clamped into the booth
 *   ranges, with the booth defaults applied when omitted or non-finite, so the
 *   result is always in range (Requirement 7.4).
 * - `seed` is included only when the caller provides one.
 *
 * @throws {import("./effects").UnknownEffectError} when `effectId` is unknown.
 */
export function buildAsyncRequest({
  effectId,
  photo,
  steps,
  guidance,
  seed,
}: BuildAsyncRequestParams): AsyncRequest {
  const request: AsyncRequest = {
    inputs: getPromptForEffect(effectId),
    // Strip any data-URI prefix so the endpoint receives raw base64 only.
    images: [stripDataUriPrefix(photo)],
    num_inference_steps: resolveInRange(
      steps,
      DEFAULTS.num_inference_steps,
      STEPS_RANGE.min,
      STEPS_RANGE.max,
    ),
    guidance_scale: resolveInRange(
      guidance,
      DEFAULTS.guidance_scale,
      GUIDANCE_RANGE.min,
      GUIDANCE_RANGE.max,
    ),
  };

  if (seed !== undefined) {
    request.seed = seed;
  }

  return request;
}
