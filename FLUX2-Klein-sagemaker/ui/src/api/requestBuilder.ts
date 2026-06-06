/**
 * Async_Request builder for the SPA (Requirements 7.2, 7.3, 7.4).
 *
 * Mirrors `backend/src/lib/request-builder.ts`. The UI package is type-checked
 * in isolation and cannot import `backend/src` (TS6307), so the same pure,
 * property-tested logic is duplicated here. Keeping the booth ranges and
 * defaults identical guarantees the browser builds the same request a
 * server-side caller would.
 */

import { getPromptForEffect } from "../booth/effects";

/** A single JSON inference request submitted to the FLUX2_Endpoint. */
export interface AsyncRequest {
  /** Prompt text mapped from the selected effect. */
  inputs: string;
  /** Base64-encoded reference image(s); the booth always sends `[photo]`. */
  images: string[];
  /** Denoising steps, clamped to the booth range [4, 20]. */
  num_inference_steps: number;
  /** Guidance scale, clamped to the booth range [1, 10]. */
  guidance_scale: number;
  /** Optional fixed seed for reproducible generation. */
  seed?: number;
}

/** Inclusive booth bounds for `num_inference_steps`. */
export const STEPS_RANGE = { min: 4, max: 20 } as const;

/** Inclusive booth bounds for `guidance_scale`. */
export const GUIDANCE_RANGE = { min: 1, max: 10 } as const;

/** Booth defaults applied when a caller omits a parameter. */
export const DEFAULTS = { num_inference_steps: 6, guidance_scale: 2.5 } as const;

/** Parameters for {@link buildAsyncRequest}. */
export interface BuildAsyncRequestParams {
  /** Effect id selected by the visitor; resolved to a prompt via the catalog. */
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
