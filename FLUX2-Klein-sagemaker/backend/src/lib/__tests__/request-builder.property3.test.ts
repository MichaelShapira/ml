import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  buildAsyncRequest,
  STEPS_RANGE,
  GUIDANCE_RANGE,
} from "../request-builder.js";

// Feature: ai-photo-booth, Property 3: Inference parameters are clamped into the booth ranges — for any numeric inputs the result stays in [4, 20] / [1, 10], and in-range inputs are preserved unchanged

describe("Property 3: inference parameters are clamped into the booth ranges", () => {
  it("clamps any numeric steps/guidance into [4,20] and [1,10]", () => {
    fc.assert(
      fc.property(
        // Cover out-of-range, negative, zero, fractional, and very large values,
        // including non-finite ones (NaN / +/-Infinity).
        fc.oneof(
          fc.double({ noNaN: false }),
          fc.integer(),
          fc.constantFrom(
            Number.NaN,
            Number.POSITIVE_INFINITY,
            Number.NEGATIVE_INFINITY,
            0,
            -1,
            1e9,
          ),
        ),
        fc.oneof(
          fc.double({ noNaN: false }),
          fc.integer(),
          fc.constantFrom(
            Number.NaN,
            Number.POSITIVE_INFINITY,
            Number.NEGATIVE_INFINITY,
            0,
            -1,
            1e9,
          ),
        ),
        (steps, guidance) => {
          const request = buildAsyncRequest({
            effectId: "bg_spaceship",
            photo: "x",
            steps,
            guidance,
          });

          expect(request.num_inference_steps).toBeGreaterThanOrEqual(STEPS_RANGE.min);
          expect(request.num_inference_steps).toBeLessThanOrEqual(STEPS_RANGE.max);
          expect(request.guidance_scale).toBeGreaterThanOrEqual(GUIDANCE_RANGE.min);
          expect(request.guidance_scale).toBeLessThanOrEqual(GUIDANCE_RANGE.max);
        },
      ),
      { numRuns: 100 },
    );
  });

  it("preserves any finite in-range input unchanged", () => {
    fc.assert(
      fc.property(
        fc.double({
          min: STEPS_RANGE.min,
          max: STEPS_RANGE.max,
          noNaN: true,
          noDefaultInfinity: true,
        }),
        fc.double({
          min: GUIDANCE_RANGE.min,
          max: GUIDANCE_RANGE.max,
          noNaN: true,
          noDefaultInfinity: true,
        }),
        (steps, guidance) => {
          const request = buildAsyncRequest({
            effectId: "bg_spaceship",
            photo: "x",
            steps,
            guidance,
          });

          expect(request.num_inference_steps).toBe(steps);
          expect(request.guidance_scale).toBe(guidance);
        },
      ),
      { numRuns: 100 },
    );
  });
});
