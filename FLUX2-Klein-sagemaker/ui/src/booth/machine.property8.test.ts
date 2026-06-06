// Feature: ai-photo-booth, Property 8: Continue retains the photo; Reset discards it and returns to camera
//
// Validates: Requirements 4.1, 4.2
//
// For any Review state holding a captured photo, dispatching CONTINUE
// transitions to Effects with the same photo retained, and dispatching RESET
// transitions to Camera with the captured photo discarded.

import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { transition, type ReviewState } from "./machine";

const photoArb = fc.string();

const reviewArb: fc.Arbitrary<ReviewState> = fc.record({
  name: fc.constant("Review" as const),
  capturedPhoto: photoArb,
});

describe("Property 8: Review CONTINUE retains the photo; RESET discards it", () => {
  it("CONTINUE from Review -> Effects retaining the same photo", () => {
    fc.assert(
      fc.property(reviewArb, (review) => {
        const next = transition(review, { type: "CONTINUE" });
        expect(next).toEqual({
          name: "Effects",
          capturedPhoto: review.capturedPhoto,
        });
      }),
      { numRuns: 100 },
    );
  });

  it("RESET from Review -> Camera discarding the captured photo", () => {
    fc.assert(
      fc.property(reviewArb, (review) => {
        const next = transition(review, { type: "RESET" });
        expect(next.name).toBe("Camera");
        // The captured photo is discarded: Camera carries no photo.
        expect("capturedPhoto" in next).toBe(false);
      }),
      { numRuns: 100 },
    );
  });
});
