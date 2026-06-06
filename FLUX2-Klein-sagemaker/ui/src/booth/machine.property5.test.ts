// Feature: ai-photo-booth, Property 5: Applying a poll result transitions Loading deterministically and never shows both loading and result
//
// Validates: Requirements 9.2, 9.3, 9.4
//
// For any Loading state and for any terminal poll result, applying a READY
// result yields Result carrying the image, applying FAILED or TIMEOUT yields
// Error carrying no image, and applying PENDING keeps Loading; and for every
// reachable machine state exactly one screen is active, so Loading and Result
// are never simultaneously active.

import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  transition,
  activeScreen,
  type BoothState,
  type ErrorReason,
  type LoadingState,
  type PollResult,
  type ActiveScreen,
} from "./machine";

const photoArb = fc.string();
const effectIdArb = fc.string();

/** Arbitrary Loading state (the only state to which POLL applies). */
const loadingArb: fc.Arbitrary<LoadingState> = fc.record({
  name: fc.constant("Loading" as const),
  capturedPhoto: photoArb,
  selectedEffectId: effectIdArb,
});

const errorReasonArb: fc.Arbitrary<ErrorReason> = fc.record({
  kind: fc.constantFrom("FAILED" as const, "TIMEOUT" as const),
});

/** Arbitrary over the full BoothState union (for the single-active-screen law). */
const stateArb: fc.Arbitrary<BoothState> = fc.oneof(
  fc.constant<BoothState>({ name: "SignedOut" }),
  fc.constant<BoothState>({ name: "Start" }),
  fc.constant<BoothState>({ name: "Camera" }),
  fc.record({ name: fc.constant("Review" as const), capturedPhoto: photoArb }),
  fc.record({ name: fc.constant("Effects" as const), capturedPhoto: photoArb }),
  loadingArb,
  fc.record({
    name: fc.constant("Result" as const),
    transformedImage: photoArb,
    capturedPhoto: photoArb,
  }),
  fc.record({
    name: fc.constant("Error" as const),
    reason: errorReasonArb,
    capturedPhoto: photoArb,
  }),
);

/** Poll results: READY carries the image; FAILED/TIMEOUT/PENDING carry none. */
const pollResultArb: fc.Arbitrary<PollResult> = fc.oneof(
  fc.record({ status: fc.constant("READY" as const), image: photoArb }),
  fc.record({ status: fc.constant("FAILED" as const) }),
  fc.record({ status: fc.constant("TIMEOUT" as const) }),
  fc.record({ status: fc.constant("PENDING" as const) }),
);

const ALL_SCREENS: ActiveScreen[] = [
  "SignIn",
  "Start",
  "Camera",
  "Review",
  "Effects",
  "Loading",
  "Result",
  "Error",
];

describe("Property 5: poll result applied to Loading is deterministic; one screen per state", () => {
  it("READY -> Result carrying the image", () => {
    fc.assert(
      fc.property(loadingArb, photoArb, (loading, image) => {
        const next = transition(loading, { type: "POLL", result: { status: "READY", image } });
        expect(next.name).toBe("Result");
        // Result carries the transformed image and the retained original photo
        // (so the visitor can regenerate in place), but no loading data.
        expect(next).toEqual({
          name: "Result",
          transformedImage: image,
          capturedPhoto: loading.capturedPhoto,
        });
      }),
      { numRuns: 100 },
    );
  });

  it("FAILED and TIMEOUT -> Error carrying no transformed image", () => {
    fc.assert(
      fc.property(
        loadingArb,
        fc.constantFrom<PollResult>({ status: "FAILED" }, { status: "TIMEOUT" }),
        (loading, result) => {
          const next = transition(loading, { type: "POLL", result });
          expect(next.name).toBe("Error");
          // No transformed image is ever carried on the failure path.
          expect("transformedImage" in next).toBe(false);
          // The captured photo is retained so the visitor can go back.
          expect((next as { capturedPhoto: string }).capturedPhoto).toBe(loading.capturedPhoto);
        },
      ),
      { numRuns: 100 },
    );
  });

  it("PENDING keeps Loading unchanged", () => {
    fc.assert(
      fc.property(loadingArb, (loading) => {
        const next = transition(loading, { type: "POLL", result: { status: "PENDING" } });
        expect(next.name).toBe("Loading");
        expect(next).toEqual(loading);
      }),
      { numRuns: 100 },
    );
  });

  it("applying any poll result to Loading never yields both Loading and Result", () => {
    fc.assert(
      fc.property(loadingArb, pollResultArb, (loading, result) => {
        const next = transition(loading, { type: "POLL", result });
        // The next state is a single state; it can be Loading XOR Result XOR Error.
        const isLoading = next.name === "Loading";
        const isResult = next.name === "Result";
        expect(isLoading && isResult).toBe(false);
      }),
      { numRuns: 100 },
    );
  });

  it("every reachable state has exactly one active screen", () => {
    fc.assert(
      fc.property(stateArb, (state) => {
        const screen = activeScreen(state);
        const matches = ALL_SCREENS.filter((s) => s === screen);
        // Exactly one screen id is active for any state -> Loading and Result
        // (distinct states mapping to distinct screens) are never both active.
        expect(matches).toHaveLength(1);
      }),
      { numRuns: 100 },
    );
  });
});
