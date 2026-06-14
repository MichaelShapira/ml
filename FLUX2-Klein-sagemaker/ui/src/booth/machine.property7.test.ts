// Feature: ai-photo-booth, Property 7: Returning to Start by any path clears all session data
//
// Validates: Requirements 10.4, 10.5
//
// For all machine states and for any event that results in the Start state
// (including New Session and idle auto-reset), the resulting state carries
// neither a captured photo nor a transformed image.

import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  transition,
  type BoothState,
  type BoothEvent,
  type ErrorReason,
} from "./machine";

const photoArb = fc.string();
const effectIdArb = fc.string();

const errorReasonArb: fc.Arbitrary<ErrorReason> = fc.record({
  kind: fc.constantFrom("FAILED" as const, "TIMEOUT" as const),
});

const stateArb: fc.Arbitrary<BoothState> = fc.oneof(
  fc.constant<BoothState>({ name: "SignedOut" }),
  fc.constant<BoothState>({ name: "Start" }),
  fc.constant<BoothState>({ name: "Camera" }),
  fc.record({ name: fc.constant("Review" as const), capturedPhoto: photoArb }),
  fc.record({ name: fc.constant("Effects" as const), capturedPhoto: photoArb, results: fc.constant({}) }),
  fc.record({
    name: fc.constant("Loading" as const),
    capturedPhoto: photoArb,
    job: fc.record({
      kind: fc.constant("effect" as const),
      effectId: effectIdArb,
      slot: fc.constantFrom("background" as const, "person" as const),
    }),
    results: fc.constant({}),
  }),
  fc.record({
    name: fc.constant("Result" as const),
    capturedPhoto: photoArb,
    results: fc.constant({}),
  }),
  fc.record({
    name: fc.constant("Error" as const),
    reason: errorReasonArb,
    capturedPhoto: photoArb,
    results: fc.constant({}),
  }),
);

const eventArb: fc.Arbitrary<BoothEvent> = fc.oneof(
  fc.constant<BoothEvent>({ type: "AUTHENTICATED" }),
  fc.constant<BoothEvent>({ type: "SIGN_OUT" }),
  fc.constant<BoothEvent>({ type: "START" }),
  fc.record({ type: fc.constant("CAPTURE" as const), photo: photoArb }),
  fc.constant<BoothEvent>({ type: "RESET" }),
  fc.constant<BoothEvent>({ type: "CONTINUE" }),
  fc.record({
    type: fc.constant("SELECT" as const),
    effectId: effectIdArb,
    slot: fc.constantFrom("background" as const, "person" as const),
  }),
  fc.constant<BoothEvent>({ type: "MERGE" }),
  fc.constant<BoothEvent>({ type: "NEW_SESSION" }),
  fc.constant<BoothEvent>({ type: "IDLE_TIMEOUT" }),
  fc.record({ type: fc.constant("CAMERA_ERROR" as const), reason: fc.string() }),
  fc.constant<BoothEvent>({ type: "RETRY" }),
);

/** Visitor-facing states from which IDLE_TIMEOUT auto-resets to Start. */
const visitorFacingStateArb: fc.Arbitrary<BoothState> = fc.oneof(
  fc.constant<BoothState>({ name: "Start" }),
  fc.constant<BoothState>({ name: "Camera" }),
  fc.record({ name: fc.constant("Review" as const), capturedPhoto: photoArb }),
  fc.record({
    name: fc.constant("Effects" as const),
    capturedPhoto: photoArb,
    results: fc.constant({}),
  }),
  fc.record({
    name: fc.constant("Loading" as const),
    capturedPhoto: photoArb,
    job: fc.record({
      kind: fc.constant("effect" as const),
      effectId: effectIdArb,
      slot: fc.constantFrom("background" as const, "person" as const),
    }),
    results: fc.constant({}),
  }),
  fc.record({
    name: fc.constant("Result" as const),
    capturedPhoto: photoArb,
    results: fc.constant({}),
  }),
  fc.record({
    name: fc.constant("Error" as const),
    reason: errorReasonArb,
    capturedPhoto: photoArb,
    results: fc.constant({}),
  }),
);

/** A Start state must carry no captured photo and no generated results. */
function expectCleanStart(state: BoothState): void {
  expect(state.name).toBe("Start");
  expect("capturedPhoto" in state).toBe(false);
  expect("results" in state).toBe(false);
  expect("job" in state).toBe(false);
}

describe("Property 7: any path back to Start clears all session data", () => {
  it("whenever an arbitrary (state, event) yields Start, it carries no session data", () => {
    fc.assert(
      fc.property(stateArb, eventArb, (state, event) => {
        const next = transition(state, event);
        if (next.name === "Start") {
          expectCleanStart(next);
        }
      }),
      { numRuns: 100 },
    );
  });

  it("NEW_SESSION from Result returns to a clean Start", () => {
    fc.assert(
      fc.property(photoArb, photoArb, effectIdArb, (image, capturedPhoto, effectId) => {
        const next = transition(
          {
            name: "Result",
            capturedPhoto,
            results: { background: { image, effectId } },
          },
          { type: "NEW_SESSION" },
        );
        expectCleanStart(next);
      }),
      { numRuns: 100 },
    );
  });

  it("IDLE_TIMEOUT from any visitor-facing state returns to a clean Start", () => {
    fc.assert(
      fc.property(visitorFacingStateArb, (state) => {
        const next = transition(state, { type: "IDLE_TIMEOUT" });
        expectCleanStart(next);
      }),
      { numRuns: 100 },
    );
  });
});
