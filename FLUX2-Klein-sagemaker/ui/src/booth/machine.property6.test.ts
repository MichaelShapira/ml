// Feature: ai-photo-booth, Property 6: Effect selection records the choice, enters Loading, and locks out further selections
//
// Validates: Requirements 5.5, 6.5, 6.6, 9.1
//
// For any Effects state and for any valid effect id, dispatching SELECT(id)
// transitions to Loading with the selected effect recorded; and for any
// sequence of subsequent SELECT events while in Loading, the state remains
// Loading and the recorded effect stays the first selection.

import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  transition,
  type EffectsState,
  type BoothEvent,
} from "./machine";

const photoArb = fc.string();
const effectIdArb = fc.string();

const effectsArb: fc.Arbitrary<EffectsState> = fc.record({
  name: fc.constant("Effects" as const),
  capturedPhoto: photoArb,
});

describe("Property 6: effect selection enters Loading and locks the first selection", () => {
  it("SELECT from Effects -> Loading with the chosen effect and retained photo", () => {
    fc.assert(
      fc.property(effectsArb, effectIdArb, (effects, effectId) => {
        const next = transition(effects, { type: "SELECT", effectId });
        expect(next).toEqual({
          name: "Loading",
          capturedPhoto: effects.capturedPhoto,
          selectedEffectId: effectId,
        });
      }),
      { numRuns: 100 },
    );
  });

  it("subsequent SELECT events while Loading are ignored; the first selection is retained", () => {
    fc.assert(
      fc.property(
        effectsArb,
        effectIdArb,
        fc.array(effectIdArb, { maxLength: 10 }),
        (effects, firstEffectId, laterEffectIds) => {
          // First selection locks in the effect and begins Loading.
          const loading = transition(effects, { type: "SELECT", effectId: firstEffectId });
          expect(loading.name).toBe("Loading");

          // Replay any sequence of further SELECT events while Loading.
          const selectEvents: BoothEvent[] = laterEffectIds.map((effectId) => ({
            type: "SELECT",
            effectId,
          }));
          const result = selectEvents.reduce(transition, loading);

          // State stays Loading and the recorded effect is still the first one.
          expect(result).toEqual({
            name: "Loading",
            capturedPhoto: effects.capturedPhoto,
            selectedEffectId: firstEffectId,
          });
        },
      ),
      { numRuns: 100 },
    );
  });
});
