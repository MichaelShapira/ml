import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { EFFECTS, type EffectOption } from "../effects.js";

// Feature: ai-photo-booth, Property 1: Effect catalog shape and total prompt mapping — exactly 12 options (6 background + 6 person), unique ids, non-empty label and prompt for every option

describe("Property 1: effect catalog shape and total prompt mapping", () => {
  it("has exactly 12 options split 6 background + 6 person with unique ids", () => {
    expect(EFFECTS).toHaveLength(12);

    const background = EFFECTS.filter((e) => e.category === "background");
    const person = EFFECTS.filter((e) => e.category === "person");
    expect(background).toHaveLength(6);
    expect(person).toHaveLength(6);

    const ids = EFFECTS.map((e) => e.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("every catalog option has a non-empty label and prompt, a known category, and a unique id", () => {
    const ids = EFFECTS.map((e) => e.id);

    fc.assert(
      fc.property(fc.constantFrom<EffectOption>(...EFFECTS), (option) => {
        // Non-empty label and prompt (so the prompt mapping is total).
        expect(option.label.trim().length).toBeGreaterThan(0);
        expect(option.prompt.trim().length).toBeGreaterThan(0);
        // Category is one of the two valid categories.
        expect(option.category === "background" || option.category === "person").toBe(true);
        // The id appears exactly once across the catalog (uniqueness).
        expect(ids.filter((id) => id === option.id)).toHaveLength(1);
      }),
      { numRuns: 100 },
    );
  });
});
