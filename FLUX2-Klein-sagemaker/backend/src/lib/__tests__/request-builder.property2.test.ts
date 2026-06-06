import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { buildAsyncRequest, stripDataUriPrefix } from "../request-builder.js";
import { EFFECTS, getPromptForEffect } from "../effects.js";

// Feature: ai-photo-booth, Property 2: Async_Request maps the selected prompt and embeds the photo — inputs equals the mapped prompt and images equals exactly [photo] (length 1, <= 4)

const EFFECT_IDS = EFFECTS.map((e) => e.id);

describe("Property 2: Async_Request maps the selected prompt and embeds the photo", () => {
  it("sets inputs to the mapped prompt and images to exactly [photo] (data-URI prefix stripped)", () => {
    fc.assert(
      fc.property(fc.constantFrom(...EFFECT_IDS), fc.string(), (effectId, photo) => {
        const request = buildAsyncRequest({ effectId, photo });

        // inputs equals the prompt mapped to the selected effect.
        expect(request.inputs).toBe(getPromptForEffect(effectId));

        // images is exactly the photo with any data-URI prefix stripped: the
        // endpoint requires raw base64. For a plain string this is unchanged.
        expect(request.images).toEqual([stripDataUriPrefix(photo)]);
        expect(request.images).toHaveLength(1);
        expect(request.images.length).toBeLessThanOrEqual(4);
      }),
      { numRuns: 100 },
    );
  });

  it("strips a data:image/png;base64 prefix, keeping only the base64 payload", () => {
    const request = buildAsyncRequest({
      effectId: EFFECT_IDS[0],
      photo: "data:image/png;base64,iVBORw0KGgo=",
    });
    expect(request.images).toEqual(["iVBORw0KGgo="]);
  });
});
