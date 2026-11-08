/**
 * Request building per model family: FLUX.2 keeps the booth's tuned prompts and
 * sampling exactly as before; FLUX 3 gets its own prompts, 16 steps and CFG 1.
 */
import { describe, it, expect, afterEach } from "vitest";

import {
  buildAsyncRequest,
  buildMergeRequest,
  DEFAULTS,
  SAMPLING,
} from "./requestBuilder";
import {
  EFFECTS,
  applyPromptOverrides,
  buildMergePrompt,
  getDefaultPromptForEffect,
  getPromptForEffect,
} from "../booth/effects";

afterEach(() => {
  applyPromptOverrides(new Map(), "flux2");
  applyPromptOverrides(new Map(), "flux3");
});

describe("buildAsyncRequest", () => {
  it("is unchanged for FLUX.2 (the default family)", () => {
    const req = buildAsyncRequest({ effectId: "bg_spaceship", photo: "data:image/png;base64,AAAA" });
    expect(req.inputs).toBe(EFFECTS.find((e) => e.id === "bg_spaceship")!.prompt);
    expect(req.images).toEqual(["AAAA"]);
    expect(req.num_inference_steps).toBe(DEFAULTS.num_inference_steps);
    expect(req.guidance_scale).toBe(DEFAULTS.guidance_scale);
  });

  it("sends the FLUX 3 prompt, 16 steps and guidance 1 for FLUX 3", () => {
    const req = buildAsyncRequest({ effectId: "person_viking", photo: "AAAA", family: "flux3" });
    expect(req.inputs).toBe(EFFECTS.find((e) => e.id === "person_viking")!.flux3Prompt);
    expect(req.num_inference_steps).toBe(16);
    expect(req.guidance_scale).toBe(1);
  });

  it("clamps FLUX 3 steps to the distilled model's 4..16 range", () => {
    expect(
      buildAsyncRequest({ effectId: "bg_lunar", photo: "A", family: "flux3", steps: 30 }).num_inference_steps,
    ).toBe(16);
    expect(
      buildAsyncRequest({ effectId: "bg_lunar", photo: "A", family: "flux3", steps: 1 }).num_inference_steps,
    ).toBe(4);
    expect(
      buildAsyncRequest({ effectId: "bg_lunar", photo: "A", family: "flux3", guidance: 7 }).guidance_scale,
    ).toBe(1);
  });

  it("uses a family's admin override only for that family", () => {
    applyPromptOverrides(new Map([["bg_tropical", "custom flux3 prompt"]]), "flux3");
    expect(buildAsyncRequest({ effectId: "bg_tropical", photo: "A", family: "flux3" }).inputs).toBe(
      "custom flux3 prompt",
    );
    expect(buildAsyncRequest({ effectId: "bg_tropical", photo: "A" }).inputs).toBe(
      getDefaultPromptForEffect("bg_tropical", "flux2"),
    );
  });
});

describe("buildMergeRequest", () => {
  it("uses the family's sampling", () => {
    expect(buildMergeRequest({ prompt: "p", images: ["a", "b"] }).num_inference_steps).toBe(
      DEFAULTS.num_inference_steps,
    );
    const flux3 = buildMergeRequest({ prompt: "p", images: ["a", "b"], family: "flux3" });
    expect(flux3.num_inference_steps).toBe(SAMPLING.flux3.defaults.num_inference_steps);
    expect(flux3.guidance_scale).toBe(1);
  });
});

describe("FLUX 3 prompts (docs.bfl.ai editing guidance)", () => {
  it("every effect has a FLUX 3 prompt distinct from its FLUX.2 prompt", () => {
    for (const effect of EFFECTS) {
      expect(effect.flux3Prompt.length, effect.id).toBeGreaterThan(40);
      expect(effect.flux3Prompt, effect.id).not.toBe(effect.prompt);
      expect(getPromptForEffect(effect.id, "flux3")).toBe(effect.flux3Prompt);
    }
  });

  it("phrases what stays positively: no 'do not' / 'don't' negations", () => {
    for (const effect of EFFECTS) {
      expect(effect.flux3Prompt, effect.id).not.toMatch(/\bdo not\b|\bdon't\b|\bnever\b/i);
    }
    expect(buildMergePrompt("bg_spaceship", "person_viking", "flux3")).not.toMatch(
      /\bdo not\b|\bdon't\b|\bnever\b/i,
    );
  });

  it("refers to the merge references by position, base image first", () => {
    const prompt = buildMergePrompt("bg_neon_city", "person_knight", "flux3");
    expect(prompt).toMatch(/^In image 1, replace each person with the same person from image 2/);
    expect(prompt).toContain("polished plate armor");
    expect(prompt).toContain("neon-lit city street");
  });

  it("keeps the FLUX.2 merge prompt unchanged", () => {
    expect(buildMergePrompt("bg_spaceship", "person_viking")).toMatch(
      /^Create a single photorealistic photograph that contains EXACTLY ONE person\./,
    );
  });
});
