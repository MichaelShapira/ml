/**
 * Per-family admin prompt overrides in the Schedule_Store: FLUX.2 rows keep
 * their original keys (sk = EFFECT#<id>); FLUX 3 rows are EFFECT#<id>#flux3.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { PutCommand } from "@aws-sdk/lib-dynamodb";

const dynamoSend = vi.fn();
vi.mock("./awsClients", () => ({
  getDynamoClient: () => ({ send: dynamoSend }),
  withAuthRetry: <T>(run: () => Promise<T>) => run(),
}));
vi.mock("../config", () => ({
  getConfig: () => ({ scheduleTable: "Schedule" }),
}));

import { loadEffectPrompts, promptOverrideSk, restorePromptDefault, savePromptOverride } from "./prompts";
import { applyPromptOverrides, getDefaultPromptForEffect, getPromptForEffect } from "../booth/effects";

beforeEach(() => {
  dynamoSend.mockReset();
  applyPromptOverrides(new Map(), "flux2");
  applyPromptOverrides(new Map(), "flux3");
});

describe("promptOverrideSk", () => {
  it("keeps the original FLUX.2 key and suffixes FLUX 3", () => {
    expect(promptOverrideSk("bg_lunar")).toBe("EFFECT#bg_lunar");
    expect(promptOverrideSk("bg_lunar", "flux2")).toBe("EFFECT#bg_lunar");
    expect(promptOverrideSk("bg_lunar", "flux3")).toBe("EFFECT#bg_lunar#flux3");
  });
});

describe("loadEffectPrompts", () => {
  it("splits custom rows by family; rows without a family are FLUX.2", async () => {
    dynamoSend.mockResolvedValue({
      Items: [
        { effectId: "bg_lunar", prompt: "old flux2 custom", isCustom: true },
        { effectId: "bg_lunar", family: "flux3", prompt: "flux3 custom", isCustom: true },
        { effectId: "bg_tropical", family: "flux3", prompt: "seeded", isCustom: false },
      ],
    });
    const overrides = await loadEffectPrompts();
    expect([...overrides.flux2]).toEqual([["bg_lunar", "old flux2 custom"]]);
    expect([...overrides.flux3]).toEqual([["bg_lunar", "flux3 custom"]]);
    expect(getPromptForEffect("bg_lunar", "flux2")).toBe("old flux2 custom");
    expect(getPromptForEffect("bg_lunar", "flux3")).toBe("flux3 custom");
    expect(getPromptForEffect("bg_tropical", "flux3")).toBe(getDefaultPromptForEffect("bg_tropical", "flux3"));
  });
});

describe("save / restore", () => {
  it("writes a FLUX 3 override under its own key with family=flux3", async () => {
    dynamoSend.mockResolvedValue({ Items: [] });
    await savePromptOverride("person_knight", "  a knight  ", "admin", "flux3");
    const put = dynamoSend.mock.calls.find((c) => c[0] instanceof PutCommand)![0];
    expect(put.input.Item).toMatchObject({
      pk: "PROMPT#OVERRIDE",
      sk: "EFFECT#person_knight#flux3",
      family: "flux3",
      prompt: "a knight",
      isCustom: true,
    });
  });

  it("restores the FLUX 3 default for FLUX 3, and writes FLUX.2 rows without a family", async () => {
    dynamoSend.mockResolvedValue({ Items: [] });
    await restorePromptDefault("person_knight", "admin", "flux3");
    await savePromptOverride("person_knight", "flux2 text");
    const puts = dynamoSend.mock.calls.filter((c) => c[0] instanceof PutCommand).map((c) => c[0].input.Item);
    expect(puts[0]).toMatchObject({
      sk: "EFFECT#person_knight#flux3",
      prompt: getDefaultPromptForEffect("person_knight", "flux3"),
      isCustom: false,
    });
    expect(puts[1].sk).toBe("EFFECT#person_knight");
    expect(puts[1]).not.toHaveProperty("family");
  });
});
