/**
 * Unit tests for the rotating loading messages (Requirement 9.1/9.2).
 */
import { describe, it, expect } from "vitest";
import { LOADING_MESSAGES, messageForElapsed } from "./loadingMessages";

describe("LOADING_MESSAGES", () => {
  it("has at least 30 non-empty, unique messages", () => {
    expect(LOADING_MESSAGES.length).toBeGreaterThanOrEqual(30);
    for (const m of LOADING_MESSAGES) {
      expect(m.trim().length).toBeGreaterThan(0);
    }
    expect(new Set(LOADING_MESSAGES).size).toBe(LOADING_MESSAGES.length);
  });
});

describe("messageForElapsed", () => {
  it("returns the first message at t=0 and rotates by interval", () => {
    expect(messageForElapsed(0, 2500)).toBe(LOADING_MESSAGES[0]);
    expect(messageForElapsed(2500, 2500)).toBe(LOADING_MESSAGES[1]);
    expect(messageForElapsed(5000, 2500)).toBe(LOADING_MESSAGES[2]);
  });

  it("wraps around after the last message", () => {
    const n = LOADING_MESSAGES.length;
    expect(messageForElapsed(n * 2500, 2500)).toBe(LOADING_MESSAGES[0]);
  });

  it("handles negative / non-finite elapsed safely", () => {
    expect(messageForElapsed(-100, 2500)).toBe(LOADING_MESSAGES[0]);
    expect(messageForElapsed(Number.NaN, 2500)).toBe(LOADING_MESSAGES[0]);
  });
});
