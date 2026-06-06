/**
 * Unit tests for the SPA timezone helpers — the logic that must agree with the
 * scheduler's wall-clock computation so manual-start windows land correctly.
 */
import { describe, it, expect, vi } from "vitest";

vi.mock("../config", () => ({
  getConfig: () => ({ timezone: "Asia/Jerusalem" }),
}));

import {
  computeWallClock,
  toMinutes,
  fromMinutes,
  addMinutesCapped,
  minTime,
  maxTime,
} from "./timezone";

describe("computeWallClock", () => {
  it("converts a UTC instant to the target timezone wall-clock", () => {
    // 2026-06-05T20:50:00Z → Jerusalem is UTC+3 (DST) → 23:50 same day.
    const at = new Date("2026-06-05T20:50:00Z");
    expect(computeWallClock("Asia/Jerusalem", at)).toEqual({
      day: "2026-06-05",
      time: "23:50",
    });
  });

  it("rolls the day when the timezone offset crosses midnight", () => {
    // 2026-06-05T22:30:00Z → Jerusalem +3 → 01:30 on 2026-06-06.
    const at = new Date("2026-06-05T22:30:00Z");
    expect(computeWallClock("Asia/Jerusalem", at)).toEqual({
      day: "2026-06-06",
      time: "01:30",
    });
  });
});

describe("HH:mm arithmetic", () => {
  it("toMinutes / fromMinutes round-trip", () => {
    expect(toMinutes("00:00")).toBe(0);
    expect(toMinutes("23:59")).toBe(1439);
    expect(toMinutes("bad")).toBeNull();
    expect(fromMinutes(0)).toBe("00:00");
    expect(fromMinutes(1439)).toBe("23:59");
  });

  it("addMinutesCapped adds and caps at 23:59", () => {
    expect(addMinutesCapped("22:00", 20)).toBe("22:20");
    expect(addMinutesCapped("23:50", 20)).toBe("23:59");
  });

  it("minTime / maxTime pick the earlier / later time", () => {
    expect(minTime("22:00", "09:00")).toBe("09:00");
    expect(maxTime("22:00", "23:00")).toBe("23:00");
  });
});
