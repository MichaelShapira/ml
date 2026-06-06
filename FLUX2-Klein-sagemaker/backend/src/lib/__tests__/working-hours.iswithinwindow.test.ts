import { describe, it, expect } from "vitest";
import { isWithinWindow } from "../working-hours.js";

// Unit test (not a property test) for isWithinWindow half-open interval semantics.
// Validates: Requirements 21.3, 21.4

describe("isWithinWindow — half-open interval [startTime, endTime)", () => {
  const start = "09:00";
  const end = "17:00";

  it("includes the start time (inclusive lower bound)", () => {
    expect(isWithinWindow("09:00", start, end)).toBe(true);
  });

  it("excludes the end time (exclusive upper bound)", () => {
    expect(isWithinWindow("17:00", start, end)).toBe(false);
  });

  it("treats a time strictly inside the window as inside", () => {
    expect(isWithinWindow("12:30", start, end)).toBe(true);
    expect(isWithinWindow("16:59", start, end)).toBe(true);
  });

  it("treats a time before the start as outside", () => {
    expect(isWithinWindow("08:59", start, end)).toBe(false);
    expect(isWithinWindow("00:00", start, end)).toBe(false);
  });

  it("treats a time after the end as outside", () => {
    expect(isWithinWindow("17:01", start, end)).toBe(false);
    expect(isWithinWindow("23:59", start, end)).toBe(false);
  });

  it("returns false for a malformed 'now' time", () => {
    expect(isWithinWindow("9:00", start, end)).toBe(false);
    expect(isWithinWindow("24:00", start, end)).toBe(false);
    expect(isWithinWindow("", start, end)).toBe(false);
    expect(isWithinWindow("noon", start, end)).toBe(false);
  });

  it("returns false for an invalid window (end <= start), even for a time between them numerically", () => {
    // end equals start -> not a valid window.
    expect(isWithinWindow("12:00", "12:00", "12:00")).toBe(false);
    // end strictly before start -> invalid window.
    expect(isWithinWindow("18:00", "17:00", "09:00")).toBe(false);
  });

  it("returns false for malformed window time strings", () => {
    expect(isWithinWindow("12:00", "9:00", "17:00")).toBe(false);
    expect(isWithinWindow("12:00", "09:00", "25:00")).toBe(false);
  });
});
