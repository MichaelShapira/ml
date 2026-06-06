import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { toItem, fromItem, type WorkingHours } from "../working-hours.js";

// Feature: ai-photo-booth, Property 13: Working_Hours serialization round-trips — serializing a valid Working_Hours to a Schedule_Store item and parsing it back yields the same day, start time, and end time

/** Build a well-formed 24-hour HH:mm string from hours/minutes. */
function hhmm(hours: number, minutes: number): string {
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
}

/** Build a YYYY-MM-DD date string from components. */
function isoDay(year: number, month: number, dayOfMonth: number): string {
  return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(
    dayOfMonth,
  ).padStart(2, "0")}`;
}

const workingHoursArb: fc.Arbitrary<WorkingHours> = fc
  .record({
    endpointName: fc.string({ minLength: 1 }),
    year: fc.integer({ min: 2000, max: 2099 }),
    month: fc.integer({ min: 1, max: 12 }),
    dayOfMonth: fc.integer({ min: 1, max: 28 }),
    startHour: fc.integer({ min: 0, max: 22 }),
    startMinute: fc.integer({ min: 0, max: 59 }),
    endHour: fc.integer({ min: 1, max: 23 }),
    endMinute: fc.integer({ min: 0, max: 59 }),
    updatedBy: fc.string(),
    updatedAt: fc.string(),
  })
  .map((r) => ({
    endpointName: r.endpointName,
    day: isoDay(r.year, r.month, r.dayOfMonth),
    startTime: hhmm(r.startHour, r.startMinute),
    endTime: hhmm(r.endHour, r.endMinute),
    updatedBy: r.updatedBy,
    updatedAt: r.updatedAt,
  }));

describe("Property 13: Working_Hours serialization round-trips", () => {
  it("fromItem(toItem(x)) preserves day, startTime, and endTime", () => {
    fc.assert(
      fc.property(workingHoursArb, (hours) => {
        const roundTripped = fromItem(toItem(hours));
        expect(roundTripped.day).toBe(hours.day);
        expect(roundTripped.startTime).toBe(hours.startTime);
        expect(roundTripped.endTime).toBe(hours.endTime);
        // The endpoint name also recovers cleanly from the pk prefix.
        expect(roundTripped.endpointName).toBe(hours.endpointName);
      }),
      { numRuns: 100 },
    );
  });
});
