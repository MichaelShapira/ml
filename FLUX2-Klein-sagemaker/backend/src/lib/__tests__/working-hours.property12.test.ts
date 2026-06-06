import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { validateWorkingHours } from "../working-hours.js";

// Feature: ai-photo-booth, Property 12: Working_Hours validity requires end strictly after start — valid iff the end time is strictly later than the start time

/** Build a well-formed 24-hour HH:mm string from hours/minutes. */
function hhmm(hours: number, minutes: number): string {
  const hh = String(hours).padStart(2, "0");
  const mm = String(minutes).padStart(2, "0");
  return `${hh}:${mm}`;
}

const hoursArb = fc.integer({ min: 0, max: 23 });
const minutesArb = fc.integer({ min: 0, max: 59 });

describe("Property 12: Working_Hours validity requires end strictly after start", () => {
  it("is valid iff end minutes-of-day are strictly greater than start", () => {
    fc.assert(
      fc.property(
        hoursArb,
        minutesArb,
        hoursArb,
        minutesArb,
        (sh, sm, eh, em) => {
          const startTime = hhmm(sh, sm);
          const endTime = hhmm(eh, em);
          const startMinutes = sh * 60 + sm;
          const endMinutes = eh * 60 + em;

          expect(validateWorkingHours(startTime, endTime)).toBe(
            endMinutes > startMinutes,
          );
        },
      ),
      { numRuns: 100 },
    );
  });

  it("rejects malformed time strings", () => {
    fc.assert(
      fc.property(
        fc.constantFrom("24:00", "12:60", "9:00", "0900", "", "ab:cd", "1:2"),
        fc.constantFrom("10:00", "23:59", "00:00"),
        (bad, good) => {
          expect(validateWorkingHours(bad, good)).toBe(false);
          expect(validateWorkingHours(good, bad)).toBe(false);
        },
      ),
      { numRuns: 100 },
    );
  });
});
