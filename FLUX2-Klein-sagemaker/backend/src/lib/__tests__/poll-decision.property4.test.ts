import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { decidePoll, PollDecision, POLL_TIMEOUT_MS } from "../poll-decision.js";

// Feature: ai-photo-booth, Property 4: Async poll decision honors precedence (output, failure, timeout, pending) — output -> READY, else failure -> FAILED, else elapsed > 120000 -> TIMEOUT, else PENDING

describe("Property 4: async poll decision honors precedence", () => {
  it("follows output > failure > timeout > pending across all inputs", () => {
    fc.assert(
      fc.property(
        fc.boolean(),
        fc.boolean(),
        // Spread elapsed times across well below, around, and well above the
        // 120000 ms deadline, and pepper in the exact boundary values.
        fc.oneof(
          fc.integer({ min: 0, max: POLL_TIMEOUT_MS * 2 }),
          fc.constantFrom(
            0,
            POLL_TIMEOUT_MS - 1,
            POLL_TIMEOUT_MS,
            POLL_TIMEOUT_MS + 1,
          ),
        ),
        (outputPresent, failurePresent, elapsedMs) => {
          const decision = decidePoll({ outputPresent, failurePresent, elapsedMs });

          let expected: PollDecision;
          if (outputPresent) {
            expected = PollDecision.READY;
          } else if (failurePresent) {
            expected = PollDecision.FAILED;
          } else if (elapsedMs > POLL_TIMEOUT_MS) {
            expected = PollDecision.TIMEOUT;
          } else {
            expected = PollDecision.PENDING;
          }

          expect(decision).toBe(expected);
        },
      ),
      { numRuns: 100 },
    );
  });

  it("treats exactly the deadline (120000 ms) as still PENDING, one past it as TIMEOUT", () => {
    expect(decidePoll({ outputPresent: false, failurePresent: false, elapsedMs: POLL_TIMEOUT_MS })).toBe(
      PollDecision.PENDING,
    );
    expect(
      decidePoll({ outputPresent: false, failurePresent: false, elapsedMs: POLL_TIMEOUT_MS + 1 }),
    ).toBe(PollDecision.TIMEOUT);
  });
});
