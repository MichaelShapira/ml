import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { EndpointStatus, canStart, canStop } from "../status-map.js";

// Feature: ai-photo-booth, Property 10: Endpoint action guards depend only on current status — start permitted iff NOT_DEPLOYED, stop permitted iff not NOT_DEPLOYED

describe("Property 10: endpoint action guards depend only on current status", () => {
  it("permits start iff NOT_DEPLOYED and stop iff not NOT_DEPLOYED", () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...Object.values(EndpointStatus)),
        (status) => {
          expect(canStart(status)).toBe(status === EndpointStatus.NOT_DEPLOYED);
          expect(canStop(status)).toBe(status !== EndpointStatus.NOT_DEPLOYED);
          // Stop is the exact complement of start across all statuses.
          expect(canStop(status)).toBe(!canStart(status));
        },
      ),
      { numRuns: 100 },
    );
  });
});
