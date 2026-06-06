import { describe, it, expect } from "vitest";
import fc from "fast-check";
import {
  EndpointStatus,
  mapStatus,
  SAGEMAKER_IN_SERVICE,
} from "../status-map.js";

// Feature: ai-photo-booth, Property 9: SageMaker status mapping is total and never over-reports availability — every input maps inside the enum, not-found -> NOT_DEPLOYED, only InService -> IN_SERVICE

const ENUM_VALUES = new Set<string>(Object.values(EndpointStatus));

// A spread of real SageMaker endpoint status strings plus the not-found sentinels.
const KNOWN_STATUSES = [
  "Creating",
  "Updating",
  "SystemUpdating",
  "InService",
  "Deleting",
  "Failed",
  "OutOfService",
  "RollingBack",
  "RollbackFailed",
  "UpdateRollbackFailed",
];

describe("Property 9: status mapping is total and never over-reports availability", () => {
  it("maps any status string (and not-found) into the enum, with not-found -> NOT_DEPLOYED and only InService -> IN_SERVICE", () => {
    fc.assert(
      fc.property(
        fc.oneof(
          fc.string(),
          fc.constantFrom(...KNOWN_STATUSES),
          fc.constant(null),
          fc.constant(undefined),
        ),
        (status) => {
          const mapped = mapStatus(status as string | null | undefined);

          // Totality: the result is always a member of the enum.
          expect(ENUM_VALUES.has(mapped)).toBe(true);

          // Not-found condition maps to NOT_DEPLOYED (never IN_SERVICE).
          if (status === null || status === undefined) {
            expect(mapped).toBe(EndpointStatus.NOT_DEPLOYED);
          }

          // Only the exact "InService" string is ever reported as available.
          if (mapped === EndpointStatus.IN_SERVICE) {
            expect(status).toBe(SAGEMAKER_IN_SERVICE);
          }
          if (status === SAGEMAKER_IN_SERVICE) {
            expect(mapped).toBe(EndpointStatus.IN_SERVICE);
          }
        },
      ),
      { numRuns: 100 },
    );
  });
});
