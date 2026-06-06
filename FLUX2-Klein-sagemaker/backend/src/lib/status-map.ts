/**
 * Endpoint status mapping and action guards (pure logic).
 *
 * Collapses SageMaker's raw `EndpointStatus` strings — and the
 * "endpoint does not exist" condition — into the requirement's enumerated
 * set, and derives the start/stop action guards from the current status.
 *
 * This module performs NO I/O and makes NO AWS SDK calls; it is shared by the
 * endpoint admin handlers (`backend/src/endpoint/*`) and the automated
 * scheduler (`backend/src/scheduler/apply.ts`).
 *
 * Requirements: 15.1, 15.4, 16.1, 16.3, 17.1, 17.4
 * Design Properties: 9 (mapping is total, never over-reports availability),
 *                    10 (action guards depend only on current status).
 */

/**
 * The normalized endpoint status surfaced to the Admin_UI and scheduler.
 *
 * `NOT_DEPLOYED` covers the "endpoint does not exist" case so callers never
 * have to distinguish a `ValidationException` from a real status.
 */
export enum EndpointStatus {
  NOT_DEPLOYED = "NOT_DEPLOYED",
  CREATING = "CREATING",
  IN_SERVICE = "IN_SERVICE",
  DELETING = "DELETING",
  FAILED = "FAILED",
}

/**
 * The single SageMaker status string that is allowed to map to
 * `IN_SERVICE`. Kept as a named constant so the "never over-report
 * availability" rule (Property 9) is explicit and easy to audit.
 */
export const SAGEMAKER_IN_SERVICE = "InService";

/**
 * SageMaker statuses that represent an endpoint that is coming up or being
 * updated (not yet, or no longer cleanly, serving traffic).
 */
const CREATING_STATUSES: ReadonlySet<string> = new Set([
  "Creating",
  "Updating",
  "SystemUpdating",
]);

/**
 * SageMaker statuses that represent an endpoint that is being torn down.
 */
const DELETING_STATUSES: ReadonlySet<string> = new Set(["Deleting"]);

/**
 * SageMaker statuses that represent an endpoint in a failed or unusable
 * state. Any status not otherwise recognized also falls through to
 * `FAILED` so the mapping stays total without ever over-reporting
 * availability.
 */
const FAILED_STATUSES: ReadonlySet<string> = new Set([
  "Failed",
  "OutOfService",
  "RollingBack",
  "RollbackFailed",
  "UpdateRollbackFailed",
]);

/**
 * Map any SageMaker `EndpointStatus` string — or the not-found condition
 * (represented by `null`/`undefined`) — into the normalized
 * {@link EndpointStatus} enum.
 *
 * This function is TOTAL: every possible input produces a value inside the
 * enum.
 *
 * Guarantees (Property 9):
 * - The not-found condition (`null`/`undefined`) maps to `NOT_DEPLOYED`.
 * - ONLY the exact string `"InService"` maps to `IN_SERVICE`; no other input
 *   (including unknown strings or differently-cased variants) is ever
 *   reported as available.
 * - Every other input maps to one of `CREATING`, `DELETING`, or `FAILED`,
 *   with unrecognized strings defaulting to `FAILED`.
 *
 * @param status The raw SageMaker status string, or `null`/`undefined` when
 *   `DescribeEndpoint` reports the endpoint does not exist
 *   (e.g. `ValidationException`).
 */
export function mapStatus(status: string | null | undefined): EndpointStatus {
  // Not-found condition -> the endpoint is not deployed.
  if (status === null || status === undefined) {
    return EndpointStatus.NOT_DEPLOYED;
  }

  // Only the exact SageMaker "InService" status reports availability.
  if (status === SAGEMAKER_IN_SERVICE) {
    return EndpointStatus.IN_SERVICE;
  }

  if (CREATING_STATUSES.has(status)) {
    return EndpointStatus.CREATING;
  }

  if (DELETING_STATUSES.has(status)) {
    return EndpointStatus.DELETING;
  }

  // Known failure states and any unrecognized status are treated as FAILED
  // so the mapping is total and never over-reports availability.
  if (FAILED_STATUSES.has(status)) {
    return EndpointStatus.FAILED;
  }

  return EndpointStatus.FAILED;
}

/**
 * Whether a start (create) action is permitted for the given status.
 *
 * Depends ONLY on the current status (Property 10): starting is permitted
 * if and only if the endpoint is not deployed (Requirements 16.1, 16.3).
 */
export function canStart(status: EndpointStatus): boolean {
  return status === EndpointStatus.NOT_DEPLOYED;
}

/**
 * Whether a stop (delete) action is permitted for the given status.
 *
 * Depends ONLY on the current status (Property 10): stopping is permitted
 * if and only if the endpoint is deployed in any form, i.e. its status is
 * not `NOT_DEPLOYED` (Requirements 17.1, 17.4).
 */
export function canStop(status: EndpointStatus): boolean {
  return status !== EndpointStatus.NOT_DEPLOYED;
}
