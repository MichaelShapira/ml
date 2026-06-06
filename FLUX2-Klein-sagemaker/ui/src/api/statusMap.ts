/**
 * Endpoint status mapping + action guards for the SPA (Requirements 16, 17, 18).
 *
 * Mirrors `backend/src/lib/status-map.ts`. The mapping is total and never
 * over-reports availability: only the exact "InService" string maps to
 * IN_SERVICE, and the not-found condition maps to NOT_DEPLOYED.
 */

/** The normalized endpoint status surfaced to the Admin UI. */
export enum EndpointStatus {
  NOT_DEPLOYED = "NOT_DEPLOYED",
  CREATING = "CREATING",
  IN_SERVICE = "IN_SERVICE",
  DELETING = "DELETING",
  FAILED = "FAILED",
}

/** The single SageMaker status that may map to IN_SERVICE. */
export const SAGEMAKER_IN_SERVICE = "InService";

const CREATING_STATUSES: ReadonlySet<string> = new Set([
  "Creating",
  "Updating",
  "SystemUpdating",
]);
const DELETING_STATUSES: ReadonlySet<string> = new Set(["Deleting"]);
const FAILED_STATUSES: ReadonlySet<string> = new Set([
  "Failed",
  "OutOfService",
  "RollingBack",
  "RollbackFailed",
  "UpdateRollbackFailed",
]);

/** Map any SageMaker status (or the not-found condition) into the enum. */
export function mapStatus(status: string | null | undefined): EndpointStatus {
  if (status === null || status === undefined) {
    return EndpointStatus.NOT_DEPLOYED;
  }
  if (status === SAGEMAKER_IN_SERVICE) {
    return EndpointStatus.IN_SERVICE;
  }
  if (CREATING_STATUSES.has(status)) {
    return EndpointStatus.CREATING;
  }
  if (DELETING_STATUSES.has(status)) {
    return EndpointStatus.DELETING;
  }
  if (FAILED_STATUSES.has(status)) {
    return EndpointStatus.FAILED;
  }
  return EndpointStatus.FAILED;
}

/** Start permitted iff the endpoint is not deployed. */
export function canStart(status: EndpointStatus): boolean {
  return status === EndpointStatus.NOT_DEPLOYED;
}

/** Stop permitted iff the endpoint is deployed in any form. */
export function canStop(status: EndpointStatus): boolean {
  return status !== EndpointStatus.NOT_DEPLOYED;
}

/** Human-friendly labels for each status (no SCREAMING_SNAKE in the UI). */
const STATUS_LABELS: Record<EndpointStatus, string> = {
  [EndpointStatus.NOT_DEPLOYED]: "Not deployed",
  [EndpointStatus.CREATING]: "Starting…",
  [EndpointStatus.IN_SERVICE]: "Running",
  [EndpointStatus.DELETING]: "Stopping…",
  [EndpointStatus.FAILED]: "Failed",
};

/** Render an {@link EndpointStatus} as a user-friendly label. */
export function statusLabel(status: EndpointStatus): string {
  return STATUS_LABELS[status] ?? "Unknown";
}
