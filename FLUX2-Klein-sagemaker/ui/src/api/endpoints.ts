/**
 * Endpoint_Manager — browser-side SageMaker endpoint management (Reqs 15-18).
 *
 * Calls SageMaker management APIs directly under the Admin_Role credentials
 * vended by the Identity Pool. A Standard_User's Authenticated_Role lacks these
 * permissions, so the calls fail at the IAM layer (AccessDeniedException) — the
 * UI tab gate is cosmetic only.
 *
 * All calls run through {@link withAuthRetry} for silent token/STS refresh.
 *
 * Requirements: 15.1, 15.4, 15.5, 16.1, 16.4, 17.1, 17.3, 18.1, 18.4.
 */

import {
  ListEndpointsCommand,
  DescribeEndpointCommand,
  CreateEndpointCommand,
  DeleteEndpointCommand,
  ListEndpointConfigsCommand,
} from "@aws-sdk/client-sagemaker";

import { getSageMakerClient, withAuthRetry } from "./awsClients";
import { EndpointStatus, mapStatus, canStart, canStop } from "./statusMap";
import { putWorkingHours, listWorkingHours, deleteWorkingHours, getCurrentConfigName, setCurrentConfigName, listManagedConfigs, addManagedConfig, removeManagedConfig } from "./schedule";
import {
  nowInScheduleTz,
  addMinutesCapped,
  subtractMinutesFloored,
  minTime,
  maxTime,
  toMinutes,
} from "./timezone";

export { EndpointStatus, statusLabel } from "./statusMap";

/** Minutes to extend the manual-start protection window past "now". */
export const MANUAL_START_WINDOW_MINUTES = 20;

/**
 * The cost-allocation tag stamped on every booth resource (matches
 * `Tags.of(stack).add("AiPhoto","true")` in CDK). Endpoints created at runtime
 * (manual start + scheduler) are NOT created by CloudFormation, so they must be
 * tagged explicitly on `CreateEndpoint` to show up under the same cost tag.
 */
export const AI_PHOTO_TAG = { Key: "AiPhoto", Value: "true" } as const;

/** A SageMaker endpoint summary surfaced to the Admin UI. */
export interface EndpointSummary {
  name: string;
  status: EndpointStatus;
}

/**
 * An endpoint configuration the admin can select/start/stop. The endpoint
 * created from a config is named identically to the config, so `name` is both
 * the config name and the endpoint name. `status` is the live status of that
 * endpoint (NOT_DEPLOYED until started). `isCurrent` reflects the booth's
 * current-config pointer.
 */
export interface EndpointConfigSummary {
  name: string;
  status: EndpointStatus;
  isCurrent: boolean;
}

/**
 * A configuration error (invalid credentials / region), surfaced as a type
 * distinct from an empty-list result so the UI can tell the two apart
 * (Requirement 15.4).
 */
export class EndpointConfigError extends Error {
  constructor(message: string, public readonly cause?: unknown) {
    super(message);
    this.name = "EndpointConfigError";
  }
}

/** Whether a SageMaker error indicates the endpoint does not exist. */
function isNotFound(err: unknown): boolean {
  if (typeof err !== "object" || err === null) {
    return false;
  }
  const e = err as { name?: string; Code?: string; __type?: string; message?: string };
  const name = `${e.name ?? ""} ${e.Code ?? ""} ${e.__type ?? ""}`;
  if (/ValidationException|ResourceNotFound/i.test(name)) {
    return true;
  }
  const msg = (e.message ?? "").toLowerCase();
  return msg.includes("could not find") || msg.includes("does not exist");
}

/** Whether an error is a credentials/region configuration problem. */
function isConfigError(err: unknown): boolean {
  if (typeof err !== "object" || err === null) {
    return false;
  }
  const e = err as {
    name?: string;
    Code?: string;
    __type?: string;
    message?: string;
  };
  const haystack = `${e.name ?? ""} ${e.Code ?? ""} ${e.__type ?? ""} ${e.message ?? ""}`.toLowerCase();
  return (
    haystack.includes("invalidclienttokenid") ||
    haystack.includes("unrecognizedclient") ||
    haystack.includes("invalidsignature") ||
    haystack.includes("could not load credentials") ||
    haystack.includes("credential") ||
    haystack.includes("region is missing") ||
    haystack.includes("unknownendpoint") ||
    haystack.includes("endpoint url")
  );
}

/**
 * List the account's SageMaker endpoints (Requirement 15.1). Returns each with
 * its mapped status. An invalid-credentials/region failure throws an
 * {@link EndpointConfigError} (distinct from an empty list, Requirement 15.4).
 * An empty account yields `[]`, which the UI renders as "no endpoints
 * available" (Requirement 15.5).
 */
export async function listEndpoints(): Promise<EndpointSummary[]> {
  const sm = getSageMakerClient();
  try {
    const response = await withAuthRetry(() =>
      sm.send(new ListEndpointsCommand({ MaxResults: 100 })),
    );
    const endpoints = response.Endpoints ?? [];
    return endpoints.map((e) => ({
      name: e.EndpointName ?? "",
      status: mapStatus(e.EndpointStatus ?? null),
    }));
  } catch (err) {
    if (isConfigError(err)) {
      throw new EndpointConfigError(
        "AWS credentials or region are invalid; cannot list endpoints.",
        err,
      );
    }
    throw err;
  }
}

/**
 * List ALL endpoint configuration names in the account (for the admin's
 * "add config" autocomplete picker). Throws {@link EndpointConfigError} on a
 * credentials/region failure.
 */
export async function listAllConfigNames(): Promise<string[]> {
  const sm = getSageMakerClient();
  try {
    const response = await withAuthRetry(() =>
      sm.send(new ListEndpointConfigsCommand({ MaxResults: 100 })),
    );
    return (response.EndpointConfigs ?? [])
      .map((c) => c.EndpointConfigName ?? "")
      .filter((n) => n.length > 0)
      .sort((a, b) => a.localeCompare(b));
  } catch (err) {
    if (isConfigError(err)) {
      throw new EndpointConfigError(
        "AWS credentials or region are invalid; cannot list endpoint configs.",
        err,
      );
    }
    throw err;
  }
}

/**
 * List the admin's CURATED managed configs (the ones explicitly added), each
 * annotated with the live status of the endpoint created from it (named
 * identically to the config) and whether it is the booth's current config.
 *
 * Auto-current rule: if no current pointer is set yet and the managed list is
 * non-empty, the FIRST managed config (most recently added is sorted first, so
 * we take the last in add-order) becomes current. We simply pick the first
 * managed entry that exists when none is current.
 */
export async function listManagedConfigsWithStatus(): Promise<EndpointConfigSummary[]> {
  const managed = await listManagedConfigs();
  const names = managed.map((m) => m.configName);
  if (names.length === 0) {
    return [];
  }

  let current = await getCurrentConfigName();
  if (!current || !names.includes(current)) {
    current = names[0];
    try {
      await setCurrentConfigName(current);
    } catch {
      // Non-fatal: the badge just may not persist.
    }
  }

  return Promise.all(
    names.map(async (name) => ({
      name,
      status: await describeEndpointStatus(name),
      isCurrent: name === current,
    })),
  );
}

/** Add an endpoint config to the curated managed list. */
export async function addConfig(configName: string): Promise<void> {
  await addManagedConfig(configName);
}

/**
 * Remove an endpoint config from the curated managed list. If it was the
 * current config, the pointer is left as-is; the next list will auto-pick a new
 * current from the remaining managed configs.
 */
export async function removeConfig(configName: string): Promise<void> {
  await removeManagedConfig(configName);
}

/**
 * Make `configName` the booth's current config. Writing the single fixed-key
 * pointer overwrites any previous value, so exactly one config is ever current.
 */
export async function makeConfigCurrent(configName: string): Promise<void> {
  await setCurrentConfigName(configName);
}

/** Read the booth's current config name (or null when unset). */
export async function getCurrentConfig(): Promise<string | null> {
  return getCurrentConfigName();
}

/**
 * Describe a single endpoint's current status (Requirement 16.1). A not-found
 * endpoint maps to NOT_DEPLOYED and never reports an in-service status
 * (Requirement 16.4).
 */
export async function describeEndpointStatus(
  name: string,
): Promise<EndpointStatus> {
  const sm = getSageMakerClient();
  try {
    const response = await withAuthRetry(() =>
      sm.send(new DescribeEndpointCommand({ EndpointName: name })),
    );
    return mapStatus(response.EndpointStatus ?? null);
  } catch (err) {
    if (isNotFound(err)) {
      return EndpointStatus.NOT_DEPLOYED;
    }
    throw err;
  }
}

/** Result of a start/stop action. */
export interface EndpointActionResult {
  ok: boolean;
  message: string;
}

/**
 * Start (create) the endpoint from the configured EndpointConfig (Req 17.1).
 * Rejected with "endpoint already exists" when the endpoint is already deployed
 * (Requirement 17.3).
 */
export async function startEndpoint(name: string): Promise<EndpointActionResult> {
  const status = await describeEndpointStatus(name);
  if (!canStart(status)) {
    return { ok: false, message: "Endpoint already exists." };
  }
  const sm = getSageMakerClient();
  await withAuthRetry(() =>
    sm.send(
      new CreateEndpointCommand({
        EndpointName: name,
        // The endpoint created from a config is named identically to the config.
        EndpointConfigName: name,
        // Tag on creation so the runtime-created endpoint carries the same
        // AiPhoto cost-allocation tag CDK applies to the rest of the stack.
        Tags: [{ Key: AI_PHOTO_TAG.Key, Value: AI_PHOTO_TAG.Value }],
      }),
    ),
  );

  // Protect a manual start from the scheduler: ensure today's Working_Hours
  // window (in the scheduler's timezone) covers [now, now+20min) so the next
  // reconcile sees the endpoint as "desired running" and does not delete the
  // freshly-created endpoint. If a window already exists for today, widen it to
  // include the manual run rather than discarding the admin's schedule.
  try {
    const now = nowInScheduleTz();
    const runEnd = addMinutesCapped(now.time, MANUAL_START_WINDOW_MINUTES);
    const existing = (await listWorkingHours()).find((wh) => wh.day === now.day);

    let startTime = now.time;
    let endTime = runEnd;
    if (existing) {
      // Union the existing window with [now, now+20min].
      startTime = minTime(existing.startTime, now.time);
      endTime = maxTime(existing.endTime, runEnd);
    }

    await putWorkingHours({
      day: now.day,
      startTime,
      endTime,
      updatedBy: "manual-start",
    });
  } catch {
    // Best-effort: a failure here doesn't block the start. Worst case the
    // scheduler may stop it on the next tick if it's outside any window.
  }

  return {
    ok: true,
    message:
      "Endpoint creation started. This can take several minutes, and also " +
      "depends on the availability of the GPU instance type for the model. " +
      `A ${MANUAL_START_WINDOW_MINUTES}-minute run window was added so the ` +
      "scheduler keeps it running; extend it from the Schedule tab if needed.",
  };
}

/**
 * Stop (delete) the endpoint (Requirement 18.1). Returns "no endpoint to
 * delete" when the endpoint is not deployed (Requirement 18.4). The UI is
 * responsible for requiring an explicit confirm before calling this
 * (Requirement 18.2).
 */
export async function stopEndpoint(name: string): Promise<EndpointActionResult> {
  const status = await describeEndpointStatus(name);
  if (!canStop(status)) {
    return { ok: false, message: "No endpoint to delete." };
  }
  const sm = getSageMakerClient();
  await withAuthRetry(() =>
    sm.send(new DeleteEndpointCommand({ EndpointName: name })),
  );

  // Close today's Working_Hours window so the scheduler does NOT immediately
  // recreate the endpoint we just deleted. The scheduler keeps the endpoint
  // running while "now" is within today's [startTime, endTime); shortening the
  // end to one minute before "now" (in the scheduler's timezone) moves "now"
  // past the window so the next reconcile leaves it stopped. Best-effort: a
  // failure here doesn't block the stop.
  try {
    const now = nowInScheduleTz();
    const today = (await listWorkingHours()).find((wh) => wh.day === now.day);
    if (today) {
      const startM = toMinutes(today.startTime);
      const endM = toMinutes(today.endTime);
      const nowM = toMinutes(now.time);
      // Only act when the scheduler would otherwise restart it, i.e. "now" is
      // currently inside today's window. (Outside it, the scheduler already
      // keeps it stopped, so there is nothing to close.)
      if (startM !== null && endM !== null && nowM !== null && nowM >= startM && nowM < endM) {
        const newEnd = subtractMinutesFloored(now.time, 1);
        if (toMinutes(newEnd)! > startM) {
          // Shorten the window to end just before now (stays valid: end > start).
          await putWorkingHours({
            day: now.day,
            startTime: today.startTime,
            endTime: newEnd,
            updatedBy: "manual-stop",
          });
        } else {
          // now is at/just after the start minute; a valid shortened window is
          // not possible, so remove today's window entirely to guarantee the
          // scheduler does not restart it today.
          await deleteWorkingHours(now.day);
        }
      }
    }
  } catch {
    // Best-effort: if closing the window fails, the worst case is the scheduler
    // recreates the endpoint on its next tick.
  }

  return {
    ok: true,
    message:
      "Endpoint deletion started. Today's schedule was closed so the " +
      "scheduler will not restart it; reopen it from the Schedule tab if needed.",
  };
}
