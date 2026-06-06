/**
 * Scheduler_Function — the only Lambda in the AI Photo Booth system.
 *
 * This module is the Amazon EventBridge target that reconciles the running
 * state of the managed SageMaker endpoint against the day's Working_Hours
 * stored in the Schedule_Store (DynamoDB). It runs on a recurring cadence
 * (two EventBridge rules offset by ~30 s, per Requirement 21.1) and on each
 * invocation it:
 *
 *   1. Resolves configuration from environment variables.
 *   2. Computes "today" (`YYYY-MM-DD`) and "now" (`HH:mm`) as wall-clock values
 *      in the configured IANA timezone using `Intl.DateTimeFormat` — no
 *      external timezone library.
 *   3. Reads today's Working_Hours item from DynamoDB with a `GetItem` on the
 *      `pk = ENDPOINT#<name>` / `sk = DAY#<YYYY-MM-DD>` key. When there is no
 *      item for today, the desired state is "stopped" (outside any window).
 *   4. Determines `desiredRunning` = an item exists for today AND `now` is
 *      within its `[startTime, endTime)` window (via `isWithinWindow`, which is
 *      fail-closed on malformed data).
 *   5. `DescribeEndpoint` and maps the result through `mapStatus`
 *      (`ValidationException` / not-found → `NOT_DEPLOYED`).
 *   6. Reconciles idempotently:
 *        - `desiredRunning` && `NOT_DEPLOYED` (`canStart`) → `CreateEndpoint`.
 *        - `!desiredRunning` && `IN_SERVICE` (`canStop`) → `DeleteEndpoint`.
 *        - otherwise (desired == actual, or a transitional CREATING/DELETING/
 *          FAILED state) → no action.
 *
 * The "no schedule today" path is a normal stopped desired-state and never
 * throws.
 *
 * The core {@link reconcile} function takes injected SageMaker + DynamoDB
 * clients and an explicit wall-clock, so it is unit/integration testable with
 * `aws-sdk-client-mock`. The {@link handler} resolves config + wall-clock from
 * the environment and supports an optional `delaySeconds` input flag (the
 * offset technique for the second EventBridge rule).
 *
 * Requirements: 21.2, 21.3, 21.4, 21.5.
 */

import {
  CreateEndpointCommand,
  DeleteEndpointCommand,
  DescribeEndpointCommand,
  SageMakerClient,
} from "@aws-sdk/client-sagemaker";
import {
  DynamoDBClient,
  GetItemCommand,
  type AttributeValue,
} from "@aws-sdk/client-dynamodb";

import { EndpointStatus, canStart, canStop, mapStatus } from "../lib/status-map.js";
import { isWithinWindow, makePk, makeSk } from "../lib/working-hours.js";
import { currentConfigKey, BOOTH_SCHEDULE_NAME } from "../lib/current-config.js";

/** Default endpoint name when `ENDPOINT_NAME` is unset (the booth's endpoint). */
export const DEFAULT_ENDPOINT_NAME = "flux2-klein-9b-g6e2";

/** Default IANA timezone when `TIMEZONE` is unset. */
export const DEFAULT_TIMEZONE = "America/Los_Angeles";

/** Resolved scheduler configuration. */
export interface SchedulerConfig {
  /** Endpoint to manage, e.g. `flux2-klein-9b-g6e2`. */
  endpointName: string;
  /** Schedule_Store DynamoDB table name. */
  scheduleTable: string;
  /** IANA timezone used to compute the wall-clock day/time. */
  timezone: string;
  /** EndpointConfig name to recreate the endpoint from on `CreateEndpoint`. */
  endpointConfigName: string;
}

/** The AWS SDK clients the scheduler needs; injected for testability. */
export interface ReconcileClients {
  sagemaker: SageMakerClient;
  dynamodb: DynamoDBClient;
}

/** A wall-clock instant in the configured timezone. */
export interface WallClock {
  /** ISO date, `YYYY-MM-DD`. */
  day: string;
  /** 24-hour time, `HH:mm`. */
  time: string;
}

/** The reconciliation action taken (or not) on this invocation. */
export type ReconcileAction = "CREATE" | "DELETE" | "NONE";

/** Outcome of a single reconciliation, returned for logging/testing. */
export interface ReconcileResult {
  /** The action taken. */
  action: ReconcileAction;
  /** Whether the endpoint is intended to be running right now. */
  desiredRunning: boolean;
  /** The actual (normalized) endpoint status observed. */
  status: EndpointStatus;
  /** Whether a Working_Hours item existed for today. */
  hasSchedule: boolean;
  /** The wall-clock day used. */
  day: string;
  /** The wall-clock time used. */
  time: string;
}

/** Input event shape for the EventBridge target. */
export interface SchedulerEvent {
  /**
   * Optional offset, in seconds, to wait before reconciling. The second
   * EventBridge rule sends `{ delaySeconds: 30 }` so the pair approximates a
   * ~30 s cadence given EventBridge's 1-minute minimum granularity.
   */
  delaySeconds?: number;
}

/** Overridable dependencies for {@link handler}; all default to real impls. */
export interface HandlerDeps {
  /** Injected AWS SDK clients (defaults to real SageMaker + DynamoDB clients). */
  clients?: ReconcileClients;
  /** Resolved config (defaults to {@link resolveConfig} over `process.env`). */
  config?: SchedulerConfig;
  /** Clock source (defaults to `() => new Date()`). */
  now?: () => Date;
  /** Sleep implementation (defaults to a real timer). */
  sleep?: (ms: number) => Promise<void>;
}

/**
 * Resolve {@link SchedulerConfig} from environment variables, applying the
 * documented defaults for `ENDPOINT_NAME` and `TIMEZONE`.
 */
export function resolveConfig(env: NodeJS.ProcessEnv = process.env): SchedulerConfig {
  return {
    endpointName: env.ENDPOINT_NAME ?? DEFAULT_ENDPOINT_NAME,
    scheduleTable: env.SCHEDULE_TABLE ?? "",
    timezone: env.TIMEZONE ?? DEFAULT_TIMEZONE,
    endpointConfigName: env.ENDPOINT_CONFIG_NAME ?? "",
  };
}

/**
 * Compute the wall-clock day (`YYYY-MM-DD`) and time (`HH:mm`) for an instant
 * in the given IANA timezone, using `Intl.DateTimeFormat` only (no external
 * timezone library).
 */
export function computeWallClock(timezone: string, at: Date = new Date()): WallClock {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(at);

  const map: Record<string, string> = {};
  for (const part of parts) {
    map[part.type] = part.value;
  }

  // `h23` keeps hours in 00–23, but guard against the occasional "24" that some
  // runtimes emit at midnight so the result is always a valid `HH:mm`.
  const hour = map.hour === "24" ? "00" : map.hour;

  return {
    day: `${map.year}-${map.month}-${map.day}`,
    time: `${hour}:${map.minute}`,
  };
}

/** Read a DynamoDB string (`S`) attribute, or `undefined` when absent/non-string. */
function readString(
  item: Record<string, AttributeValue> | undefined,
  key: string,
): string | undefined {
  const attr = item?.[key];
  return attr !== undefined && typeof attr.S === "string" ? attr.S : undefined;
}

/**
 * Whether a `DescribeEndpoint` error represents a non-existent endpoint, which
 * the design maps to `NOT_DEPLOYED`. SageMaker reports this as a
 * `ValidationException` ("Could not find endpoint ..."); we also accept the
 * `ResourceNotFound`-style families and "not found"/"does not exist" messages
 * defensively.
 */
function isEndpointNotFound(err: unknown): boolean {
  if (typeof err !== "object" || err === null) {
    return false;
  }
  const e = err as { name?: string; Code?: string; __type?: string; message?: string };
  const name = e.name ?? e.Code ?? e.__type ?? "";
  if (name === "ValidationException" || name.endsWith("ValidationException")) {
    return true;
  }
  if (name === "ResourceNotFound" || name.endsWith("ResourceNotFoundException")) {
    return true;
  }
  const message = (e.message ?? "").toLowerCase();
  return (
    message.includes("could not find") ||
    message.includes("does not exist") ||
    message.includes("not found")
  );
}

/**
 * `DescribeEndpoint` and map the result into the normalized
 * {@link EndpointStatus}, collapsing the not-found condition to
 * `NOT_DEPLOYED`.
 */
async function describeStatus(
  sagemaker: SageMakerClient,
  endpointName: string,
): Promise<EndpointStatus> {
  try {
    const response = await sagemaker.send(
      new DescribeEndpointCommand({ EndpointName: endpointName }),
    );
    return mapStatus(response.EndpointStatus ?? null);
  } catch (err) {
    if (isEndpointNotFound(err)) {
      return EndpointStatus.NOT_DEPLOYED;
    }
    throw err;
  }
}

/**
 * Read today's Working_Hours from the Schedule_Store and decide whether the
 * endpoint is intended to be running right now.
 *
 * Returns `{ hasSchedule, desiredRunning }`. When no item exists for today,
 * `hasSchedule` is `false` and `desiredRunning` is `false` (a normal stopped
 * desired-state — this path never throws). `isWithinWindow` is fail-closed, so
 * a present-but-malformed item also yields `desiredRunning: false`.
 */
async function readDesiredRunning(
  dynamodb: DynamoDBClient,
  config: SchedulerConfig,
  now: WallClock,
): Promise<{ hasSchedule: boolean; desiredRunning: boolean }> {
  const response = await dynamodb.send(
    new GetItemCommand({
      TableName: config.scheduleTable,
      Key: {
        // Schedule is booth-wide (stable key), not keyed by the current config.
        pk: { S: makePk(BOOTH_SCHEDULE_NAME) },
        sk: { S: makeSk(now.day) },
      },
    }),
  );

  const item = response.Item;
  if (item === undefined) {
    return { hasSchedule: false, desiredRunning: false };
  }

  const startTime = readString(item, "startTime");
  const endTime = readString(item, "endTime");
  if (startTime === undefined || endTime === undefined) {
    return { hasSchedule: true, desiredRunning: false };
  }

  return {
    hasSchedule: true,
    desiredRunning: isWithinWindow(now.time, startTime, endTime),
  };
}

/**
 * Resolve the current endpoint config (and thus endpoint name) from the
 * Schedule_Store pointer, falling back to the configured default. The endpoint
 * created from a config is named identically to the config.
 */
async function resolveCurrentEndpoint(
  dynamodb: DynamoDBClient,
  config: SchedulerConfig,
): Promise<{ endpointName: string; endpointConfigName: string }> {
  try {
    const key = currentConfigKey();
    const response = await dynamodb.send(
      new GetItemCommand({
        TableName: config.scheduleTable,
        Key: { pk: { S: key.pk }, sk: { S: key.sk } },
      }),
    );
    const name = readString(response.Item, "configName");
    if (name && name.length > 0) {
      return { endpointName: name, endpointConfigName: name };
    }
  } catch {
    // Fall through to the configured defaults.
  }
  return {
    endpointName: config.endpointName,
    endpointConfigName: config.endpointConfigName || config.endpointName,
  };
}

/**
 * The core reconciliation. Reads today's Working_Hours, observes the endpoint
 * status, and idempotently creates, deletes, or leaves the endpoint unchanged.
 *
 * This function performs the AWS I/O through the injected {@link ReconcileClients},
 * so tests can supply `aws-sdk-client-mock`'d clients and a fixed wall-clock.
 *
 * Requirements: 21.2 (read today's hours), 21.3 (inside window + not running →
 * create), 21.4 (outside window + running → delete), 21.5 (desired == actual →
 * no change).
 */
export async function reconcile(
  clients: ReconcileClients,
  config: SchedulerConfig,
  now: WallClock,
): Promise<ReconcileResult> {
  const { hasSchedule, desiredRunning } = await readDesiredRunning(
    clients.dynamodb,
    config,
    now,
  );

  // Resolve which endpoint/config is currently selected by the admin.
  const { endpointName, endpointConfigName } = await resolveCurrentEndpoint(
    clients.dynamodb,
    config,
  );

  const status = await describeStatus(clients.sagemaker, endpointName);

  let action: ReconcileAction = "NONE";

  if (desiredRunning && canStart(status)) {
    // Inside the window and the endpoint is not deployed → create it.
    // `canStart(status)` is true iff status === NOT_DEPLOYED.
    await clients.sagemaker.send(
      new CreateEndpointCommand({
        EndpointName: endpointName,
        EndpointConfigName: endpointConfigName,
        // Tag on creation so the scheduler-created endpoint carries the same
        // AiPhoto cost-allocation tag CDK applies to the rest of the stack.
        Tags: [{ Key: "AiPhoto", Value: "true" }],
      }),
    );
    action = "CREATE";
  } else if (!desiredRunning && status === EndpointStatus.IN_SERVICE && canStop(status)) {
    // Outside the window and the endpoint is in service → delete it. We act
    // only on IN_SERVICE, leaving transitional CREATING/DELETING/FAILED states
    // untouched so the reconcile stays idempotent and never thrashes.
    await clients.sagemaker.send(
      new DeleteEndpointCommand({ EndpointName: endpointName }),
    );
    action = "DELETE";
  }

  const result: ReconcileResult = {
    action,
    desiredRunning,
    status,
    hasSchedule,
    day: now.day,
    time: now.time,
  };

  console.info("[scheduler] reconcile decision", {
    // Log the endpoint the scheduler actually acted on — the resolved CURRENT
    // endpoint — not the static fallback, so logs aren't misleading.
    endpointName,
    ...result,
  });

  return result;
}

/** Default real AWS SDK clients, constructed lazily so importing is side-effect free. */
function defaultClients(): ReconcileClients {
  return {
    sagemaker: new SageMakerClient({}),
    dynamodb: new DynamoDBClient({}),
  };
}

/** Default timer-based sleep. */
function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * EventBridge target handler.
 *
 * Resolves config + wall-clock from the environment, honors the optional
 * `delaySeconds` offset (used by the second EventBridge rule to approximate a
 * ~30 s cadence), then delegates to {@link reconcile}. Accepts injected
 * dependencies (clients/config/clock/sleep) that all default to real
 * implementations, so the whole handler is testable.
 */
export async function handler(
  event: SchedulerEvent = {},
  deps: HandlerDeps = {},
): Promise<ReconcileResult> {
  const config = deps.config ?? resolveConfig();
  const clients = deps.clients ?? defaultClients();
  const clock = deps.now ?? (() => new Date());
  const sleep = deps.sleep ?? defaultSleep;

  const delaySeconds =
    typeof event?.delaySeconds === "number" && Number.isFinite(event.delaySeconds)
      ? Math.max(0, event.delaySeconds)
      : 0;

  if (delaySeconds > 0) {
    await sleep(delaySeconds * 1000);
  }

  const now = computeWallClock(config.timezone, clock());
  return reconcile(clients, config, now);
}
