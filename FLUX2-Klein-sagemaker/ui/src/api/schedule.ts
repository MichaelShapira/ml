/**
 * Schedule CRUD — browser-side Schedule_Store access (Requirement 20).
 *
 * Reads and writes Working_Hours directly in DynamoDB under the Admin_Role
 * credentials vended by the Identity Pool (a Standard_User's Authenticated_Role
 * lacks DynamoDB permissions, so these fail at the IAM layer). Uses the
 * DynamoDB Document client so items are plain JS objects.
 *
 * All calls run through {@link withAuthRetry} for silent token/STS refresh.
 *
 * Requirements: 20.1, 20.2, 20.4, 20.5.
 */

import {
  QueryCommand,
  PutCommand,
  DeleteCommand,
  GetCommand,
} from "@aws-sdk/lib-dynamodb";

import { getConfig } from "../config";
import { getDynamoClient, withAuthRetry } from "./awsClients";
import { isHomeRegion, regionOrHome, type ConfigRef } from "./regions";
import {
  makePk,
  makeSk,
  validateWorkingHours,
  type WorkingHours,
  type WorkingHoursItem,
} from "./workingHours";

/** Stable booth-wide key for Working_Hours (decoupled from the current config). */
export const BOOTH_SCHEDULE_NAME = "booth";

/** Fixed key for the single "current endpoint config" pointer item. */
export const CURRENT_CONFIG_PK = "CONFIG#CURRENT";
export const CURRENT_CONFIG_SK = "CONFIG#CURRENT";

/** Partition key under which the admin's curated "managed config" list lives. */
export const MANAGED_CONFIG_PK = "CONFIG#MANAGED";
/**
 * Build the sort key for a managed config entry. Home-region configs keep the
 * original `CONFIG#<name>` key; a config in another region is
 * `CONFIG#<region>#<name>`, because the same name can exist in two regions.
 * (SageMaker names cannot contain `#`.)
 */
export function managedConfigSk(configName: string, region?: string | null): string {
  return isHomeRegion(region)
    ? `CONFIG#${configName}`
    : `CONFIG#${regionOrHome(region)}#${configName}`;
}

/** Thrown when an upsert is rejected by client-side validation (end <= start). */
export class InvalidWorkingHoursError extends Error {
  constructor(message = "End time must be strictly after start time.") {
    super(message);
    this.name = "InvalidWorkingHoursError";
  }
}

/**
 * Load all Working_Hours for the managed endpoint (Requirement 20.5).
 * `Query(pk = ENDPOINT#<name>)` returns every persisted day.
 */
export async function listWorkingHours(): Promise<WorkingHours[]> {
  const config = getConfig();
  const doc = getDynamoClient();
  const response = await withAuthRetry(() =>
    doc.send(
      new QueryCommand({
        TableName: config.scheduleTable,
        KeyConditionExpression: "pk = :pk",
        ExpressionAttributeValues: { ":pk": makePk(BOOTH_SCHEDULE_NAME) },
      }),
    ),
  );
  const items = (response.Items ?? []) as WorkingHoursItem[];
  return items.map((item) => ({
    endpointName: BOOTH_SCHEDULE_NAME,
    day: item.day,
    startTime: item.startTime,
    endTime: item.endTime,
    updatedBy: item.updatedBy,
    updatedAt: item.updatedAt,
  }));
}

/** Input to {@link putWorkingHours}. */
export interface PutWorkingHoursInput {
  day: string;
  startTime: string;
  endTime: string;
  updatedBy?: string;
}

/**
 * Upsert a day's Working_Hours (Requirements 20.1, 20.2). Re-validates
 * `endTime > startTime` before writing and rejects with
 * {@link InvalidWorkingHoursError} otherwise (defense in depth behind the UI's
 * own validation).
 */
export async function putWorkingHours(input: PutWorkingHoursInput): Promise<void> {
  if (!validateWorkingHours(input.startTime, input.endTime)) {
    throw new InvalidWorkingHoursError();
  }
  const config = getConfig();
  const doc = getDynamoClient();
  const item: WorkingHoursItem = {
    pk: makePk(BOOTH_SCHEDULE_NAME),
    sk: makeSk(input.day),
    day: input.day,
    startTime: input.startTime,
    endTime: input.endTime,
    updatedBy: input.updatedBy ?? "admin",
    updatedAt: new Date().toISOString(),
  };
  await withAuthRetry(() =>
    doc.send(new PutCommand({ TableName: config.scheduleTable, Item: item })),
  );
}

/** Remove a day's Working_Hours (Requirement 20.4). */
export async function deleteWorkingHours(day: string): Promise<void> {
  const config = getConfig();
  const doc = getDynamoClient();
  await withAuthRetry(() =>
    doc.send(
      new DeleteCommand({
        TableName: config.scheduleTable,
        Key: { pk: makePk(BOOTH_SCHEDULE_NAME), sk: makeSk(day) },
      }),
    ),
  );
}

/**
 * Read the currently-selected endpoint config (the booth's "current" pointer)
 * and its region, or `null` when none is set yet. A pointer written before
 * regions existed has no `region` and means the home region.
 */
export async function getCurrentConfigRef(): Promise<ConfigRef | null> {
  const config = getConfig();
  const doc = getDynamoClient();
  const response = await withAuthRetry(() =>
    doc.send(
      new GetCommand({
        TableName: config.scheduleTable,
        Key: { pk: CURRENT_CONFIG_PK, sk: CURRENT_CONFIG_SK },
      }),
    ),
  );
  const item = response.Item as { configName?: string; region?: string } | undefined;
  const name = item?.configName;
  return name && name.length > 0 ? { name, region: regionOrHome(item?.region) } : null;
}

/** The current pointer's config name only (see {@link getCurrentConfigRef}). */
export async function getCurrentConfigName(): Promise<string | null> {
  return (await getCurrentConfigRef())?.name ?? null;
}

/**
 * Set the current endpoint config (the booth's "current" pointer). Writing the
 * single fixed-key item overwrites any previous value, so exactly one config is
 * ever current — selecting a new one automatically un-currents the old one. The
 * region is always written, so the invoke proxy and the scheduler drive the
 * endpoint in that region.
 */
export async function setCurrentConfigName(
  configName: string,
  region?: string | null,
): Promise<void> {
  const config = getConfig();
  const doc = getDynamoClient();
  await withAuthRetry(() =>
    doc.send(
      new PutCommand({
        TableName: config.scheduleTable,
        Item: {
          pk: CURRENT_CONFIG_PK,
          sk: CURRENT_CONFIG_SK,
          configName,
          region: regionOrHome(region),
          updatedAt: new Date().toISOString(),
        },
      }),
    ),
  );
}

/** A managed config entry the admin has explicitly added to the booth. */
export interface ManagedConfigItem {
  /** The SageMaker endpoint configuration name. */
  configName: string;
  /** Region of the config (and of the endpoint created from it). */
  region: string;
  /** ISO timestamp the config was added. */
  addedAt: string;
}

/**
 * List the admin's curated managed configs (the ones explicitly "Added" to the
 * booth), newest first. These are a subset of the account's endpoint configs.
 */
export async function listManagedConfigs(): Promise<ManagedConfigItem[]> {
  const config = getConfig();
  const doc = getDynamoClient();
  const response = await withAuthRetry(() =>
    doc.send(
      new QueryCommand({
        TableName: config.scheduleTable,
        KeyConditionExpression: "pk = :pk",
        ExpressionAttributeValues: { ":pk": MANAGED_CONFIG_PK },
      }),
    ),
  );
  const items = (response.Items ?? []) as Array<{
    configName?: string;
    region?: string;
    addedAt?: string;
  }>;
  return items
    .filter((i): i is { configName: string; region?: string; addedAt?: string } =>
      Boolean(i.configName),
    )
    .map((i) => ({
      configName: i.configName,
      region: regionOrHome(i.region),
      addedAt: i.addedAt ?? "",
    }))
    .sort((a, b) => b.addedAt.localeCompare(a.addedAt));
}

/** Add an endpoint config (in `region`, default home) to the curated managed list (idempotent upsert). */
export async function addManagedConfig(
  configName: string,
  region?: string | null,
): Promise<void> {
  const config = getConfig();
  const doc = getDynamoClient();
  await withAuthRetry(() =>
    doc.send(
      new PutCommand({
        TableName: config.scheduleTable,
        Item: {
          pk: MANAGED_CONFIG_PK,
          sk: managedConfigSk(configName, region),
          configName,
          region: regionOrHome(region),
          addedAt: new Date().toISOString(),
        },
      }),
    ),
  );
}

/** Remove an endpoint config (in `region`, default home) from the curated managed list. */
export async function removeManagedConfig(
  configName: string,
  region?: string | null,
): Promise<void> {
  const config = getConfig();
  const doc = getDynamoClient();
  await withAuthRetry(() =>
    doc.send(
      new DeleteCommand({
        TableName: config.scheduleTable,
        Key: { pk: MANAGED_CONFIG_PK, sk: managedConfigSk(configName, region) },
      }),
    ),
  );
}
