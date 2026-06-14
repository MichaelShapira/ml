/**
 * Effect-prompt overrides — browser-side Schedule_Store access.
 *
 * Admins can edit the prompt used for each effect. Overrides are persisted in
 * the shared Schedule_Store DynamoDB table (single-table design) under a
 * dedicated partition, separate from Working_Hours and the managed-config items:
 *
 *   pk = "PROMPT#OVERRIDE", sk = "EFFECT#<effectId>"
 *
 * Each item records the admin-edited `prompt`, an `isCustom` flag, and audit
 * fields. The booth treats a row as an override ONLY when `isCustom === true`;
 * a non-custom row (e.g. the CDK-seeded default) is ignored at runtime so the
 * catalog default in `effects.ts` always wins. "Restore to default" resets the
 * row to the catalog default with `isCustom = false`.
 *
 * Reads run under any signed-in user's role (standard users get a scoped,
 * read-only grant on just this partition, so the booth can pick up admin
 * edits); writes require the Admin_Role and fail at the IAM layer otherwise.
 * All calls run through {@link withAuthRetry} for silent token/STS refresh.
 *
 * Uses the AWS SDK for JavaScript v3 (DynamoDB Document client), consistent
 * with the rest of the project.
 */

import { QueryCommand, PutCommand } from "@aws-sdk/lib-dynamodb";

import { getConfig } from "../config";
import { getDynamoClient, withAuthRetry } from "./awsClients";
import {
  applyPromptOverrides,
  findEffect,
  getDefaultPromptForEffect,
} from "../booth/effects";

/** Partition key under which all per-effect prompt overrides live. */
export const PROMPT_OVERRIDE_PK = "PROMPT#OVERRIDE";

/** Build the sort key for an effect's prompt-override item. */
export function promptOverrideSk(effectId: string): string {
  return `EFFECT#${effectId}`;
}

/** The shape persisted per effect in the Schedule_Store. */
interface PromptOverrideItem {
  pk: string;
  sk: string;
  effectId: string;
  /** The prompt text (admin-edited when `isCustom`, else the seeded default). */
  prompt: string;
  /** True only when an admin has explicitly customized this effect's prompt. */
  isCustom: boolean;
  updatedAt: string;
  updatedBy: string;
}

/**
 * Load all persisted prompt rows and push the custom ones into the booth's
 * in-memory override registry (so the next generation uses them). Returns the
 * map of effectId → prompt for every CUSTOM override (non-custom rows are
 * ignored, matching the booth's runtime behavior).
 *
 * Best-effort: callers should treat a thrown error as "no overrides" and fall
 * back to catalog defaults (e.g. when a standard user's role lacks the read).
 */
export async function loadEffectPrompts(): Promise<Map<string, string>> {
  const overrides = new Map<string, string>();
  try {
    const config = getConfig();
    const doc = getDynamoClient();
    // Best-effort: a plain send (NOT withAuthRetry). The credentials provider
    // still does a silent token refresh, but we must NEVER trigger the
    // sign-out path here — a standard user whose role cannot read this
    // partition (or any transient error) must fall back to catalog defaults,
    // not get bounced to the sign-in screen. So all errors are swallowed.
    const response = await doc.send(
      new QueryCommand({
        TableName: config.scheduleTable,
        KeyConditionExpression: "pk = :pk",
        ExpressionAttributeValues: { ":pk": PROMPT_OVERRIDE_PK },
      }),
    );
    const items = (response.Items ?? []) as Partial<PromptOverrideItem>[];
    for (const item of items) {
      if (
        item.isCustom === true &&
        typeof item.effectId === "string" &&
        typeof item.prompt === "string" &&
        item.prompt.length > 0 &&
        // Ignore rows for effect ids the current build no longer knows about.
        findEffect(item.effectId) !== undefined
      ) {
        overrides.set(item.effectId, item.prompt);
      }
    }
  } catch {
    // Swallow — fall back to catalog defaults. Do not sign the user out.
  }
  applyPromptOverrides(overrides);
  return overrides;
}

/** Write an admin-customized prompt for an effect, then refresh the registry. */
export async function savePromptOverride(
  effectId: string,
  prompt: string,
  updatedBy = "admin",
): Promise<void> {
  if (findEffect(effectId) === undefined) {
    throw new Error(`Unknown effectId: ${JSON.stringify(effectId)}`);
  }
  const trimmed = prompt.trim();
  if (trimmed.length === 0) {
    throw new Error("Prompt must not be empty.");
  }
  await putItem(effectId, trimmed, true, updatedBy);
}

/**
 * Restore an effect's prompt to the built-in catalog default: writes the
 * catalog default back with `isCustom = false`, so the row reflects the default
 * and the booth uses the catalog value again.
 */
export async function restorePromptDefault(
  effectId: string,
  updatedBy = "admin",
): Promise<void> {
  const def = getDefaultPromptForEffect(effectId); // throws on unknown id
  await putItem(effectId, def, false, updatedBy);
}

/** Internal: upsert a prompt row and refresh the in-memory override registry. */
async function putItem(
  effectId: string,
  prompt: string,
  isCustom: boolean,
  updatedBy: string,
): Promise<void> {
  const config = getConfig();
  const doc = getDynamoClient();
  const item: PromptOverrideItem = {
    pk: PROMPT_OVERRIDE_PK,
    sk: promptOverrideSk(effectId),
    effectId,
    prompt,
    isCustom,
    updatedAt: new Date().toISOString(),
    updatedBy,
  };
  await withAuthRetry(() =>
    doc.send(new PutCommand({ TableName: config.scheduleTable, Item: item })),
  );
  // Re-load so the registry reflects this change immediately for generation.
  await loadEffectPrompts();
}
