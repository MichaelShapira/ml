/**
 * Model family of the booth's current endpoint config.
 *
 * The booth can drive two image models, and they want different requests:
 *
 *   - FLUX.2 [klein] 9B (`code/inference.py`, g6e): instruction prompts tuned
 *     for FLUX.2 (composition lock first), 6 steps, guidance 2.5.
 *   - FLUX 3 Image Klein (`../FLUX3-Klein-sagemaker-g7e`, g7e): short edit
 *     instructions that the endpoint's prompt upsampler expands into a full
 *     caption, the distilled checkpoint's 16 steps, guidance 1 (no CFG).
 *
 * The family is derived from the current config's NAME: any name containing
 * "flux3" or "flux-3" (case-insensitive) is FLUX 3, everything else is FLUX.2.
 * deploy.sh names the FLUX 3 config `FluxAiBoot-flux3-g7e`.
 *
 * The current config name lives in the Schedule_Store pointer
 * (pk = sk = "CONFIG#CURRENT"). Every signed-in role may GetItem that one item
 * (see `auth-construct.ts`, statement ReadCurrentConfig), so the visitor's
 * booth reads it before each generation; the invoke proxy resolves the same
 * pointer server-side, so the request and the endpoint stay in step.
 */
import { GetCommand } from "@aws-sdk/lib-dynamodb";

import { getConfig } from "../config";
import { getDynamoClient } from "./awsClients";

/** The image models the booth knows how to prompt. */
export type ModelFamily = "flux2" | "flux3";

/** Config names matching this are served by FLUX 3 Image Klein. */
export const FLUX3_CONFIG_PATTERN = /flux-?3/i;

/** Partition/sort key of the current-config pointer item. */
const CURRENT_CONFIG_KEY = { pk: "CONFIG#CURRENT", sk: "CONFIG#CURRENT" } as const;

/** The family a config (and the endpoint created from it) belongs to. */
export function modelFamilyForConfig(configName: string | null | undefined): ModelFamily {
  return typeof configName === "string" && FLUX3_CONFIG_PATTERN.test(configName)
    ? "flux3"
    : "flux2";
}

/** Short human-readable model name for admin screens. */
export function modelFamilyLabel(family: ModelFamily): string {
  return family === "flux3" ? "FLUX 3 Image Klein" : "FLUX.2 [klein]";
}

/** Last known family; FLUX.2 until the pointer has been read. */
let currentFamily: ModelFamily = "flux2";

/** The last known family of the current config (no I/O). */
export function getModelFamily(): ModelFamily {
  return currentFamily;
}

/**
 * Record the family of `configName` as current. Called by the admin panel when
 * it makes a config current, so the next generation uses it without a re-read.
 */
export function setModelFamilyFromConfig(configName: string | null | undefined): ModelFamily {
  currentFamily = modelFamilyForConfig(configName);
  return currentFamily;
}

/**
 * Re-read the current-config pointer and update the family. Best-effort, like
 * `loadEffectPrompts`: a plain send (NOT `withAuthRetry`, which can sign the
 * user out), and any error keeps the last known family instead of failing the
 * generation.
 */
export async function refreshModelFamily(): Promise<ModelFamily> {
  try {
    const response = await getDynamoClient().send(
      new GetCommand({
        TableName: getConfig().scheduleTable,
        Key: CURRENT_CONFIG_KEY,
        ProjectionExpression: "configName",
      }),
    );
    const name = (response.Item as { configName?: unknown } | undefined)?.configName;
    if (typeof name === "string" && name.length > 0) {
      currentFamily = modelFamilyForConfig(name);
    }
  } catch {
    // Keep the last known family.
  }
  return currentFamily;
}
