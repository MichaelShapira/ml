/**
 * Generation_Service — the browser-side async generation module (Requirement 8).
 *
 * The booth has no backend, so the SPA submits and polls directly under the
 * visitor's Identity-Pool credentials:
 *
 *   submit():
 *     1. validate `{ effectId, photo }`,
 *     2. build the Async_Request (shared, clamped) via `requestBuilder`,
 *     3. PutObject the JSON to `flux2-klein-inputs/{jobId}.json`,
 *     4. invoke the endpoint via the Invoke_Proxy Function URL (the SageMaker
 *        runtime API has no CORS, so the call is SigV4-signed to an
 *        IAM-authenticated Lambda Function URL instead of called directly),
 *     5. return `{ jobId, submittedAt }`.
 *
 *   poll():
 *     HeadObject the output then failure keys, apply the shared poll-decision
 *     with a client-side 120 s deadline, and on READY GetObject the PNG bytes
 *     and hand back an object URL for `<img>` rendering. Terminal statuses are
 *     READY | FAILED | TIMEOUT (no BLOCKED — there is no moderation step).
 *
 * Every AWS call runs through {@link withAuthRetry} so an expired token / STS
 * credential is silently refreshed once before surfacing an auth error.
 *
 * Requirements: 8.1, 8.2, 8.3, 8.4, 8.5.
 */

import {
  PutObjectCommand,
  HeadObjectCommand,
  GetObjectCommand,
} from "@aws-sdk/client-s3";

import { getConfig } from "../config";
import {
  getS3Client,
  withAuthRetry,
} from "./awsClients";
import { invokeEndpointAsyncViaProxy } from "./invokeProxy";
import { buildAsyncRequest } from "./requestBuilder";
import { decidePoll, POLL_TIMEOUT_MS, type PollDecision } from "./pollDecision";

/** S3 key prefixes used by the existing async inference flow. */
export const INPUTS_PREFIX = "flux2-klein-inputs/";
export const OUTPUTS_PREFIX = "flux2-klein-outputs/";
export const FAILURES_PREFIX = "flux2-klein-failures/";

/** Request to {@link submitGeneration}. */
export interface SubmitRequest {
  /** Selected effect id (resolved to a prompt by the shared catalog). */
  effectId: string;
  /** Captured photo as a base64-encoded PNG/JPEG string. */
  photo: string;
}

/** Handle returned by {@link submitGeneration}, fed back into {@link pollGeneration}. */
export interface SubmitResult {
  /** Opaque job id (used only for the input key + logging). */
  jobId: string;
  /** Epoch ms when the request was submitted (drives the client timeout). */
  submittedAt: number;
  /**
   * The S3 key the endpoint writes the result to, as returned by
   * `InvokeEndpointAsync` (`OutputLocation`). SageMaker async inference uses a
   * server-generated UUID key here — it is NOT derived from the input key — so
   * the browser MUST poll this exact key, not a key built from `jobId`.
   */
  outputKey: string;
  /** The S3 key the endpoint writes a failure record to (`FailureLocation`). */
  failureKey?: string;
}

/** Result of a single {@link pollGeneration} call. */
export type PollResult =
  | { status: "PENDING" }
  | { status: "READY"; imageUrl: string; aiGenerated: true }
  | { status: "FAILED"; reason?: string }
  | { status: "TIMEOUT" };

/** The object key under the inputs prefix for a job. */
function inputKey(jobId: string): string {
  return `${INPUTS_PREFIX}${jobId}.json`;
}

/** Parse the S3 key from an `s3://bucket/key` URI, or return undefined. */
function keyFromS3Uri(uri: string | undefined): string | undefined {
  if (typeof uri !== "string" || !uri.startsWith("s3://")) {
    return undefined;
  }
  const withoutScheme = uri.slice("s3://".length);
  const slash = withoutScheme.indexOf("/");
  return slash >= 0 ? withoutScheme.slice(slash + 1) : undefined;
}

/** Generate a reasonably unique, key-safe job id. */
function newJobId(): string {
  const rand =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2);
  return `booth-${Date.now()}-${rand}`;
}

/**
 * Submit an Async_Request: upload the request JSON to the inputs prefix, then
 * invoke the endpoint asynchronously with that InputLocation (Requirement 8.1).
 *
 * @throws {import("../booth/effects").UnknownEffectError} for an unknown effect.
 * @throws Error when `photo` is missing/empty, or on an SDK failure.
 */
export async function submitGeneration(req: SubmitRequest): Promise<SubmitResult> {
  if (!req || typeof req.effectId !== "string" || req.effectId.length === 0) {
    throw new Error("submitGeneration: effectId is required");
  }
  if (typeof req.photo !== "string" || req.photo.length === 0) {
    throw new Error("submitGeneration: photo is required");
  }

  const config = getConfig();
  const jobId = newJobId();
  const asyncRequest = buildAsyncRequest({ effectId: req.effectId, photo: req.photo });
  const body = JSON.stringify(asyncRequest);

  const s3 = getS3Client();

  // 1) Upload the request JSON to the inputs prefix.
  await withAuthRetry(() =>
    s3.send(
      new PutObjectCommand({
        Bucket: config.ioBucket,
        Key: inputKey(jobId),
        Body: body,
        ContentType: "application/json",
      }),
    ),
  );

  // 2) Invoke the endpoint asynchronously via the Invoke_Proxy Function URL.
  //    The browser cannot call the SageMaker runtime API directly (no CORS), so
  //    this one call is SigV4-signed to an IAM-authenticated Lambda Function URL.
  //    The response carries the SERVER-GENERATED output/failure locations; we
  //    must poll those exact keys (they are NOT derived from the input key).
  const inputLocation = `s3://${config.ioBucket}/${inputKey(jobId)}`;
  const invokeResult = await withAuthRetry(() =>
    invokeEndpointAsyncViaProxy({
      inputLocation,
      contentType: "application/json",
    }),
  );

  const outputKey = keyFromS3Uri(invokeResult.outputLocation);
  if (!outputKey) {
    throw new Error(
      "submitGeneration: endpoint did not return an OutputLocation to poll",
    );
  }
  const failureKey = keyFromS3Uri(invokeResult.failureLocation);

  return {
    jobId,
    submittedAt: Date.now(),
    outputKey,
    ...(failureKey ? { failureKey } : {}),
  };
}

/** True iff a HeadObject succeeds (object exists); false on a 404/NotFound. */
async function objectExists(bucket: string, key: string): Promise<boolean> {
  const s3 = getS3Client();
  try {
    await withAuthRetry(() =>
      s3.send(new HeadObjectCommand({ Bucket: bucket, Key: key })),
    );
    return true;
  } catch (err) {
    if (isNotFound(err)) {
      return false;
    }
    throw err;
  }
}

/** Whether an S3 error represents a missing object (404 / NotFound / NoSuchKey). */
function isNotFound(err: unknown): boolean {
  if (typeof err !== "object" || err === null) {
    return false;
  }
  const e = err as {
    name?: string;
    Code?: string;
    $metadata?: { httpStatusCode?: number };
  };
  if (e.$metadata?.httpStatusCode === 404) {
    return true;
  }
  const code = `${e.name ?? ""} ${e.Code ?? ""}`;
  return /NotFound|NoSuchKey/i.test(code);
}

/** Read the result PNG bytes and turn them into an object URL for `<img>`. */
async function readResultObjectUrl(bucket: string, key: string): Promise<string> {
  const s3 = getS3Client();
  const response = await withAuthRetry(() =>
    s3.send(new GetObjectCommand({ Bucket: bucket, Key: key })),
  );
  // The browser SDK Body is a Blob (web streams); normalize to a Blob.
  const body = response.Body as unknown as {
    transformToByteArray?: () => Promise<Uint8Array>;
    blob?: () => Promise<Blob>;
  };
  let blob: Blob;
  if (typeof body?.transformToByteArray === "function") {
    const bytes = await body.transformToByteArray();
    // Copy into a fresh ArrayBuffer-backed view so the BlobPart type is exact
    // (the SDK's Uint8Array may be typed over ArrayBufferLike).
    const copy = new Uint8Array(bytes.byteLength);
    copy.set(bytes);
    blob = new Blob([copy], { type: "image/png" });
  } else if (typeof body?.blob === "function") {
    blob = await body.blob();
  } else {
    blob = new Blob([], { type: "image/png" });
  }
  return URL.createObjectURL(blob);
}

/**
 * Poll once for a submitted job (Requirements 8.2-8.5).
 *
 * HEADs the output then failure keys, applies the shared poll-decision with a
 * client-side 120 s deadline, and on READY downloads the PNG and returns an
 * object URL. Callers should revoke the URL (see {@link revokeResult}) when
 * leaving the result screen.
 */
export async function pollGeneration(handle: SubmitResult): Promise<PollResult> {
  const config = getConfig();
  const elapsedMs = Date.now() - handle.submittedAt;

  const outputPresent = await objectExists(config.ioBucket, handle.outputKey);
  // Only HEAD the failure key when there is no output (output wins precedence)
  // and a failure key was provided by the endpoint.
  const failurePresent =
    outputPresent || !handle.failureKey
      ? false
      : await objectExists(config.ioBucket, handle.failureKey);

  const decision: PollDecision = decidePoll({
    outputPresent,
    failurePresent,
    elapsedMs,
    timeoutMs: POLL_TIMEOUT_MS,
  });

  switch (decision) {
    case "READY": {
      const imageUrl = await readResultObjectUrl(
        config.ioBucket,
        handle.outputKey,
      );
      return { status: "READY", imageUrl, aiGenerated: true };
    }
    case "FAILED":
      return { status: "FAILED" };
    case "TIMEOUT":
      return { status: "TIMEOUT" };
    case "PENDING":
    default:
      return { status: "PENDING" };
  }
}

/** Revoke an object URL produced by a READY {@link PollResult}. */
export function revokeResult(imageUrl: string | null | undefined): void {
  if (imageUrl && imageUrl.startsWith("blob:")) {
    URL.revokeObjectURL(imageUrl);
  }
}
