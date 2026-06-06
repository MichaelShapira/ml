/**
 * Browser-side AWS SDK v3 client wiring for the AI Photo Booth (`api/awsClients`).
 *
 * The booth has **no per-request backend**: the SPA calls S3, SageMaker
 * Runtime, SageMaker management, and DynamoDB *directly* from the browser,
 * SigV4-signed with temporary STS credentials vended by the Cognito Identity
 * Pool and authorized by IAM (design "Authorization flow"). This module owns:
 *
 *   1. {@link credentialsProvider} — a lazy AWS SDK v3 credentials provider
 *      that defers to {@link authService} for the silent token + STS refresh
 *      contract, so callers never hold static keys.
 *   2. Lazily-created singleton service clients (S3, SageMaker Runtime,
 *      SageMaker management, DynamoDB Document) built with that provider and
 *      the runtime region. Creation is on first use, so importing this module
 *      has no side effects and tests can reset the cache.
 *   3. {@link withAuthRetry} — the design's **refresh-once-then-retry** rule
 *      that every AWS call path runs through.
 *   4. {@link isAuthError} — the lenient auth-flavoured-failure heuristic that
 *      decides whether {@link withAuthRetry} should refresh and retry.
 *
 * The `api/*` modules built in later tasks (generation 7.3, endpoints 7.5,
 * schedule 7.7) consume the client getters and {@link withAuthRetry} from here;
 * they are intentionally NOT implemented in this module.
 *
 * Requirements: 11.2, 12.1, 12.2, 12.3, 12.4.
 */

import { S3Client } from "@aws-sdk/client-s3";
import { SageMakerRuntimeClient } from "@aws-sdk/client-sagemaker-runtime";
import { SageMakerClient } from "@aws-sdk/client-sagemaker";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient } from "@aws-sdk/lib-dynamodb";
import { SESv2Client } from "@aws-sdk/client-sesv2";
import { CostExplorerClient } from "@aws-sdk/client-cost-explorer";
import type { AwsCredentialIdentityProvider } from "@aws-sdk/types";

import { authService } from "../auth/authService";
import { getConfig } from "../config";

/**
 * Lazy AWS SDK v3 credentials provider.
 *
 * The SDK invokes this whenever a client needs credentials to sign a request.
 * It implements the design's **silent-refresh contract** by delegating to
 * {@link authService}:
 *
 *   - `getSession()` returns the current Cognito session, *implicitly
 *     refreshing* the access/ID token when it has expired but the refresh
 *     token is still valid (Requirement 12.1, 12.3).
 *   - `getCredentials(session)` mints (or reuses) the Identity-Pool STS
 *     credentials for the session's mapped IAM role (Requirement 11.2, 12.2);
 *     `authService` builds these via `fromCognitoIdentityPool`, so the
 *     logins-key construction lives in exactly one place.
 *
 * When `getSession()` returns `null` the refresh token is dead and silent
 * refresh is impossible, so this throws a `NotAuthorized`-style error
 * (Requirement 12.4). {@link isAuthError} recognizes that name, so a failure
 * surfacing through {@link withAuthRetry} drives the sign-in fallback rather
 * than an opaque crash.
 */
export const credentialsProvider: AwsCredentialIdentityProvider = async () => {
  const session = await authService.getSession(); // implicit token refresh
  if (!session) {
    // Refresh token dead -> the SPA must present the sign-in interface.
    throw new Error("NotAuthorized: no valid Cognito session");
  }
  return authService.getCredentials(session); // STS creds for the mapped role
};

// --- Lazily-created singleton service clients --------------------------------
//
// Clients are created on first use (not at import time) so that:
//   * importing this module has no side effects (no config read, no client
//     construction) — safe for tree-shaking and for tests that import the
//     pure helpers; and
//   * tests can reset the cache between cases via resetAwsClients().
// Every client shares the same refreshing credentials provider and the runtime
// region from getConfig(), so they all assume the visitor's mapped IAM role.

let s3Client: S3Client | undefined;
let sageMakerRuntimeClient: SageMakerRuntimeClient | undefined;
let sageMakerClient: SageMakerClient | undefined;
let dynamoBaseClient: DynamoDBClient | undefined;
let dynamoDocClient: DynamoDBDocumentClient | undefined;
let sesClient: SESv2Client | undefined;
let costExplorerClient: CostExplorerClient | undefined;

/** S3 client for async-inference input upload and output/failure polling. */
export function getS3Client(): S3Client {
  if (!s3Client) {
    s3Client = new S3Client({
      region: getConfig().region,
      credentials: credentialsProvider,
    });
  }
  return s3Client;
}

/** SageMaker Runtime client for `InvokeEndpointAsync`. */
export function getSageMakerRuntimeClient(): SageMakerRuntimeClient {
  if (!sageMakerRuntimeClient) {
    sageMakerRuntimeClient = new SageMakerRuntimeClient({
      region: getConfig().region,
      credentials: credentialsProvider,
    });
  }
  return sageMakerRuntimeClient;
}

/** SageMaker management client for List/Describe/Create/DeleteEndpoint (admin). */
export function getSageMakerClient(): SageMakerClient {
  if (!sageMakerClient) {
    sageMakerClient = new SageMakerClient({
      region: getConfig().region,
      credentials: credentialsProvider,
    });
  }
  return sageMakerClient;
}

/**
 * DynamoDB Document client for the Schedule_Store (admin).
 *
 * The Schedule_Store items are plain attribute maps (see
 * `backend/src/lib/working-hours.ts` `WorkingHoursItem`), so the Document
 * client's automatic (un)marshalling lets `schedule.ts` work with native JS
 * objects rather than DynamoDB AttributeValues. `removeUndefinedValues` keeps
 * optional fields from producing invalid items.
 */
export function getDynamoClient(): DynamoDBDocumentClient {
  if (!dynamoDocClient) {
    dynamoBaseClient = new DynamoDBClient({
      region: getConfig().region,
      credentials: credentialsProvider,
    });
    dynamoDocClient = DynamoDBDocumentClient.from(dynamoBaseClient, {
      marshallOptions: { removeUndefinedValues: true },
    });
  }
  return dynamoDocClient;
}

/** SES v2 client for sending the visitor's photo by email (raw MIME). */
export function getSesClient(): SESv2Client {
  if (!sesClient) {
    sesClient = new SESv2Client({
      region: getConfig().region,
      credentials: credentialsProvider,
    });
  }
  return sesClient;
}

/**
 * Cost Explorer client for the admin cost panel (read-only GetCostAndUsage).
 *
 * Cost Explorer is a GLOBAL service whose endpoint lives in `us-east-1`
 * regardless of where the stack is deployed, so the client is pinned there.
 */
export function getCostExplorerClient(): CostExplorerClient {
  if (!costExplorerClient) {
    costExplorerClient = new CostExplorerClient({
      region: "us-east-1",
      credentials: credentialsProvider,
    });
  }
  return costExplorerClient;
}

/**
 * Destroys and clears every cached client. Primarily for tests so each case
 * starts from a clean slate; also usable on sign-out to drop clients bound to
 * stale credentials. Safe to call when nothing has been created yet.
 */
export function resetAwsClients(): void {
  s3Client?.destroy();
  sageMakerRuntimeClient?.destroy();
  sageMakerClient?.destroy();
  // Destroying the Document client tears down its underlying base client.
  dynamoDocClient?.destroy();
  sesClient?.destroy();
  costExplorerClient?.destroy();
  s3Client = undefined;
  sageMakerRuntimeClient = undefined;
  sageMakerClient = undefined;
  dynamoBaseClient = undefined;
  dynamoDocClient = undefined;
  sesClient = undefined;
  costExplorerClient = undefined;
}

// --- Auth-flavoured failure heuristic ----------------------------------------

/** Throttling families that MUST NOT be retried by the auth path. */
const THROTTLING_MARKERS = [
  "throttl", // ThrottlingException, Throttling, RequestThrottled
  "toomanyrequests", // TooManyRequestsException
  "provisionedthroughputexceeded", // ProvisionedThroughputExceededException
  "requestlimitexceeded", // RequestLimitExceeded
  "slowdown", // S3 SlowDown
];

/** Network/abort families that MUST NOT be retried by the auth path. */
const NETWORK_NAME_MARKERS = [
  "aborterror",
  "timeouterror",
  "networkerror",
];
const NETWORK_MESSAGE_MARKERS = [
  "failed to fetch",
  "network request failed",
  "network error",
  "load failed",
];
const NETWORK_CODE_MARKERS = [
  "econnreset",
  "econnrefused",
  "enotfound",
  "etimedout",
  "epipe",
  "eai_again",
  " abort", // AbortError surfaced as code/message fragments
];

/** Auth-flavoured families that SHOULD trigger a single refresh-and-retry. */
const AUTH_MARKERS = [
  "expiredtoken", // ExpiredToken / ExpiredTokenException
  "invalidclienttokenid", // InvalidClientTokenId
  "notauthorized", // NotAuthorized / NotAuthorizedException
  "accessdenied", // AccessDeniedException (403 token-expiry sense)
  "unauthorized", // UnauthorizedException (403 token-expiry sense)
];

/** Lowercased, whitespace-collapsed concatenation of an error's identifying fields. */
function authHaystack(err: Record<string, unknown>): string {
  const parts: unknown[] = [
    err.name,
    err.message,
    err.Code, // S3/XML-style error code
    err.code, // node system error code
    err.__type, // JSON-protocol shape id
  ];
  return parts
    .filter((p): p is string => typeof p === "string")
    .join(" ")
    .toLowerCase();
}

/** Best-effort extraction of an HTTP status code from an SDK error. */
function httpStatus(err: Record<string, unknown>): number | undefined {
  const metadata = err.$metadata as { httpStatusCode?: unknown } | undefined;
  if (metadata && typeof metadata.httpStatusCode === "number") {
    return metadata.httpStatusCode;
  }
  if (typeof err.statusCode === "number") {
    return err.statusCode;
  }
  return undefined;
}

/**
 * Decide whether an error is an **auth-flavoured failure** — i.e. the kind of
 * failure that a one-shot token + STS refresh might cure — per the design's
 * lenient-but-precise heuristic.
 *
 * Returns `true` when the error's `name`/`message`/`Code`/`__type` contains any
 * of `ExpiredToken(Exception)`, an expired *security token* phrase (e.g. "The
 * security token included in the request is expired"), `InvalidClientTokenId`,
 * `NotAuthorized`, `AccessDenied`/`Unauthorized`, **or** it maps to an HTTP
 * 401/403 status.
 *
 * Returns `false` — so {@link withAuthRetry} does NOT retry on this path — for
 * throttling (`ThrottlingException`, `TooManyRequestsException`,
 * `ProvisionedThroughputExceeded`), 5xx server errors, and network/abort
 * errors. Those are transient or non-auth and are left to the SDK's own retry
 * strategy or the caller. Non-object/empty errors are treated as non-auth.
 */
export function isAuthError(err: unknown): boolean {
  if (err === null || typeof err !== "object") {
    return false;
  }
  const e = err as Record<string, unknown>;
  const haystack = authHaystack(e);
  const status = httpStatus(e);

  // 1) Exclusions first: these are never the auth path, even if some unrelated
  //    substring would otherwise look auth-flavoured.
  if (THROTTLING_MARKERS.some((m) => haystack.includes(m))) {
    return false;
  }
  if (
    NETWORK_NAME_MARKERS.some((m) => haystack.includes(m)) ||
    NETWORK_MESSAGE_MARKERS.some((m) => haystack.includes(m)) ||
    NETWORK_CODE_MARKERS.some((m) => haystack.includes(m))
  ) {
    return false;
  }
  if (status !== undefined && status >= 500) {
    return false;
  }

  // 2) Expired-security-token phrasing, e.g.
  //    "The security token included in the request is expired".
  if (haystack.includes("security token") && haystack.includes("expired")) {
    return true;
  }

  // 3) Named auth families.
  if (AUTH_MARKERS.some((m) => haystack.includes(m))) {
    return true;
  }

  // 4) 401/403-style status (covers AccessDenied/Unauthorized senses too).
  if (status === 401 || status === 403) {
    return true;
  }

  return false;
}

/**
 * Run an AWS SDK call through the design's **refresh-once-then-retry** rule.
 *
 * Behavior:
 *   1. Run `run()` once.
 *   2. If it fails with a NON-auth error ({@link isAuthError} `false` —
 *      throttling, 5xx, network/abort), rethrow immediately; this path never
 *      retries those (the SDK's own retry strategy handles transient ones).
 *   3. On an auth-flavoured failure, force a Cognito token refresh
 *      (`refreshSession`) and an STS re-mint (`refreshCredentials`), then retry
 *      `run()` **exactly once**.
 *   4. If the retry also fails with an auth error, the refresh token is dead:
 *      fire `requireSignIn()` so the SPA returns to the sign-in interface, then
 *      rethrow. A non-auth error from the retry is simply rethrown.
 *
 * The single retry guarantees this never loops: at most one refresh + one
 * extra attempt per call. Both expiry paths (Cognito token, Identity-Pool STS
 * credentials) are covered because both `refreshSession` and
 * `refreshCredentials` run before the retry (Requirement 12.1, 12.2, 12.3);
 * exhausting them maps to the re-auth prompt (Requirement 12.4).
 */
export async function withAuthRetry<T>(run: () => Promise<T>): Promise<T> {
  try {
    return await run();
  } catch (err) {
    if (!isAuthError(err)) {
      throw err; // throttling / 5xx / network -> do NOT retry here
    }
    // Refresh both independently-expiring things, then retry exactly once.
    await authService.refreshSession(); // force Cognito token refresh
    await authService.refreshCredentials(); // force STS re-mint
    try {
      return await run();
    } catch (err2) {
      if (isAuthError(err2)) {
        // Second auth failure => refresh token is dead; re-auth required.
        authService.requireSignIn();
      }
      throw err2;
    }
  }
}
