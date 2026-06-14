/**
 * Share_Signer — mints a short, time-limited CloudFront SIGNED URL for a shared
 * image. Exposed via a Lambda Function URL with `authType: AWS_IAM` (NOT public)
 * — the same auth posture as the Invoke proxy — so only the authenticated SPA
 * (SigV4-signed with the visitor's Cognito Identity-Pool credentials) can call
 * it.
 *
 * Why CloudFront signed URLs (not S3 presigned): an S3 presigned URL made with
 * Cognito temporary credentials embeds the huge STS session token (~1 KB),
 * which makes the QR code unscannably dense. A CloudFront signed URL is ~300
 * chars (Expires + Signature + Key-Pair-Id over a short distribution domain),
 * so the QR is low-density and scans reliably. The bucket stays fully private
 * behind CloudFront Origin Access Control; the signed URL is the only read path
 * and it expires in {@link DEFAULT_TTL_SECONDS}.
 *
 * AWS SDK v3 (`@aws-sdk/cloudfront-signer`, `@aws-sdk/client-secrets-manager`).
 */

import { getSignedUrl } from "@aws-sdk/cloudfront-signer";
import {
  SecretsManagerClient,
  GetSecretValueCommand,
} from "@aws-sdk/client-secrets-manager";

/** Key prefix shared images live under (matches the SPA + CDK). */
export const SHARE_PREFIX = "shared/";

/** Default link validity, in seconds (15 minutes). */
export const DEFAULT_TTL_SECONDS = 15 * 60;

/** Resolved signer configuration. */
export interface SignerConfig {
  /** CloudFront distribution domain serving the share bucket (no scheme). */
  cloudfrontDomain: string;
  /** CloudFront public-key id the signature is verified against. */
  keyPairId: string;
  /** Secrets Manager id/ARN holding the RSA private key (PEM). */
  privateKeySecretId: string;
  /** Link validity in seconds. */
  ttlSeconds: number;
}

/** Resolve {@link SignerConfig} from environment variables. */
export function resolveConfig(env: NodeJS.ProcessEnv = process.env): SignerConfig {
  const ttl = Number(env.SHARE_TTL_SECONDS);
  return {
    cloudfrontDomain: env.SHARE_CLOUDFRONT_DOMAIN ?? "",
    keyPairId: env.SHARE_KEY_PAIR_ID ?? "",
    privateKeySecretId: env.SHARE_PRIVATE_KEY_SECRET_ID ?? "",
    ttlSeconds: Number.isFinite(ttl) && ttl > 0 ? ttl : DEFAULT_TTL_SECONDS,
  };
}

/** Lambda Function URL event subset this handler reads. */
export interface FunctionUrlEvent {
  body?: string | null;
  isBase64Encoded?: boolean;
  requestContext?: { http?: { method?: string } };
}

/** Lambda Function URL response shape. */
export interface FunctionUrlResult {
  statusCode: number;
  headers?: Record<string, string>;
  body?: string;
}

/** Request body the browser sends. */
export interface SignRequest {
  /** The share object id (the SPA uploaded `shared/{id}.png`). */
  id: string;
}

/**
 * Validate a share id: only `[A-Za-z0-9._-]`, ≤200 chars, no path traversal.
 * Returns the id or `null`.
 */
export function validateId(id: unknown): string | null {
  if (typeof id !== "string" || id.length === 0 || id.length > 200) {
    return null;
  }
  return /^[A-Za-z0-9._-]+$/.test(id) ? id : null;
}

function json(statusCode: number, payload: unknown): FunctionUrlResult {
  return {
    statusCode,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  };
}

// Cache the private key across warm invocations (it never changes for the
// lifetime of the deployed key pair).
let cachedPrivateKey: string | undefined;

async function loadPrivateKey(
  secrets: SecretsManagerClient,
  secretId: string,
): Promise<string> {
  if (cachedPrivateKey) {
    return cachedPrivateKey;
  }
  const out = await secrets.send(
    new GetSecretValueCommand({ SecretId: secretId }),
  );
  const key = out.SecretString;
  if (!key || !key.includes("PRIVATE KEY")) {
    throw new Error("share signer private key is not configured");
  }
  cachedPrivateKey = key;
  return key;
}

/** Overridable dependencies for {@link handler}; default to real impls. */
export interface HandlerDeps {
  secrets?: SecretsManagerClient;
  config?: SignerConfig;
  now?: number;
}

function defaultSecrets(): SecretsManagerClient {
  return new SecretsManagerClient({});
}

/**
 * Build the CloudFront signed URL for `shared/{id}.png`, valid for `ttlSeconds`.
 * Separated from the Function-URL plumbing so it is unit-testable.
 */
export function buildSignedUrl(
  config: SignerConfig,
  id: string,
  privateKey: string,
  now: number,
): string {
  const resourceUrl = `https://${config.cloudfrontDomain}/${SHARE_PREFIX}${id}.png`;
  const dateLessThan = new Date(now + config.ttlSeconds * 1000).toISOString();
  return getSignedUrl({
    url: resourceUrl,
    keyPairId: config.keyPairId,
    privateKey,
    dateLessThan,
  });
}

/**
 * Lambda Function URL handler (AWS_IAM): parse `{ id }`, mint a 15-minute
 * CloudFront signed URL, and return it.
 */
export async function handler(
  event: FunctionUrlEvent = {},
  deps: HandlerDeps = {},
): Promise<FunctionUrlResult> {
  const config = deps.config ?? resolveConfig();
  const secrets = deps.secrets ?? defaultSecrets();
  const now = deps.now ?? Date.now();

  const method = event.requestContext?.http?.method ?? "POST";
  if (method !== "POST") {
    return json(405, { error: "method not allowed" });
  }
  if (!config.cloudfrontDomain || !config.keyPairId || !config.privateKeySecretId) {
    return json(500, { error: "share signer not configured" });
  }

  let request: SignRequest;
  try {
    const raw =
      event.isBase64Encoded && event.body
        ? Buffer.from(event.body, "base64").toString("utf-8")
        : event.body ?? "";
    request = JSON.parse(raw) as SignRequest;
  } catch {
    return json(400, { error: "invalid JSON body" });
  }

  const id = validateId(request.id);
  if (!id) {
    return json(400, { error: "invalid id" });
  }

  try {
    const privateKey = await loadPrivateKey(secrets, config.privateKeySecretId);
    const url = buildSignedUrl(config, id, privateKey, now);
    return json(200, { url, expiresInSeconds: config.ttlSeconds });
  } catch (err) {
    const message = err instanceof Error ? err.message : "sign failed";
    console.error("[share-signer] error", { message });
    return json(502, { error: message });
  }
}
