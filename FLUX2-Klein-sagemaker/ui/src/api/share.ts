/**
 * Share_Service — "Share with me" QR download flow (Option A).
 *
 * Security: there is NO public endpoint. The download is a short, 15-minute
 * **CloudFront signed URL** served from a FULLY PRIVATE bucket via Origin Access
 * Control. The browser never holds the signing key — it asks the Share_Signer
 * Lambda (behind an `AWS_IAM` Function URL, same auth as the Invoke proxy) to
 * mint the signed URL, SigV4-signed with the visitor's Cognito credentials.
 *
 * Flow:
 *   1. read the image bytes (from the generated object URL),
 *   2. PutObject to `shared/{id}.png` (visitor credentials; bucket stays private),
 *   3. call the IAM-authed signer with `{ id }` → receive a CloudFront signed URL.
 *
 * The signed URL is ~300 chars (vs ~1 KB for an S3 presigned URL bloated by the
 * STS token), so the QR is low-density and scans reliably.
 */

import { PutObjectCommand } from "@aws-sdk/client-s3";
import { SignatureV4 } from "@smithy/signature-v4";
import { HttpRequest } from "@smithy/protocol-http";
import { Sha256 } from "@aws-crypto/sha256-js";

import { getConfig } from "../config";
import { getS3Client, credentialsProvider, withAuthRetry } from "./awsClients";

/** S3 key prefix the Share_Bucket lifecycle rule + IAM policy are scoped to. */
export const SHARE_PREFIX = "shared/";

/** The signed-URL access window, in seconds (15 minutes; matches the signer). */
export const SHARE_TTL_SECONDS = 15 * 60;

/** Result of {@link shareImage}. */
export interface ShareResult {
  /** The CloudFront signed download URL (valid for {@link SHARE_TTL_SECONDS}). */
  url: string;
  /** The TTL, so the UI can show "Valid for N minutes". */
  expiresInSeconds: number;
}

/** Generate a reasonably-unique, key-safe object id (matches the signer's rules). */
function newShareId(): string {
  const rand =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2);
  return `${Date.now()}-${rand}`;
}

/** Read an image reference (object URL or data URL) into PNG bytes. */
async function readImageBytes(imageSrc: string): Promise<Uint8Array> {
  const response = await fetch(imageSrc);
  const blob = await response.blob();
  const buffer = await blob.arrayBuffer();
  return new Uint8Array(buffer);
}

/** Response from the Share_Signer Lambda. */
interface SignerResponse {
  url?: string;
  expiresInSeconds?: number;
  error?: string;
}

/**
 * SigV4-sign and POST `{ id }` to the IAM-authenticated signer Function URL,
 * returning the CloudFront signed URL. Mirrors `invokeProxy.ts`.
 */
async function requestSignedUrl(id: string): Promise<SignerResponse> {
  const config = getConfig();
  const url = new URL(config.shareSignerUrl);
  const body = JSON.stringify({ id });

  const credentials = await credentialsProvider();
  const signer = new SignatureV4({
    service: "lambda",
    region: config.region,
    credentials,
    sha256: Sha256,
  });

  const request = new HttpRequest({
    method: "POST",
    protocol: url.protocol,
    hostname: url.hostname,
    path: url.pathname,
    headers: { host: url.hostname, "content-type": "application/json" },
    body,
  });
  const signed = await signer.sign(request);

  const response = await fetch(config.shareSignerUrl, {
    method: "POST",
    headers: signed.headers,
    body,
  });

  if (!response.ok) {
    const err = new Error(`Share signer failed (${response.status})`) as Error & {
      $metadata?: { httpStatusCode: number };
    };
    // Surface the status so isAuthError() can drive refresh+retry on 401/403.
    err.$metadata = { httpStatusCode: response.status };
    throw err;
  }
  return (await response.json()) as SignerResponse;
}

/**
 * Upload a generated image to the private Share_Bucket and return a 15-minute
 * CloudFront signed download URL for it.
 *
 * @param imageSrc the image to share — the object URL (or data URL) of the
 *   generated result currently shown in the studio.
 * @throws Error when `imageSrc` is missing, or on an SDK / signer failure.
 */
export async function shareImage(imageSrc: string): Promise<ShareResult> {
  if (typeof imageSrc !== "string" || imageSrc.length === 0) {
    throw new Error("shareImage: imageSrc is required");
  }

  const config = getConfig();
  const id = newShareId();
  const key = `${SHARE_PREFIX}${id}.png`;
  const bytes = await readImageBytes(imageSrc);
  const s3 = getS3Client();

  // 1) Upload to the fully-private share bucket (PutObject only).
  await withAuthRetry(() =>
    s3.send(
      new PutObjectCommand({
        Bucket: config.shareBucket,
        Key: key,
        Body: bytes,
        ContentType: "image/png",
      }),
    ),
  );

  // 2) Ask the IAM-authed signer to mint a short CloudFront signed URL.
  const signed = await withAuthRetry(() => requestSignedUrl(id));
  if (!signed.url) {
    throw new Error(signed.error ?? "share signer returned no URL");
  }

  return {
    url: signed.url,
    expiresInSeconds: signed.expiresInSeconds ?? SHARE_TTL_SECONDS,
  };
}
