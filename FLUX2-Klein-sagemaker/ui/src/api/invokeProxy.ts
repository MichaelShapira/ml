/**
 * invokeProxy — SigV4-signed call to the Invoke_Proxy Lambda Function URL.
 *
 * The SageMaker **runtime** API (`runtime.sagemaker.<region>.amazonaws.com`)
 * does not support CORS, so the browser cannot call `InvokeEndpointAsync`
 * directly — the preflight is never answered and the request is blocked. The
 * one un-CORS-able call is therefore proxied by a tiny Lambda exposed via a
 * Function URL with `authType: AWS_IAM` (NOT public). This module signs the
 * request to that URL with the visitor's Cognito Identity-Pool credentials
 * (SigV4, service `lambda`), exactly as the SDK would, so IAM authorizes the
 * call via the role's `lambda:InvokeFunctionUrl` permission.
 *
 * S3 input upload and output/failure polling remain browser-direct (S3 supports
 * CORS); only this invoke step goes through the proxy.
 */

import { SignatureV4 } from "@smithy/signature-v4";
import { HttpRequest } from "@smithy/protocol-http";
import { Sha256 } from "@aws-crypto/sha256-js";

import { getConfig } from "../config";
import { credentialsProvider } from "./awsClients";

/** Request to {@link invokeEndpointAsyncViaProxy}. */
export interface ProxyInvokeRequest {
  /** `s3://bucket/flux2-klein-inputs/<jobId>.json` — the uploaded input. */
  inputLocation: string;
  /** Content type of the input payload (defaults to `application/json`). */
  contentType?: string;
}

/** Response from the Invoke_Proxy on success. */
export interface ProxyInvokeResponse {
  outputLocation?: string;
  failureLocation?: string;
  inferenceId?: string;
}

/**
 * Sign and POST the invoke request to the IAM-authenticated Function URL.
 *
 * @throws Error on a non-2xx response (carrying the proxy's error message) or a
 *   network failure. Auth-flavoured failures bubble up so {@link withAuthRetry}
 *   (in the caller) can refresh and retry.
 */
export async function invokeEndpointAsyncViaProxy(
  req: ProxyInvokeRequest,
): Promise<ProxyInvokeResponse> {
  const config = getConfig();
  const url = new URL(config.invokeFunctionUrl);
  const body = JSON.stringify({
    inputLocation: req.inputLocation,
    contentType: req.contentType ?? "application/json",
  });

  // Resolve the current STS credentials (silent refresh handled by the provider).
  const credentials = await credentialsProvider();

  const signer = new SignatureV4({
    service: "lambda",
    region: config.region,
    credentials,
    sha256: Sha256,
  });

  // Build the canonical request. The host header is required for SigV4; the
  // content-type must be signed because the Function URL CORS allows it.
  const request = new HttpRequest({
    method: "POST",
    protocol: url.protocol,
    hostname: url.hostname,
    path: url.pathname,
    headers: {
      host: url.hostname,
      "content-type": "application/json",
    },
    body,
  });

  const signed = await signer.sign(request);

  const response = await fetch(config.invokeFunctionUrl, {
    method: "POST",
    headers: signed.headers,
    body,
  });

  if (!response.ok) {
    let detail = "";
    try {
      const payload = (await response.json()) as { error?: string };
      detail = payload?.error ? `: ${payload.error}` : "";
    } catch {
      // ignore body parse errors
    }
    const err = new Error(
      `Invoke proxy failed (${response.status})${detail}`,
    ) as Error & { $metadata?: { httpStatusCode: number } };
    // Attach the status so isAuthError() can recognize 401/403 for refresh+retry.
    err.$metadata = { httpStatusCode: response.status };
    throw err;
  }

  return (await response.json()) as ProxyInvokeResponse;
}
