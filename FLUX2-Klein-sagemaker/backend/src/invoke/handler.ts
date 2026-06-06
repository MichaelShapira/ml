/**
 * Invoke_Proxy — a minimal Lambda that performs the one AWS call the browser
 * cannot make directly: `sagemaker:InvokeEndpointAsync`.
 *
 * Why this Lambda exists: the SageMaker **runtime** API
 * (`runtime.sagemaker.<region>.amazonaws.com`) does **not** support CORS — it
 * never returns `Access-Control-Allow-Origin` and does not answer the browser's
 * preflight, so a browser `fetch`/SDK call to `InvokeEndpointAsync` is always
 * blocked. (S3 supports CORS, so input upload and output/failure polling stay
 * browser-direct; only this invoke step is proxied.)
 *
 * Security:
 *   - Exposed via a Lambda **Function URL with `authType: AWS_IAM`** (NOT
 *     public). Every request must be SigV4-signed with the caller's Cognito
 *     Identity-Pool credentials; the Authenticated_Role is granted only
 *     `lambda:InvokeFunctionUrl` on this function.
 *   - The Lambda's own role holds the single least-privilege permission
 *     `sagemaker:InvokeEndpointAsync` on the configured endpoint ARN.
 *   - The handler validates that the requested `inputLocation` points at the
 *     configured I/O bucket + inputs prefix, so a caller cannot coerce the
 *     endpoint into reading an arbitrary S3 object (defense in depth).
 *
 * AWS SDK for JavaScript v3 (`@aws-sdk/client-sagemaker-runtime`).
 */

import {
  InvokeEndpointAsyncCommand,
  SageMakerRuntimeClient,
} from "@aws-sdk/client-sagemaker-runtime";
import { DynamoDBClient, GetItemCommand } from "@aws-sdk/client-dynamodb";

import { currentConfigKey } from "../lib/current-config.js";

/** Default endpoint name when `ENDPOINT_NAME` is unset (the booth's endpoint). */
export const DEFAULT_ENDPOINT_NAME = "flux2-klein-9b-g6e2";

/** Inputs prefix the browser uploads request JSON to (must match the UI/CDK). */
export const INPUTS_PREFIX = "flux2-klein-inputs/";

/** Resolved invoke-proxy configuration. */
export interface InvokeConfig {
  /** Fallback endpoint to invoke when no current config is set in DynamoDB. */
  endpointName: string;
  /** The I/O bucket the input location must reside in (defense in depth). */
  ioBucket: string;
  /** Schedule_Store table holding the current-config pointer (optional). */
  scheduleTable: string;
}

/** Request body the browser sends to the Function URL. */
export interface InvokeRequest {
  /** `s3://bucket/flux2-klein-inputs/<jobId>.json` — the uploaded input. */
  inputLocation: string;
  /** Content type of the input payload (defaults to `application/json`). */
  contentType?: string;
}

/** Response returned to the browser on a successful invoke. */
export interface InvokeResponse {
  /** SageMaker async `OutputLocation` (informational; the UI derives keys itself). */
  outputLocation?: string;
  /** SageMaker async `FailureLocation` when provided by the service. */
  failureLocation?: string;
  /** The async inference id, when returned. */
  inferenceId?: string;
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

/** Overridable dependencies for {@link handler}; all default to real impls. */
export interface HandlerDeps {
  /** Injected SageMaker Runtime client (defaults to a real client). */
  runtime?: SageMakerRuntimeClient;
  /** Injected DynamoDB client for the current-config lookup. */
  dynamodb?: DynamoDBClient;
  /** Resolved config (defaults to {@link resolveConfig} over `process.env`). */
  config?: InvokeConfig;
}

/** Resolve {@link InvokeConfig} from environment variables. */
export function resolveConfig(env: NodeJS.ProcessEnv = process.env): InvokeConfig {
  return {
    endpointName: env.ENDPOINT_NAME ?? DEFAULT_ENDPOINT_NAME,
    ioBucket: env.IO_BUCKET ?? "",
    scheduleTable: env.SCHEDULE_TABLE ?? "",
  };
}

/**
 * Resolve the endpoint to invoke: the admin-chosen "current" config from the
 * Schedule_Store pointer, falling back to the env `ENDPOINT_NAME` when no
 * pointer is set or the lookup fails. The endpoint created from a config is
 * named identically to the config.
 */
export async function resolveCurrentEndpointName(
  dynamodb: DynamoDBClient,
  config: InvokeConfig,
): Promise<string> {
  if (!config.scheduleTable) {
    return config.endpointName;
  }
  try {
    const key = currentConfigKey();
    const response = await dynamodb.send(
      new GetItemCommand({
        TableName: config.scheduleTable,
        Key: { pk: { S: key.pk }, sk: { S: key.sk } },
      }),
    );
    const name = response.Item?.configName?.S;
    return name && name.length > 0 ? name : config.endpointName;
  } catch {
    return config.endpointName;
  }
}

/** A JSON Function URL response with no CORS headers (the Function URL adds them). */
function json(statusCode: number, payload: unknown): FunctionUrlResult {
  return {
    statusCode,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  };
}

/**
 * Validate that an `s3://` input location targets the configured I/O bucket and
 * the inputs prefix. Returns the parsed `{ bucket, key }` or throws.
 */
export function validateInputLocation(
  inputLocation: string,
  ioBucket: string,
): { bucket: string; key: string } {
  if (typeof inputLocation !== "string" || !inputLocation.startsWith("s3://")) {
    throw new Error("inputLocation must be an s3:// URI");
  }
  const withoutScheme = inputLocation.slice("s3://".length);
  const slash = withoutScheme.indexOf("/");
  if (slash <= 0) {
    throw new Error("inputLocation must include a bucket and key");
  }
  const bucket = withoutScheme.slice(0, slash);
  const key = withoutScheme.slice(slash + 1);

  if (ioBucket && bucket !== ioBucket) {
    throw new Error("inputLocation bucket is not the configured I/O bucket");
  }
  if (!key.startsWith(INPUTS_PREFIX)) {
    throw new Error("inputLocation key is not under the inputs prefix");
  }
  return { bucket, key };
}

/**
 * Core invoke: validate the input location, call `InvokeEndpointAsync` on the
 * resolved endpoint, and return the async locations. Separated from the
 * Function-URL plumbing so it is unit-testable with `aws-sdk-client-mock`.
 */
export async function invokeAsync(
  runtime: SageMakerRuntimeClient,
  config: InvokeConfig,
  request: InvokeRequest,
  endpointName: string,
): Promise<InvokeResponse> {
  validateInputLocation(request.inputLocation, config.ioBucket);

  const response = await runtime.send(
    new InvokeEndpointAsyncCommand({
      EndpointName: endpointName,
      ContentType: request.contentType ?? "application/json",
      InputLocation: request.inputLocation,
    }),
  );

  return {
    outputLocation: response.OutputLocation,
    failureLocation: response.FailureLocation,
    inferenceId: response.InferenceId,
  };
}

/** Lazily-constructed real clients so importing the module is side-effect free. */
function defaultRuntime(): SageMakerRuntimeClient {
  return new SageMakerRuntimeClient({});
}
function defaultDynamo(): DynamoDBClient {
  return new DynamoDBClient({});
}

/**
 * Lambda Function URL handler. Parses the JSON body, runs {@link invokeAsync},
 * and returns a JSON response. The Function URL's own CORS config supplies the
 * `Access-Control-*` headers, so this handler does not set them.
 */
export async function handler(
  event: FunctionUrlEvent = {},
  deps: HandlerDeps = {},
): Promise<FunctionUrlResult> {
  const config = deps.config ?? resolveConfig();
  const runtime = deps.runtime ?? defaultRuntime();
  const dynamodb = deps.dynamodb ?? defaultDynamo();

  const method = event.requestContext?.http?.method ?? "POST";
  if (method !== "POST") {
    return json(405, { error: "method not allowed" });
  }

  let request: InvokeRequest;
  try {
    const raw = event.isBase64Encoded && event.body
      ? Buffer.from(event.body, "base64").toString("utf-8")
      : event.body ?? "";
    request = JSON.parse(raw) as InvokeRequest;
  } catch {
    return json(400, { error: "invalid JSON body" });
  }

  try {
    const endpointName = await resolveCurrentEndpointName(dynamodb, config);
    const result = await invokeAsync(runtime, config, request, endpointName);
    return json(200, result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "invoke failed";
    // Validation problems are client errors; everything else is a 502 upstream.
    const isValidation = /inputLocation/.test(message);
    console.error("[invoke-proxy] error", { message });
    return json(isValidation ? 400 : 502, { error: message });
  }
}
