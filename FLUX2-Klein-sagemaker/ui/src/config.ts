/**
 * Runtime configuration accessor for the AI Photo Booth SPA.
 *
 * There is no per-request backend; the browser talks to AWS directly using
 * temporary credentials vended by a Cognito Identity Pool. The IDs and region
 * those calls need are not known at build time — they are produced by the CDK
 * stack (task 14.1) and injected at runtime as a single global config object
 * (`window.__BOOTH_CONFIG__`) loaded before the bundle. For local development a
 * `VITE_*` environment fallback is provided.
 *
 * This module is the single source of truth for that config. `authService.ts`
 * consumes the auth subset via {@link getAuthConfig}; the `api/*` modules
 * (tasks 7.x) consume the full object via {@link getConfig}.
 */

/**
 * The complete runtime configuration object injected into the page.
 *
 * The auth fields are required by the always-on sign-in gate. The AWS-resource
 * fields are required by the direct-to-AWS modules built in later tasks and are
 * validated by those consumers, not by {@link getAuthConfig}.
 */
export interface BoothConfig {
  /** AWS region hosting the Cognito pools and all called services. */
  region: string;
  /** Cognito user pool id (e.g. `us-east-1_ABC123`). */
  userPoolId: string;
  /** Cognito user pool app client id (public SPA client, no secret). */
  userPoolClientId: string;
  /** Cognito identity pool id that vends the STS credentials. */
  identityPoolId: string;
  /** Name of the FLUX.2 SageMaker async endpoint (`flux2-klein-9b-g6e2`). */
  endpointName: string;
  /** Name of the SageMaker EndpointConfig used when (re)creating the endpoint. */
  endpointConfigName: string;
  /** Bucket holding the async inputs/outputs/failures prefixes. */
  ioBucket: string;
  /** DynamoDB table name backing the Schedule_Store. */
  scheduleTable: string;
  /** Short-lived bucket for the "Share with me" QR download flow. */
  shareBucket: string;
  /** IAM-authenticated Share_Signer Function URL (mints CloudFront signed URLs). */
  shareSignerUrl: string;
  /** IANA timezone the scheduler uses to evaluate Working_Hours (e.g. Asia/Jerusalem). */
  timezone: string;
  /**
   * IAM-authenticated Lambda Function URL the SPA SigV4-signs to invoke the
   * async SageMaker endpoint. Needed because the SageMaker runtime API does not
   * support CORS, so the browser cannot call InvokeEndpointAsync directly.
   */
  invokeFunctionUrl: string;
}

/** The subset of {@link BoothConfig} required to authenticate and mint credentials. */
export type AuthConfig = Pick<
  BoothConfig,
  "region" | "userPoolId" | "userPoolClientId" | "identityPoolId"
>;

declare global {
  interface Window {
    /** Runtime config injected by the hosting layer before the SPA bundle loads. */
    __BOOTH_CONFIG__?: Partial<BoothConfig>;
  }
}

/**
 * Reads the raw, possibly-incomplete runtime config from the injected global,
 * falling back to Vite `VITE_*` environment variables for local development.
 */
function readRawConfig(): Partial<BoothConfig> {
  const injected =
    typeof window !== "undefined" ? window.__BOOTH_CONFIG__ : undefined;

  // Vite exposes string env vars on import.meta.env; treat as optional.
  const env = (import.meta as unknown as { env?: Record<string, string | undefined> })
    .env;

  return {
    region: injected?.region ?? env?.VITE_AWS_REGION,
    userPoolId: injected?.userPoolId ?? env?.VITE_COGNITO_USER_POOL_ID,
    userPoolClientId:
      injected?.userPoolClientId ?? env?.VITE_COGNITO_USER_POOL_CLIENT_ID,
    identityPoolId:
      injected?.identityPoolId ?? env?.VITE_COGNITO_IDENTITY_POOL_ID,
    endpointName: injected?.endpointName ?? env?.VITE_ENDPOINT_NAME,
    endpointConfigName:
      injected?.endpointConfigName ?? env?.VITE_ENDPOINT_CONFIG_NAME,
    ioBucket: injected?.ioBucket ?? env?.VITE_IO_BUCKET,
    scheduleTable: injected?.scheduleTable ?? env?.VITE_SCHEDULE_TABLE,
    shareBucket: injected?.shareBucket ?? env?.VITE_SHARE_BUCKET,
    shareSignerUrl: injected?.shareSignerUrl ?? env?.VITE_SHARE_SIGNER_URL,
    timezone: injected?.timezone ?? env?.VITE_TIMEZONE,
    invokeFunctionUrl:
      injected?.invokeFunctionUrl ?? env?.VITE_INVOKE_FUNCTION_URL,
  };
}

/** Throws a descriptive error when a required config field is missing or blank. */
function requireField(value: string | undefined, name: string): string {
  if (value === undefined || value === null || value === "") {
    throw new Error(
      `Missing runtime config: "${name}". Ensure window.__BOOTH_CONFIG__ is ` +
        `injected before the app bundle (see CDK hosting/runtime config).`
    );
  }
  return value;
}

/**
 * Returns the auth subset of the runtime config, validating that every field
 * the sign-in gate and credential mint need is present.
 *
 * @throws Error if any auth field is missing — surfaced as a clear setup error
 * rather than an opaque Cognito failure later.
 */
export function getAuthConfig(): AuthConfig {
  const raw = readRawConfig();
  return {
    region: requireField(raw.region, "region"),
    userPoolId: requireField(raw.userPoolId, "userPoolId"),
    userPoolClientId: requireField(raw.userPoolClientId, "userPoolClientId"),
    identityPoolId: requireField(raw.identityPoolId, "identityPoolId"),
  };
}

/**
 * Returns the complete runtime config, validating every field.
 *
 * Consumed by the direct-to-AWS modules (`api/*`, tasks 7.x). The auth layer
 * uses the narrower {@link getAuthConfig} so it does not depend on resource
 * fields it never reads.
 *
 * @throws Error if any field is missing.
 */
export function getConfig(): BoothConfig {
  const raw = readRawConfig();
  return {
    ...getAuthConfig(),
    endpointName: requireField(raw.endpointName, "endpointName"),
    endpointConfigName: requireField(raw.endpointConfigName, "endpointConfigName"),
    ioBucket: requireField(raw.ioBucket, "ioBucket"),
    scheduleTable: requireField(raw.scheduleTable, "scheduleTable"),
    shareBucket: requireField(raw.shareBucket, "shareBucket"),
    shareSignerUrl: requireField(raw.shareSignerUrl, "shareSignerUrl"),
    // Optional: defaults to the scheduler's default timezone when not injected.
    timezone: raw.timezone && raw.timezone !== "" ? raw.timezone : "Asia/Jerusalem",
    invokeFunctionUrl: requireField(raw.invokeFunctionUrl, "invokeFunctionUrl"),
  };
}
