import * as path from "path";
import { Duration, RemovalPolicy, Stack } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { NodejsFunction, OutputFormat } from "aws-cdk-lib/aws-lambda-nodejs";
import * as logs from "aws-cdk-lib/aws-logs";
import type { DataConstruct } from "./data-construct";

/**
 * Absolute path to the `Invoke_Proxy` Lambda source
 * (`backend/src/invoke/handler.ts`), resolved relative to this construct file.
 */
const INVOKE_ENTRY = path.join(
  __dirname,
  "..",
  "..",
  "backend",
  "src",
  "invoke",
  "handler.ts",
);

/** Lock file for the `backend/` package (deterministic NodejsFunction bundling). */
const INVOKE_DEPS_LOCK = path.join(
  __dirname,
  "..",
  "..",
  "backend",
  "package-lock.json",
);

/** Name of the exported handler in `backend/src/invoke/handler.ts`. */
const INVOKE_HANDLER = "handler";

/**
 * Props for {@link InvokeConstruct}.
 */
export interface InvokeConstructProps {
  /**
   * The data layer providing the managed endpoint name/ARN and the I/O bucket
   * name used to scope the proxy's least-privilege policy and validation.
   */
  readonly data: DataConstruct;

  /**
   * Allowed CORS origins for the Function URL — the CloudFront distribution
   * origin plus any local-dev origins. The browser-direct flow signs requests
   * to the Function URL, which (unlike the SageMaker runtime API) supports CORS.
   */
  readonly allowedOrigins: string[];
}

/**
 * Invoke_Proxy: a tiny Lambda + IAM-authenticated Function URL that performs the
 * one call the browser cannot make directly — `sagemaker:InvokeEndpointAsync`.
 *
 * The SageMaker runtime API does not support CORS, so the SPA cannot invoke the
 * async endpoint from the browser. This construct proxies just that call:
 *
 *   - a Node 20 Lambda bundled from `backend/src/invoke/handler.ts`;
 *   - a dedicated least-privilege role holding only
 *     `sagemaker:InvokeEndpointAsync` on the endpoint ARN (+ CloudWatch Logs via
 *     the AWS-managed basic execution policy);
 *   - a **Function URL with `authType: AWS_IAM`** (never public) and a native
 *     CORS config allowing the booth's CloudFront origin + local dev.
 *
 * The browser SigV4-signs its request to the Function URL with the visitor's
 * Cognito Identity-Pool credentials; {@link grantInvoke} wires the matching
 * `lambda:InvokeFunctionUrl` permission onto the caller roles.
 *
 * Every resource carries {@link RemovalPolicy.DESTROY} so the stack tears down
 * cleanly.
 */
export class InvokeConstruct extends Construct {
  /** The Node 20 `Invoke_Proxy` Lambda. */
  public readonly invokeFunction: NodejsFunction;

  /** The IAM role assumed by {@link invokeFunction}. */
  public readonly invokeRole: iam.Role;

  /** The IAM-authenticated Function URL the browser signs requests to. */
  public readonly functionUrl: lambda.FunctionUrl;

  constructor(scope: Construct, id: string, props: InvokeConstructProps) {
    super(scope, id);

    const { data } = props;

    // --- IAM role: only InvokeEndpointAsync on the endpoint ARN ------------
    this.invokeRole = new iam.Role(this, "InvokeRole", {
      description:
        "Invoke_Proxy role: sagemaker:InvokeEndpointAsync on the booth endpoint + CloudWatch Logs.",
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AWSLambdaBasicExecutionRole",
        ),
      ],
    });
    this.invokeRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "InvokeAsyncEndpoint",
        effect: iam.Effect.ALLOW,
        actions: ["sagemaker:InvokeEndpointAsync"],
        // Any endpoint in this account/region: the current endpoint is whichever
        // config the admin selected (endpoint name === config name), not a
        // single hardcoded endpoint.
        resources: [
          Stack.of(this).formatArn({
            service: "sagemaker",
            resource: "endpoint",
            resourceName: "*",
          }),
        ],
      }),
    );
    // Read the current-config pointer from the Schedule_Store to resolve which
    // endpoint to invoke.
    this.invokeRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "ReadCurrentConfig",
        effect: iam.Effect.ALLOW,
        actions: ["dynamodb:GetItem"],
        resources: [data.scheduleTable.tableArn],
      }),
    );

    // --- Log group (explicit, DESTROY) -------------------------------------
    const logGroup = new logs.LogGroup(this, "InvokeLogs", {
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // --- Lambda (Node 20, bundled from backend/src/invoke/handler.ts) ------
    this.invokeFunction = new NodejsFunction(this, "InvokeFunction", {
      description:
        "Invoke_Proxy: browser-signed proxy for sagemaker:InvokeEndpointAsync (CORS workaround).",
      runtime: lambda.Runtime.NODEJS_20_X,
      entry: INVOKE_ENTRY,
      handler: INVOKE_HANDLER,
      depsLockFilePath: INVOKE_DEPS_LOCK,
      role: this.invokeRole,
      logGroup,
      timeout: Duration.seconds(30),
      environment: {
        ENDPOINT_NAME: data.endpointName,
        // Scopes input-location validation to the configured I/O bucket.
        IO_BUCKET: data.ioBucket.bucketName,
        // Resolve the admin-selected current endpoint config from this table.
        SCHEDULE_TABLE: data.scheduleTable.tableName,
      },
      bundling: {
        target: "node20",
        format: OutputFormat.CJS,
        // The AWS SDK v3 clients ship in the Node 20 Lambda runtime.
        externalModules: ["@aws-sdk/*"],
      },
    });

    // --- Function URL: IAM auth (NOT public) + native CORS -----------------
    this.functionUrl = this.invokeFunction.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
      cors: {
        allowedOrigins: props.allowedOrigins,
        allowedMethods: [lambda.HttpMethod.POST],
        allowedHeaders: [
          "content-type",
          "authorization",
          "x-amz-date",
          "x-amz-security-token",
          "x-amz-content-sha256",
        ],
        maxAge: Duration.hours(1),
      },
    });
  }

  /**
   * Grant a caller role permission to invoke the Function URL (SigV4-signed).
   *
   * A Function URL with `AWS_IAM` auth requires BOTH the caller's identity
   * policy AND a **resource-based** policy on the function. CDK's
   * `addFunctionUrl` adds neither for arbitrary principals, so this method adds
   * both: the identity statement on the grantee role, and a
   * `lambda:InvokeFunctionUrl` resource permission on the function scoped to the
   * grantee's ARN with the `AWS_IAM` auth-type condition.
   */
  public grantInvoke(grantee: iam.IRole): void {
    // A Lambda Function URL with AWS_IAM auth requires the caller to hold BOTH
    // `lambda:InvokeFunctionUrl` AND `lambda:InvokeFunction` on the function.
    // Granting only InvokeFunctionUrl yields a 403 from the URL authorizer even
    // though the action name suggests it should suffice (verified empirically:
    // InvokeFunctionUrl alone -> 403; both -> reaches the function). Scope both
    // to this function's ARN to keep least privilege.
    grantee.addToPrincipalPolicy(
      new iam.PolicyStatement({
        sid: "InvokeProxyFunctionUrl",
        effect: iam.Effect.ALLOW,
        actions: ["lambda:InvokeFunctionUrl", "lambda:InvokeFunction"],
        resources: [this.invokeFunction.functionArn],
      }),
    );
  }
}
