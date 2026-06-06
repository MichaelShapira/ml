import { RemovalPolicy, Stack } from "aws-cdk-lib";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as iam from "aws-cdk-lib/aws-iam";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

/**
 * The existing FLUX.2 [klein] 9B SageMaker asynchronous inference endpoint that
 * the photo booth drives. This is the default endpoint name; callers may
 * override it via {@link DataConstructProps.endpointName}.
 *
 * Source: requirements glossary (`FLUX2_Endpoint`) and `flux2-klein-sagemaker.ipynb`.
 */
export const FLUX2_ENDPOINT_NAME = "flux2-klein-9b-g6e2";

/**
 * S3 key prefixes used by the existing async inference flow. These mirror the
 * `flux2-klein-sagemaker.ipynb` notebook's `invoke()` input prefix and the
 * `AsyncInferenceConfig` output/failure paths, so the CDK-provisioned (or
 * referenced) I/O bucket and the IAM policies stay aligned with the endpoint.
 */
export const INPUTS_PREFIX = "flux2-klein-inputs/";
export const OUTPUTS_PREFIX = "flux2-klein-outputs/";
export const FAILURES_PREFIX = "flux2-klein-failures/";

export interface DataConstructProps {
  /**
   * The name of the managed SageMaker endpoint. Used to derive the endpoint ARN
   * for least-privilege IAM statements. Defaults to {@link FLUX2_ENDPOINT_NAME}.
   */
  readonly endpointName?: string;

  /**
   * If set, the async I/O bucket is **referenced** by name via
   * `Bucket.fromBucketName` instead of being created. Use this when the
   * deployment reuses the notebook's pre-existing default bucket (which holds
   * the `flux2-klein-inputs/`, `flux2-klein-outputs/`, and
   * `flux2-klein-failures/` prefixes).
   *
   * When omitted (the default), a **new** bucket is created with
   * `autoDeleteObjects: true` and `RemovalPolicy.DESTROY` so the demo stack
   * tears down cleanly (Requirement 22.4).
   */
  readonly existingIoBucketName?: string;
}

/**
 * Provisions the system's persistence layer and centralizes the reusable IAM
 * policy statements that the auth and scheduler constructs attach to their
 * roles, so least-privilege policy definitions live in exactly one place.
 *
 * Resources:
 * - `Schedule_Store` DynamoDB table — partition key `pk` (string), sort key
 *   `sk` (string), on-demand billing, `RemovalPolicy.DESTROY` (Requirements
 *   20.1, 22.2, 22.4).
 * - Async I/O S3 bucket for the `flux2-klein-inputs/`, `flux2-klein-outputs/`,
 *   and `flux2-klein-failures/` prefixes — created with `autoDeleteObjects` +
 *   `RemovalPolicy.DESTROY` and all public access blocked, OR a reference to an
 *   existing bucket by name when {@link DataConstructProps.existingIoBucketName}
 *   is supplied (Requirements 22.2, 22.4).
 *
 * Policy factories (consumed by `auth-construct` and `scheduler-construct`):
 * - {@link authenticatedRoleStatements} — capture/effect flow least-privilege.
 * - {@link adminExtraStatements} — endpoint management + schedule writes.
 * - {@link schedulerStatements} — endpoint management + read-only schedule.
 *
 * The construct does not build the IAM roles themselves; it only mints the
 * `PolicyStatement`s so that `auth-construct` (task 13.1) and
 * `scheduler-construct` (task 13.4) attach identical, single-sourced policies.
 */
export class DataConstruct extends Construct {
  /** The `Schedule_Store` DynamoDB table that persists Working_Hours. */
  public readonly scheduleTable: dynamodb.Table;

  /**
   * The async I/O bucket (created by this stack or referenced by name). Holds
   * the inputs/outputs/failures prefixes used by the async inference flow.
   */
  public readonly ioBucket: s3.IBucket;

  /** The managed SageMaker endpoint name used to derive endpoint ARNs. */
  public readonly endpointName: string;

  constructor(scope: Construct, id: string, props: DataConstructProps = {}) {
    super(scope, id);

    this.endpointName = props.endpointName ?? FLUX2_ENDPOINT_NAME;

    // --- Schedule_Store DynamoDB table -------------------------------------
    // Single-table design: pk = ENDPOINT#<name>, sk = DAY#<YYYY-MM-DD>.
    // On-demand billing keeps the demo trivially cheap; DESTROY removal policy
    // tears the table down with the stack (Requirements 20.1, 22.4).
    this.scheduleTable = new dynamodb.Table(this, "ScheduleStore", {
      partitionKey: { name: "pk", type: dynamodb.AttributeType.STRING },
      sortKey: { name: "sk", type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // --- Async I/O S3 bucket -----------------------------------------------
    if (props.existingIoBucketName) {
      // Reuse the notebook's pre-existing bucket. Referenced (not created), so
      // it is unaffected by this stack's removal policy.
      this.ioBucket = s3.Bucket.fromBucketName(
        this,
        "AsyncIoBucket",
        props.existingIoBucketName,
      );
    } else {
      // Create a fresh bucket scoped to this stack. autoDeleteObjects empties
      // it on destroy so the DESTROY removal policy can delete a non-empty
      // bucket cleanly (Requirement 22.4).
      this.ioBucket = new s3.Bucket(this, "AsyncIoBucket", {
        blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
        encryption: s3.BucketEncryption.S3_MANAGED,
        enforceSSL: true,
        autoDeleteObjects: true,
        removalPolicy: RemovalPolicy.DESTROY,
      });
    }
  }

  // ---------------------------------------------------------------------------
  // ARN helpers
  // ---------------------------------------------------------------------------

  /**
   * The ARN of the managed SageMaker endpoint, derived from the stack's
   * partition/region/account so no ARN is hardcoded.
   */
  public get endpointArn(): string {
    return Stack.of(this).formatArn({
      service: "sagemaker",
      resource: "endpoint",
      resourceName: this.endpointName,
    });
  }

  /**
   * ARN covering ANY SageMaker endpoint in this account/region. The scheduler
   * and admin manage the endpoint of whichever config is currently selected
   * (endpoint name === config name), so management is not limited to one name.
   */
  public get anyEndpointArn(): string {
    return Stack.of(this).formatArn({
      service: "sagemaker",
      resource: "endpoint",
      resourceName: "*",
    });
  }

  /**
   * The ARN pattern covering SageMaker endpoint configurations in this
   * account/region. `CreateEndpoint`/`DeleteEndpoint` reference the endpoint
   * config in addition to the endpoint itself.
   */
  public get endpointConfigArn(): string {
    return Stack.of(this).formatArn({
      service: "sagemaker",
      resource: "endpoint-config",
      resourceName: "*",
    });
  }

  // ---------------------------------------------------------------------------
  // Reusable IAM policy statement factories
  // ---------------------------------------------------------------------------

  /**
   * Least-privilege statements for the **capture/effect flow** only, attached to
   * both `Authenticated_Role` and (transitively) `Admin_Role`.
   *
   * Grants:
   * - `sagemaker:InvokeEndpointAsync` on the endpoint ARN.
   * - `s3:PutObject` on the inputs prefix.
   * - `s3:GetObject` on the outputs + failures prefixes.
   * - `s3:ListBucket` on the bucket, scoped by an `s3:prefix` condition to the
   *   outputs + failures prefixes (enables `HeadObject` 404-vs-403 semantics).
   *
   * No SageMaker management, no DynamoDB, no write access to outputs/failures.
   *
   * @param endpointArn ARN of the `FLUX2_Endpoint` (defaults to {@link endpointArn}).
   */
  public authenticatedRoleStatements(
    endpointArn: string = this.endpointArn,
  ): iam.PolicyStatement[] {
    const bucketArn = this.ioBucket.bucketArn;

    return [
      new iam.PolicyStatement({
        sid: "InvokeAsyncEndpoint",
        effect: iam.Effect.ALLOW,
        actions: ["sagemaker:InvokeEndpointAsync"],
        resources: [endpointArn],
      }),
      new iam.PolicyStatement({
        sid: "PutInputObjects",
        effect: iam.Effect.ALLOW,
        actions: ["s3:PutObject"],
        resources: [`${bucketArn}/${INPUTS_PREFIX}*`],
      }),
      new iam.PolicyStatement({
        sid: "GetResultObjects",
        effect: iam.Effect.ALLOW,
        actions: ["s3:GetObject"],
        resources: [
          `${bucketArn}/${OUTPUTS_PREFIX}*`,
          `${bucketArn}/${FAILURES_PREFIX}*`,
        ],
      }),
      new iam.PolicyStatement({
        sid: "ListResultPrefixes",
        effect: iam.Effect.ALLOW,
        actions: ["s3:ListBucket"],
        resources: [bucketArn],
        conditions: {
          StringLike: {
            "s3:prefix": [`${OUTPUTS_PREFIX}*`, `${FAILURES_PREFIX}*`],
          },
        },
      }),
    ];
  }

  /**
   * Endpoint-management and schedule-write statements layered on top of
   * {@link authenticatedRoleStatements} for the `Admin_Role`. Returns only the
   * **extra** statements; attach these alongside the authenticated ones (or use
   * {@link adminRoleStatements} for the combined set).
   *
   * Grants:
   * - `sagemaker:ListEndpoints` on `*` (List has no per-endpoint scoping).
   * - `sagemaker:DescribeEndpoint`/`CreateEndpoint`/`DeleteEndpoint` on the
   *   endpoint ARN and endpoint-config ARN.
   * - `dynamodb:Query`/`PutItem`/`DeleteItem` on the `Schedule_Store` table.
   *
   * No `iam:PassRole` is granted: the booth only calls `CreateEndpoint`
   * (referencing the pre-existing EndpointConfig by name) and `DeleteEndpoint`,
   * neither of which passes a role.
   *
   * @param endpointArn       ARN of the `FLUX2_Endpoint` (defaults to {@link endpointArn}).
   * @param endpointConfigArn ARN of the endpoint config (defaults to {@link endpointConfigArn}).
   */
  public adminExtraStatements(
    endpointArn: string = this.endpointArn,
    endpointConfigArn: string = this.endpointConfigArn,
  ): iam.PolicyStatement[] {
    return [
      ...this.endpointManagementStatements(endpointArn, endpointConfigArn),
      new iam.PolicyStatement({
        sid: "ScheduleReadWrite",
        effect: iam.Effect.ALLOW,
        actions: [
          "dynamodb:Query",
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:DeleteItem",
        ],
        resources: [this.scheduleTable.tableArn],
      }),
    ];
  }

  /**
   * The full `Admin_Role` statement set: the capture/effect flow statements
   * plus the admin-only endpoint-management and schedule-write statements.
   *
   * @see authenticatedRoleStatements
   * @see adminExtraStatements
   */
  public adminRoleStatements(
    endpointArn: string = this.endpointArn,
    endpointConfigArn: string = this.endpointConfigArn,
  ): iam.PolicyStatement[] {
    return [
      ...this.authenticatedRoleStatements(endpointArn),
      ...this.adminExtraStatements(endpointArn, endpointConfigArn),
    ];
  }

  /**
   * Statements for the `Scheduler_Function` Lambda role (Requirement 21.6):
   * endpoint management plus **read-only** schedule access. The scheduler never
   * writes the schedule, so DynamoDB is limited to `Query`/`GetItem`.
   *
   * Grants:
   * - `sagemaker:ListEndpoints` on `*`.
   * - `sagemaker:DescribeEndpoint`/`CreateEndpoint`/`DeleteEndpoint` on the
   *   endpoint ARN and endpoint-config ARN.
   * - `dynamodb:Query`/`GetItem` (read-only) on the `Schedule_Store` table.
   *
   * No `iam:PassRole` is granted: `CreateEndpoint` references the pre-existing
   * EndpointConfig by name and passes no role. CloudWatch Logs permissions are
   * provided by the Lambda's default execution role in `scheduler-construct`.
   *
   * @param endpointArn       ARN of the `FLUX2_Endpoint` (defaults to {@link endpointArn}).
   * @param endpointConfigArn ARN of the endpoint config (defaults to {@link endpointConfigArn}).
   */
  public schedulerStatements(
    endpointArn: string = this.endpointArn,
    endpointConfigArn: string = this.endpointConfigArn,
  ): iam.PolicyStatement[] {
    return [
      ...this.endpointManagementStatements(endpointArn, endpointConfigArn),
      new iam.PolicyStatement({
        sid: "ScheduleReadOnly",
        effect: iam.Effect.ALLOW,
        actions: ["dynamodb:Query", "dynamodb:GetItem"],
        resources: [this.scheduleTable.tableArn],
      }),
    ];
  }

  /**
   * Shared SageMaker management statements (List + Describe/Create/Delete)
   * reused by both the admin role and the scheduler role so the management
   * surface is defined exactly once. No `iam:PassRole` — neither
   * `CreateEndpoint` (by config name) nor `DeleteEndpoint` passes a role.
   */
  private endpointManagementStatements(
    endpointArn: string,
    endpointConfigArn: string,
  ): iam.PolicyStatement[] {
    return [
      new iam.PolicyStatement({
        sid: "ListEndpoints",
        effect: iam.Effect.ALLOW,
        actions: ["sagemaker:ListEndpoints"],
        // ListEndpoints has no per-endpoint resource scoping.
        resources: ["*"],
      }),
      new iam.PolicyStatement({
        sid: "ManageEndpoint",
        effect: iam.Effect.ALLOW,
        actions: [
          "sagemaker:DescribeEndpoint",
          "sagemaker:CreateEndpoint",
          "sagemaker:DeleteEndpoint",
          // Required to tag the endpoint on CreateEndpoint(Tags=[...]).
          "sagemaker:AddTags",
        ],
        resources: [endpointArn, endpointConfigArn],
      }),
    ];
  }

  // ---------------------------------------------------------------------------
  // Grant convenience helpers (delegate to the table/bucket grants)
  // ---------------------------------------------------------------------------

  /** Grants read+write schedule access (Query/PutItem/DeleteItem semantics). */
  public grantScheduleReadWrite(grantee: iam.IGrantable): iam.Grant {
    return this.scheduleTable.grantReadWriteData(grantee);
  }

  /** Grants read-only schedule access (used by the scheduler). */
  public grantScheduleRead(grantee: iam.IGrantable): iam.Grant {
    return this.scheduleTable.grantReadData(grantee);
  }
}
