import * as path from "path";
import { Duration, RemovalPolicy } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { NodejsFunction, OutputFormat } from "aws-cdk-lib/aws-lambda-nodejs";
import * as logs from "aws-cdk-lib/aws-logs";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import type { DataConstruct } from "./data-construct";

/** Default kiosk timezone used to evaluate wall-clock Working_Hours. */
const DEFAULT_TIMEZONE = "America/Los_Angeles";

/**
 * Absolute path to the `Scheduler_Function` Lambda source
 * (`backend/src/scheduler/apply.ts`), resolved relative to this construct file
 * so the bundle works regardless of the current working directory.
 *
 * The cdk package is CommonJS (its `package.json` declares no `"type":
 * "module"`), so `__dirname` is available here. From `cdk/lib/` we walk up two
 * levels to the repo root, then into the `backend/` package.
 *
 * NOTE: `apply.ts` is implemented by task 11.1. NodejsFunction only reads this
 * entry at **synth** time (esbuild bundling), not when `tsc` builds this
 * construct, so typechecking this file does not require the source to exist
 * yet. The entry must export a `handler` (see {@link SCHEDULER_HANDLER}).
 */
const SCHEDULER_ENTRY = path.join(
  __dirname,
  "..",
  "..",
  "backend",
  "src",
  "scheduler",
  "apply.ts",
);

/**
 * Lock file for the `backend/` package. Set explicitly so NodejsFunction
 * installs/bundles deterministically against the backend's dependency tree
 * rather than searching upward for an ambiguous lock file.
 */
const SCHEDULER_DEPS_LOCK = path.join(
  __dirname,
  "..",
  "..",
  "backend",
  "package-lock.json",
);

/** Name of the exported handler in `backend/src/scheduler/apply.ts`. */
const SCHEDULER_HANDLER = "handler";

/**
 * Props for {@link SchedulerConstruct}.
 *
 * Dependencies are injected rather than hardcoded. The {@link DataConstruct} is
 * the single source of truth for the managed endpoint name, its ARNs, the
 * `Schedule_Store` table, and the least-privilege `schedulerStatements` policy
 * factory (Requirement 21.6).
 */
export interface SchedulerConstructProps {
  /**
   * The data layer providing the `Schedule_Store` table, the managed endpoint
   * name/ARNs, and the reusable `schedulerStatements` IAM policy factory.
   */
  readonly data: DataConstruct;

  /**
   * Name of the SageMaker endpoint configuration the scheduler passes to
   * `CreateEndpoint` when it starts the endpoint. Surfaced to the handler as
   * the `ENDPOINT_CONFIG_NAME` environment variable.
   */
  readonly endpointConfigName: string;

  /**
   * IANA timezone used to interpret the wall-clock Working_Hours stored in
   * `Schedule_Store`. Surfaced to the handler as the `TIMEZONE` environment
   * variable.
   *
   * @default "America/Los_Angeles"
   */
  readonly timezone?: string;
}

/**
 * Scheduler_Function (the only Lambda) + its IAM role + the two EventBridge
 * rules that drive it (Requirements 21.1, 21.6, 22.2, 22.4).
 *
 * Provisions, as described in the design's "Scheduler_Function (the only
 * Lambda)" section:
 *   - a Node 20 Lambda bundled from `backend/src/scheduler/apply.ts` that, on
 *     each tick, reads today's Working_Hours, computes in/out of the
 *     `[startTime, endTime)` window in the configured timezone, and reconciles
 *     the endpoint idempotently (create when inside-window + NOT_DEPLOYED,
 *     delete when outside-window + IN_SERVICE, otherwise no-op);
 *   - an IAM role carrying the least-privilege `schedulerStatements` from
 *     {@link DataConstruct} (SageMaker List/Describe/Create/Delete + read-only
 *     DynamoDB on `Schedule_Store`) plus the AWS-managed basic execution policy
 *     for CloudWatch Logs; and
 *   - two EventBridge `rate(1 minute)` rules offset by ~30 s to approximate a
 *     30 s reconcile cadence (see the constructor for the offset technique).
 *
 * Every resource that supports it carries {@link RemovalPolicy.DESTROY} so the
 * demo/kiosk stack tears down cleanly (Requirement 22.4); in particular the
 * Lambda's log group is created explicitly with DESTROY (CloudWatch log groups
 * otherwise default to RETAIN).
 */
export class SchedulerConstruct extends Construct {
  /** The Node 20 `Scheduler_Function` Lambda. */
  public readonly schedulerFunction: NodejsFunction;

  /** The IAM role assumed by {@link schedulerFunction}. */
  public readonly schedulerRole: iam.Role;

  /** EventBridge rule A — fires every minute, invokes the function with no delay. */
  public readonly ruleA: events.Rule;

  /**
   * EventBridge rule B — fires every minute, invokes the function with a
   * constant `{ delaySeconds: 30 }` input so the handler self-delays ~30 s,
   * offsetting it from rule A.
   */
  public readonly ruleB: events.Rule;

  constructor(scope: Construct, id: string, props: SchedulerConstructProps) {
    super(scope, id);

    const { data } = props;
    const timezone = props.timezone ?? DEFAULT_TIMEZONE;

    // --- IAM role ----------------------------------------------------------
    // Created explicitly so it can be exposed as a concrete iam.Role. The
    // AWS-managed basic execution policy supplies the CloudWatch Logs
    // permissions (CreateLogGroup/CreateLogStream/PutLogEvents); the
    // endpoint-management + read-only schedule permissions come from the
    // single-sourced schedulerStatements factory (Requirement 21.6).
    this.schedulerRole = new iam.Role(this, "SchedulerRole", {
      description:
        "Scheduler_Function role: SageMaker endpoint mgmt + read-only Schedule_Store + CloudWatch Logs.",
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AWSLambdaBasicExecutionRole",
        ),
      ],
    });
    for (const statement of data.schedulerStatements(
      data.anyEndpointArn,
      data.endpointConfigArn,
    )) {
      this.schedulerRole.addToPolicy(statement);
    }

    // --- Log group (explicit, DESTROY) -------------------------------------
    // CloudWatch log groups default to RETAIN; create one explicitly with
    // DESTROY so the demo/kiosk stack tears down cleanly (Requirement 22.4).
    const logGroup = new logs.LogGroup(this, "SchedulerLogs", {
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // --- Lambda (Node 20, bundled from backend/src/scheduler/apply.ts) ------
    // Timeout is 60 s — comfortably covering the optional ~30 s self-delay used
    // by rule B (see below) plus the reconcile work (DynamoDB read +
    // Describe/Create/Delete endpoint).
    this.schedulerFunction = new NodejsFunction(this, "SchedulerFunction", {
      description:
        "Scheduler_Function: reconciles the FLUX.2 endpoint against Working_Hours (the only Lambda).",
      runtime: lambda.Runtime.NODEJS_20_X,
      entry: SCHEDULER_ENTRY,
      handler: SCHEDULER_HANDLER,
      depsLockFilePath: SCHEDULER_DEPS_LOCK,
      role: this.schedulerRole,
      logGroup,
      timeout: Duration.seconds(60),
      environment: {
        // Read by the handler to scope SageMaker management calls.
        ENDPOINT_NAME: data.endpointName,
        // Schedule_Store table the handler Query/GetItem-s for today's hours.
        SCHEDULE_TABLE: data.scheduleTable.tableName,
        // IANA timezone used to interpret wall-clock Working_Hours.
        TIMEZONE: timezone,
        // Endpoint configuration name passed to CreateEndpoint on start.
        ENDPOINT_CONFIG_NAME: props.endpointConfigName,
      },
      bundling: {
        target: "node20",
        format: OutputFormat.CJS,
        // The AWS SDK v3 clients are present in the Node 20 Lambda runtime, so
        // exclude them from the bundle to keep the artifact small.
        externalModules: ["@aws-sdk/*"],
      },
    });

    // --- EventBridge trigger: ~30 s cadence from two rate(1 minute) rules ---
    // EventBridge's minimum schedule granularity is 1 minute, so a single rule
    // cannot fire every 30 s. We approximate a 30 s cadence with TWO rules,
    // each rate(1 minute), targeting the same function (Requirement 21.1):
    //   - Rule A invokes the function directly (no input).
    //   - Rule B invokes the function with a constant { delaySeconds: 30 }
    //     input; the handler awaits ~30 s before reconciling (the "self-delay"
    //     offset technique). This guarantees the offset at the cost of a short
    //     idle wait inside rule B's invocation, so the function effectively
    //     reconciles about twice a minute.
    // The reconcile logic is idempotent, so an imperfect offset only affects
    // latency-to-reconcile, never correctness.
    this.ruleA = new events.Rule(this, "RuleA", {
      description:
        "Scheduler tick A: rate(1 minute), invokes Scheduler_Function with no delay.",
      schedule: events.Schedule.rate(Duration.minutes(1)),
    });
    this.ruleA.addTarget(new targets.LambdaFunction(this.schedulerFunction));

    this.ruleB = new events.Rule(this, "RuleB", {
      description:
        "Scheduler tick B: rate(1 minute), invokes Scheduler_Function with a 30 s self-delay (offset from rule A).",
      schedule: events.Schedule.rate(Duration.minutes(1)),
    });
    this.ruleB.addTarget(
      new targets.LambdaFunction(this.schedulerFunction, {
        // Constant event input consumed by the handler to self-delay ~30 s,
        // offsetting rule B from rule A.
        event: events.RuleTargetInput.fromObject({ delaySeconds: 30 }),
      }),
    );
  }
}
