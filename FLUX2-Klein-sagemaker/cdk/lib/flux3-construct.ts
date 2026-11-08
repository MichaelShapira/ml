import { Stack } from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import type * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

import { FAILURES_PREFIX, INPUTS_PREFIX, OUTPUTS_PREFIX } from "./data-construct";
import { bucketArn, objectArn, type EndpointRegion } from "./endpoint-regions";

/**
 * S3 prefix (in the async I/O bucket) holding everything FLUX 3 Image Klein
 * needs: `model/` is mirrored to `/opt/ml/model` (Ray Serve config, serving
 * code, staged weights) and `staging/` holds the weight-staging script. Same
 * layout as the standalone notebook in `../FLUX3-Klein-sagemaker-g7e`, so the
 * two share one copy of the weights.
 */
export const FLUX3_PREFIX = "flux3-klein-g7e/";

/**
 * Secrets Manager name of the Hugging Face read token deploy.sh stores (the
 * FLUX 3 weights are gated). Only the weight-staging Processing job reads it;
 * the endpoint never talks to the Hub.
 */
export const FLUX3_HF_TOKEN_SECRET_NAME = "AmazonSageMaker-flux3-klein-hf-token";

/** Account that publishes the AWS Deep Learning Containers in commercial regions. */
export const DLC_ACCOUNT = "763104351884";

/** Props for {@link Flux3Construct}. */
export interface Flux3ConstructProps {
  /** The async I/O bucket (referenced or created by the data construct). */
  readonly ioBucket: s3.IBucket;
  /**
   * Extra regions the booth may run the FLUX 3 endpoint in, each with its own
   * I/O bucket (see endpoint-regions.ts). The home region is always covered.
   */
  readonly extraRegions?: readonly EndpointRegion[];
}

/**
 * Execution role for the FLUX 3 Image Klein endpoint and its weight-staging job.
 *
 * deploy.sh uses this role for the SageMaker Model behind the
 * `FluxAiBoot-flux3-g7e` endpoint config and for the Processing job that copies
 * the weights from Hugging Face to S3. The Model and EndpointConfig themselves
 * are created by deploy.sh, not CloudFormation: the booth creates and deletes
 * the endpoint at runtime from a config of the same name, and a fixed-name
 * EndpointConfig cannot be replaced in place by CloudFormation.
 *
 * Least privilege, and deliberately WITHOUT IAM conditions: SageMaker validates
 * an execution role against an empty request context, so a condition such as
 * `s3:prefix` makes CreateModel reject the role. Scoping is by resource ARN.
 */
export class Flux3Construct extends Construct {
  /** The SageMaker execution role (trusted by sagemaker.amazonaws.com). */
  public readonly role: iam.Role;

  constructor(scope: Construct, id: string, props: Flux3ConstructProps) {
    super(scope, id);
    const stack = Stack.of(this);
    const bucket = props.ioBucket;
    const extraRegions = props.extraRegions ?? [];
    const regions = [stack.region, ...extraRegions.map((r) => r.region)];

    this.role = new iam.Role(this, "SageMakerRole", {
      description:
        "FLUX 3 Image Klein: SageMaker execution role for the booth's g7e endpoint and its weight-staging job.",
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
    });

    const statements = [
      // Model data (config.yaml, code/, weights/) + the staging script, and the
      // request JSON the booth uploads for each async invocation.
      new iam.PolicyStatement({
        sid: "ReadModelAndRequests",
        actions: ["s3:GetObject"],
        resources: [bucket.arnForObjects(`${FLUX3_PREFIX}*`), bucket.arnForObjects(`${INPUTS_PREFIX}*`)],
      }),
      // The staging job writes the weights; the endpoint writes results and
      // failure records where the booth polls for them.
      new iam.PolicyStatement({
        sid: "WriteWeightsAndResults",
        actions: ["s3:PutObject", "s3:AbortMultipartUpload"],
        resources: [
          bucket.arnForObjects(`${FLUX3_PREFIX}model/weights/*`),
          bucket.arnForObjects(`${OUTPUTS_PREFIX}*`),
          bucket.arnForObjects(`${FAILURES_PREFIX}*`),
        ],
      }),
      // Listing the model prefix (model download, staging resume). No s3:prefix
      // condition, see the class comment.
      new iam.PolicyStatement({
        sid: "ListIoBucket",
        actions: ["s3:ListBucket"],
        resources: [bucket.bucketArn],
      }),
      // Pull the AWS Ray Serve DLC image. GetAuthorizationToken has no resource
      // scoping.
      new iam.PolicyStatement({
        sid: "EcrAuth",
        actions: ["ecr:GetAuthorizationToken"],
        resources: ["*"],
      }),
      new iam.PolicyStatement({
        sid: "PullRayServeDlc",
        actions: ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
        resources: regions.map((r) => `arn:${stack.partition}:ecr:${r}:${DLC_ACCOUNT}:repository/ray`),
      }),
      // Container and job logs.
      new iam.PolicyStatement({
        sid: "WriteSageMakerLogs",
        actions: ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
        resources: regions.flatMap((r) => [
          `arn:${stack.partition}:logs:${r}:${stack.account}:log-group:/aws/sagemaker/*`,
          `arn:${stack.partition}:logs:${r}:${stack.account}:log-group:/aws/sagemaker/*:log-stream:*`,
        ]),
      }),
      // Endpoint invocation / utilisation metrics. PutMetricData has no
      // resource scoping.
      new iam.PolicyStatement({
        sid: "PutMetrics",
        actions: ["cloudwatch:PutMetricData"],
        resources: ["*"],
      }),
      // The Hugging Face token, for the staging job only. Secrets Manager adds
      // a random 6-character suffix to secret ARNs.
      new iam.PolicyStatement({
        sid: "ReadHfTokenSecret",
        actions: ["secretsmanager:GetSecretValue"],
        resources: [
          `arn:${stack.partition}:secretsmanager:${stack.region}:${stack.account}:secret:${FLUX3_HF_TOKEN_SECRET_NAME}-??????`,
        ],
      }),
    ];
    // The same endpoint in another region reads its model data and requests
    // from, and writes its results to, that region's bucket. deploy.sh copies
    // the staged weights there with the operator's credentials, so the role
    // gets no write access to the model prefix outside the home region.
    for (const extra of extraRegions) {
      const b = extra.ioBucketName;
      statements.push(
        new iam.PolicyStatement({
          actions: ["s3:GetObject"],
          resources: [objectArn(stack, b, `${FLUX3_PREFIX}*`), objectArn(stack, b, `${INPUTS_PREFIX}*`)],
        }),
        new iam.PolicyStatement({
          actions: ["s3:PutObject", "s3:AbortMultipartUpload"],
          resources: [objectArn(stack, b, `${OUTPUTS_PREFIX}*`), objectArn(stack, b, `${FAILURES_PREFIX}*`)],
        }),
        new iam.PolicyStatement({
          actions: ["s3:ListBucket"],
          resources: [bucketArn(stack, b)],
        }),
      );
    }
    for (const statement of statements) {
      this.role.addToPolicy(statement);
    }
  }
}
