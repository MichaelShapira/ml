import {
  custom_resources as cr,
  aws_iam as iam,
  Stack,
} from "aws-cdk-lib";
import { Construct } from "constructs";

export interface IoBucketCorsProps {
  /**
   * Name of the async I/O bucket to apply CORS to. This is the SageMaker async
   * inference bucket that the browser writes inputs to and reads outputs/failures
   * from. It is referenced (not created) by this stack, so its CORS cannot be set
   * via the L2 `Bucket` `cors` prop — we set it imperatively at deploy time.
   */
  readonly ioBucketName: string;

  /**
   * The CloudFront distribution domain (without scheme) that serves the SPA, e.g.
   * `d1wzenbghyc6sk.cloudfront.net`. Added as an allowed CORS origin so the
   * browser-direct PutObject / Head / GetObject calls pass preflight.
   */
  readonly distributionDomainName: string;

  /**
   * Extra allowed origins (e.g. local dev). Defaults to `http://localhost:5173`.
   */
  readonly additionalAllowedOrigins?: string[];
}

/**
 * Sets the CORS configuration on the **referenced** async I/O bucket via an
 * `AwsCustomResource` (`s3:PutBucketCors`).
 *
 * Why a custom resource: the I/O bucket is the pre-existing SageMaker default
 * bucket, referenced by name (`Bucket.fromBucketName`), so CDK does not own it
 * and the L2 `cors` prop is unavailable. We must call PutBucketCors imperatively.
 *
 * Important: PutBucketCors **replaces** the entire CORS config, so this construct
 * writes BOTH the original SageMaker rule (so SageMaker Studio/processing keep
 * working) AND the booth rule (CloudFront origin + localhost). On stack delete we
 * intentionally do nothing — the bucket is shared and pre-existing, so we leave
 * its CORS untouched rather than risk clobbering the SageMaker rule.
 */
export class IoBucketCorsConstruct extends Construct {
  constructor(scope: Construct, id: string, props: IoBucketCorsProps) {
    super(scope, id);

    const localhostOrigins = props.additionalAllowedOrigins ?? [
      "http://localhost:5173",
      "http://localhost:5174",
    ];

    const corsConfiguration = {
      CORSRules: [
        // Original SageMaker rule — preserved so Studio/processing keep working.
        {
          AllowedHeaders: ["*"],
          AllowedMethods: ["POST", "PUT", "GET", "HEAD", "DELETE"],
          AllowedOrigins: ["https://*.sagemaker.aws", "http://localhost:3000"],
          ExposeHeaders: [
            "ETag",
            "x-amz-delete-marker",
            "x-amz-id-2",
            "x-amz-request-id",
            "x-amz-server-side-encryption",
            "x-amz-version-id",
          ],
        },
        // Booth rule — CloudFront origin + local dev for the browser-direct flow.
        {
          AllowedHeaders: ["*"],
          AllowedMethods: ["GET", "PUT", "HEAD"],
          AllowedOrigins: [
            `https://${props.distributionDomainName}`,
            ...localhostOrigins,
          ],
          ExposeHeaders: ["ETag"],
        },
      ],
    };

    const bucketArn = Stack.of(this).formatArn({
      service: "s3",
      region: "",
      account: "",
      resource: props.ioBucketName,
    });

    new cr.AwsCustomResource(this, "PutBucketCors", {
      // Re-run when the bucket name or distribution domain changes.
      onCreate: {
        service: "S3",
        action: "putBucketCors",
        parameters: {
          Bucket: props.ioBucketName,
          CORSConfiguration: corsConfiguration,
        },
        physicalResourceId: cr.PhysicalResourceId.of(
          `io-bucket-cors-${props.ioBucketName}`,
        ),
      },
      onUpdate: {
        service: "S3",
        action: "putBucketCors",
        parameters: {
          Bucket: props.ioBucketName,
          CORSConfiguration: corsConfiguration,
        },
        physicalResourceId: cr.PhysicalResourceId.of(
          `io-bucket-cors-${props.ioBucketName}`,
        ),
      },
      // No onDelete: leave the shared bucket's CORS untouched on stack delete.
      policy: cr.AwsCustomResourcePolicy.fromStatements([
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["s3:PutBucketCORS"],
          resources: [bucketArn],
        }),
      ]),
    });
  }
}
