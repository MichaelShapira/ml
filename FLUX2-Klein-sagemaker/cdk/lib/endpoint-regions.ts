import { Stack } from "aws-cdk-lib";

/**
 * A region the booth may run its SageMaker endpoint in, with that region's
 * async I/O bucket. SageMaker reads model data from the endpoint's own region,
 * and the booth keeps each endpoint's requests and results in the same region,
 * so every endpoint region has its own bucket.
 */
export interface EndpointRegion {
  /** AWS region, e.g. `us-east-2`. */
  readonly region: string;
  /** Async I/O bucket in that region (inputs/outputs/failures + FLUX 3 model data). */
  readonly ioBucketName: string;
  /** True for the stack's own region (the booth's original endpoint region). */
  readonly home: boolean;
}

/** `us-east-2`, `eu-west-1`, `ap-southeast-2`, `us-gov-west-1`, ... */
const REGION_PATTERN = /^[a-z]{2}(-[a-z]+)+-\d$/;

/**
 * Parse the comma-separated `endpointRegions` context value (extra regions the
 * admin may pick endpoints from). Blank entries are ignored, duplicates and the
 * stack's own region are dropped, and anything that is not a region name throws
 * so a typo fails at synth time instead of producing an unusable policy.
 */
export function parseExtraRegions(value: unknown, homeRegion: string): string[] {
  if (value === undefined || value === null) {
    return [];
  }
  if (typeof value !== "string") {
    throw new Error(`endpointRegions must be a comma-separated string, got ${typeof value}`);
  }
  const regions: string[] = [];
  for (const raw of value.split(",")) {
    const region = raw.trim();
    if (region.length === 0) {
      continue;
    }
    if (!REGION_PATTERN.test(region)) {
      throw new Error(`endpointRegions: "${region}" is not an AWS region name`);
    }
    if (region !== homeRegion && !regions.includes(region)) {
      regions.push(region);
    }
  }
  return regions;
}

/**
 * The booth's endpoint regions: the stack's region (with the stack's I/O
 * bucket) first, then each extra region from the `endpointRegions` context with
 * its default SageMaker bucket, `sagemaker-<region>-<account>`. deploy.sh makes
 * sure those buckets exist before the stack is deployed.
 */
export function resolveEndpointRegions(stack: Stack, homeIoBucketName: string): EndpointRegion[] {
  const extra = parseExtraRegions(stack.node.tryGetContext("endpointRegions"), stack.region);
  return [
    { region: stack.region, ioBucketName: homeIoBucketName, home: true },
    ...extra.map((region) => ({
      region,
      ioBucketName: `sagemaker-${region}-${stack.account}`,
      home: false,
    })),
  ];
}

/** `arn:<partition>:s3:::<bucket>` */
export function bucketArn(stack: Stack, bucketName: string): string {
  return `arn:${stack.partition}:s3:::${bucketName}`;
}

/** `arn:<partition>:s3:::<bucket>/<pattern>` */
export function objectArn(stack: Stack, bucketName: string, pattern: string): string {
  return `arn:${stack.partition}:s3:::${bucketName}/${pattern}`;
}

/** `arn:<partition>:sagemaker:<region>:<account>:<resource>/<name>` */
export function sageMakerArn(stack: Stack, region: string, resource: string, name: string): string {
  return `arn:${stack.partition}:sagemaker:${region}:${stack.account}:${resource}/${name}`;
}
