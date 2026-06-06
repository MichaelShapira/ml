import { Construct } from "constructs";
import { RemovalPolicy } from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";

/**
 * Properties for {@link HostingConstruct}.
 *
 * The construct is intentionally self-contained and takes no required props so
 * it can be reused as-is by the root stack. An optional comment string can be
 * supplied for the CloudFront distribution.
 */
export interface HostingConstructProps {
  /**
   * Optional comment applied to the CloudFront distribution, useful for
   * identifying the distribution in the AWS console.
   *
   * @default - "AI Photo Booth UI distribution"
   */
  readonly distributionComment?: string;
}

/**
 * UI hosting for the AI Photo Booth SPA (Requirement 24).
 *
 * Provisions a **private** S3 bucket (`UI_Bucket`) that holds the built React
 * assets and a CloudFront distribution (`Web_Distribution`) that serves those
 * assets to clients. The bucket blocks all public access; access is granted
 * exclusively to the distribution via CloudFront **Origin Access Control
 * (OAC)**. Using {@link origins.S3BucketOrigin.withOriginAccessControl} makes
 * CDK auto-attach a bucket policy that restricts `s3:GetObject` to this
 * distribution (Requirement 24.1–24.3).
 *
 * For client-side routing, HTTP 403 and 404 responses from the origin are
 * rewritten to `/index.html` with an HTTP 200 status so deep links resolve to
 * the SPA entry document (Requirement 24.4).
 *
 * This construct does **not** deploy any assets into the bucket — the
 * `BucketDeployment` of the built `ui/` assets is wired separately (task 14.1).
 * The bucket is exposed via {@link uiBucket} so that deployment can target it.
 *
 * All resources use {@link RemovalPolicy.DESTROY}, and the bucket sets
 * `autoDeleteObjects: true` so the demo/kiosk stack tears down cleanly
 * (Requirement 22.4).
 */
export class HostingConstruct extends Construct {
  /**
   * The private S3 bucket holding the built UI assets. Exposed so a later task
   * (14.1) can deploy assets into it via a `BucketDeployment`.
   */
  public readonly uiBucket: s3.Bucket;

  /** The CloudFront distribution serving the UI from {@link uiBucket}. */
  public readonly distribution: cloudfront.Distribution;

  /** The CloudFront-assigned domain name of {@link distribution}. */
  public readonly distributionDomainName: string;

  constructor(scope: Construct, id: string, props: HostingConstructProps = {}) {
    super(scope, id);

    // Private UI_Bucket: all public access blocked, encrypted with S3-managed
    // keys, and torn down with the stack (Requirements 24.1, 22.4).
    this.uiBucket = new s3.Bucket(this, "UiBucket", {
      blockPublicAccess: new s3.BlockPublicAccess({
        blockPublicAcls: true,
        blockPublicPolicy: true,
        ignorePublicAcls: true,
        restrictPublicBuckets: true,
      }),
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      autoDeleteObjects: true,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // CloudFront serves the bucket as its origin using Origin Access Control
    // (OAC). `withOriginAccessControl` auto-attaches a bucket policy that
    // restricts s3:GetObject to this distribution, so the bucket stays private
    // and is reachable only through CloudFront (Requirements 24.2, 24.3).
    this.distribution = new cloudfront.Distribution(this, "WebDistribution", {
      comment: props.distributionComment ?? "AI Photo Booth UI distribution",
      defaultRootObject: "index.html",
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(this.uiBucket),
        viewerProtocolPolicy:
          cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: cloudfront.AllowedMethods.ALLOW_GET_HEAD,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
      },
      // SPA fallback: map origin 403/404 to the entry document with HTTP 200 so
      // client-side routing resolves deep links (Requirement 24.4).
      errorResponses: [
        {
          httpStatus: 403,
          responseHttpStatus: 200,
          responsePagePath: "/index.html",
        },
        {
          httpStatus: 404,
          responseHttpStatus: 200,
          responsePagePath: "/index.html",
        },
      ],
    });

    this.distributionDomainName = this.distribution.distributionDomainName;
  }
}
