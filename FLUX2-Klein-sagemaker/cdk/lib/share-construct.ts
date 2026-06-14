import * as path from "path";
import { Duration, RemovalPolicy } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { NodejsFunction, OutputFormat } from "aws-cdk-lib/aws-lambda-nodejs";
import * as logs from "aws-cdk-lib/aws-logs";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";

/** Key prefix under which the browser uploads images to be shared. */
export const SHARE_PREFIX = "shared/";

/** Access window for a shared image (CloudFront signed-URL expiry). */
export const SHARE_TTL_SECONDS = 15 * 60;

/**
 * RSA public key (SPKI PEM) used by CloudFront to VERIFY the signed-URL
 * signatures. Public keys are not secret, so this is safe to embed. The MATCHING
 * private key is NOT in the repo — it is stored in Secrets Manager (seeded
 * post-deploy) and read only by the signer Lambda.
 *
 * To rotate: generate a new pair (`openssl genrsa` + `openssl rsa -pubout`),
 * replace this PEM, and put the new private key into the Secrets Manager secret.
 */
const SHARE_SIGNER_PUBLIC_KEY_PEM = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA2KmuJ+nL0m22DGSRVDdK
lMlR9vkAbGS6zsB/LKsdqMRY3l7YBQi8/JitDGhkHuNTWSPP3fKwcRuEuq33YcwS
oV8SzRYdcI+N/Y287W5xAqqQIqTzMrwNdMQoYx7DkNMeV5WQehwAQaBjesdCN8ux
fQ3UdHSrRaXc8FmrR2UKKubYZSG5VHuAJp1yDMjD/pNhuvnYrhVMEEfdyZ7Pgcwy
Wx5XNo2bzxFmngnfcOTIgvDRSiWF2/plrFMEGaYJzPx9RswmYMsAUWCCaqedlPPt
BzFUtjZ4iFscl1aYJrxvx1CRA3B4uMe3S5H6Vs26S+7+GF2qIM7IoxUxkUwZSFDz
mwIDAQAB
-----END PUBLIC KEY-----`;

/** Absolute path to the Share_Signer Lambda source. */
const SHARE_SIGNER_ENTRY = path.join(
  __dirname,
  "..",
  "..",
  "backend",
  "src",
  "share",
  "signer.ts",
);

/** Lock file for the `backend/` package (deterministic NodejsFunction bundling). */
const BACKEND_DEPS_LOCK = path.join(
  __dirname,
  "..",
  "..",
  "backend",
  "package-lock.json",
);

/** Props for {@link ShareConstruct}. */
export interface ShareConstructProps {
  /** Browser origins allowed to PUT (upload) on the share bucket. */
  readonly allowedOrigins: string[];
}

/**
 * Share_Bucket + CloudFront + Share_Signer — the "Share with me" QR download.
 *
 * Design (Option A — secure AND scannable):
 *   1. The SPA uploads the chosen image to `shared/{id}.png` (PutObject only,
 *      visitor's Cognito credentials). The bucket stays FULLY PRIVATE.
 *   2. The SPA calls the Share_Signer Lambda via an **AWS_IAM** Function URL
 *      (same auth posture as the Invoke proxy — NOT public). The signer mints a
 *      short, 15-minute **CloudFront signed URL** and returns it.
 *   3. The QR encodes that CloudFront signed URL (~300 chars → low-density,
 *      reliably scannable). A phone scans it; CloudFront validates the signature
 *      + expiry against the trusted key group and serves the object from the
 *      private bucket via Origin Access Control (OAC).
 *
 * Security:
 *   - No public (`NONE`) endpoint anywhere; the signer is `AWS_IAM`.
 *   - Bucket blocks ALL public access; only CloudFront (OAC) can read it, and
 *     only for a request carrying a valid signature.
 *   - The signing private key lives in Secrets Manager, read only by the signer
 *     role. The public key is embedded (public keys are not secret).
 *   - Least privilege: visitors get only `s3:PutObject` on `shared/*` plus
 *     invoke on the signer; the signer gets only `secretsmanager:GetSecretValue`
 *     on its key secret.
 *   - 15-minute expiry enforced by the CloudFront signature; bucket lifecycle
 *     (1 day, the S3 minimum) is the cleanup backstop.
 */
export class ShareConstruct extends Construct {
  /** The private bucket holding short-lived shared images. */
  public readonly bucket: s3.Bucket;

  /** CloudFront distribution serving the share bucket via OAC (signed URLs). */
  public readonly distribution: cloudfront.Distribution;

  /** The IAM-authenticated signer Lambda. */
  public readonly signerFunction: NodejsFunction;

  /** The AWS_IAM Function URL the SPA calls to mint a signed URL. */
  public readonly signerUrl: lambda.FunctionUrl;

  /** Secrets Manager secret holding the RSA private key (seeded post-deploy). */
  public readonly privateKeySecret: secretsmanager.Secret;

  constructor(scope: Construct, id: string, props: ShareConstructProps) {
    super(scope, id);

    // --- Private share bucket (upload target) ------------------------------
    this.bucket = new s3.Bucket(this, "ShareBucket", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      autoDeleteObjects: true,
      removalPolicy: RemovalPolicy.DESTROY,
      cors: [
        {
          allowedHeaders: ["*"],
          allowedMethods: [s3.HttpMethods.PUT],
          allowedOrigins: props.allowedOrigins,
          exposedHeaders: ["ETag"],
          maxAge: 3000,
        },
      ],
      lifecycleRules: [
        {
          id: "expire-shared-objects",
          enabled: true,
          prefix: SHARE_PREFIX,
          expiration: Duration.days(1),
          abortIncompleteMultipartUploadAfter: Duration.days(1),
        },
      ],
    });

    // --- CloudFront signing key group (public key) -------------------------
    const publicKey = new cloudfront.PublicKey(this, "ShareSignerPublicKey", {
      encodedKey: SHARE_SIGNER_PUBLIC_KEY_PEM,
      comment: "AI Photo Booth share-download URL signing key",
    });
    const keyGroup = new cloudfront.KeyGroup(this, "ShareKeyGroup", {
      items: [publicKey],
      comment: "AI Photo Booth share-download trusted key group",
    });

    // --- CloudFront distribution: OAC origin + trusted key group -----------
    // The bucket stays private; OAC lets ONLY this distribution read it, and the
    // trusted key group means CloudFront only serves a request with a valid
    // signature (i.e. a signed URL minted by the signer Lambda).
    this.distribution = new cloudfront.Distribution(this, "ShareDistribution", {
      comment: "AI Photo Booth share downloads (signed URLs, OAC, private bucket)",
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(this.bucket),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: cloudfront.AllowedMethods.ALLOW_GET_HEAD,
        // Each signed URL is unique + short-lived; don't cache responses.
        cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
        trustedKeyGroups: [keyGroup],
      },
    });

    // --- Secrets Manager: the RSA private key (seeded post-deploy) ---------
    this.privateKeySecret = new secretsmanager.Secret(this, "ShareSignerPrivateKey", {
      description:
        "RSA private key (PEM) used by the Share_Signer to sign CloudFront URLs. Seed via put-secret-value.",
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // --- Signer Lambda role: read ONLY the private-key secret + Logs -------
    const signerRole = new iam.Role(this, "ShareSignerRole", {
      description:
        "Share_Signer role: secretsmanager:GetSecretValue on the signing key + CloudWatch Logs.",
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AWSLambdaBasicExecutionRole",
        ),
      ],
    });
    this.privateKeySecret.grantRead(signerRole);

    const logGroup = new logs.LogGroup(this, "ShareSignerLogs", {
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    this.signerFunction = new NodejsFunction(this, "ShareSignerFunction", {
      description:
        "Share_Signer: mints a 15-minute CloudFront signed URL for a shared image (IAM-authed).",
      runtime: lambda.Runtime.NODEJS_20_X,
      entry: SHARE_SIGNER_ENTRY,
      handler: "handler",
      depsLockFilePath: BACKEND_DEPS_LOCK,
      role: signerRole,
      logGroup,
      timeout: Duration.seconds(15),
      environment: {
        SHARE_CLOUDFRONT_DOMAIN: this.distribution.distributionDomainName,
        SHARE_KEY_PAIR_ID: publicKey.publicKeyId,
        SHARE_PRIVATE_KEY_SECRET_ID: this.privateKeySecret.secretArn,
        SHARE_TTL_SECONDS: String(SHARE_TTL_SECONDS),
      },
      bundling: {
        target: "node20",
        format: OutputFormat.CJS,
        // cloudfront-signer is bundled; client-secrets-manager is in the runtime.
        externalModules: ["@aws-sdk/client-secrets-manager"],
      },
    });

    // --- Function URL: AWS_IAM (NOT public) + CORS for the SPA origin ------
    this.signerUrl = this.signerFunction.addFunctionUrl({
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
   * Grant a role least-privilege upload access: only `s3:PutObject` on
   * `shared/*`. Reads happen via CloudFront signed URLs, not the visitor's role.
   */
  public grantUpload(grantee: iam.IGrantable): void {
    grantee.grantPrincipal.addToPrincipalPolicy(
      new iam.PolicyStatement({
        sid: "ShareImageUpload",
        effect: iam.Effect.ALLOW,
        actions: ["s3:PutObject"],
        resources: [this.bucket.arnForObjects(`${SHARE_PREFIX}*`)],
      }),
    );
  }

  /**
   * Grant a role permission to invoke the signer Function URL (SigV4-signed).
   * An AWS_IAM Function URL requires BOTH `lambda:InvokeFunctionUrl` AND
   * `lambda:InvokeFunction` (verified with the Invoke proxy), scoped to this
   * function.
   */
  public grantInvokeSigner(grantee: iam.IRole): void {
    grantee.addToPrincipalPolicy(
      new iam.PolicyStatement({
        sid: "InvokeShareSignerFunctionUrl",
        effect: iam.Effect.ALLOW,
        actions: ["lambda:InvokeFunctionUrl", "lambda:InvokeFunction"],
        resources: [this.signerFunction.functionArn],
      }),
    );
  }
}

export default ShareConstruct;
