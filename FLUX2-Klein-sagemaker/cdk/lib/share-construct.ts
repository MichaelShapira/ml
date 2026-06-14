import * as path from "path";
import { CustomResource, Duration, Names, RemovalPolicy } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { NodejsFunction, OutputFormat } from "aws-cdk-lib/aws-lambda-nodejs";
import * as logs from "aws-cdk-lib/aws-logs";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { Provider } from "aws-cdk-lib/custom-resources";

/** Key prefix under which the browser uploads images to be shared. */
export const SHARE_PREFIX = "shared/";

/** Access window for a shared image (CloudFront signed-URL expiry). */
export const SHARE_TTL_SECONDS = 15 * 60;

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

    // --- Secrets Manager: the RSA private key (generated at deploy) --------
    // Created empty; the KeyGen custom resource below writes the generated
    // private key into it. Never committed, never on a developer machine.
    this.privateKeySecret = new secretsmanager.Secret(this, "ShareSignerPrivateKey", {
      description:
        "RSA private key (PEM) used by the Share_Signer to sign CloudFront URLs. Generated at deploy by the KeyGen custom resource.",
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // --- KeyGen custom resource: generate the RSA pair AT DEPLOY -----------
    // On create it generates a fresh 2048-bit RSA pair, stores the PRIVATE key
    // in Secrets Manager, and returns ONLY the PUBLIC key (PEM) as an attribute
    // (public keys are not secret). On update it derives the public key from the
    // stored private key so the value is STABLE across deploys (no CloudFront
    // key replacement, no broken signatures). This removes the committed public
    // key and the manual post-deploy seeding entirely, and gives each account
    // its own isolated key pair.
    const keyGenFn = new lambda.Function(this, "ShareKeyGenFn", {
      runtime: lambda.Runtime.NODEJS_20_X,
      handler: "index.handler",
      timeout: Duration.minutes(1),
      code: lambda.Code.fromInline(KEYGEN_FN_SOURCE),
    });
    this.privateKeySecret.grantRead(keyGenFn);
    this.privateKeySecret.grantWrite(keyGenFn);

    const keyGenProvider = new Provider(this, "ShareKeyGenProvider", {
      onEventHandler: keyGenFn,
    });
    const keyGen = new CustomResource(this, "ShareKeyGen", {
      serviceToken: keyGenProvider.serviceToken,
      resourceType: "Custom::ShareSignerKeyGen",
      properties: { SecretArn: this.privateKeySecret.secretArn },
    });
    keyGen.node.addDependency(this.privateKeySecret);

    // --- CloudFront signing key group (public key from KeyGen) -------------
    // L1 resources: the encoded key is a custom-resource attribute (a token),
    // which the L2 PublicKey would reject at synth via its PEM regex.
    const cfnPublicKey = new cloudfront.CfnPublicKey(this, "ShareSignerPublicKey", {
      publicKeyConfig: {
        name: `${Names.uniqueId(this)}-share-signer-key`.slice(-128),
        callerReference: `${Names.uniqueId(this)}-share-signer-key`,
        encodedKey: keyGen.getAttString("PublicKeyPem"),
        comment: "AI Photo Booth share-download URL signing key (deploy-generated)",
      },
    });
    const cfnKeyGroup = new cloudfront.CfnKeyGroup(this, "ShareKeyGroup", {
      keyGroupConfig: {
        name: `${Names.uniqueId(this)}-share-key-group`.slice(-128),
        items: [cfnPublicKey.attrId],
        comment: "AI Photo Booth share-download trusted key group",
      },
    });
    const keyGroup = cloudfront.KeyGroup.fromKeyGroupId(
      this,
      "ShareKeyGroupRef",
      cfnKeyGroup.attrId,
    );

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
        SHARE_KEY_PAIR_ID: cfnPublicKey.attrId,
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

/**
 * Inline handler source for the KeyGen custom resource (Node.js 20). On Create
 * it generates a fresh RSA-2048 pair, stores the PRIVATE key in Secrets Manager,
 * and returns the PUBLIC key (SPKI PEM) as a resource attribute. On Update it
 * derives the public key from the stored private key so the value is stable
 * (no CloudFront key replacement). Delete is a no-op. Uses only Node built-ins
 * (`crypto`) + the Secrets Manager SDK bundled in the Lambda runtime.
 */
const KEYGEN_FN_SOURCE = `
const { generateKeyPairSync, createPublicKey } = require("crypto");
const { SecretsManagerClient, GetSecretValueCommand, PutSecretValueCommand } = require("@aws-sdk/client-secrets-manager");
const sm = new SecretsManagerClient({});

async function readKey(secretId) {
  try {
    const out = await sm.send(new GetSecretValueCommand({ SecretId: secretId }));
    if (out.SecretString && out.SecretString.includes("PRIVATE KEY")) return out.SecretString;
  } catch (e) { /* not set yet */ }
  return null;
}

exports.handler = async (event) => {
  const physicalId = "share-signer-keygen";
  if (event.RequestType === "Delete") return { PhysicalResourceId: physicalId };

  const secretId = event.ResourceProperties.SecretArn;
  let privateKeyPem = null;

  // On update, reuse the existing key so the public key (and CloudFront config)
  // stays stable. On create, always generate a fresh pair (rotates away from any
  // previously-seeded shared key, avoiding a duplicate-encoded-key collision).
  if (event.RequestType === "Update") {
    privateKeyPem = await readKey(secretId);
  }
  if (!privateKeyPem) {
    const { privateKey } = generateKeyPairSync("rsa", {
      modulusLength: 2048,
      publicKeyEncoding: { type: "spki", format: "pem" },
      privateKeyEncoding: { type: "pkcs8", format: "pem" },
    });
    privateKeyPem = privateKey;
    await sm.send(new PutSecretValueCommand({ SecretId: secretId, SecretString: privateKeyPem }));
  }

  const publicKeyPem = createPublicKey(privateKeyPem).export({ type: "spki", format: "pem" }).toString();
  return { PhysicalResourceId: physicalId, Data: { PublicKeyPem: publicKeyPem } };
};
`;
