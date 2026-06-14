import * as path from "path";
import * as fs from "fs";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as s3deploy from "aws-cdk-lib/aws-s3-deployment";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";

/**
 * Absolute path to the built SPA assets (`ui/dist`), resolved relative to this
 * construct file so it is independent of the current working directory. The
 * `ui/` package must be built (`npm run build`) before `cdk deploy`, which
 * emits `ui/dist`.
 */
const UI_DIST_DIR = path.join(__dirname, "..", "..", "ui", "dist");

/**
 * The runtime configuration the SPA reads from `window.__BOOTH_CONFIG__`.
 *
 * These values are only known after the auth/data/hosting constructs are
 * created, so they are injected at deploy time as a generated `config.js`
 * asset rather than baked into the build (the build is environment-agnostic).
 * Token-bearing values (ids, names) are resolved by CloudFormation when the
 * `BucketDeployment` writes the object.
 */
export interface UiRuntimeConfig {
  region: string;
  userPoolId: string;
  userPoolClientId: string;
  identityPoolId: string;
  endpointName: string;
  endpointConfigName: string;
  ioBucket: string;
  scheduleTable: string;
  /** Short-lived Share_Bucket for the "Share with me" QR download flow. */
  shareBucket: string;
  /** IAM-authenticated Share_Signer Function URL (mints CloudFront signed URLs). */
  shareSignerUrl: string;
  timezone: string;
  /** IAM-authenticated Lambda Function URL the SPA signs to invoke the endpoint. */
  invokeFunctionUrl: string;
}

/** Props for {@link UiDeploymentConstruct}. */
export interface UiDeploymentConstructProps {
  /** The private UI bucket to deploy the built assets into. */
  readonly uiBucket: s3.IBucket;
  /** The CloudFront distribution to invalidate after deployment. */
  readonly distribution: cloudfront.IDistribution;
  /** The runtime config injected as `config.js` (`window.__BOOTH_CONFIG__`). */
  readonly runtimeConfig: UiRuntimeConfig;
}

/**
 * Deploys the built SPA into the `UI_Bucket` behind CloudFront and injects the
 * runtime config (Requirements 22.2, 24.2, 24.4).
 *
 * Two sources are deployed together so they land in the same bucket without
 * pruning each other:
 *   1. {@link s3deploy.Source.asset} of `ui/dist` — the built Vite bundle and
 *      `index.html` (which references `/config.js`).
 *   2. {@link s3deploy.Source.jsonData}-style inline `config.js` that sets
 *      `window.__BOOTH_CONFIG__` from the stack's resolved values.
 *
 * The deployment invalidates the distribution's cache so a redeploy serves the
 * new assets and config immediately.
 */
export class UiDeploymentConstruct extends Construct {
  constructor(scope: Construct, id: string, props: UiDeploymentConstructProps) {
    super(scope, id);

    // Inline config.js sets window.__BOOTH_CONFIG__ before the app bundle runs.
    // JSON.stringify over a record of (possibly token) strings yields a valid
    // object literal; CloudFormation substitutes the token values on deploy.
    const configJs = `window.__BOOTH_CONFIG__ = ${this.stringifyConfig(
      props.runtimeConfig,
    )};`;

    // The built SPA assets are only present after `npm run build` in ui/.
    // Deploy them when available; always deploy the generated config.js so a
    // redeploy refreshes runtime config even if the bundle is unchanged. This
    // also lets `cdk synth` / assertion tests run without a UI build.
    const sources = [s3deploy.Source.data("config.js", configJs)];
    const hasBuiltUi =
      fs.existsSync(UI_DIST_DIR) &&
      fs.existsSync(path.join(UI_DIST_DIR, "index.html"));
    if (hasBuiltUi) {
      sources.unshift(s3deploy.Source.asset(UI_DIST_DIR));
    }

    new s3deploy.BucketDeployment(this, "DeployUi", {
      destinationBucket: props.uiBucket,
      distribution: props.distribution,
      // Invalidate everything so index.html + config.js + hashed assets refresh.
      distributionPaths: ["/*"],
      sources,
      // Prune only when deploying the full built site; otherwise a config-only
      // deploy would delete the previously-uploaded bundle.
      prune: hasBuiltUi,
    });
  }

  /**
   * Build the object literal for `config.js`. Uses `JSON.stringify` so any
   * resolved string is correctly quoted; CDK tokens stringify to placeholders
   * that CloudFormation resolves when the deployment custom resource writes the
   * file.
   */
  private stringifyConfig(config: UiRuntimeConfig): string {
    return JSON.stringify(
      {
        region: config.region,
        userPoolId: config.userPoolId,
        userPoolClientId: config.userPoolClientId,
        identityPoolId: config.identityPoolId,
        endpointName: config.endpointName,
        endpointConfigName: config.endpointConfigName,
        ioBucket: config.ioBucket,
        scheduleTable: config.scheduleTable,
        shareBucket: config.shareBucket,
        shareSignerUrl: config.shareSignerUrl,
        timezone: config.timezone,
        invokeFunctionUrl: config.invokeFunctionUrl,
      },
      null,
      2,
    );
  }
}

export default UiDeploymentConstruct;
