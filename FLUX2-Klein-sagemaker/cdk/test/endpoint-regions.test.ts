/**
 * Extra endpoint regions (`endpointRegions` context, deploy.sh
 * --endpoint-regions): every piece of the booth that touches the endpoint or
 * its I/O bucket gets that region, and nothing changes without the context.
 */
import { describe, it, expect, beforeAll } from "vitest";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import * as cdk from "aws-cdk-lib";
import { Template } from "aws-cdk-lib/assertions";
import { PhotoBoothStack } from "../lib/photo-booth-stack";
import { parseExtraRegions } from "../lib/endpoint-regions";

type Statement = { Sid?: string; Action?: string | string[]; Resource?: unknown; Condition?: unknown };

const ACCOUNT = "123456789012";
const EXTRA_BUCKET = `sagemaker-us-east-2-${ACCOUNT}`;

/** The staged UI runtime config (config.js) of the last {@link synth}. */
let lastConfigJs = "";

function synth(context: Record<string, string>): Template {
  const outdir = fs.mkdtempSync(path.join(os.tmpdir(), "regions-"));
  const app = new cdk.App({ outdir, context: { existingIoBucketName: "test-io-bucket", ...context } });
  const stack = new PhotoBoothStack(app, "RegionsStack", { env: { account: ACCOUNT, region: "us-east-1" } });
  const template = Template.fromStack(stack);
  app.synth();
  lastConfigJs = fs
    .readdirSync(outdir)
    .map((d) => path.join(outdir, d, "config.js"))
    .filter((f) => fs.existsSync(f))
    .map((f) => fs.readFileSync(f, "utf8"))
    .join("\n");
  return template;
}

/** Replace `{"Fn::Join": ["", [... {"Ref": "AWS::Partition"} ...]]}` with the plain "aws" string. */
function flatten(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(flatten);
  }
  if (value !== null && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const join = obj["Fn::Join"] as [string, unknown[]] | undefined;
    if (join && join[0] === "" && Array.isArray(join[1])) {
      const parts = join[1].map((p) =>
        p !== null && typeof p === "object" && (p as { Ref?: string }).Ref === "AWS::Partition" ? "aws" : p,
      );
      if (parts.every((p) => typeof p === "string")) {
        return parts.join("");
      }
    }
    return Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, flatten(v)]));
  }
  return value;
}

function logicalIdOf(template: Template, type: string, pattern: RegExp): string {
  const id = Object.keys(template.findResources(type)).find((k) => pattern.test(k));
  if (!id) throw new Error(`no ${type} matching ${pattern}`);
  return id;
}

function statementsForRole(template: Template, roleLogicalId: string): Statement[] {
  const out: Statement[] = [];
  for (const policy of Object.values(template.findResources("AWS::IAM::Policy"))) {
    const props = policy.Properties as { Roles?: { Ref?: string }[]; PolicyDocument?: { Statement?: Statement[] } };
    if ((props.Roles ?? []).some((r) => r.Ref === roleLogicalId)) {
      out.push(...((flatten(props.PolicyDocument?.Statement ?? []) as Statement[])));
    }
  }
  return out;
}

/** Every resource string a role's statements grant, for substring checks. */
function grantedResources(template: Template, rolePattern: RegExp): string {
  return JSON.stringify(statementsForRole(template, logicalIdOf(template, "AWS::IAM::Role", rolePattern)));
}

function lambdaEnv(template: Template, pattern: RegExp): Record<string, unknown> {
  const fn = template.findResources("AWS::Lambda::Function")[logicalIdOf(template, "AWS::Lambda::Function", pattern)];
  return (fn.Properties as { Environment: { Variables: Record<string, unknown> } }).Environment.Variables;
}

describe("parseExtraRegions", () => {
  it("drops blanks, duplicates and the home region, and rejects non-regions", () => {
    expect(parseExtraRegions(undefined, "us-east-1")).toEqual([]);
    expect(parseExtraRegions(" us-east-2, ,us-west-2,us-east-2,us-east-1", "us-east-1")).toEqual([
      "us-east-2",
      "us-west-2",
    ]);
    expect(() => parseExtraRegions("us-east-2,useast2", "us-east-1")).toThrow(/useast2/);
    expect(() => parseExtraRegions(["us-east-2"], "us-east-1")).toThrow(/comma-separated/);
  });
});

describe("with endpointRegions=us-east-2", () => {
  let template: Template;
  beforeAll(() => {
    template = synth({ endpointRegions: "us-east-2" });
  });

  it("lets admins manage endpoints in us-east-2, and visitors not", () => {
    const admin = statementsForRole(template, logicalIdOf(template, "AWS::IAM::Role", /^AuthAdminRole/));
    const manage = admin.find((s) => s.Sid === "ManageEndpointsInExtraRegions");
    expect(manage?.Resource).toEqual([
      `arn:aws:sagemaker:us-east-2:${ACCOUNT}:endpoint/*`,
      `arn:aws:sagemaker:us-east-2:${ACCOUNT}:endpoint-config/*`,
    ]);
    expect(grantedResources(template, /^AuthAuthenticatedRole/)).not.toContain("sagemaker:us-east-2");
  });

  it("gives both booth roles the capture-flow S3 access on the us-east-2 bucket", () => {
    for (const role of [/^AuthAdminRole/, /^AuthAuthenticatedRole/]) {
      const statements = statementsForRole(template, logicalIdOf(template, "AWS::IAM::Role", role));
      const put = statements.find((s) => s.Sid === "PutInputObjectsExtraRegions");
      const get = statements.find((s) => s.Sid === "GetOutputObjectsExtraRegions");
      const list = statements.find((s) => s.Sid === "ListOutputPrefixesExtraRegions");
      // CDK writes a one-element resource list as a plain string.
      expect(put?.Resource).toEqual(`arn:aws:s3:::${EXTRA_BUCKET}/flux2-klein-inputs/*`);
      expect(get?.Resource).toEqual([
        `arn:aws:s3:::${EXTRA_BUCKET}/flux2-klein-outputs/*`,
        `arn:aws:s3:::${EXTRA_BUCKET}/flux2-klein-failures/*`,
      ]);
      expect(list?.Condition).toEqual({
        StringLike: { "s3:prefix": ["flux2-klein-outputs/*", "flux2-klein-failures/*"] },
      });
    }
  });

  it("lets the invoke proxy and the scheduler reach us-east-2 and tells them its bucket", () => {
    const invoke = grantedResources(template, /^InvokeInvokeRole/);
    expect(invoke).toContain(`arn:aws:sagemaker:us-east-2:${ACCOUNT}:endpoint/*`);
    const scheduler = statementsForRole(template, logicalIdOf(template, "AWS::IAM::Role", /^SchedulerSchedulerRole/));
    expect(scheduler.find((s) => s.Sid === "ManageEndpointInExtraRegions")?.Resource).toEqual([
      `arn:aws:sagemaker:us-east-2:${ACCOUNT}:endpoint/*`,
      `arn:aws:sagemaker:us-east-2:${ACCOUNT}:endpoint-config/*`,
    ]);
    for (const fn of [/^InvokeInvokeFunction/, /^SchedulerSchedulerFunction/]) {
      expect(JSON.parse(lambdaEnv(template, fn).ENDPOINT_REGIONS as string)).toEqual({ "us-east-2": EXTRA_BUCKET });
    }
  });

  it("gives the FLUX 3 role the us-east-2 image, logs and bucket, but no weight writes there", () => {
    const flux3 = grantedResources(template, /^Flux3SageMakerRole/);
    expect(flux3).toContain("arn:aws:ecr:us-east-2:763104351884:repository/ray");
    expect(flux3).toContain(`arn:aws:logs:us-east-2:${ACCOUNT}:log-group:/aws/sagemaker/*`);
    expect(flux3).toContain(`arn:aws:s3:::${EXTRA_BUCKET}/flux3-klein-g7e/*`);
    expect(flux3).toContain(`arn:aws:s3:::${EXTRA_BUCKET}/flux2-klein-outputs/*`);
    expect(flux3).not.toContain(`${EXTRA_BUCKET}/flux3-klein-g7e/model/weights/*`);
  });

  it("sets the browser CORS on the us-east-2 bucket with a us-east-2 S3 call", () => {
    const cors = Object.values(template.findResources("Custom::AWS")).map((r) => JSON.stringify(r.Properties));
    const extra = cors.filter((c) => c.includes(EXTRA_BUCKET));
    expect(extra).toHaveLength(1);
    expect(extra[0]).toContain('\\"region\\":\\"us-east-2\\"');
    expect(extra[0]).toContain("putBucketCors");
  });

  it("puts the region and its bucket in the UI runtime config and the outputs", () => {
    const config = JSON.parse(lastConfigJs.replace(/^window\.__BOOTH_CONFIG__ = /, "").replace(/;\s*$/, "")) as {
      endpointRegions: unknown;
    };
    expect(config.endpointRegions).toEqual([{ region: "us-east-2", ioBucket: EXTRA_BUCKET }]);
    template.hasOutput("EndpointRegions", {
      Value: `us-east-1=test-io-bucket,us-east-2=${EXTRA_BUCKET}`,
    });
  });
});

describe("without endpointRegions", () => {
  it("adds no extra-region grants and an empty region map", () => {
    const template = synth({});
    const all = JSON.stringify(template.findResources("AWS::IAM::Policy"));
    expect(all).not.toContain("us-east-2");
    expect(all).not.toContain("ExtraRegions");
    expect(lambdaEnv(template, /^InvokeInvokeFunction/).ENDPOINT_REGIONS).toBe("{}");
    expect(Object.keys(template.findResources("Custom::AWS")).filter((k) => /IoBucketCors/.test(k))).toHaveLength(1);
    template.hasOutput("EndpointRegions", { Value: "us-east-1=test-io-bucket" });
  });
});
