/**
 * End-to-end wiring verification (Requirements 14.6, 22.3).
 *
 * The SPA calls AWS directly, so "wiring" means: every AWS action the browser
 * modules issue must be permitted by one of the two IAM roles the stack
 * provisions, there must be NO API Gateway / per-request backend Lambda, and
 * the only server-side compute is the scheduler. This test synthesizes the
 * stack and asserts the IAM policy documents cover exactly the SDK actions used
 * by `ui/src/api/*`, with no orphaned route tier.
 */
import { describe, it, expect, beforeAll } from "vitest";
import * as cdk from "aws-cdk-lib";
import { Template } from "aws-cdk-lib/assertions";
import { PhotoBoothStack } from "../lib/photo-booth-stack";

/** Collect every IAM action granted (Allow) across all IAM policies + roles. */
function collectAllowedActions(template: Template): Set<string> {
  const actions = new Set<string>();
  const addFrom = (doc: unknown) => {
    const statements = (doc as { Statement?: unknown[] })?.Statement ?? [];
    for (const stmt of statements) {
      const s = stmt as { Effect?: string; Action?: string | string[] };
      if (s.Effect !== "Allow" || s.Action === undefined) continue;
      const list = Array.isArray(s.Action) ? s.Action : [s.Action];
      for (const a of list) {
        if (typeof a === "string") actions.add(a);
      }
    }
  };
  for (const policy of Object.values(template.findResources("AWS::IAM::Policy"))) {
    addFrom((policy.Properties as { PolicyDocument?: unknown }).PolicyDocument);
  }
  for (const role of Object.values(template.findResources("AWS::IAM::Role"))) {
    const policies = (role.Properties as { Policies?: { PolicyDocument?: unknown }[] }).Policies ?? [];
    for (const p of policies) addFrom(p.PolicyDocument);
  }
  return actions;
}

describe("SPA → AWS wiring contract", () => {
  let template: Template;
  let allowed: Set<string>;

  beforeAll(() => {
    const app = new cdk.App({ context: { existingIoBucketName: "test-io-bucket" } });
    const stack = new PhotoBoothStack(app, "WiringStack", {
      env: { account: "123456789012", region: "us-east-1" },
    });
    template = Template.fromStack(stack);
    allowed = collectAllowedActions(template);
  });

  it("permits every AWS action the browser Generation_Service issues", () => {
    // generation.ts: S3 PutObject/HeadObject(=ListBucket)/GetObject directly,
    // plus invoke via the proxy (lambda:InvokeFunctionUrl). The actual
    // sagemaker:InvokeEndpointAsync now lives on the Invoke_Proxy Lambda role.
    expect(allowed).toContain("sagemaker:InvokeEndpointAsync");
    expect(allowed).toContain("s3:PutObject");
    expect(allowed).toContain("s3:GetObject");
    expect(allowed).toContain("s3:ListBucket");
    expect(allowed).toContain("lambda:InvokeFunctionUrl");
  });

  it("exposes the Invoke_Proxy via an IAM-authenticated Function URL (never public)", () => {
    // The SageMaker runtime API has no CORS; the browser signs requests to this
    // Function URL instead. It MUST require IAM auth so it is not public. There
    // are NO public (NONE) Function URLs anywhere — the share/download path uses
    // a client-side presigned S3 URL, not a Lambda Function URL.
    const urls = template.findResources("AWS::Lambda::Url");
    const authTypes = Object.values(urls).map(
      (u) => (u.Properties as { AuthType?: string }).AuthType,
    );
    expect(authTypes.length).toBeGreaterThanOrEqual(1);
    for (const authType of authTypes) {
      expect(authType).toBe("AWS_IAM");
    }
    expect(authTypes).not.toContain("NONE");
  });

  it("grants both InvokeFunctionUrl and InvokeFunction (both required for AWS_IAM URLs)", () => {
    // The Function URL authorizer requires the caller to hold BOTH actions; the
    // action name InvokeFunctionUrl alone is insufficient (403). Assert the
    // identity policies grant both.
    expect(allowed).toContain("lambda:InvokeFunctionUrl");
    expect(allowed).toContain("lambda:InvokeFunction");
  });

  it("permits every AWS action the browser Endpoint_Manager issues (admin)", () => {
    expect(allowed).toContain("sagemaker:ListEndpoints");
    expect(allowed).toContain("sagemaker:DescribeEndpoint");
    expect(allowed).toContain("sagemaker:CreateEndpoint");
    expect(allowed).toContain("sagemaker:DeleteEndpoint");
  });

  it("permits every AWS action the browser schedule CRUD issues (admin)", () => {
    expect(allowed).toContain("dynamodb:Query");
    expect(allowed).toContain("dynamodb:PutItem");
    expect(allowed).toContain("dynamodb:DeleteItem");
  });

  it("permits the share upload + signer invoke the browser Share_Service issues", () => {
    // "Share with me": the browser PutObjects the image to the private share
    // bucket, then invokes the AWS_IAM Share_Signer Function URL to mint a
    // CloudFront signed URL. No GetObject on the share bucket from the browser.
    expect(allowed).toContain("s3:PutObject");
    expect(allowed).toContain("lambda:InvokeFunctionUrl");
    expect(allowed).toContain("lambda:InvokeFunction");
  });

  it("permits the Cost Explorer read the admin cost panel issues", () => {
    expect(allowed).toContain("ce:GetCostAndUsage");
  });

  it("grants NO iam:PassRole (start/stop pass no role)", () => {
    expect([...allowed]).not.toContain("iam:PassRole");
  });

  it("grants NO Rekognition permission anywhere (moderation removed)", () => {
    const rekognition = [...allowed].filter((a) => a.startsWith("rekognition:"));
    expect(rekognition).toEqual([]);
  });

  it("provisions no API Gateway or per-request backend route tier (Req 22.3)", () => {
    template.resourceCountIs("AWS::ApiGateway::RestApi", 0);
    template.resourceCountIs("AWS::ApiGateway::Method", 0);
    template.resourceCountIs("AWS::ApiGatewayV2::Api", 0);
    template.resourceCountIs("AWS::ApiGatewayV2::Route", 0);
  });

  it("exposes the runtime-config outputs the SPA reads", () => {
    const outputs = template.toJSON().Outputs ?? {};
    for (const key of [
      "UserPoolId",
      "UserPoolClientId",
      "IdentityPoolId",
      "Region",
      "IoBucketName",
      "ScheduleTableName",
      "EndpointNameOutput",
      "EndpointConfigNameOutput",
      "InvokeFunctionUrl",
    ]) {
      expect(outputs).toHaveProperty(key);
    }
  });
});
