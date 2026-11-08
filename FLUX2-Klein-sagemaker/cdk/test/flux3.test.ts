/**
 * FLUX 3 Image Klein wiring: the SageMaker execution role deploy.sh uses for
 * the g7e model + weight-staging job, and the visitor's read of the
 * current-config pointer (which picks the FLUX.2 or FLUX 3 prompt dialect).
 */
import { describe, it, expect, beforeAll } from "vitest";
import * as cdk from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { PhotoBoothStack } from "../lib/photo-booth-stack";

type Statement = {
  Sid?: string;
  Effect?: string;
  Action?: string | string[];
  Resource?: unknown;
  Condition?: unknown;
};

/** All statements of the IAM::Policy resources attached to a role logical id. */
function statementsForRole(template: Template, roleLogicalId: string): Statement[] {
  const out: Statement[] = [];
  for (const policy of Object.values(template.findResources("AWS::IAM::Policy"))) {
    const props = policy.Properties as {
      Roles?: { Ref?: string }[];
      PolicyDocument?: { Statement?: Statement[] };
    };
    if ((props.Roles ?? []).some((r) => r.Ref === roleLogicalId)) {
      out.push(...(props.PolicyDocument?.Statement ?? []));
    }
  }
  return out;
}

function logicalIdOf(template: Template, type: string, pattern: RegExp): string {
  const id = Object.keys(template.findResources(type)).find((k) => pattern.test(k));
  if (!id) throw new Error(`no ${type} matching ${pattern}`);
  return id;
}

describe("FLUX 3 Image Klein wiring", () => {
  let template: Template;

  beforeAll(() => {
    const app = new cdk.App({ context: { existingIoBucketName: "test-io-bucket" } });
    const stack = new PhotoBoothStack(app, "Flux3Stack", {
      env: { account: "123456789012", region: "us-east-1" },
    });
    template = Template.fromStack(stack);
  });

  it("creates a SageMaker-trusted execution role with no trust conditions", () => {
    const roleId = logicalIdOf(template, "AWS::IAM::Role", /^Flux3SageMakerRole/);
    const role = template.findResources("AWS::IAM::Role")[roleId];
    const trust = (role.Properties as { AssumeRolePolicyDocument: { Statement: Statement[] } })
      .AssumeRolePolicyDocument.Statement;
    expect(trust).toHaveLength(1);
    expect(trust[0]).toMatchObject({
      Effect: "Allow",
      Action: "sts:AssumeRole",
      Principal: { Service: "sagemaker.amazonaws.com" },
    });
    expect(trust[0].Condition).toBeUndefined();
  });

  it("scopes the role by resource ARN only (SageMaker rejects roles with conditions)", () => {
    const roleId = logicalIdOf(template, "AWS::IAM::Role", /^Flux3SageMakerRole/);
    const statements = statementsForRole(template, roleId);
    const sids = statements.map((s) => s.Sid);
    expect(sids).toEqual(
      expect.arrayContaining([
        "ReadModelAndRequests",
        "WriteWeightsAndResults",
        "ListIoBucket",
        "EcrAuth",
        "PullRayServeDlc",
        "WriteSageMakerLogs",
        "PutMetrics",
        "ReadHfTokenSecret",
      ]),
    );
    for (const s of statements) {
      expect(s.Condition, `statement ${s.Sid} must have no Condition`).toBeUndefined();
    }
    const actions = statements.flatMap((s) => (Array.isArray(s.Action) ? s.Action : [s.Action]));
    expect(actions).not.toContain("s3:DeleteObject");
    expect(actions).not.toContain("iam:PassRole");
    expect(actions.some((a) => typeof a === "string" && a.endsWith(":*"))).toBe(false);
  });

  it("limits the role's S3 writes to the weights, outputs and failures prefixes", () => {
    const roleId = logicalIdOf(template, "AWS::IAM::Role", /^Flux3SageMakerRole/);
    const write = statementsForRole(template, roleId).find((s) => s.Sid === "WriteWeightsAndResults");
    const resources = JSON.stringify(write?.Resource);
    expect(resources).toContain("flux3-klein-g7e/model/weights/*");
    expect(resources).toContain("flux2-klein-outputs/*");
    expect(resources).toContain("flux2-klein-failures/*");
    expect(resources).not.toContain("flux2-klein-inputs");
  });

  it("lets the role read only the Hugging Face token secret", () => {
    const roleId = logicalIdOf(template, "AWS::IAM::Role", /^Flux3SageMakerRole/);
    const secret = statementsForRole(template, roleId).find((s) => s.Sid === "ReadHfTokenSecret");
    expect(secret?.Action).toBe("secretsmanager:GetSecretValue");
    expect(JSON.stringify(secret?.Resource)).toContain(
      "secret:AmazonSageMaker-flux3-klein-hf-token-??????",
    );
  });

  it("lets every signed-in role GetItem the current-config pointer, and nothing else in it", () => {
    template.hasResourceProperties("AWS::IAM::Policy", {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Sid: "ReadCurrentConfig",
            Effect: "Allow",
            Action: "dynamodb:GetItem",
            Condition: {
              "ForAllValues:StringEquals": { "dynamodb:LeadingKeys": ["CONFIG#CURRENT"] },
            },
          }),
        ]),
      },
    });
  });

  it("exposes the outputs deploy.sh reads to provision the FLUX 3 endpoint config", () => {
    const outputs = template.toJSON().Outputs ?? {};
    for (const key of ["Flux3SageMakerRoleArn", "Flux3ModelS3Uri", "Flux3HfTokenSecretName"]) {
      expect(outputs).toHaveProperty(key);
    }
    expect(JSON.stringify(outputs.Flux3ModelS3Uri.Value)).toContain("flux3-klein-g7e/model/");
    expect(outputs.Flux3HfTokenSecretName.Value).toBe("AmazonSageMaker-flux3-klein-hf-token");
  });
});
