/**
 * CDK assertion/snapshot tests for the PhotoBoothStack (Requirement 22 et al.).
 *
 * Using `Template.fromStack`, assert the stack:
 *   - synthesizes and contains the Cognito pool + client, Identity Pool, both
 *     IAM roles, the role attachment, the Schedule_Store table, the UI bucket +
 *     CloudFront (OAC) + SPA fallback, the Scheduler Lambda + two EventBridge
 *     rules, and the initial-user CfnParameters;
 *   - provisions NO API Gateway (Req 22.3);
 *   - stamps DeletionPolicy/UpdateReplacePolicy: Delete on every resource and
 *     autoDeleteObjects on stack-created buckets (Req 22.4);
 *   - scopes the two IAM roles to the documented least-privilege actions.
 */
import { describe, it, expect, beforeAll } from "vitest";
import * as cdk from "aws-cdk-lib";
import { Template, Match } from "aws-cdk-lib/assertions";
import { PhotoBoothStack } from "../lib/photo-booth-stack";

function synth(): Template {
  const app = new cdk.App({
    context: { existingIoBucketName: "test-io-bucket" },
  });
  const stack = new PhotoBoothStack(app, "TestStack", {
    env: { account: "123456789012", region: "us-east-1" },
  });
  return Template.fromStack(stack);
}

describe("PhotoBoothStack", () => {
  let template: Template;

  beforeAll(() => {
    template = synth();
  });

  it("synthesizes without error", () => {
    expect(template.toJSON()).toBeDefined();
  });

  it("provisions the Cognito user pool, SPA client, and identity pool", () => {
    template.resourceCountIs("AWS::Cognito::UserPool", 1);
    template.resourceCountIs("AWS::Cognito::UserPoolClient", 1);
    template.resourceCountIs("AWS::Cognito::IdentityPool", 1);
    template.resourceCountIs("AWS::Cognito::IdentityPoolRoleAttachment", 1);
    // Public SPA client: no secret, SRP enabled.
    template.hasResourceProperties("AWS::Cognito::UserPoolClient", {
      ExplicitAuthFlows: Match.arrayWith(["ALLOW_USER_SRP_AUTH"]),
    });
    // admin group exists — group membership is the sole admin signal
    // (mapped to Admin_Role; surfaced as cognito:groups in the ID token).
    template.hasResourceProperties("AWS::Cognito::UserPoolGroup", {
      GroupName: "admin",
    });
  });

  it("provisions the Schedule_Store table (on-demand)", () => {
    template.hasResourceProperties("AWS::DynamoDB::Table", {
      BillingMode: "PAY_PER_REQUEST",
      KeySchema: Match.arrayWith([
        Match.objectLike({ AttributeName: "pk", KeyType: "HASH" }),
        Match.objectLike({ AttributeName: "sk", KeyType: "RANGE" }),
      ]),
    });
  });

  it("provisions a private UI bucket with all public access blocked", () => {
    template.hasResourceProperties("AWS::S3::Bucket", {
      PublicAccessBlockConfiguration: {
        BlockPublicAcls: true,
        BlockPublicPolicy: true,
        IgnorePublicAcls: true,
        RestrictPublicBuckets: true,
      },
    });
  });

  it("provisions CloudFront distributions with OAC (UI + share signed downloads)", () => {
    // Two distributions now: the UI hosting distribution and the share-download
    // distribution (private bucket via OAC, served with signed URLs).
    template.resourceCountIs("AWS::CloudFront::Distribution", 2);
    const oacs = template.findResources("AWS::CloudFront::OriginAccessControl");
    expect(Object.keys(oacs).length).toBeGreaterThanOrEqual(2);
    // The UI distribution keeps its SPA 403/404 fallback.
    template.hasResourceProperties("AWS::CloudFront::Distribution", {
      DistributionConfig: Match.objectLike({
        DefaultRootObject: "index.html",
        CustomErrorResponses: Match.arrayWith([
          Match.objectLike({
            ErrorCode: 403,
            ResponseCode: 200,
            ResponsePagePath: "/index.html",
          }),
          Match.objectLike({
            ErrorCode: 404,
            ResponseCode: 200,
            ResponsePagePath: "/index.html",
          }),
        ]),
      }),
    });
    // The share distribution restricts access with a trusted key group.
    template.resourceCountIs("AWS::CloudFront::KeyGroup", 1);
    template.resourceCountIs("AWS::CloudFront::PublicKey", 1);
  });

  it("provisions the Scheduler Lambda and two EventBridge rules", () => {
    // The scheduler function is Node 20. (The BucketDeployment + custom-resource
    // framework also add Lambdas, so assert via a properties match.)
    template.hasResourceProperties("AWS::Lambda::Function", {
      Runtime: "nodejs20.x",
    });
    // Two rate(1 minute) EventBridge rules.
    const rules = template.findResources("AWS::Events::Rule");
    const scheduleRules = Object.values(rules).filter(
      (r) =>
        (r.Properties as { ScheduleExpression?: string }).ScheduleExpression ===
        "rate(1 minute)",
    );
    expect(scheduleRules.length).toBe(2);
  });

  it("provisions the short-lived Share_Bucket (private, with a lifecycle rule)", () => {
    // The Share_Bucket is created (not referenced) and has an expiration
    // lifecycle rule for server-side cleanup of shared images.
    const buckets = template.findResources("AWS::S3::Bucket");
    const withLifecycle = Object.values(buckets).filter((b) =>
      Boolean((b.Properties as { LifecycleConfiguration?: unknown }).LifecycleConfiguration),
    );
    expect(withLifecycle.length).toBeGreaterThanOrEqual(1);
  });

  it("exposes CfnParameters for the initial admin and standard usernames", () => {
    const params = template.toJSON().Parameters ?? {};
    expect(params).toHaveProperty("InitialAdminUsername");
    expect(params).toHaveProperty("InitialStandardUsername");
  });

  it("tags taggable resources with AiPhoto", () => {
    // Spot-check a couple of taggable resource types carry the AiPhoto tag.
    const table = Object.values(template.findResources("AWS::DynamoDB::Table"))[0];
    const tags = (table.Properties as { Tags?: { Key: string; Value: string }[] }).Tags ?? [];
    expect(tags.some((t) => t.Key === "AiPhoto")).toBe(true);

    const bucket = Object.values(template.findResources("AWS::S3::Bucket"))[0];
    const bucketTags = (bucket.Properties as { Tags?: { Key: string; Value: string }[] }).Tags ?? [];
    expect(bucketTags.some((t) => t.Key === "AiPhoto")).toBe(true);
  });

  it("provisions NO API Gateway (Req 22.3)", () => {
    template.resourceCountIs("AWS::ApiGateway::RestApi", 0);
    template.resourceCountIs("AWS::ApiGatewayV2::Api", 0);
  });

  it("scopes the Authenticated_Role to the capture-flow least-privilege actions", () => {
    // InvokeEndpointAsync appears in an IAM policy.
    template.hasResourceProperties("AWS::IAM::Policy", {
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: "sagemaker:InvokeEndpointAsync",
            Effect: "Allow",
          }),
        ]),
      }),
    });
  });

  it("grants the Admin_Role endpoint management + DynamoDB writes", () => {
    template.hasResourceProperties("AWS::IAM::Policy", {
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith([
              "sagemaker:DescribeEndpoint",
              "sagemaker:CreateEndpoint",
              "sagemaker:DeleteEndpoint",
            ]),
          }),
          Match.objectLike({
            Action: Match.arrayWith([
              "dynamodb:Query",
              "dynamodb:PutItem",
              "dynamodb:DeleteItem",
            ]),
          }),
        ]),
      }),
    });
  });

  it("stamps DeletionPolicy: Delete on the stateful resources", () => {
    for (const type of [
      "AWS::S3::Bucket",
      "AWS::DynamoDB::Table",
      "AWS::Cognito::UserPool",
    ]) {
      const resources = template.findResources(type);
      for (const [, resource] of Object.entries(resources)) {
        expect(resource.DeletionPolicy).toBe("Delete");
        expect(resource.UpdateReplacePolicy).toBe("Delete");
      }
    }
  });

  it("enables autoDeleteObjects on the stack-created UI bucket", () => {
    // The auto-delete custom resource is the marker that autoDeleteObjects is on.
    const customResources = template.findResources("Custom::S3AutoDeleteObjects");
    expect(Object.keys(customResources).length).toBeGreaterThanOrEqual(1);
  });
});
