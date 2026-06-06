#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { PhotoBoothStack } from "../lib/photo-booth-stack";

/**
 * AI Photo Booth CDK application entry point.
 *
 * Instantiates the single {@link PhotoBoothStack} that wires every construct
 * (data, auth, hosting, scheduler, initial-users). Deploy-time inputs (initial
 * usernames, scheduler timezone, endpoint config/name, SageMaker execution-role
 * ARN) are modeled as CfnParameters on the stack; the optional
 * `existingIoBucketName` / `initialUserTemporaryPassword` are read from CDK
 * context.
 *
 * The stack is environment-agnostic by default (region/account resolved at
 * deploy time). Set `CDK_DEFAULT_ACCOUNT` / `CDK_DEFAULT_REGION` (or pass
 * `env`) to pin it to a specific target.
 */
const app = new cdk.App();

new PhotoBoothStack(app, "AiPhotoBoothStack", {
  description: "AI Photo Booth infrastructure (Cognito, CloudFront, DynamoDB, scheduler Lambda).",
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});

app.synth();
