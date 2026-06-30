#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { Sam3dStudioStack } from "../lib/sam3d-studio-stack";

const app = new cdk.App();

new Sam3dStudioStack(app, "Sam3dStudioStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  description: "SAM3D Studio — Cognito-auth React app + API for the SAM 3D Objects endpoint",
});
