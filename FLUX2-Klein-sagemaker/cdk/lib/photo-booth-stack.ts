import {
  Aspects,
  CfnOutput,
  CfnParameter,
  CfnResource,
  RemovalPolicy,
  Stack,
  Tags,
  type IAspect,
  type StackProps,
} from "aws-cdk-lib";
import { Construct, type IConstruct } from "constructs";

import { DataConstruct, FLUX2_ENDPOINT_NAME } from "./data-construct";
import { AuthConstruct } from "./auth-construct";
import { HostingConstruct } from "./hosting-construct";
import { SchedulerConstruct } from "./scheduler-construct";
import { InitialUsersCustomResource } from "./initial-users-cr";
import { UiDeploymentConstruct } from "./ui-deployment-construct";
import { EmailConstruct } from "./email-construct";
import { IoBucketCorsConstruct } from "./io-bucket-cors-construct";
import { InvokeConstruct } from "./invoke-construct";
import { CostAllocationTagConstruct } from "./cost-allocation-tag-construct";

/** Default SageMaker endpoint configuration name the admin/scheduler `CreateEndpoint` uses. */
const DEFAULT_ENDPOINT_CONFIG_NAME = "flux2-klein-9b-g6e2-config";

/** Default IANA timezone used to evaluate wall-clock Working_Hours. */
const DEFAULT_TIMEZONE = "America/Los_Angeles";

/**
 * CDK aspect that stamps `DeletionPolicy: Delete` and `UpdateReplacePolicy:
 * Delete` onto **every** L1 resource in the stack (Requirement 22.4).
 *
 * The individual constructs already set `RemovalPolicy.DESTROY` on the
 * resources they own; this aspect is a belt-and-braces guarantee that also
 * covers resources synthesized indirectly by L2/L3 constructs and the custom
 * resource / auto-delete-objects provider framework (e.g. the singleton
 * provider Lambdas), so no provisioned resource is left with the implicit
 * `Retain` default. Because it applies the same `DESTROY` policy the constructs
 * already use, it never conflicts with them.
 */
class ApplyDestroyPolicyAspect implements IAspect {
  public visit(node: IConstruct): void {
    if (node instanceof CfnResource) {
      node.applyRemovalPolicy(RemovalPolicy.DESTROY);
    }
  }
}

/**
 * Props for {@link PhotoBoothStack}. Only the standard CDK {@link StackProps}
 * are accepted; all deploy-time inputs are modeled as {@link CfnParameter}s (for
 * values resolved at deploy time) or CDK **context** (for values that must be
 * known at synth time, e.g. whether to create or reference the async I/O
 * bucket).
 */
export interface PhotoBoothStackProps extends StackProps {}

/**
 * Root stack that wires together every AI Photo Booth construct
 * (Requirement 22.1).
 *
 * Provisioning order and shared-reference wiring:
 *  1. {@link DataConstruct} — owns the `Schedule_Store` table and the async I/O
 *     bucket, and is the single source of truth for the managed endpoint
 *     name/ARNs and the reusable IAM policy factories.
 *  2. {@link AuthConstruct} — Cognito user pool + app client + identity pool +
 *     `Authenticated_Role`/`Admin_Role`, scoped with `data.ioBucket`,
 *     `data.scheduleTable`, the SageMaker execution-role ARN, and the endpoint
 *     name.
 *  3. {@link HostingConstruct} — private `UI_Bucket` + CloudFront (OAC) +
 *     SPA fallback.
 *  4. {@link SchedulerConstruct} — the only Lambda + its role + the two
 *     EventBridge rules, driven by the {@link DataConstruct}.
 *  5. {@link InitialUsersCustomResource} — creates the initial admin/standard
 *     users in the user pool from the deploy-time username parameters.
 *
 * Deliberately provisions **no** API Gateway and **no** per-request Lambda — the
 * SPA calls AWS directly under the vended IAM credentials, and the only
 * server-side compute is the scheduler Lambda (Requirement 22.3).
 *
 * An {@link ApplyDestroyPolicyAspect} guarantees `DELETE` removal coverage
 * across all resources (Requirement 22.4).
 */
export class PhotoBoothStack extends Stack {
  constructor(scope: Construct, id: string, props: PhotoBoothStackProps = {}) {
    super(scope, id, props);

    // -------------------------------------------------------------------------
    // Deploy-time parameters (Requirement 23.1, 23.2)
    // -------------------------------------------------------------------------

    // Initial Admin_User username (Requirement 23.1). Created with
    // custom:profile = ADMIN and added to the `admin` group by the
    // initial-users custom resource.
    const initialAdminUsername = new CfnParameter(this, "InitialAdminUsername", {
      type: "String",
      description:
        "Username of the initial Admin_User created in the Cognito user pool (custom:profile = ADMIN).",
      default: "admin",
      minLength: 1,
    });

    // Initial Standard_User username (Requirement 23.2). Created without the
    // admin profile attribute and without admin group membership.
    const initialStandardUsername = new CfnParameter(
      this,
      "InitialStandardUsername",
      {
        type: "String",
        description:
          "Username of the initial Standard_User created in the Cognito user pool (no admin profile).",
        default: "visitor",
        minLength: 1,
      },
    );

    // IANA timezone used by the Scheduler_Function to interpret wall-clock
    // Working_Hours. Defaults to the kiosk's lobby timezone.
    const schedulerTimezone = new CfnParameter(this, "SchedulerTimezone", {
      type: "String",
      description:
        "IANA timezone used to evaluate Working_Hours in the Scheduler_Function (e.g. America/Los_Angeles).",
      default: DEFAULT_TIMEZONE,
      minLength: 1,
    });

    // SageMaker endpoint configuration name the admin "start" action and the
    // Scheduler_Function pass to CreateEndpoint. Defaults to the existing
    // FLUX.2 endpoint's config name.
    const endpointConfigName = new CfnParameter(this, "EndpointConfigName", {
      type: "String",
      description:
        "SageMaker endpoint configuration name used by CreateEndpoint (admin start + scheduler).",
      default: DEFAULT_ENDPOINT_CONFIG_NAME,
      minLength: 1,
    });

    // Verified SES sender email address used as the From of "email my photo"
    // messages. Required (no default) and account/operator-specific. SES emails
    // a verification link to this address on first deploy.
    const senderEmail = new CfnParameter(this, "SenderEmail", {
      type: "String",
      description:
        "Verified SES sender email address used as the From of result emails (e.g. booth@example.com).",
      minLength: 3,
    });

    // Name of the managed FLUX.2 async endpoint. Defaults to the existing
    // endpoint; surfaced as a parameter so the stack can target a differently
    // named endpoint without code changes.
    const endpointName = new CfnParameter(this, "EndpointName", {
      type: "String",
      description: "Name of the managed FLUX.2 [klein] 9B async SageMaker endpoint.",
      default: FLUX2_ENDPOINT_NAME,
      minLength: 1,
    });

    // -------------------------------------------------------------------------
    // Synth-time context (controls a synth-time branch, so it cannot be a
    // CfnParameter): when set, the async I/O bucket is REFERENCED by name
    // instead of created. Leave unset to have the stack create a fresh bucket.
    // -------------------------------------------------------------------------
    const existingIoBucketName = this.node.tryGetContext("existingIoBucketName") as
      | string
      | undefined;

    // Optional temporary password for the initial users (context-driven so it
    // is not baked into the template parameters surface).
    const initialUserTemporaryPassword = this.node.tryGetContext(
      "initialUserTemporaryPassword",
    ) as string | undefined;

    // When set, the SES sender identity is REFERENCED, not created — use this
    // when the sender email is already a verified SES identity in the account
    // (CloudFormation cannot create an identity that already exists).
    const senderEmailAlreadyVerified =
      this.node.tryGetContext("senderEmailAlreadyVerified") === true ||
      this.node.tryGetContext("senderEmailAlreadyVerified") === "true";

    // -------------------------------------------------------------------------
    // 1. Data layer — Schedule_Store + async I/O bucket + policy factories.
    // -------------------------------------------------------------------------
    const data = new DataConstruct(this, "Data", {
      endpointName: endpointName.valueAsString,
      ...(existingIoBucketName ? { existingIoBucketName } : {}),
    });

    // -------------------------------------------------------------------------
    // 1b. Email — verified SES sender identity for "email my photo".
    // -------------------------------------------------------------------------
    const emailService = new EmailConstruct(this, "Email", {
      senderEmail: senderEmail.valueAsString,
      useExistingIdentity: senderEmailAlreadyVerified,
    });

    // -------------------------------------------------------------------------
    // 2. Auth — Cognito + Identity Pool + Authenticated_Role/Admin_Role.
    //    Scoped with the data layer's bucket + table and the SES sender so any
    //    signed-in visitor can email their photo from the verified address.
    // -------------------------------------------------------------------------
    const auth = new AuthConstruct(this, "Auth", {
      ioBucket: data.ioBucket,
      scheduleTable: data.scheduleTable,
      endpointName: endpointName.valueAsString,
      senderIdentityArn: emailService.senderIdentityArn,
      senderEmail: senderEmail.valueAsString,
    });

    // -------------------------------------------------------------------------
    // 3. Hosting — private UI_Bucket + CloudFront (OAC) + SPA fallback.
    // -------------------------------------------------------------------------
    const hosting = new HostingConstruct(this, "Hosting", {
      distributionComment: "AI Photo Booth UI distribution",
    });

    // -------------------------------------------------------------------------
    // 3b. I/O bucket CORS — allow the browser-direct flow (PutObject inputs,
    //     Head/GetObject outputs/failures) from the CloudFront origin + local
    //     dev. The I/O bucket is referenced (not created), so its CORS is set
    //     imperatively via a custom resource that wires in the just-created
    //     CloudFront distribution domain automatically.
    // -------------------------------------------------------------------------
    new IoBucketCorsConstruct(this, "IoBucketCors", {
      ioBucketName: data.ioBucket.bucketName,
      distributionDomainName: hosting.distributionDomainName,
    });

    // -------------------------------------------------------------------------
    // 3c. Invoke_Proxy — the SageMaker runtime API does not support CORS, so the
    //     browser cannot call InvokeEndpointAsync directly. This tiny Lambda
    //     behind an IAM-authenticated Function URL (with native CORS) performs
    //     that one call; S3 input upload + output/failure polling stay
    //     browser-direct. Both caller roles are granted lambda:InvokeFunctionUrl.
    // -------------------------------------------------------------------------
    const invoke = new InvokeConstruct(this, "Invoke", {
      data,
      allowedOrigins: [
        `https://${hosting.distributionDomainName}`,
        "http://localhost:5173",
        "http://localhost:5174",
      ],
    });
    invoke.grantInvoke(auth.authenticatedRole);
    invoke.grantInvoke(auth.adminRole);

    // -------------------------------------------------------------------------
    // 4. Scheduler — the only Lambda + its role + the two EventBridge rules.
    // -------------------------------------------------------------------------
    new SchedulerConstruct(this, "Scheduler", {
      data,
      endpointConfigName: endpointConfigName.valueAsString,
      timezone: schedulerTimezone.valueAsString,
    });

    // -------------------------------------------------------------------------
    // 5. Initial users — created from the deploy-time username parameters.
    //    auth.adminGroup is passed as the ordering dependency so the `admin`
    //    group exists before the membership call runs.
    // -------------------------------------------------------------------------
    new InitialUsersCustomResource(this, "InitialUsers", {
      userPool: auth.userPool,
      adminUsername: initialAdminUsername.valueAsString,
      standardUsername: initialStandardUsername.valueAsString,
      adminGroup: auth.adminGroup,
      ...(initialUserTemporaryPassword
        ? { temporaryPassword: initialUserTemporaryPassword }
        : {}),
    });

    // -------------------------------------------------------------------------
    // 6. UI deployment — push the built SPA + injected runtime config into the
    //    UI_Bucket behind CloudFront (Requirements 22.2, 24.2, 24.4). The
    //    runtime config carries the ids/names the browser needs to call AWS
    //    directly; it is written as config.js (window.__BOOTH_CONFIG__).
    // -------------------------------------------------------------------------
    new UiDeploymentConstruct(this, "UiDeployment", {
      uiBucket: hosting.uiBucket,
      distribution: hosting.distribution,
      runtimeConfig: {
        region: this.region,
        userPoolId: auth.userPool.userPoolId,
        userPoolClientId: auth.userPoolClient.userPoolClientId,
        identityPoolId: auth.identityPool.ref,
        endpointName: data.endpointName,
        endpointConfigName: endpointConfigName.valueAsString,
        ioBucket: data.ioBucket.bucketName,
        scheduleTable: data.scheduleTable.tableName,
        senderEmail: senderEmail.valueAsString,
        timezone: schedulerTimezone.valueAsString,
        invokeFunctionUrl: invoke.functionUrl.url,
      },
    });

    // -------------------------------------------------------------------------
    // Tagging: stamp every taggable resource in the stack with AiPhoto.
    // -------------------------------------------------------------------------
    Tags.of(this).add("AiPhoto", "true");

    // Activate `AiPhoto` as a cost allocation tag so the admin cost panel can
    // group/filter spend by it in Cost Explorer (best-effort; see construct).
    new CostAllocationTagConstruct(this, "CostAllocationTag", {
      tagKey: "AiPhoto",
    });

    // -------------------------------------------------------------------------
    // Removal-policy guarantee (Requirement 22.4): stamp DELETE on every
    // resource, including provider-framework resources created indirectly.
    // -------------------------------------------------------------------------
    Aspects.of(this).add(new ApplyDestroyPolicyAspect());

    // -------------------------------------------------------------------------
    // Outputs — feed the SPA runtime config (task 14.1).
    // -------------------------------------------------------------------------
    new CfnOutput(this, "DistributionDomainName", {
      description: "CloudFront domain name serving the Photo_Booth_App SPA.",
      value: hosting.distributionDomainName,
    });
    new CfnOutput(this, "UiBucketName", {
      description: "Private S3 bucket holding the built UI assets (deploy target).",
      value: hosting.uiBucket.bucketName,
    });
    new CfnOutput(this, "UserPoolId", {
      description: "Cognito user pool id (Auth_Service).",
      value: auth.userPool.userPoolId,
    });
    new CfnOutput(this, "UserPoolClientId", {
      description: "Cognito SPA app client id.",
      value: auth.userPoolClient.userPoolClientId,
    });
    new CfnOutput(this, "IdentityPoolId", {
      description: "Cognito identity pool id (vends temporary AWS credentials).",
      value: auth.identityPool.ref,
    });
    new CfnOutput(this, "ScheduleTableName", {
      description: "Schedule_Store DynamoDB table name.",
      value: data.scheduleTable.tableName,
    });
    new CfnOutput(this, "IoBucketName", {
      description: "Async I/O S3 bucket name (inputs/outputs/failures prefixes).",
      value: data.ioBucket.bucketName,
    });
    new CfnOutput(this, "Region", {
      description: "AWS region the stack is deployed to.",
      value: this.region,
    });
    new CfnOutput(this, "EndpointNameOutput", {
      description: "Managed FLUX.2 async SageMaker endpoint name.",
      value: data.endpointName,
    });
    new CfnOutput(this, "EndpointConfigNameOutput", {
      description: "SageMaker endpoint configuration name used by CreateEndpoint.",
      value: endpointConfigName.valueAsString,
    });
    new CfnOutput(this, "SenderEmailOutput", {
      description: "Verified SES sender address (verify the link SES emails before sending).",
      value: emailService.senderEmail,
    });
    new CfnOutput(this, "InvokeFunctionUrl", {
      description:
        "IAM-authenticated Lambda Function URL the SPA SigV4-signs to invoke the async endpoint (CORS workaround).",
      value: invoke.functionUrl.url,
    });
  }
}
