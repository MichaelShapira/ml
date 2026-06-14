import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as iam from "aws-cdk-lib/aws-iam";
import type * as s3 from "aws-cdk-lib/aws-s3";
import type * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import { PROMPT_OVERRIDE_PK } from "./data-construct";

/**
 * The S3 prefixes used by the existing FLUX.2 async inference endpoint.
 *
 * These match the prefixes the deployment notebook configures
 * (`flux2-klein-sagemaker.ipynb`) and that the SPA's `Generation_Service`
 * reads/writes:
 *   - inputs:   request JSON uploaded by the browser (PutObject)
 *   - outputs:  result PNG written by the endpoint (GetObject)
 *   - failures: failure records written by the endpoint (GetObject)
 */
const INPUTS_PREFIX = "flux2-klein-inputs/";
const OUTPUTS_PREFIX = "flux2-klein-outputs/";
const FAILURES_PREFIX = "flux2-klein-failures/";

/** Default name of the existing FLUX.2 [klein] 9B async endpoint. */
const DEFAULT_ENDPOINT_NAME = "flux2-klein-9b-g6e2";

/** Cognito group whose members are mapped to the {@link AuthConstruct.adminRole}. */
const ADMIN_GROUP_NAME = "admin";

/**
 * Props for {@link AuthConstruct}.
 *
 * The construct itself owns no application data resources; the root stack wires
 * in the dependencies it needs to scope the IAM role policies. Values that are
 * not known at construct time (e.g. the existing SageMaker execution role ARN)
 * are accepted as props/parameters rather than hardcoded.
 */
export interface AuthConstructProps {
  /**
   * The async I/O bucket that holds the FLUX.2 inputs/outputs/failures prefixes.
   * Provided by the data construct (or a reference to the pre-existing bucket)
   * and used to scope the S3 statements on both roles to exact prefixes.
   */
  readonly ioBucket: s3.IBucket;

  /**
   * The Schedule_Store DynamoDB table. Only the {@link AuthConstruct.adminRole}
   * is granted Query/PutItem/DeleteItem on it; the Authenticated_Role gets no
   * DynamoDB access at all.
   */
  readonly scheduleTable: dynamodb.ITable;

  /**
   * Name of the FLUX.2 async endpoint to scope SageMaker statements to.
   * Defaults to {@link DEFAULT_ENDPOINT_NAME} (`flux2-klein-9b-g6e2`).
   */
  readonly endpointName?: string;

  /** Optional explicit name for the Cognito user pool. */
  readonly userPoolName?: string;
}

/**
 * Auth_Service (Cognito) + Identity_Pool + the two IAM roles.
 *
 * Provisions, as described in the design's "Auth_Service (Cognito) and IAM
 * roles" section:
 *   - a Cognito user pool with a `custom:profile` attribute and an `admin`
 *     group (Requirement 11.1, 14.3),
 *   - a public SPA app client (no secret) with SRP auth,
 *   - a Cognito identity pool (authenticated identities only),
 *   - `Authenticated_Role` and `Admin_Role` with least-privilege policies, and
 *   - a role attachment mapping the `admin` group to `Admin_Role` and everyone
 *     else to `Authenticated_Role` (Requirement 14.4, 14.5).
 *
 * Authorization is enforced by IAM at the service layer: the SPA's reading of
 * the `profile` claim to show/hide the Admin tab is cosmetic only.
 *
 * Every resource that supports it carries `RemovalPolicy.DESTROY` so the
 * demo/kiosk stack tears down cleanly (Requirement 22.4).
 *
 * The created resources are exposed as public readonly properties so the root
 * stack and other constructs (hosting, data, scheduler, initial-users) can
 * reference them.
 */
export class AuthConstruct extends Construct {
  /** The Cognito user pool (Auth_Service). */
  public readonly userPool: cognito.UserPool;

  /** The public SPA app client (no secret, SRP enabled). */
  public readonly userPoolClient: cognito.UserPoolClient;

  /** The Cognito identity pool (authenticated identities only). */
  public readonly identityPool: cognito.CfnIdentityPool;

  /** IAM role assumed by a Standard_User (capture/effect flow only). */
  public readonly authenticatedRole: iam.Role;

  /** IAM role assumed by an Admin_User (adds endpoint mgmt + schedule writes). */
  public readonly adminRole: iam.Role;

  /** The Cognito `admin` group whose `roleArn` points at {@link adminRole}. */
  public readonly adminGroup: cognito.CfnUserPoolGroup;

  constructor(scope: Construct, id: string, props: AuthConstructProps) {
    super(scope, id);

    const stack = cdk.Stack.of(this);
    const endpointName = props.endpointName ?? DEFAULT_ENDPOINT_NAME;

    // --- ARNs the IAM policies are scoped to -------------------------------
    // SageMaker endpoint ARN (e.g. arn:aws:sagemaker:<region>:<account>:endpoint/flux2-klein-9b-g6e2)
    const endpointArn = stack.formatArn({
      service: "sagemaker",
      resource: "endpoint",
      resourceName: endpointName,
    });
    // Endpoint-config ARNs (CreateEndpoint references an endpoint config).
    const endpointConfigArn = stack.formatArn({
      service: "sagemaker",
      resource: "endpoint-config",
      resourceName: "*",
    });
    // Admin endpoint management spans ANY endpoint name in this account/region,
    // since endpoints are created per selected config name (endpoint name ===
    // config name), not a single hardcoded endpoint.
    const anyEndpointArn = stack.formatArn({
      service: "sagemaker",
      resource: "endpoint",
      resourceName: "*",
    });

    // --- Cognito user pool (Auth_Service) ----------------------------------
    // Admin identity is conveyed by membership in the `admin` Cognito group
    // (mapped to Admin_Role by the Identity Pool and surfaced as cognito:groups
    // in the ID token). No custom attribute is used: a custom-attribute schema
    // is unreliable to write at create time and the group is the authoritative
    // signal anyway.
    this.userPool = new cognito.UserPool(this, "UserPool", {
      userPoolName: props.userPoolName,
      selfSignUpEnabled: false,
      signInAliases: { username: true },
      standardAttributes: {
        email: { required: false, mutable: true },
      },
      // Password policy: require a minimum length plus an uppercase letter, a
      // lowercase letter, and a special character so the initial admin/visitor
      // credentials (and any later-created users) cannot be trivially weak.
      passwordPolicy: {
        minLength: 8,
        requireUppercase: true,
        requireLowercase: true,
        requireSymbols: true,
        requireDigits: false,
      },
      accountRecovery: cognito.AccountRecovery.NONE,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Public SPA app client: no secret, SRP (USER_SRP_AUTH) enabled. CDK adds
    // ALLOW_REFRESH_TOKEN_AUTH automatically (needed for silent refresh).
    this.userPoolClient = this.userPool.addClient("SpaClient", {
      userPoolClientName: "photo-booth-spa",
      generateSecret: false,
      authFlows: { userSrp: true },
      preventUserExistenceErrors: true,
    });

    // --- IAM roles (created before the identity pool / group reference them) -
    // Placeholder for the federated trust; aud is bound to the identity pool
    // id once it exists. We create the identity pool first so we can reference
    // its ref in the role trust conditions.
    this.identityPool = new cognito.CfnIdentityPool(this, "IdentityPool", {
      allowUnauthenticatedIdentities: false,
      cognitoIdentityProviders: [
        {
          clientId: this.userPoolClient.userPoolClientId,
          providerName: this.userPool.userPoolProviderName,
        },
      ],
    });
    this.identityPool.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);

    // Authenticated_Role: capture/effect flow only.
    this.authenticatedRole = new iam.Role(this, "AuthenticatedRole", {
      description:
        "Photo booth Standard_User role: capture/effect flow + read-only effect prompts (no SageMaker mgmt, no schedule writes).",
      assumedBy: this.identityPoolPrincipal(),
    });
    for (const statement of this.captureFlowStatements(
      props.ioBucket,
      endpointArn,
      props.scheduleTable.tableArn,
    )) {
      this.authenticatedRole.addToPolicy(statement);
    }

    // Admin_Role: everything the Authenticated_Role can do PLUS endpoint
    // management and schedule writes.
    this.adminRole = new iam.Role(this, "AdminRole", {
      description:
        "Photo booth Admin_User role: capture/effect flow PLUS SageMaker endpoint mgmt + Schedule_Store writes.",
      assumedBy: this.identityPoolPrincipal(),
    });
    for (const statement of this.captureFlowStatements(
      props.ioBucket,
      endpointArn,
      props.scheduleTable.tableArn,
    )) {
      this.adminRole.addToPolicy(statement);
    }
    for (const statement of this.adminStatements(
      anyEndpointArn,
      endpointConfigArn,
      props.scheduleTable.tableArn,
    )) {
      this.adminRole.addToPolicy(statement);
    }

    // --- Cognito admin group -> Admin_Role ---------------------------------
    // Setting the group's roleArn lets the identity pool's Token-type role
    // mapping resolve admin members to the Admin_Role via cognito:preferred_role.
    this.adminGroup = new cognito.CfnUserPoolGroup(this, "AdminGroup", {
      userPoolId: this.userPool.userPoolId,
      groupName: ADMIN_GROUP_NAME,
      description: "Administrators mapped to the Admin_Role by the identity pool.",
      roleArn: this.adminRole.roleArn,
      precedence: 0,
    });

    // --- Identity pool role attachment -------------------------------------
    // Default authenticated role = Authenticated_Role. A Token-type role
    // mapping uses the cognito:preferred_role / cognito:roles claims (driven by
    // the admin group's roleArn above) so admin-group members assume the
    // Admin_Role; everyone else falls back to the default Authenticated_Role.
    const providerIdentifier = `${this.userPool.userPoolProviderName}:${this.userPoolClient.userPoolClientId}`;
    const roleAttachment = new cognito.CfnIdentityPoolRoleAttachment(
      this,
      "RoleAttachment",
      {
        identityPoolId: this.identityPool.ref,
        roles: {
          authenticated: this.authenticatedRole.roleArn,
        },
        roleMappings: {
          cognitoProvider: {
            type: "Token",
            ambiguousRoleResolution: "AuthenticatedRole",
            identityProvider: providerIdentifier,
          },
        },
      },
    );
    roleAttachment.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);
  }

  /**
   * Trust policy principal for a role assumable through the identity pool:
   * the Cognito federated principal, scoped to this identity pool's id
   * (`aud`) and to authenticated identities (`amr`).
   */
  private identityPoolPrincipal(): iam.FederatedPrincipal {
    return new iam.FederatedPrincipal(
      "cognito-identity.amazonaws.com",
      {
        StringEquals: {
          "cognito-identity.amazonaws.com:aud": this.identityPool.ref,
        },
        "ForAnyValue:StringLike": {
          "cognito-identity.amazonaws.com:amr": "authenticated",
        },
      },
      "sts:AssumeRoleWithWebIdentity",
    );
  }

  /**
   * Least-privilege statements for the capture/effect flow, shared by both the
   * Authenticated_Role and the Admin_Role. Fresh statement objects are returned
   * on each call so they can be attached to multiple roles independently.
   *
   * No Rekognition. No SageMaker management. No DynamoDB. No write access to
   * the outputs/failures prefixes.
   */
  private captureFlowStatements(
    ioBucket: s3.IBucket,
    endpointArn: string,
    scheduleTableArn: string,
  ): iam.PolicyStatement[] {
    return [
      // Submit the async inference request to the FLUX.2 endpoint.
      new iam.PolicyStatement({
        sid: "InvokeFlux2Endpoint",
        effect: iam.Effect.ALLOW,
        actions: ["sagemaker:InvokeEndpointAsync"],
        resources: [endpointArn],
      }),
      // Upload the request JSON to the inputs prefix only.
      new iam.PolicyStatement({
        sid: "PutInputObjects",
        effect: iam.Effect.ALLOW,
        actions: ["s3:PutObject"],
        resources: [ioBucket.arnForObjects(`${INPUTS_PREFIX}*`)],
      }),
      // Read result / failure objects from the outputs + failures prefixes only.
      new iam.PolicyStatement({
        sid: "GetOutputObjects",
        effect: iam.Effect.ALLOW,
        actions: ["s3:GetObject"],
        resources: [
          ioBucket.arnForObjects(`${OUTPUTS_PREFIX}*`),
          ioBucket.arnForObjects(`${FAILURES_PREFIX}*`),
        ],
      }),
      // ListBucket (enables HeadObject 404 vs 403 polling semantics), scoped by
      // a prefix condition to the outputs + failures prefixes.
      new iam.PolicyStatement({
        sid: "ListOutputPrefixes",
        effect: iam.Effect.ALLOW,
        actions: ["s3:ListBucket"],
        resources: [ioBucket.bucketArn],
        conditions: {
          StringLike: {
            "s3:prefix": [`${OUTPUTS_PREFIX}*`, `${FAILURES_PREFIX}*`],
          },
        },
      }),
      // Read-only access to the per-effect prompt overrides, scoped (via the
      // DynamoDB leading-key condition) to ONLY the prompt partition so a
      // standard user can pick up admin-customized prompts but cannot read or
      // touch Working_Hours / managed-config items. The Admin_Role additionally
      // gets full table access via adminStatements (overlap is harmless).
      new iam.PolicyStatement({
        sid: "ReadEffectPrompts",
        effect: iam.Effect.ALLOW,
        actions: ["dynamodb:Query", "dynamodb:GetItem"],
        resources: [scheduleTableArn],
        conditions: {
          "ForAllValues:StringEquals": {
            "dynamodb:LeadingKeys": [PROMPT_OVERRIDE_PK],
          },
        },
      }),
    ];
  }

  /**
   * Additional statements granted only to the Admin_Role: SageMaker endpoint
   * management and Schedule_Store writes.
   *
   * No `iam:PassRole` is needed: the booth only calls `CreateEndpoint`
   * (referencing the pre-existing EndpointConfig by name) and `DeleteEndpoint`,
   * neither of which passes a role. The SageMaker *execution* role belongs to
   * the Model/EndpointConfig (created by the deployment notebook) and is never
   * passed by this flow.
   */
  private adminStatements(
    endpointArn: string,
    endpointConfigArn: string,
    scheduleTableArn: string,
  ): iam.PolicyStatement[] {
    return [
      // ListEndpoints has no per-endpoint resource scoping.
      new iam.PolicyStatement({
        sid: "ListEndpoints",
        effect: iam.Effect.ALLOW,
        actions: ["sagemaker:ListEndpoints", "sagemaker:ListEndpointConfigs"],
        resources: ["*"],
      }),
      // Describe / Create / Delete the booth's endpoint (+ endpoint configs).
      // AddTags is required to tag the endpoint on CreateEndpoint(Tags=[...]).
      new iam.PolicyStatement({
        sid: "ManageFlux2Endpoint",
        effect: iam.Effect.ALLOW,
        actions: [
          "sagemaker:DescribeEndpoint",
          "sagemaker:CreateEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:AddTags",
        ],
        resources: [endpointArn, endpointConfigArn],
      }),
      // Schedule_Store writes (the SPA admin calendar CRUD) + reads (the
      // current-config pointer GetItem and schedule queries).
      new iam.PolicyStatement({
        sid: "ScheduleStoreWrite",
        effect: iam.Effect.ALLOW,
        actions: [
          "dynamodb:Query",
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:DeleteItem",
        ],
        resources: [scheduleTableArn],
      }),
      // User management for the admin Users panel: list users and force a
      // global sign-out. Scoped to this user pool's ARN.
      new iam.PolicyStatement({
        sid: "ManageUsers",
        effect: iam.Effect.ALLOW,
        actions: [
          "cognito-idp:ListUsers",
          "cognito-idp:AdminUserGlobalSignOut",
        ],
        resources: [this.userPool.userPoolArn],
      }),
      // Cost Explorer read for the admin cost panel. Cost Explorer is a global
      // service whose actions do not support resource-level scoping, so the
      // resource must be "*". GetCostAndUsage is read-only.
      new iam.PolicyStatement({
        sid: "ReadCostExplorer",
        effect: iam.Effect.ALLOW,
        actions: ["ce:GetCostAndUsage"],
        resources: ["*"],
      }),
    ];
  }
}
