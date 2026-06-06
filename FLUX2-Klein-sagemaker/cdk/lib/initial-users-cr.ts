import { RemovalPolicy } from "aws-cdk-lib";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as iam from "aws-cdk-lib/aws-iam";
import {
  AwsCustomResource,
  AwsCustomResourcePolicy,
  PhysicalResourceId,
} from "aws-cdk-lib/custom-resources";
import { Construct, type IDependable } from "constructs";

/**
 * Default Cognito group whose members the Identity_Pool maps to the Admin_Role.
 * Mirrors the `admin` group created by {@link AuthConstruct}.
 */
const DEFAULT_ADMIN_GROUP_NAME = "admin";

/**
 * The AWS SDK v3 client used by the underlying custom-resource Lambda to call
 * the Cognito Identity Provider admin APIs. This package ships in the Lambda
 * runtime's bundled SDK, so `installLatestAwsSdk` can stay `false`.
 */
const COGNITO_IDP_SERVICE = "@aws-sdk/client-cognito-identity-provider";

/**
 * Props for {@link InitialUsersCustomResource}.
 *
 * The usernames are accepted as plain strings. The root stack (task 13.6)
 * defines the `CfnParameter`s for the initial admin/standard usernames
 * (Requirements 23.1, 23.2) and passes their resolved values in here; this
 * construct deliberately does **not** define those parameters itself.
 */
export interface InitialUsersCustomResourceProps {
  /**
   * The Cognito user pool (Auth_Service) the initial users are created in.
   * Used both to target the admin APIs (`UserPoolId`) and to scope the
   * custom-resource Lambda's IAM policy to this pool's ARN.
   */
  readonly userPool: cognito.IUserPool;

  /**
   * Username of the initial Admin_User. Created with
   * `custom:profile = ADMIN` and added to the {@link adminGroupName} group so
   * the Identity_Pool maps them to the Admin_Role (Requirement 23.3).
   */
  readonly adminUsername: string;

  /**
   * Username of the initial Standard_User. Created **without**
   * `custom:profile = ADMIN` and **not** added to the admin group
   * (Requirement 23.4).
   */
  readonly standardUsername: string;

  /**
   * Name of the Cognito group the admin user is added to.
   *
   * @default "admin"
   */
  readonly adminGroupName?: string;

  /**
   * Optional dependency used to guarantee the admin group exists before the
   * `AdminAddUserToGroup` call runs. The root stack typically passes
   * {@link AuthConstruct.adminGroup} here so CloudFormation orders the group
   * creation ahead of the membership call. When omitted, the root stack is
   * responsible for ordering.
   */
  readonly adminGroup?: IDependable;

  /**
   * Optional temporary password applied to both initial users at create time
   * (passed as `TemporaryPassword` to `AdminCreateUser`). When supplied, the
   * users start in `FORCE_CHANGE_PASSWORD` state and must set a new password on
   * first sign-in.
   *
   * When omitted, Cognito auto-generates a temporary password that is **not**
   * delivered (message delivery is suppressed), so supply this if operators
   * need a known first-login credential.
   *
   * @default - Cognito generates a suppressed temporary password
   */
  readonly temporaryPassword?: string;
}

/**
 * Creates the initial Admin_User and Standard_User in the Cognito user pool at
 * deploy time (Requirement 23.3, 23.4).
 *
 * Implemented with {@link AwsCustomResource}s that call the Cognito Identity
 * Provider admin APIs from a singleton Lambda:
 *
 * - **Admin_User:** `AdminCreateUser` with `custom:profile = ADMIN`, followed by
 *   `AdminAddUserToGroup` into the `admin` group so the Identity_Pool resolves
 *   the user to the Admin_Role.
 * - **Standard_User:** `AdminCreateUser` with **no** `custom:profile` attribute
 *   and **no** group membership.
 *
 * **Idempotency / updates:** `AdminCreateUser` calls set
 * `ignoreErrorCodesMatching` for `UsernameExistsException`, so re-deploys and
 * stack updates (which re-run the `onUpdate` handler) are safe and never fail
 * if the user already exists. `AdminAddUserToGroup` is itself idempotent.
 * Message delivery is suppressed (`MessageAction: "SUPPRESS"`) so creation does
 * not depend on a configured email/SMS channel.
 *
 * **Teardown:** no `onDelete` handler is registered and every resource carries
 * {@link RemovalPolicy.DESTROY}, so deleting the stack simply removes the custom
 * resources without issuing Cognito calls. The user pool is destroyed with the
 * stack anyway (Requirement 22.4), so the custom resource never blocks
 * teardown.
 *
 * The Lambda's role is granted only the Cognito admin actions it needs, scoped
 * to the target user pool's ARN.
 */
export class InitialUsersCustomResource extends Construct {
  /** Custom resource that creates the initial Admin_User. */
  public readonly adminUserResource: AwsCustomResource;

  /** Custom resource that adds the Admin_User to the admin group. */
  public readonly adminGroupMembershipResource: AwsCustomResource;

  /** Custom resource that creates the initial Standard_User. */
  public readonly standardUserResource: AwsCustomResource;

  /** The admin group name the Admin_User was added to. */
  public readonly adminGroupName: string;

  constructor(
    scope: Construct,
    id: string,
    props: InitialUsersCustomResourceProps,
  ) {
    super(scope, id);

    this.adminGroupName = props.adminGroupName ?? DEFAULT_ADMIN_GROUP_NAME;
    const userPoolId = props.userPool.userPoolId;
    const userPoolArn = props.userPool.userPoolArn;

    // Policy scoped to the Cognito admin actions this construct invokes, limited
    // to the target user pool. fromSdkCalls cannot be used because it derives
    // action names like `cognito-identity-provider:adminCreateUser`, whereas the
    // real IAM actions are `cognito-idp:AdminCreateUser` etc.
    const adminApiPolicy = AwsCustomResourcePolicy.fromStatements([
      new iam.PolicyStatement({
        effect: iam.Effect.ALLOW,
        actions: [
          "cognito-idp:AdminCreateUser",
          "cognito-idp:AdminSetUserPassword",
          "cognito-idp:AdminAddUserToGroup",
        ],
        resources: [userPoolArn],
      }),
    ]);

    // --- Initial Admin_User --------------------------------------------------
    // Admin identity is conveyed solely by membership in the `admin` group
    // (see AdminUserGroupMembership below), which the Identity Pool maps to the
    // Admin_Role and Cognito surfaces as `cognito:groups` in the ID token. No
    // custom attribute is set.
    this.adminUserResource = new AwsCustomResource(this, "AdminUser", {
      resourceType: "Custom::InitialAdminUser",
      policy: adminApiPolicy,
      installLatestAwsSdk: false,
      removalPolicy: RemovalPolicy.DESTROY,
      onCreate: this.adminCreateUserCall(
        userPoolId,
        props.adminUsername,
        [],
        props.temporaryPassword,
        `InitialAdminUser-${props.adminUsername}`,
      ),
      // Re-run create on stack updates; idempotent via ignoreErrorCodesMatching.
      onUpdate: this.adminCreateUserCall(
        userPoolId,
        props.adminUsername,
        [],
        props.temporaryPassword,
        `InitialAdminUser-${props.adminUsername}`,
      ),
    });

    // --- Admin_User -> admin group membership (Admin_Role mapping) ---------
    const addToGroupCall = {
      service: COGNITO_IDP_SERVICE,
      action: "adminAddUserToGroup",
      parameters: {
        UserPoolId: userPoolId,
        Username: props.adminUsername,
        GroupName: this.adminGroupName,
      },
      // AdminAddUserToGroup is idempotent (re-adding an existing member is a
      // no-op success), so the same physical id is reused across updates.
      physicalResourceId: PhysicalResourceId.of(
        `InitialAdminUserGroup-${props.adminUsername}-${this.adminGroupName}`,
      ),
    };
    this.adminGroupMembershipResource = new AwsCustomResource(
      this,
      "AdminUserGroupMembership",
      {
        resourceType: "Custom::InitialAdminUserGroup",
        policy: adminApiPolicy,
        installLatestAwsSdk: false,
        removalPolicy: RemovalPolicy.DESTROY,
        onCreate: addToGroupCall,
        onUpdate: addToGroupCall,
      },
    );
    // The user must exist before it can be added to the group.
    this.adminGroupMembershipResource.node.addDependency(this.adminUserResource);
    // ...and the admin group must exist before the membership call runs.
    if (props.adminGroup) {
      this.adminGroupMembershipResource.node.addDependency(props.adminGroup);
    }

    // --- Initial Standard_User (no custom:profile = ADMIN, no group) -------
    this.standardUserResource = new AwsCustomResource(this, "StandardUser", {
      resourceType: "Custom::InitialStandardUser",
      policy: adminApiPolicy,
      installLatestAwsSdk: false,
      removalPolicy: RemovalPolicy.DESTROY,
      // No custom:profile attribute is set, so the user is never an admin.
      onCreate: this.adminCreateUserCall(
        userPoolId,
        props.standardUsername,
        [],
        props.temporaryPassword,
        `InitialStandardUser-${props.standardUsername}`,
      ),
      onUpdate: this.adminCreateUserCall(
        userPoolId,
        props.standardUsername,
        [],
        props.temporaryPassword,
        `InitialStandardUser-${props.standardUsername}`,
      ),
    });
  }

  /**
   * Builds an `AdminCreateUser` SDK call that suppresses message delivery, sets
   * the supplied user attributes, optionally sets a temporary password, and is
   * idempotent by ignoring `UsernameExistsException`.
   */
  private adminCreateUserCall(
    userPoolId: string,
    username: string,
    userAttributes: Array<{ Name: string; Value: string }>,
    temporaryPassword: string | undefined,
    physicalId: string,
  ) {
    const parameters: Record<string, unknown> = {
      UserPoolId: userPoolId,
      Username: username,
      // Suppress the welcome email/SMS so creation does not require a
      // configured delivery channel.
      MessageAction: "SUPPRESS",
    };
    if (userAttributes.length > 0) {
      parameters.UserAttributes = userAttributes;
    }
    if (temporaryPassword) {
      parameters.TemporaryPassword = temporaryPassword;
    }

    return {
      service: COGNITO_IDP_SERVICE,
      action: "adminCreateUser",
      parameters,
      physicalResourceId: PhysicalResourceId.of(physicalId),
      // Idempotent on re-deploy / stack update: a pre-existing user is fine.
      ignoreErrorCodesMatching: "UsernameExistsException",
    };
  }
}
