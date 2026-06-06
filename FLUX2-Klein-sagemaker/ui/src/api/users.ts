/**
 * User_Admin — browser-side Cognito user management (admin only).
 *
 * Lists the user pool's users and force-signs-out a user, calling the Cognito
 * Identity Provider admin APIs directly under the Admin_Role credentials. A
 * Standard_User's Authenticated_Role lacks these permissions, so the calls fail
 * at the IAM layer.
 *
 * This is an admin console surface, so usernames ARE shown here — that is
 * distinct from the visitor-facing UI, which never reveals identity (Req 13.4).
 *
 * All calls run through {@link withAuthRetry} for silent token/STS refresh.
 */

import {
  CognitoIdentityProviderClient,
  ListUsersCommand,
  AdminUserGlobalSignOutCommand,
} from "@aws-sdk/client-cognito-identity-provider";

import { getConfig } from "../config";
import { credentialsProvider, withAuthRetry } from "./awsClients";

let cognitoClient: CognitoIdentityProviderClient | undefined;

/** Lazily-created Cognito IdP client bound to the refreshing credentials. */
function getCognitoClient(): CognitoIdentityProviderClient {
  if (!cognitoClient) {
    cognitoClient = new CognitoIdentityProviderClient({
      region: getConfig().region,
      credentials: credentialsProvider,
    });
  }
  return cognitoClient;
}

/** A user pool member surfaced to the admin Users panel. */
export interface BoothUser {
  /** Cognito username. */
  username: string;
  /** Account status, e.g. CONFIRMED / FORCE_CHANGE_PASSWORD. */
  status: string;
  /** Whether the account is enabled. */
  enabled: boolean;
  /** Whether the user is an admin (member of the `admin` group). */
  isAdmin?: boolean;
}

/**
 * List users in the configured user pool. The user pool id is not in the SPA
 * runtime config today, so it is accepted as a parameter; the Users panel reads
 * it from config (see note in {@link listUsers}).
 */
export async function listUsers(userPoolId: string): Promise<BoothUser[]> {
  const client = getCognitoClient();
  const response = await withAuthRetry(() =>
    client.send(new ListUsersCommand({ UserPoolId: userPoolId, Limit: 60 })),
  );
  return (response.Users ?? []).map((u) => ({
    username: u.Username ?? "",
    status: u.UserStatus ?? "UNKNOWN",
    enabled: u.Enabled ?? false,
  }));
}

/**
 * Force a global sign-out of the given user: revokes all their refresh tokens
 * so their active sessions can no longer silently refresh. They keep any
 * still-valid access token until it expires (≤ 1h), then must sign in again.
 */
export async function signOutUser(
  userPoolId: string,
  username: string,
): Promise<void> {
  const client = getCognitoClient();
  await withAuthRetry(() =>
    client.send(
      new AdminUserGlobalSignOutCommand({
        UserPoolId: userPoolId,
        Username: username,
      }),
    ),
  );
}
