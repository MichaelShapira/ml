// Cognito auth via amazon-cognito-identity-js (no hosted UI). We use the ID
// token for API calls because custom attributes (custom:role) are present in the
// ID token, not the access token.
import {
  CognitoUserPool,
  CognitoUser,
  AuthenticationDetails,
  CognitoUserSession,
} from "amazon-cognito-identity-js";
import { config } from "./config";

const pool = new CognitoUserPool({
  UserPoolId: config.userPoolId,
  ClientId: config.userPoolClientId,
});

export interface Identity {
  username: string;
  role: string;
  isAdmin: boolean;
}

function identityFromSession(session: CognitoUserSession): Identity {
  const payload = session.getIdToken().decodePayload() as Record<string, unknown>;
  const role = String(payload["custom:role"] ?? "visitor");
  return {
    username: String(payload["cognito:username"] ?? payload["username"] ?? ""),
    role,
    isAdmin: role === "admin",
  };
}

export function signIn(username: string, password: string): Promise<Identity> {
  const user = new CognitoUser({ Username: username, Pool: pool });
  const details = new AuthenticationDetails({ Username: username, Password: password });
  return new Promise((resolve, reject) => {
    user.authenticateUser(details, {
      onSuccess: (session) => resolve(identityFromSession(session)),
      onFailure: (err) => reject(err),
      newPasswordRequired: () =>
        reject(new Error("Password change required — set a permanent password (deploy.sh does this).")),
    });
  });
}

export function getCurrentIdentity(): Promise<Identity | null> {
  const user = pool.getCurrentUser();
  if (!user) return Promise.resolve(null);
  return new Promise((resolve) => {
    user.getSession((err: Error | null, session: CognitoUserSession | null) => {
      if (err || !session || !session.isValid()) return resolve(null);
      resolve(identityFromSession(session));
    });
  });
}

// Returns a valid ID token JWT, refreshing the session if needed.
export function getIdToken(): Promise<string> {
  const user = pool.getCurrentUser();
  if (!user) return Promise.reject(new Error("not signed in"));
  return new Promise((resolve, reject) => {
    user.getSession((err: Error | null, session: CognitoUserSession | null) => {
      if (err || !session || !session.isValid()) return reject(err ?? new Error("session invalid"));
      resolve(session.getIdToken().getJwtToken());
    });
  });
}

export function signOut(): void {
  pool.getCurrentUser()?.signOut();
}
