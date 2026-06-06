/**
 * Client auth and credentials layer for the AI Photo Booth (`authService`).
 *
 * The booth has **no per-request backend**: the browser calls AWS services
 * directly using temporary STS credentials vended by a Cognito Identity Pool,
 * with authorization enforced by IAM. Two independent things expire and each
 * has its own refresh path:
 *
 *   1. the Cognito **token** (access/ID) — refreshed via the user pool's
 *      refresh token, transparently by `getSession()` or forcibly by
 *      `refreshSession()`; and
 *   2. the Identity-Pool **STS credentials** — minted by `getCredentials()`
 *      and forcibly re-minted by `refreshCredentials()`.
 *
 * This module owns both paths and the `requireSignIn()` signal the app
 * subscribes to so the UI returns to the sign-in interface when the refresh
 * token is dead. It deliberately exposes only auth/credentials concerns; the
 * `useAuth` hook (task 6.2) and the SDK-client wiring + `withAuthRetry`
 * (task 7.1) build on this surface and are implemented separately.
 *
 * Requirements: 11.1, 11.2, 11.3, 11.4, 12.1, 12.2, 12.3, 12.4.
 */

import {
  AuthenticationDetails,
  CognitoUser,
  CognitoUserPool,
  type CognitoUserSession,
} from "amazon-cognito-identity-js";
import { fromCognitoIdentityPool } from "@aws-sdk/credential-providers";
import type { AwsCredentialIdentity } from "@aws-sdk/types";
import { getAuthConfig, type AuthConfig } from "../config";

/**
 * The subset of Cognito ID-token claims relevant to the cosmetic admin toggle.
 *
 * Structurally identical to `backend/src/lib/authz.ts`'s `AuthClaims`, so the
 * value returned by {@link AuthService.readClaims} can be passed straight to
 * that module's `isAdmin` predicate from the `useAuth` hook (task 6.2). Both
 * fields are optional: a standard or unauthenticated user may carry neither.
 *
 * Reading these claims is purely cosmetic — IAM, not the SPA, is the real
 * authorization boundary (design "Authorization flow").
 */
export interface AuthClaims {
  /** Cognito custom attribute `custom:profile`; `"ADMIN"` designates an admin. */
  "custom:profile"?: string;
  /** Cognito group memberships; administrators belong to the `admin` group. */
  "cognito:groups"?: string[];
}

/** A subscriber notified when the refresh token is dead and re-auth is required. */
export type RequireSignInListener = () => void;

/** Seconds of head-room treated as "about to expire" when reusing STS creds. */
const CREDENTIAL_EXPIRY_BUFFER_MS = 60_000;

/**
 * Manages Cognito authentication and Identity-Pool credential minting for the
 * booth SPA. A single shared instance is exported as {@link authService}.
 */
export class AuthService {
  private readonly config: AuthConfig;
  private readonly userPool: CognitoUserPool;

  /** Last STS credentials minted, reused until near expiry. */
  private cachedCredentials: AwsCredentialIdentity | null = null;

  /** Subscribers to the "refresh token is dead" signal. */
  private readonly requireSignInListeners = new Set<RequireSignInListener>();

  constructor(config: AuthConfig = getAuthConfig()) {
    this.config = config;
    this.userPool = new CognitoUserPool({
      UserPoolId: config.userPoolId,
      ClientId: config.userPoolClientId,
    });
  }

  /**
   * Returns the current Cognito session, **implicitly refreshing** the
   * access/ID token when it has expired but the refresh token is still valid
   * (`amazon-cognito-identity-js` does this transparently inside
   * `getSession`). Returns `null` when there is no signed-in user or the
   * refresh token is dead — the caller must then present the sign-in
   * interface.
   *
   * Requirements: 12.1, 12.3, 12.4.
   */
  async getSession(): Promise<CognitoUserSession | null> {
    const currentUser = this.userPool.getCurrentUser();
    if (!currentUser) {
      return null;
    }

    return new Promise<CognitoUserSession | null>((resolve) => {
      currentUser.getSession(
        (err: Error | null, session: CognitoUserSession | null) => {
          if (err || !session || !session.isValid()) {
            resolve(null);
            return;
          }
          resolve(session);
        }
      );
    });
  }

  /**
   * **Forces** a Cognito token refresh using the stored refresh token, even
   * when the access token still looks valid. Used by reactive retry paths
   * (`withAuthRetry`, task 7.1) after an upstream auth-flavoured failure that
   * the Cognito SDK never observed.
   *
   * Returns the refreshed session, or `null` when there is no current user or
   * the refresh token is no longer valid.
   *
   * Requirements: 12.1, 12.4.
   */
  async refreshSession(): Promise<CognitoUserSession | null> {
    const currentUser = this.userPool.getCurrentUser();
    if (!currentUser) {
      return null;
    }

    // Obtain the current session first so we hold a refresh-token handle.
    const current = await new Promise<CognitoUserSession | null>((resolve) => {
      currentUser.getSession(
        (err: Error | null, session: CognitoUserSession | null) => {
          resolve(err ? null : session);
        }
      );
    });
    if (!current) {
      return null;
    }

    const refreshToken = current.getRefreshToken();
    if (!refreshToken || !refreshToken.getToken()) {
      return null;
    }

    return new Promise<CognitoUserSession | null>((resolve) => {
      currentUser.refreshSession(
        refreshToken,
        (err: Error | null, session: CognitoUserSession | null) => {
          if (err || !session || !session.isValid()) {
            resolve(null);
            return;
          }
          // A fresh token invalidates any STS creds minted from the old one.
          this.cachedCredentials = null;
          resolve(session);
        }
      );
    });
  }

  /**
   * Obtains Identity-Pool (STS) temporary AWS credentials for the role mapped
   * to the supplied session, exchanging the session's ID token via
   * `fromCognitoIdentityPool`. The Cognito group → role mapping configured on
   * the Identity Pool decides whether the `Authenticated_Role` or `Admin_Role`
   * is assumed; this method is agnostic to which.
   *
   * Re-uses the last-minted credentials while they remain comfortably valid,
   * minting fresh ones otherwise. Use {@link refreshCredentials} to force a
   * re-mint regardless of the cache.
   *
   * Requirements: 11.2, 12.2.
   *
   * @throws Error when the Identity Pool declines to vend credentials.
   */
  async getCredentials(
    session: CognitoUserSession
  ): Promise<AwsCredentialIdentity> {
    if (this.cachedCredentials && !this.isExpiring(this.cachedCredentials)) {
      return this.cachedCredentials;
    }
    return this.mintCredentials(session);
  }

  /**
   * **Forces** a re-mint of the Identity-Pool STS credentials, distinct from a
   * Cognito token refresh. First obtains a session (implicitly refreshing the
   * token if needed); if the refresh token is dead it fires {@link requireSignIn}
   * and throws so callers fall back to the sign-in interface.
   *
   * Requirements: 12.2, 12.4.
   *
   * @throws Error (`NotAuthorized`) when no valid session is available, or when
   * the Identity Pool declines to vend credentials.
   */
  async refreshCredentials(): Promise<AwsCredentialIdentity> {
    const session = await this.getSession();
    if (!session) {
      this.requireSignIn();
      throw new Error("NotAuthorized");
    }
    this.cachedCredentials = null;
    return this.mintCredentials(session);
  }

  /**
   * Signs in with username and password, then mints STS credentials so the
   * capture flow can call AWS immediately (Requirement 11.2). Rejects with an
   * authentication error on invalid credentials (Requirement 11.3).
   *
   * @returns the established Cognito session.
   * @throws Error with a message suitable for surfacing as an auth error.
   */
  async signIn(username: string, password: string): Promise<CognitoUserSession> {
    const authDetails = new AuthenticationDetails({
      Username: username,
      Password: password,
    });
    const cognitoUser = new CognitoUser({
      Username: username,
      Pool: this.userPool,
    });

    const session = await new Promise<CognitoUserSession>((resolve, reject) => {
      cognitoUser.authenticateUser(authDetails, {
        onSuccess: (s) => resolve(s),
        onFailure: (err) => reject(err as Error),
        // The capture/effect flow never collects a new password at the kiosk;
        // surface the challenge as an actionable auth error instead.
        newPasswordRequired: () =>
          reject(
            new Error("A password change is required. Contact the operator.")
          ),
      });
    });

    // Vend credentials up-front so access is granted only once IAM creds exist.
    await this.mintCredentials(session);
    return session;
  }

  /**
   * Signs out the current user, discarding both the Cognito session and the
   * cached STS credentials so no privileged state survives the sign-out
   * (Requirement 11.4). The app then returns to the sign-in interface.
   */
  signOut(): void {
    const currentUser = this.userPool.getCurrentUser();
    currentUser?.signOut();
    this.cachedCredentials = null;
  }

  /**
   * Returns whether a user record is present in storage. This is a cheap,
   * synchronous check; it does not validate token expiry — use
   * {@link getSession} when a valid session is required.
   */
  isAuthenticated(): boolean {
    return this.userPool.getCurrentUser() !== null;
  }

  /**
   * Reads the cosmetic-admin-toggle claims (`custom:profile`,
   * `cognito:groups`) from a session's ID token. Pure given the session.
   *
   * Requirements: 14.1, 14.2, 14.3 (claim sourcing for the cosmetic toggle).
   */
  readClaims(session: CognitoUserSession): AuthClaims {
    const payload = session.getIdToken().decodePayload();
    const profile = payload["custom:profile"];
    const groups = payload["cognito:groups"];
    return {
      ...(typeof profile === "string" ? { "custom:profile": profile } : {}),
      ...(Array.isArray(groups)
        ? { "cognito:groups": groups.filter((g): g is string => typeof g === "string") }
        : {}),
    };
  }

  /**
   * Convenience accessor returning the current session's claims, or `null`
   * when no valid session exists. Used by the `useAuth` hook (task 6.2) to
   * drive the cosmetic admin toggle without re-implementing session handling.
   */
  async getClaims(): Promise<AuthClaims | null> {
    const session = await this.getSession();
    return session ? this.readClaims(session) : null;
  }

  /**
   * Returns the current signed-in user's Cognito username, or `null` when not
   * signed in. Admin-console use only (e.g. excluding the current admin from the
   * Users panel) — this is NEVER surfaced in the visitor-facing UI (Req 13.4).
   */
  async getUsername(): Promise<string | null> {
    const session = await this.getSession();
    if (!session) {
      return null;
    }
    const payload = session.getIdToken().decodePayload();
    const username = payload["cognito:username"] ?? payload["username"];
    return typeof username === "string" ? username : null;
  }

  /**
   * Subscribes to the "refresh token is dead" signal. The returned function
   * unsubscribes. The app uses this to route back to the sign-in interface
   * when continued silent refresh is impossible (Requirement 12.4).
   */
  onRequireSignIn(listener: RequireSignInListener): () => void {
    this.requireSignInListeners.add(listener);
    return () => {
      this.requireSignInListeners.delete(listener);
    };
  }

  /**
   * Fires the "refresh token is dead" signal to every subscriber. Invoked
   * internally when a forced credential refresh has no valid session, and
   * available to the reactive retry wrapper (task 7.1) on a second auth
   * failure.
   *
   * Requirements: 11.4, 12.4.
   */
  requireSignIn(): void {
    for (const listener of this.requireSignInListeners) {
      listener();
    }
  }

  /**
   * Mints fresh Identity-Pool credentials from the session's ID token and
   * caches them. Centralizes the `logins` key construction and clientConfig
   * shared by {@link getCredentials} and {@link refreshCredentials}.
   */
  private async mintCredentials(
    session: CognitoUserSession
  ): Promise<AwsCredentialIdentity> {
    const idToken = session.getIdToken().getJwtToken();
    const loginKey = `cognito-idp.${this.config.region}.amazonaws.com/${this.config.userPoolId}`;

    const provider = fromCognitoIdentityPool({
      identityPoolId: this.config.identityPoolId,
      logins: { [loginKey]: idToken },
      clientConfig: { region: this.config.region },
    });

    const credentials = await provider();
    this.cachedCredentials = credentials;
    return credentials;
  }

  /** True when credentials are missing an expiry or are within the buffer of it. */
  private isExpiring(credentials: AwsCredentialIdentity): boolean {
    if (!credentials.expiration) {
      return false;
    }
    return (
      credentials.expiration.getTime() - Date.now() <= CREDENTIAL_EXPIRY_BUFFER_MS
    );
  }
}

/** Shared auth service instance used across the SPA. */
export const authService = new AuthService();

export default authService;
