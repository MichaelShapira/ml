/**
 * `useAuth` — the React binding over {@link authService} for the booth SPA.
 *
 * This hook is the single surface the app gate (task 9.1) and the admin tab
 * (task 10.x) consume to answer three questions, and nothing more:
 *
 *   - `isAuthenticated` — is a valid Cognito session currently held?
 *   - `isAdmin` — do the session's claims designate an administrator? This is
 *     a **cosmetic** toggle for showing the Admin tab only; IAM, not the SPA,
 *     is the real authorization boundary (design "Authorization flow").
 *   - `connectionStatus` — the two-state kiosk indicator: `"connected"` iff a
 *     valid Cognito session **and** valid STS credentials are held, otherwise
 *     `"disconnected"` (Requirement 13.1–13.3).
 *
 * The hook deliberately **never** exposes the username or any identity
 * (Requirement 13.4): it returns only the booleans, the two-state indicator,
 * and the sign-in/sign-out actions. The signed-in user's name is never read
 * out of the session here and never surfaced to any screen.
 *
 * Connection state is established asynchronously on mount and re-evaluated
 * after sign-in, after sign-out, and whenever {@link authService} signals that
 * the refresh token is dead (`onRequireSignIn`). Until the first evaluation
 * resolves, the hook reports `disconnected` / not-authenticated.
 *
 * Requirements: 11.3, 11.4, 13.1, 13.2, 13.3, 13.4, 14.1, 14.2, 14.3.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { AwsCredentialIdentity } from "@aws-sdk/types";
import { authService, type AuthClaims } from "./authService";

/** The two-state connection indicator value (Requirement 13.1). */
export type ConnectionStatus = "connected" | "disconnected";

/** The `custom:profile` value that designates an administrator. */
const ADMIN_PROFILE = "ADMIN";

/** The Cognito group whose members are administrators. */
const ADMIN_GROUP = "admin";

/**
 * The reactive auth state surfaced to consumers.
 *
 * Note the absence of any `username` / identity field — that omission is
 * intentional and required (Requirement 13.4).
 */
export interface AuthState {
  /** Whether a valid Cognito session is currently held. */
  readonly isAuthenticated: boolean;
  /**
   * Whether the session's claims designate an administrator. Cosmetic only —
   * used to show/hide the Admin tab (Requirement 14.1–14.3); IAM enforces the
   * real boundary.
   */
  readonly isAdmin: boolean;
  /** The two-state connection indicator (Requirement 13.1–13.3). */
  readonly connectionStatus: ConnectionStatus;
  /**
   * A human-readable authentication error to surface after a failed sign-in
   * (Requirement 11.3), or `null` when there is nothing to show.
   */
  readonly authError: string | null;
  /**
   * Sign in with username and password, then re-evaluate connection state. On
   * invalid credentials, sets {@link AuthState.authError} instead of throwing
   * (Requirement 11.3).
   */
  readonly signIn: (username: string, password: string) => Promise<void>;
  /**
   * Sign out, discarding the session and STS credentials and clearing local
   * state so the app returns to the sign-in interface (Requirement 11.4).
   */
  readonly signOut: () => void;
  /** Clears any surfaced {@link AuthState.authError}. */
  readonly clearAuthError: () => void;
}

/** The portion of {@link AuthState} that is derived from async evaluation. */
interface CoreState {
  readonly isAuthenticated: boolean;
  readonly isAdmin: boolean;
  readonly connectionStatus: ConnectionStatus;
}

/** The state held before (and whenever) no valid session/credentials exist. */
const DISCONNECTED: CoreState = {
  isAuthenticated: false,
  isAdmin: false,
  connectionStatus: "disconnected",
};

/**
 * Local re-implementation of `backend/src/lib/authz.ts`'s `isAdmin`, matching
 * its semantics exactly: admin iff `custom:profile === "ADMIN"` OR
 * `cognito:groups` includes `"admin"`. The UI package is type-checked in
 * isolation and cannot import `backend/src` (TS6307), so the tiny predicate is
 * inlined here. Missing/malformed claims fail closed to `false`.
 */
function claimsAreAdmin(claims: AuthClaims | null): boolean {
  if (!claims) {
    return false;
  }
  if (claims["custom:profile"] === ADMIN_PROFILE) {
    return true;
  }
  const groups = claims["cognito:groups"];
  return Array.isArray(groups) && groups.includes(ADMIN_GROUP);
}

/**
 * True iff the supplied STS credentials are usable right now: present access
 * key + secret, and not past their expiry (when an expiry is known). This is
 * the credentials half of the `connected` condition (Requirement 13.2).
 */
function hasValidCredentials(
  credentials: AwsCredentialIdentity | null | undefined
): boolean {
  if (!credentials || !credentials.accessKeyId || !credentials.secretAccessKey) {
    return false;
  }
  if (credentials.expiration && credentials.expiration.getTime() <= Date.now()) {
    return false;
  }
  return true;
}

/**
 * The booth's authentication hook. See {@link AuthState} for the surfaced
 * shape and the module docstring for the connection-status semantics.
 */
export function useAuth(): AuthState {
  const [core, setCore] = useState<CoreState>(DISCONNECTED);
  const [authError, setAuthError] = useState<string | null>(null);

  // Tracks whether the hook is still mounted so async resolutions never call
  // setState after unmount. Set true on each effect run (React StrictMode in
  // development mounts → unmounts → remounts the same instance, preserving
  // refs, so we must re-arm it rather than rely on the initial value).
  const mountedRef = useRef(false);

  // Monotonic sequence so that only the most recently started evaluation may
  // commit its result. Bumping it also invalidates any in-flight evaluation
  // (e.g. when a requireSignIn signal must win over a slow getCredentials).
  const evalSeqRef = useRef(0);

  /**
   * Establish connection state from {@link authService}: read the session,
   * derive the admin flag from its claims, and confirm STS credentials. Only
   * the latest invocation, and only while mounted, commits its result.
   */
  const evaluate = useCallback(async (): Promise<void> => {
    const seq = ++evalSeqRef.current;
    let next: CoreState = DISCONNECTED;

    try {
      const session = await authService.getSession();
      if (session) {
        const isAdmin = claimsAreAdmin(authService.readClaims(session));
        let connected = false;
        try {
          const credentials = await authService.getCredentials(session);
          connected = hasValidCredentials(credentials);
        } catch {
          // No STS credentials -> authenticated but not connected.
          connected = false;
        }
        next = {
          isAuthenticated: true,
          isAdmin,
          connectionStatus: connected ? "connected" : "disconnected",
        };
      }
    } catch {
      next = DISCONNECTED;
    }

    if (mountedRef.current && seq === evalSeqRef.current) {
      setCore(next);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    // When the refresh token is dead, flip to disconnected immediately and
    // invalidate any in-flight evaluation so a late getCredentials cannot
    // re-report "connected" (Requirement 12.4 -> 13.3 / sign-in routing).
    const unsubscribe = authService.onRequireSignIn(() => {
      evalSeqRef.current++;
      if (mountedRef.current) {
        setCore(DISCONNECTED);
      }
    });

    // Establish connection state on mount.
    void evaluate();

    return () => {
      mountedRef.current = false;
      unsubscribe();
    };
  }, [evaluate]);

  const signIn = useCallback(
    async (username: string, password: string): Promise<void> => {
      setAuthError(null);
      try {
        await authService.signIn(username, password);
      } catch (err) {
        // Invalid credentials (or any sign-in failure) surface as a message,
        // never an unhandled throw, so the gate can render it (Requirement 11.3).
        setAuthError(errorMessage(err));
      }
      // Re-derive state whether sign-in succeeded (now connected) or failed
      // (still disconnected), keeping the indicator authoritative.
      await evaluate();
    },
    [evaluate]
  );

  const signOut = useCallback((): void => {
    // Discard the Cognito session and cached STS credentials (Requirement 11.4).
    authService.signOut();
    setAuthError(null);
    // Invalidate any in-flight evaluation and clear state immediately so the
    // app returns to the sign-in interface without waiting on async work.
    evalSeqRef.current++;
    if (mountedRef.current) {
      setCore(DISCONNECTED);
    }
  }, []);

  const clearAuthError = useCallback((): void => {
    setAuthError(null);
  }, []);

  return {
    isAuthenticated: core.isAuthenticated,
    isAdmin: core.isAdmin,
    connectionStatus: core.connectionStatus,
    authError,
    signIn,
    signOut,
    clearAuthError,
  };
}

/** Extracts a safe, user-facing message from an unknown thrown value. */
function errorMessage(err: unknown): string {
  if (err instanceof Error && err.message) {
    return err.message;
  }
  if (typeof err === "string" && err) {
    return err;
  }
  return "Sign-in failed. Check your credentials and try again.";
}

export default useAuth;
