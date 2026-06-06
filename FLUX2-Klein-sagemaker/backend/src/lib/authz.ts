/**
 * Admin authorization for the AI Photo Booth (pure logic — no AWS SDK, no I/O).
 *
 * The API Gateway Cognito authorizer validates the JWT at the edge; this module
 * decides whether the already-verified claims belong to an administrator. Admin
 * gating is therefore enforced server-side, never client-only.
 *
 * Requirements: 13.1, 13.2, 13.3, 13.4 — design Property 11.
 */

/** The `custom:profile` attribute value that designates an administrator. */
export const ADMIN_PROFILE = "ADMIN";

/** The Cognito group whose members are administrators. */
export const ADMIN_GROUP = "admin";

/**
 * The subset of Cognito ID-token claims relevant to admin gating.
 *
 * Both fields are optional: an unauthenticated or standard user may carry
 * neither. When present, `cognito:groups` is an array of group names.
 */
export interface AuthClaims {
  /** Cognito custom attribute `custom:profile`; `"ADMIN"` designates an admin. */
  "custom:profile"?: string;
  /** Cognito group memberships; administrators belong to the `admin` group. */
  "cognito:groups"?: string[];
}

/**
 * Returns true iff the supplied claims identify an administrator.
 *
 * A caller is an admin when EITHER `custom:profile` equals `"ADMIN"` OR
 * `cognito:groups` includes `"admin"`. Missing, undefined, empty, or malformed
 * claims are handled gracefully and yield `false` (fail-closed): unauthenticated
 * and standard users are never treated as administrators.
 */
export function isAdmin(claims: AuthClaims | null | undefined): boolean {
  if (claims == null) {
    return false;
  }

  if (claims["custom:profile"] === ADMIN_PROFILE) {
    return true;
  }

  const groups = claims["cognito:groups"];
  return Array.isArray(groups) && groups.includes(ADMIN_GROUP);
}
