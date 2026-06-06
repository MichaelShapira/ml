import { describe, it, expect } from "vitest";
import fc from "fast-check";
import { isAdmin, ADMIN_PROFILE, ADMIN_GROUP, type AuthClaims } from "../authz.js";

// Feature: ai-photo-booth, Property 11: Admin tab visibility is equivalent to admin identity — show the Admin tab iff isAdmin(claims) (custom:profile == "ADMIN" or membership in the admin group); non-admin and unauthenticated claims always hide it

// The SPA shows the Admin tab exactly when isAdmin(claims) holds, so the
// reference predicate below mirrors the design's definition of admin identity.
function referenceIsAdmin(claims: AuthClaims | null | undefined): boolean {
  if (claims == null) {
    return false;
  }
  if (claims["custom:profile"] === ADMIN_PROFILE) {
    return true;
  }
  const groups = claims["cognito:groups"];
  return Array.isArray(groups) && groups.includes(ADMIN_GROUP);
}

// Generators that explore admin / non-admin / unauthenticated shapes, including
// look-alike values that must NOT be treated as admin.
const profileArb = fc.option(
  fc.oneof(
    fc.constant(ADMIN_PROFILE),
    fc.constantFrom("admin", "Admin", "USER", "", "ADMINISTRATOR"),
    fc.string(),
  ),
  { nil: undefined },
);

const groupsArb = fc.option(
  fc.array(
    fc.oneof(
      fc.constant(ADMIN_GROUP),
      fc.constantFrom("users", "administrators", "Admin", "ADMIN"),
      fc.string(),
    ),
    { maxLength: 5 },
  ),
  { nil: undefined },
);

const claimsArb: fc.Arbitrary<AuthClaims> = fc.record(
  {
    "custom:profile": profileArb,
    "cognito:groups": groupsArb,
  },
  { requiredKeys: [] },
);

describe("Property 11: admin tab visibility is equivalent to admin identity", () => {
  it("isAdmin holds iff custom:profile == ADMIN or cognito:groups includes admin", () => {
    fc.assert(
      fc.property(claimsArb, (claims) => {
        expect(isAdmin(claims)).toBe(referenceIsAdmin(claims));
      }),
      { numRuns: 100 },
    );
  });

  it("unauthenticated claims (null/undefined) always hide the Admin tab", () => {
    fc.assert(
      fc.property(fc.constantFrom(null, undefined), (claims) => {
        expect(isAdmin(claims)).toBe(false);
      }),
      { numRuns: 100 },
    );
  });
});
