import { describe, it, expect } from "vitest";
import { isAdmin, ADMIN_PROFILE, ADMIN_GROUP, type AuthClaims } from "../authz.js";

describe("isAdmin", () => {
  it("returns true when custom:profile equals ADMIN", () => {
    expect(isAdmin({ "custom:profile": ADMIN_PROFILE })).toBe(true);
  });

  it("returns true when cognito:groups includes admin", () => {
    expect(isAdmin({ "cognito:groups": ["users", ADMIN_GROUP] })).toBe(true);
  });

  it("returns true when either signal alone is present", () => {
    expect(isAdmin({ "custom:profile": "ADMIN", "cognito:groups": [] })).toBe(true);
    expect(isAdmin({ "custom:profile": "USER", "cognito:groups": ["admin"] })).toBe(true);
  });

  it("returns false for a standard user (no admin signals)", () => {
    expect(isAdmin({ "custom:profile": "USER", "cognito:groups": ["users"] })).toBe(false);
  });

  it("returns false for empty claims (unauthenticated)", () => {
    expect(isAdmin({})).toBe(false);
  });

  it("handles missing/undefined claims fields without throwing", () => {
    expect(isAdmin(undefined)).toBe(false);
    expect(isAdmin(null)).toBe(false);
    expect(isAdmin({ "custom:profile": undefined, "cognito:groups": undefined })).toBe(false);
  });

  it("is case-sensitive on the profile value", () => {
    expect(isAdmin({ "custom:profile": "admin" })).toBe(false);
    expect(isAdmin({ "custom:profile": "Admin" })).toBe(false);
  });

  it("does not treat a non-admin group of similar name as admin", () => {
    expect(isAdmin({ "cognito:groups": ["administrators", "ADMIN"] })).toBe(false);
  });

  it("tolerates a non-array cognito:groups value", () => {
    const malformed = { "cognito:groups": "admin" } as unknown as AuthClaims;
    expect(isAdmin(malformed)).toBe(false);
  });
});
