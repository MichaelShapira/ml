/**
 * Tests for the refresh-once-then-retry wrapper and the auth-error heuristic
 * (Requirements 12.1-12.4).
 *
 * - withAuthRetry: on an auth-flavoured failure it refreshes the session + STS
 *   creds once and retries exactly once; a second auth failure fires
 *   requireSignIn; throttling/5xx/network errors are NOT retried.
 * - isAuthError: classifies the auth families and excludes throttling/5xx/net.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const refreshSession = vi.fn();
const refreshCredentials = vi.fn();
const requireSignIn = vi.fn();

vi.mock("../auth/authService", () => ({
  authService: {
    getSession: vi.fn(),
    getCredentials: vi.fn(),
    refreshSession: (...a: unknown[]) => refreshSession(...a),
    refreshCredentials: (...a: unknown[]) => refreshCredentials(...a),
    requireSignIn: (...a: unknown[]) => requireSignIn(...a),
  },
}));

// getConfig isn't needed for these helpers, but the module imports it.
vi.mock("../config", () => ({
  getConfig: () => ({
    region: "us-east-1",
    userPoolId: "p",
    userPoolClientId: "c",
    identityPoolId: "i",
    endpointName: "e",
    endpointConfigName: "ec",
    ioBucket: "b",
    scheduleTable: "t",
  }),
}));

import { withAuthRetry, isAuthError } from "./awsClients";

beforeEach(() => {
  refreshSession.mockReset().mockResolvedValue({});
  refreshCredentials.mockReset().mockResolvedValue({});
  requireSignIn.mockReset();
});

function authErr() {
  return Object.assign(new Error("The security token included in the request is expired"), {
    name: "ExpiredTokenException",
  });
}

describe("isAuthError", () => {
  it("classifies the auth-flavoured families as true", () => {
    expect(isAuthError({ name: "ExpiredToken" })).toBe(true);
    expect(isAuthError({ name: "ExpiredTokenException" })).toBe(true);
    expect(isAuthError({ name: "InvalidClientTokenId" })).toBe(true);
    expect(isAuthError({ name: "NotAuthorizedException" })).toBe(true);
    expect(
      isAuthError({ message: "The security token included in the request is expired" }),
    ).toBe(true);
    expect(isAuthError({ $metadata: { httpStatusCode: 403 } })).toBe(true);
    expect(isAuthError({ $metadata: { httpStatusCode: 401 } })).toBe(true);
  });

  it("excludes throttling, 5xx, and network/abort errors", () => {
    expect(isAuthError({ name: "ThrottlingException" })).toBe(false);
    expect(isAuthError({ name: "TooManyRequestsException" })).toBe(false);
    expect(isAuthError({ $metadata: { httpStatusCode: 500 } })).toBe(false);
    expect(isAuthError({ $metadata: { httpStatusCode: 503 } })).toBe(false);
    expect(isAuthError({ name: "AbortError" })).toBe(false);
    expect(isAuthError({ message: "Failed to fetch" })).toBe(false);
    expect(isAuthError(null)).toBe(false);
    expect(isAuthError("nope")).toBe(false);
  });
});

describe("withAuthRetry", () => {
  it("returns the value with no refresh when the call succeeds", async () => {
    const run = vi.fn().mockResolvedValue("ok");
    await expect(withAuthRetry(run)).resolves.toBe("ok");
    expect(run).toHaveBeenCalledTimes(1);
    expect(refreshSession).not.toHaveBeenCalled();
  });

  it("refreshes once and retries once on an auth-flavoured failure, then succeeds", async () => {
    const run = vi.fn().mockRejectedValueOnce(authErr()).mockResolvedValueOnce("ok");
    await expect(withAuthRetry(run)).resolves.toBe("ok");
    expect(run).toHaveBeenCalledTimes(2);
    expect(refreshSession).toHaveBeenCalledTimes(1);
    expect(refreshCredentials).toHaveBeenCalledTimes(1);
    expect(requireSignIn).not.toHaveBeenCalled();
  });

  it("fires requireSignIn and rethrows when the retry also fails with auth", async () => {
    const run = vi.fn().mockRejectedValue(authErr());
    let caught: unknown;
    try {
      await withAuthRetry(run);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    expect(run).toHaveBeenCalledTimes(2);
    expect(requireSignIn).toHaveBeenCalledTimes(1);
  });

  it("does NOT retry a non-auth error (throttling)", async () => {
    const run = vi.fn().mockRejectedValue(
      Object.assign(new Error("rate exceeded"), { name: "ThrottlingException" }),
    );
    let caught: unknown;
    try {
      await withAuthRetry(run);
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    expect(run).toHaveBeenCalledTimes(1);
    expect(refreshSession).not.toHaveBeenCalled();
  });
});
