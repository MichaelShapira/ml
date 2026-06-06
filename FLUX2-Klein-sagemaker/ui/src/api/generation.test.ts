/**
 * Integration tests for the browser Generation_Service (Requirements 8.1-8.5).
 *
 * Mocks the `awsClients` S3 getter, the `invokeProxy` module, and `config` so
 * the test exercises the real submit/poll logic in `generation.ts`:
 *   - submit PutObjects to the inputs prefix THEN invokes via the proxy with the
 *     matching InputLocation (Req 8.1);
 *   - poll: output present → GetObject + object URL + READY (Req 8.3);
 *     failure present → FAILED (Req 8.4); elapsed > 120 s → TIMEOUT (Req 8.5);
 *     neither + within deadline → PENDING (Req 8.2).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { PutObjectCommand, HeadObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";

// --- Mocks (declared before importing the module under test) -----------------
const s3Send = vi.fn();
const invokeViaProxy = vi.fn();

vi.mock("./awsClients", () => ({
  getS3Client: () => ({ send: s3Send }),
  // Pass-through: the auth retry wrapper just runs the call once here.
  withAuthRetry: <T>(run: () => Promise<T>) => run(),
}));

vi.mock("./invokeProxy", () => ({
  invokeEndpointAsyncViaProxy: (req: unknown) => invokeViaProxy(req),
}));

vi.mock("../config", () => ({
  getConfig: () => ({
    region: "us-east-1",
    userPoolId: "us-east-1_test",
    userPoolClientId: "client",
    identityPoolId: "us-east-1:pool",
    endpointName: "flux2-klein-9b-g6e2",
    endpointConfigName: "flux2-klein-9b-g6e2-config",
    ioBucket: "io-bucket",
    scheduleTable: "Schedule",
    invokeFunctionUrl: "https://fn-url.lambda-url.us-east-1.on.aws/",
  }),
}));

import {
  submitGeneration,
  pollGeneration,
  INPUTS_PREFIX,
  type SubmitResult,
} from "./generation";

// jsdom lacks object URL helpers; stub them.
beforeEach(() => {
  s3Send.mockReset();
  invokeViaProxy.mockReset();
  globalThis.URL.createObjectURL = vi.fn(() => "blob:mock-url");
  globalThis.URL.revokeObjectURL = vi.fn();
});

/** A NotFound error shaped like the S3 SDK's. */
function notFound() {
  return Object.assign(new Error("Not Found"), {
    name: "NotFound",
    $metadata: { httpStatusCode: 404 },
  });
}

describe("submitGeneration (Req 8.1)", () => {
  it("PutObjects to the inputs prefix, then invokes via the proxy with the matching InputLocation", async () => {
    s3Send.mockResolvedValue({});
    invokeViaProxy.mockResolvedValue({
      outputLocation: "s3://io-bucket/flux2-klein-outputs/uuid-1.out",
      failureLocation: "s3://io-bucket/flux2-klein-failures/uuid-1-error.out",
    });

    const result = await submitGeneration({ effectId: "bg_spaceship", photo: "BASE64" });

    expect(result.jobId).toBeTruthy();
    expect(typeof result.submittedAt).toBe("number");
    // The handle carries the SERVER-returned output/failure keys (not derived
    // from the jobId) — these are what polling must HEAD.
    expect(result.outputKey).toBe("flux2-klein-outputs/uuid-1.out");
    expect(result.failureKey).toBe("flux2-klein-failures/uuid-1-error.out");

    // First call: S3 PutObject to inputs/{jobId}.json with the request JSON.
    expect(s3Send).toHaveBeenCalledTimes(1);
    const putCmd = s3Send.mock.calls[0][0];
    expect(putCmd).toBeInstanceOf(PutObjectCommand);
    expect(putCmd.input.Bucket).toBe("io-bucket");
    expect(putCmd.input.Key).toBe(`${INPUTS_PREFIX}${result.jobId}.json`);
    const body = JSON.parse(putCmd.input.Body as string);
    expect(body.images).toEqual(["BASE64"]);
    expect(typeof body.inputs).toBe("string");

    // Then: invoke via the proxy with InputLocation pointing at that object.
    expect(invokeViaProxy).toHaveBeenCalledTimes(1);
    const invokeArg = invokeViaProxy.mock.calls[0][0];
    expect(invokeArg.inputLocation).toBe(
      `s3://io-bucket/${INPUTS_PREFIX}${result.jobId}.json`,
    );
    expect(invokeArg.contentType).toBe("application/json");
  });

  it("throws when the endpoint returns no OutputLocation to poll", async () => {
    s3Send.mockResolvedValue({});
    invokeViaProxy.mockResolvedValue({});
    await expect(
      submitGeneration({ effectId: "bg_spaceship", photo: "BASE64" }),
    ).rejects.toThrow(/OutputLocation/);
  });

  it("rejects a missing photo before any AWS call", async () => {
    await expect(
      submitGeneration({ effectId: "bg_spaceship", photo: "" }),
    ).rejects.toThrow(/photo is required/);
    expect(s3Send).not.toHaveBeenCalled();
    expect(invokeViaProxy).not.toHaveBeenCalled();
  });
});

describe("pollGeneration (Req 8.2-8.5)", () => {
  const handle: SubmitResult = {
    jobId: "job-1",
    submittedAt: Date.now(),
    outputKey: "flux2-klein-outputs/uuid-1.out",
    failureKey: "flux2-klein-failures/uuid-1-error.out",
  };

  it("output present → GetObject then READY with an object URL (Req 8.3)", async () => {
    // HeadObject(output) succeeds; then GetObject(output) returns bytes.
    s3Send.mockImplementation((cmd) => {
      if (cmd instanceof HeadObjectCommand) return Promise.resolve({});
      if (cmd instanceof GetObjectCommand) {
        return Promise.resolve({
          Body: { transformToByteArray: async () => new Uint8Array([1, 2, 3]) },
        });
      }
      return Promise.resolve({});
    });

    const result = await pollGeneration(handle);
    expect(result.status).toBe("READY");
    if (result.status === "READY") {
      expect(result.imageUrl).toBe("blob:mock-url");
      expect(result.aiGenerated).toBe(true);
    }
    expect(globalThis.URL.createObjectURL).toHaveBeenCalled();
  });

  it("failure present (no output) → FAILED (Req 8.4)", async () => {
    let head = 0;
    s3Send.mockImplementation((cmd) => {
      if (cmd instanceof HeadObjectCommand) {
        head += 1;
        // First HEAD is the output key (missing), second is the failure key (present).
        return head === 1 ? Promise.reject(notFound()) : Promise.resolve({});
      }
      return Promise.resolve({});
    });

    const result = await pollGeneration(handle);
    expect(result.status).toBe("FAILED");
  });

  it("neither present + within deadline → PENDING (Req 8.2)", async () => {
    s3Send.mockImplementation((cmd) => {
      if (cmd instanceof HeadObjectCommand) return Promise.reject(notFound());
      return Promise.resolve({});
    });

    const result = await pollGeneration({
      jobId: "job-2",
      submittedAt: Date.now(),
      outputKey: "flux2-klein-outputs/uuid-2.out",
      failureKey: "flux2-klein-failures/uuid-2-error.out",
    });
    expect(result.status).toBe("PENDING");
  });

  it("neither present + elapsed > 120 s → TIMEOUT (Req 8.5)", async () => {
    s3Send.mockImplementation((cmd) => {
      if (cmd instanceof HeadObjectCommand) return Promise.reject(notFound());
      return Promise.resolve({});
    });

    const stale: SubmitResult = {
      jobId: "job-3",
      submittedAt: Date.now() - 121_000,
      outputKey: "flux2-klein-outputs/uuid-3.out",
      failureKey: "flux2-klein-failures/uuid-3-error.out",
    };
    const result = await pollGeneration(stale);
    expect(result.status).toBe("TIMEOUT");
  });
});
