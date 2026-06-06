/**
 * Integration tests for the browser Endpoint_Manager (Requirements 15-18, 14.6).
 *
 * Mocks the SageMaker management client + config and verifies call shapes and
 * the guard/error behaviours:
 *   - list returns names + mapped status; config error vs empty list (15.4/15.5);
 *   - describe maps not-found → NOT_DEPLOYED (16.4);
 *   - start guarded by NOT_DEPLOYED, else "endpoint already exists" (17.3);
 *   - stop guarded by deployed, else "no endpoint to delete" (18.4);
 *   - an AccessDeniedException (standard-user context) surfaces and performs
 *     no further operation (14.6).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  ListEndpointsCommand,
  CreateEndpointCommand,
  DeleteEndpointCommand,
} from "@aws-sdk/client-sagemaker";

const smSend = vi.fn();

vi.mock("./awsClients", () => ({
  getSageMakerClient: () => ({ send: smSend }),
  withAuthRetry: <T>(run: () => Promise<T>) => run(),
}));

vi.mock("../config", () => ({
  getConfig: () => ({
    region: "us-east-1",
    endpointName: "flux2-klein-9b-g6e2",
    endpointConfigName: "flux2-klein-9b-g6e2-config",
    ioBucket: "io-bucket",
    scheduleTable: "Schedule",
    userPoolId: "p",
    userPoolClientId: "c",
    identityPoolId: "i",
  }),
}));

import {
  listEndpoints,
  describeEndpointStatus,
  startEndpoint,
  stopEndpoint,
  EndpointConfigError,
  EndpointStatus,
} from "./endpoints";

beforeEach(() => smSend.mockReset());

function validationException() {
  return Object.assign(new Error("Could not find endpoint"), {
    name: "ValidationException",
  });
}
function accessDenied() {
  return Object.assign(new Error("not authorized to perform sagemaker:ListEndpoints"), {
    name: "AccessDeniedException",
    $metadata: { httpStatusCode: 403 },
  });
}

describe("listEndpoints (Req 15.1, 15.4, 15.5)", () => {
  it("returns names with mapped status", async () => {
    smSend.mockResolvedValue({
      Endpoints: [
        { EndpointName: "a", EndpointStatus: "InService" },
        { EndpointName: "b", EndpointStatus: "Creating" },
      ],
    });
    const result = await listEndpoints();
    expect(smSend.mock.calls[0][0]).toBeInstanceOf(ListEndpointsCommand);
    expect(result).toEqual([
      { name: "a", status: EndpointStatus.IN_SERVICE },
      { name: "b", status: EndpointStatus.CREATING },
    ]);
  });

  it("returns an empty array for no endpoints (distinct from config error)", async () => {
    smSend.mockResolvedValue({ Endpoints: [] });
    await expect(listEndpoints()).resolves.toEqual([]);
  });

  it("throws EndpointConfigError on invalid credentials (Req 15.4)", async () => {
    smSend.mockRejectedValueOnce(
      Object.assign(new Error("The security token included in the request is invalid"), {
        name: "InvalidClientTokenId",
      }),
    );
    let caught: unknown;
    try {
      await listEndpoints();
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(EndpointConfigError);
  });
});

describe("describeEndpointStatus (Req 16.4)", () => {
  it("maps a not-found endpoint to NOT_DEPLOYED", async () => {
    smSend.mockRejectedValueOnce(validationException());
    await expect(describeEndpointStatus("x")).resolves.toBe(EndpointStatus.NOT_DEPLOYED);
  });

  it("maps InService to IN_SERVICE", async () => {
    smSend.mockResolvedValue({ EndpointStatus: "InService" });
    await expect(describeEndpointStatus("x")).resolves.toBe(EndpointStatus.IN_SERVICE);
  });
});

describe("startEndpoint (Req 17.1, 17.3)", () => {
  it("creates the endpoint when NOT_DEPLOYED", async () => {
    smSend
      .mockRejectedValueOnce(validationException()) // describe → NOT_DEPLOYED
      .mockResolvedValueOnce({}); // create
    const result = await startEndpoint("flux2-klein-9b-g6e2");
    expect(result.ok).toBe(true);
    const createCmd = smSend.mock.calls[1][0];
    expect(createCmd).toBeInstanceOf(CreateEndpointCommand);
    expect(createCmd.input.EndpointName).toBe("flux2-klein-9b-g6e2");
    // Endpoint created from the config of the same name.
    expect(createCmd.input.EndpointConfigName).toBe("flux2-klein-9b-g6e2");
    // Tagged with the AiPhoto cost-allocation tag on creation.
    expect(createCmd.input.Tags).toEqual([{ Key: "AiPhoto", Value: "true" }]);
  });

  it("rejects with 'endpoint already exists' when already deployed", async () => {
    smSend.mockResolvedValueOnce({ EndpointStatus: "InService" });
    const result = await startEndpoint("flux2-klein-9b-g6e2");
    expect(result.ok).toBe(false);
    expect(result.message).toMatch(/already exists/i);
    // No CreateEndpoint issued.
    expect(smSend).toHaveBeenCalledTimes(1);
  });
});

describe("stopEndpoint (Req 18.1, 18.4)", () => {
  it("deletes the endpoint when deployed", async () => {
    smSend
      .mockResolvedValueOnce({ EndpointStatus: "InService" }) // describe
      .mockResolvedValueOnce({}); // delete
    const result = await stopEndpoint("flux2-klein-9b-g6e2");
    expect(result.ok).toBe(true);
    expect(smSend.mock.calls[1][0]).toBeInstanceOf(DeleteEndpointCommand);
  });

  it("returns 'no endpoint to delete' when NOT_DEPLOYED", async () => {
    smSend.mockRejectedValueOnce(validationException());
    const result = await stopEndpoint("flux2-klein-9b-g6e2");
    expect(result.ok).toBe(false);
    expect(result.message).toMatch(/no endpoint to delete/i);
    expect(smSend).toHaveBeenCalledTimes(1);
  });
});

describe("IAM denial for a standard user (Req 14.6)", () => {
  it("propagates AccessDeniedException from list and performs no operation", async () => {
    smSend.mockRejectedValueOnce(accessDenied());
    let caught: unknown;
    try {
      await listEndpoints();
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toMatch(/not authorized|access denied/i);
    // Only the single denied call was attempted.
    expect(smSend).toHaveBeenCalledTimes(1);
  });
});
