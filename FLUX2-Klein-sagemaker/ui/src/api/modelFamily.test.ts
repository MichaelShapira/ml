/**
 * Model family selection: which model (FLUX.2 or FLUX 3) the current config
 * serves decides the prompt dialect and sampling of every booth request.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { GetCommand } from "@aws-sdk/lib-dynamodb";

const dynamoSend = vi.fn();
vi.mock("./awsClients", () => ({
  getDynamoClient: () => ({ send: dynamoSend }),
}));
vi.mock("../config", () => ({
  getConfig: () => ({ scheduleTable: "Schedule" }),
}));

import {
  getModelFamily,
  modelFamilyForConfig,
  modelFamilyLabel,
  refreshModelFamily,
  setModelFamilyFromConfig,
} from "./modelFamily";

beforeEach(() => {
  dynamoSend.mockReset();
  setModelFamilyFromConfig(null);
});

describe("modelFamilyForConfig", () => {
  it.each([
    ["FluxAiBoot-flux3-g7e", "flux3"],
    ["booth-FLUX3", "flux3"],
    ["my-flux-3-endpoint", "flux3"],
    ["FluxAiBoot-g6e", "flux2"],
    ["flux2-klein-9b-g6e2", "flux2"],
    ["FluxAiBoot-g7e", "flux2"],
  ])("%s -> %s", (name, family) => {
    expect(modelFamilyForConfig(name)).toBe(family);
  });

  it("treats a missing config as FLUX.2", () => {
    expect(modelFamilyForConfig(null)).toBe("flux2");
    expect(modelFamilyForConfig(undefined)).toBe("flux2");
    expect(modelFamilyForConfig("")).toBe("flux2");
  });

  it("labels both families", () => {
    expect(modelFamilyLabel("flux3")).toBe("FLUX 3 Image Klein");
    expect(modelFamilyLabel("flux2")).toBe("FLUX.2 [klein]");
  });
});

describe("refreshModelFamily", () => {
  it("reads the single current-config pointer item", async () => {
    dynamoSend.mockResolvedValue({ Item: { configName: "FluxAiBoot-flux3-g7e" } });
    await expect(refreshModelFamily()).resolves.toBe("flux3");
    const cmd = dynamoSend.mock.calls[0][0];
    expect(cmd).toBeInstanceOf(GetCommand);
    expect(cmd.input).toMatchObject({
      TableName: "Schedule",
      Key: { pk: "CONFIG#CURRENT", sk: "CONFIG#CURRENT" },
    });
    expect(getModelFamily()).toBe("flux3");
  });

  it("switches back to FLUX.2 when the pointer names a FLUX.2 config", async () => {
    setModelFamilyFromConfig("FluxAiBoot-flux3-g7e");
    dynamoSend.mockResolvedValue({ Item: { configName: "FluxAiBoot-g6e" } });
    await expect(refreshModelFamily()).resolves.toBe("flux2");
  });

  it("keeps the last known family when the read fails or finds nothing", async () => {
    setModelFamilyFromConfig("FluxAiBoot-flux3-g7e");
    dynamoSend.mockRejectedValueOnce(new Error("AccessDeniedException"));
    await expect(refreshModelFamily()).resolves.toBe("flux3");
    dynamoSend.mockResolvedValueOnce({});
    await expect(refreshModelFamily()).resolves.toBe("flux3");
  });
});
