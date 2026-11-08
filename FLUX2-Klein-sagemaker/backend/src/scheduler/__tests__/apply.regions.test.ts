import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  CreateEndpointCommand,
  DeleteEndpointCommand,
  DescribeEndpointCommand,
  SageMakerClient,
} from "@aws-sdk/client-sagemaker";
import { DynamoDBClient, GetItemCommand } from "@aws-sdk/client-dynamodb";

import { parseExtraRegions, reconcile, resolveConfig, type SchedulerConfig } from "../apply.js";

/**
 * The scheduler acts on the current config's endpoint in the config's region:
 * the pointer's `region` picks the SageMaker client, a pointer without one
 * stays in the home region, and a region outside ENDPOINT_REGIONS is refused.
 */

const CONFIG: SchedulerConfig = {
  endpointName: "flux2-klein-9b-g6e2",
  scheduleTable: "Schedule_Store",
  timezone: "Asia/Jerusalem",
  endpointConfigName: "flux2-klein-9b-g6e2",
  homeRegion: "us-east-1",
  extraRegions: ["us-east-2"],
};

/** A DynamoDB stub: today's window is 09:00-17:00 and the pointer is `pointer`. */
function dynamoWith(pointer: Record<string, { S: string }> | undefined) {
  return {
    send: vi.fn(async (cmd: GetItemCommand) => {
      if (cmd.input.Key?.pk?.S === "CONFIG#CURRENT") {
        return { Item: pointer };
      }
      return { Item: { startTime: { S: "09:00" }, endTime: { S: "17:00" } } };
    }),
  } as unknown as DynamoDBClient;
}

/** A SageMaker stub reporting `status` (or not-found) and recording calls. */
function sageMakerReporting(status: string | null) {
  const send = vi.fn(async (cmd: unknown) => {
    if (cmd instanceof DescribeEndpointCommand) {
      if (status === null) {
        throw Object.assign(new Error("Could not find endpoint"), { name: "ValidationException" });
      }
      return { EndpointStatus: status };
    }
    return {};
  });
  return { client: { send } as unknown as SageMakerClient, send };
}

beforeEach(() => {
  vi.spyOn(console, "info").mockImplementation(() => undefined);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

describe("scheduler endpoint regions", () => {
  it("creates the endpoint in the pointer's region inside the window", async () => {
    const home = sageMakerReporting(null);
    const remote = sageMakerReporting(null);
    const sagemakerFor = vi.fn(() => remote.client);

    const result = await reconcile(
      {
        sagemaker: home.client,
        dynamodb: dynamoWith({ configName: { S: "FluxAiBoot-flux3-g7e" }, region: { S: "us-east-2" } }),
        sagemakerFor,
      },
      CONFIG,
      { day: "2026-11-08", time: "12:00" },
    );

    expect(result).toMatchObject({ action: "CREATE", region: "us-east-2" });
    expect(sagemakerFor).toHaveBeenCalledWith("us-east-2");
    expect(home.send).not.toHaveBeenCalled();
    const create = remote.send.mock.calls.find((c) => c[0] instanceof CreateEndpointCommand)![0] as CreateEndpointCommand;
    expect(create.input).toMatchObject({
      EndpointName: "FluxAiBoot-flux3-g7e",
      EndpointConfigName: "FluxAiBoot-flux3-g7e",
    });
  });

  it("deletes the endpoint in the pointer's region outside the window", async () => {
    const home = sageMakerReporting("InService");
    const remote = sageMakerReporting("InService");

    const result = await reconcile(
      {
        sagemaker: home.client,
        dynamodb: dynamoWith({ configName: { S: "FluxAiBoot-flux3-g7e" }, region: { S: "us-east-2" } }),
        sagemakerFor: () => remote.client,
      },
      CONFIG,
      { day: "2026-11-08", time: "20:00" },
    );

    expect(result).toMatchObject({ action: "DELETE", region: "us-east-2" });
    expect(remote.send.mock.calls.some((c) => c[0] instanceof DeleteEndpointCommand)).toBe(true);
    expect(home.send).not.toHaveBeenCalled();
  });

  it("stays in the home region for a pointer without a region or with the home region", async () => {
    const pointers: Array<Record<string, { S: string }>> = [
      { configName: { S: "FluxAiBoot-g6e" } },
      { configName: { S: "FluxAiBoot-g6e" }, region: { S: "us-east-1" } },
    ];
    for (const pointer of pointers) {
      const home = sageMakerReporting("InService");
      const sagemakerFor = vi.fn();
      const result = await reconcile(
        { sagemaker: home.client, dynamodb: dynamoWith(pointer), sagemakerFor },
        CONFIG,
        { day: "2026-11-08", time: "12:00" },
      );
      expect(result.action).toBe("NONE");
      expect(result.region).toBeUndefined();
      expect(sagemakerFor).not.toHaveBeenCalled();
      expect(home.send).toHaveBeenCalledTimes(1);
    }
  });

  it("refuses a region the stack was not deployed for", async () => {
    const home = sageMakerReporting(null);
    const sagemakerFor = vi.fn();

    const result = await reconcile(
      {
        sagemaker: home.client,
        dynamodb: dynamoWith({ configName: { S: "cfg" }, region: { S: "eu-west-1" } }),
        sagemakerFor,
      },
      CONFIG,
      { day: "2026-11-08", time: "12:00" },
    );

    expect(result).toMatchObject({ action: "NONE", region: "eu-west-1" });
    expect(sagemakerFor).not.toHaveBeenCalled();
    expect(home.send).not.toHaveBeenCalled();
    expect(console.error).toHaveBeenCalled();
  });

  it("reads the extra regions from ENDPOINT_REGIONS", () => {
    expect(parseExtraRegions('{"us-east-2":"b2","us-west-2":"b3"}')).toEqual(["us-east-2", "us-west-2"]);
    expect(parseExtraRegions("not json")).toEqual([]);
    expect(parseExtraRegions('["us-east-2"]')).toEqual([]);
    expect(parseExtraRegions(undefined)).toEqual([]);
    expect(
      resolveConfig({ AWS_REGION: "us-east-1", ENDPOINT_REGIONS: '{"us-east-2":"b2"}' }),
    ).toMatchObject({ homeRegion: "us-east-1", extraRegions: ["us-east-2"] });
  });
});
