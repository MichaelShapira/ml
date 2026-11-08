import { beforeEach, describe, expect, it, vi } from "vitest";
import { InvokeEndpointAsyncCommand, SageMakerRuntimeClient } from "@aws-sdk/client-sagemaker-runtime";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";

import {
  handler,
  ioBucketForRegion,
  parseEndpointRegions,
  resolveConfig,
  type InvokeConfig,
} from "../handler.js";

/**
 * The invoke proxy invokes the current config's endpoint in the config's
 * region, and only accepts an input location in that region's I/O bucket.
 */

const CONFIG: InvokeConfig = {
  endpointName: "flux2-klein-9b-g6e2",
  ioBucket: "io-use1",
  scheduleTable: "Schedule",
  homeRegion: "us-east-1",
  endpointRegions: { "us-east-2": "io-use2" },
};

function dynamoWith(item: Record<string, { S: string }> | undefined): DynamoDBClient {
  return { send: vi.fn(async () => ({ Item: item })) } as unknown as DynamoDBClient;
}

function runtimeClient() {
  const send = vi.fn(async () => ({
    OutputLocation: "s3://bucket/flux2-klein-outputs/x.out",
    FailureLocation: "s3://bucket/flux2-klein-failures/x-error.out",
    InferenceId: "id-1",
  }));
  return { client: { send } as unknown as SageMakerRuntimeClient, send };
}

function event(inputLocation: string) {
  return {
    body: JSON.stringify({ inputLocation, contentType: "application/json" }),
    requestContext: { http: { method: "POST" } },
  };
}

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

describe("invoke proxy endpoint regions", () => {
  it("invokes the endpoint in the pointer's region with that region's bucket", async () => {
    const home = runtimeClient();
    const remote = runtimeClient();
    const runtimeFor = vi.fn(() => remote.client);

    const res = await handler(event("s3://io-use2/flux2-klein-inputs/job.json"), {
      config: CONFIG,
      runtime: home.client,
      runtimeFor,
      dynamodb: dynamoWith({ configName: { S: "FluxAiBoot-flux3-g7e" }, region: { S: "us-east-2" } }),
    });

    expect(res.statusCode).toBe(200);
    expect(runtimeFor).toHaveBeenCalledWith("us-east-2");
    expect(home.send).not.toHaveBeenCalled();
    const cmd = (remote.send.mock.calls[0] as unknown[])[0] as InvokeEndpointAsyncCommand;
    expect(cmd.input).toMatchObject({
      EndpointName: "FluxAiBoot-flux3-g7e",
      InputLocation: "s3://io-use2/flux2-klein-inputs/job.json",
    });
  });

  it("rejects a home-bucket input for a remote endpoint (400, nothing invoked)", async () => {
    const remote = runtimeClient();
    const res = await handler(event("s3://io-use1/flux2-klein-inputs/job.json"), {
      config: CONFIG,
      runtimeFor: () => remote.client,
      dynamodb: dynamoWith({ configName: { S: "FluxAiBoot-flux3-g7e" }, region: { S: "us-east-2" } }),
    });
    expect(res.statusCode).toBe(400);
    expect(remote.send).not.toHaveBeenCalled();
  });

  it("keeps a pointer without a region (or with the home region) on the home endpoint and bucket", async () => {
    const items: Array<Record<string, { S: string }>> = [
      { configName: { S: "FluxAiBoot-g6e" } },
      { configName: { S: "FluxAiBoot-g6e" }, region: { S: "us-east-1" } },
    ];
    for (const item of items) {
      const home = runtimeClient();
      const runtimeFor = vi.fn();
      const ok = await handler(event("s3://io-use1/flux2-klein-inputs/job.json"), {
        config: CONFIG,
        runtime: home.client,
        runtimeFor,
        dynamodb: dynamoWith(item),
      });
      expect(ok.statusCode).toBe(200);
      expect(runtimeFor).not.toHaveBeenCalled();
      const bad = await handler(event("s3://io-use2/flux2-klein-inputs/job.json"), {
        config: CONFIG,
        runtime: home.client,
        runtimeFor,
        dynamodb: dynamoWith(item),
      });
      expect(bad.statusCode).toBe(400);
      expect(home.send).toHaveBeenCalledTimes(1);
    }
  });

  it("refuses a region the stack was not deployed for (400, nothing invoked)", async () => {
    const remote = runtimeClient();
    const runtimeFor = vi.fn(() => remote.client);
    const res = await handler(event("s3://io-euw1/flux2-klein-inputs/job.json"), {
      config: CONFIG,
      runtimeFor,
      dynamodb: dynamoWith({ configName: { S: "cfg" }, region: { S: "eu-west-1" } }),
    });
    expect(res.statusCode).toBe(400);
    expect(JSON.parse(res.body ?? "{}").error).toMatch(/eu-west-1/);
    expect(remote.send).not.toHaveBeenCalled();
  });

  it("parses ENDPOINT_REGIONS and maps regions to buckets", () => {
    expect(parseEndpointRegions('{"us-east-2":"io-use2","bad":3}')).toEqual({ "us-east-2": "io-use2" });
    expect(parseEndpointRegions("[]")).toEqual({});
    expect(parseEndpointRegions("{")).toEqual({});
    expect(ioBucketForRegion(CONFIG, undefined)).toBe("io-use1");
    expect(ioBucketForRegion(CONFIG, "us-east-1")).toBe("io-use1");
    expect(ioBucketForRegion(CONFIG, "us-east-2")).toBe("io-use2");
    expect(() => ioBucketForRegion(CONFIG, "eu-west-1")).toThrow(/eu-west-1/);
    expect(
      resolveConfig({ AWS_REGION: "us-east-1", IO_BUCKET: "io-use1", ENDPOINT_REGIONS: '{"us-east-2":"io-use2"}' }),
    ).toMatchObject({ homeRegion: "us-east-1", ioBucket: "io-use1", endpointRegions: { "us-east-2": "io-use2" } });
  });
});
