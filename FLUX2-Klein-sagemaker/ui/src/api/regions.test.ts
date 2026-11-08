/**
 * Endpoint regions: a config in another region keeps its own managed-list key,
 * the current pointer carries the region, every SageMaker call for it goes to
 * that region, and booth requests use that region's I/O bucket.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { DeleteCommand, GetCommand, PutCommand, QueryCommand } from "@aws-sdk/lib-dynamodb";
import { CreateEndpointCommand, DescribeEndpointCommand } from "@aws-sdk/client-sagemaker";
import { HeadObjectCommand, PutObjectCommand } from "@aws-sdk/client-s3";

const docSend = vi.fn();
const smSend = vi.fn();
const s3Send = vi.fn();
const smRegions: string[] = [];
const s3Regions: string[] = [];
const invokeViaProxy = vi.fn();

vi.mock("./awsClients", () => ({
  getDynamoClient: () => ({ send: docSend }),
  getSageMakerClient: (region?: string) => {
    smRegions.push(region ?? "(default)");
    return { send: smSend };
  },
  getS3Client: (region?: string) => {
    s3Regions.push(region ?? "(default)");
    return { send: s3Send };
  },
  withAuthRetry: <T>(run: () => Promise<T>) => run(),
}));
vi.mock("./invokeProxy", () => ({
  invokeEndpointAsyncViaProxy: (req: unknown) => invokeViaProxy(req),
}));
vi.mock("../config", () => ({
  getConfig: () => ({
    region: "us-east-1",
    ioBucket: "io-use1",
    scheduleTable: "Schedule",
    timezone: "Asia/Jerusalem",
    endpointRegions: [
      { region: "us-east-2", ioBucket: "io-use2" },
      { region: "us-east-1", ioBucket: "ignored-duplicate-of-home" },
      { region: "", ioBucket: "ignored-blank" },
    ],
  }),
}));

import {
  configId,
  getEndpointRegions,
  ioBucketForRegion,
  regionOrHome,
  UnknownRegionError,
} from "./regions";
import {
  addManagedConfig,
  getCurrentConfigRef,
  listManagedConfigs,
  managedConfigSk,
  removeManagedConfig,
  setCurrentConfigName,
} from "./schedule";
import { listManagedConfigsWithStatus, startEndpoint, EndpointStatus } from "./endpoints";
import { submitGeneration, pollGeneration } from "./generation";
import { getCurrentEndpointRegion, setModelFamilyFromConfig } from "./modelFamily";

beforeEach(() => {
  docSend.mockReset();
  smSend.mockReset();
  s3Send.mockReset();
  invokeViaProxy.mockReset();
  smRegions.length = 0;
  s3Regions.length = 0;
  setModelFamilyFromConfig(null);
});

describe("regions", () => {
  it("lists home first and drops duplicates and blanks", () => {
    expect(getEndpointRegions()).toEqual([
      { region: "us-east-1", ioBucket: "io-use1", home: true },
      { region: "us-east-2", ioBucket: "io-use2", home: false },
    ]);
  });

  it("maps a region to its bucket, missing to home, unknown to an error", () => {
    expect(ioBucketForRegion("us-east-2")).toBe("io-use2");
    expect(ioBucketForRegion(undefined)).toBe("io-use1");
    expect(regionOrHome("")).toBe("us-east-1");
    expect(() => ioBucketForRegion("eu-west-1")).toThrow(UnknownRegionError);
  });

  it("ids are the bare name at home and name@region elsewhere", () => {
    expect(configId({ name: "cfg", region: "us-east-1" })).toBe("cfg");
    expect(configId({ name: "cfg", region: "us-east-2" })).toBe("cfg@us-east-2");
  });
});

describe("schedule store keys", () => {
  it("keeps the old key at home and adds the region elsewhere", () => {
    expect(managedConfigSk("cfg")).toBe("CONFIG#cfg");
    expect(managedConfigSk("cfg", "us-east-1")).toBe("CONFIG#cfg");
    expect(managedConfigSk("cfg", "us-east-2")).toBe("CONFIG#us-east-2#cfg");
  });

  it("writes and deletes a remote managed config under its region key", async () => {
    docSend.mockResolvedValue({});
    await addManagedConfig("cfg", "us-east-2");
    const put = docSend.mock.calls[0][0] as PutCommand;
    expect(put).toBeInstanceOf(PutCommand);
    expect(put.input.Item).toMatchObject({
      pk: "CONFIG#MANAGED",
      sk: "CONFIG#us-east-2#cfg",
      configName: "cfg",
      region: "us-east-2",
    });
    await removeManagedConfig("cfg", "us-east-2");
    const del = docSend.mock.calls[1][0] as DeleteCommand;
    expect(del.input.Key).toEqual({ pk: "CONFIG#MANAGED", sk: "CONFIG#us-east-2#cfg" });
  });

  it("reads rows without a region as home-region rows", async () => {
    docSend.mockResolvedValue({
      Items: [
        { configName: "old", addedAt: "2026-01-01" },
        { configName: "new", region: "us-east-2", addedAt: "2026-02-01" },
      ],
    });
    const rows = await listManagedConfigs();
    expect((docSend.mock.calls[0][0] as QueryCommand).input.TableName).toBe("Schedule");
    expect(rows).toEqual([
      { configName: "new", region: "us-east-2", addedAt: "2026-02-01" },
      { configName: "old", region: "us-east-1", addedAt: "2026-01-01" },
    ]);
  });

  it("writes the region into the current pointer and reads it back", async () => {
    docSend.mockResolvedValueOnce({});
    await setCurrentConfigName("cfg", "us-east-2");
    expect((docSend.mock.calls[0][0] as PutCommand).input.Item).toMatchObject({
      pk: "CONFIG#CURRENT",
      sk: "CONFIG#CURRENT",
      configName: "cfg",
      region: "us-east-2",
    });
    docSend.mockResolvedValueOnce({ Item: { configName: "cfg", region: "us-east-2" } });
    expect(await getCurrentConfigRef()).toEqual({ name: "cfg", region: "us-east-2" });
    docSend.mockResolvedValueOnce({ Item: { configName: "legacy" } });
    expect(await getCurrentConfigRef()).toEqual({ name: "legacy", region: "us-east-1" });
  });
});

describe("endpoint management per region", () => {
  it("describes each managed config in its own region and matches current by name and region", async () => {
    docSend.mockImplementation(async (cmd: unknown) => {
      if (cmd instanceof QueryCommand) {
        return {
          Items: [
            { configName: "cfg", region: "us-east-1", addedAt: "2026-01-01" },
            { configName: "cfg", region: "us-east-2", addedAt: "2026-02-01" },
          ],
        };
      }
      if (cmd instanceof GetCommand) {
        return { Item: { configName: "cfg", region: "us-east-2" } };
      }
      return {};
    });
    smSend.mockResolvedValue({ EndpointStatus: "InService" });

    const rows = await listManagedConfigsWithStatus();

    expect(rows.map((r) => [r.id, r.region, r.isCurrent])).toEqual([
      ["cfg@us-east-2", "us-east-2", true],
      ["cfg", "us-east-1", false],
    ]);
    expect(smRegions.sort()).toEqual(["us-east-1", "us-east-2"]);
    expect(getCurrentEndpointRegion()).toBe("us-east-2");
  });

  it("creates the endpoint in the config's region", async () => {
    smSend.mockImplementation(async (cmd: unknown) => {
      if (cmd instanceof DescribeEndpointCommand) {
        throw Object.assign(new Error("Could not find endpoint"), { name: "ValidationException" });
      }
      return {};
    });
    docSend.mockResolvedValue({ Items: [] });

    const result = await startEndpoint("cfg", "us-east-2");

    expect(result.ok).toBe(true);
    expect(smRegions).toEqual(["us-east-2", "us-east-2"]);
    const create = smSend.mock.calls.find((c) => c[0] instanceof CreateEndpointCommand)![0] as CreateEndpointCommand;
    expect(create.input).toMatchObject({ EndpointName: "cfg", EndpointConfigName: "cfg" });
    expect(EndpointStatus.NOT_DEPLOYED).toBeDefined();
  });
});

describe("generation per region", () => {
  it("uploads to the current region's bucket and polls the same bucket", async () => {
    docSend.mockResolvedValue({ Item: { configName: "FluxAiBoot-flux3-g7e", region: "us-east-2" } });
    s3Send.mockResolvedValue({});
    invokeViaProxy.mockResolvedValue({
      outputLocation: "s3://io-use2/flux2-klein-outputs/abc.out",
      failureLocation: "s3://io-use2/flux2-klein-failures/abc-error.out",
    });

    const handle = await submitGeneration({ effectId: "bg_lunar", photo: "aGVsbG8=" });

    const put = s3Send.mock.calls[0][0] as PutObjectCommand;
    expect(put.input.Bucket).toBe("io-use2");
    expect(put.input.Key).toMatch(/^flux2-klein-inputs\//);
    const body = JSON.parse(String(put.input.Body)) as { num_inference_steps: number };
    expect(body.num_inference_steps).toBe(16);
    expect(invokeViaProxy).toHaveBeenCalledWith({
      inputLocation: `s3://io-use2/${put.input.Key}`,
      contentType: "application/json",
    });
    expect(handle).toMatchObject({ bucket: "io-use2", region: "us-east-2", outputKey: "flux2-klein-outputs/abc.out" });

    s3Send.mockReset();
    s3Regions.length = 0;
    s3Send.mockRejectedValue(Object.assign(new Error("NotFound"), { $metadata: { httpStatusCode: 404 } }));
    expect(await pollGeneration(handle)).toEqual({ status: "PENDING" });
    const heads = s3Send.mock.calls.map((c) => c[0] as HeadObjectCommand);
    expect(heads.map((h) => h.input.Bucket)).toEqual(["io-use2", "io-use2"]);
    expect(s3Regions).toEqual(["us-east-2", "us-east-2"]);
  });

  it("uses the home bucket for a pointer without a region", async () => {
    docSend.mockResolvedValue({ Item: { configName: "FluxAiBoot-g6e" } });
    s3Send.mockResolvedValue({});
    invokeViaProxy.mockResolvedValue({ outputLocation: "s3://io-use1/flux2-klein-outputs/x.out" });

    const handle = await submitGeneration({ effectId: "bg_lunar", photo: "aGVsbG8=" });

    expect((s3Send.mock.calls[0][0] as PutObjectCommand).input.Bucket).toBe("io-use1");
    expect(handle).toMatchObject({ bucket: "io-use1", region: "us-east-1" });
  });

  it("fails before uploading when the pointer names a region the booth does not know", async () => {
    docSend.mockResolvedValue({ Item: { configName: "cfg", region: "eu-west-1" } });
    await expect(submitGeneration({ effectId: "bg_lunar", photo: "aGVsbG8=" })).rejects.toThrow(
      UnknownRegionError,
    );
    expect(s3Send).not.toHaveBeenCalled();
    expect(invokeViaProxy).not.toHaveBeenCalled();
  });
});
