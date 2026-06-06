import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mockClient } from "aws-sdk-client-mock";
import {
  CreateEndpointCommand,
  DeleteEndpointCommand,
  DescribeEndpointCommand,
  SageMakerClient,
} from "@aws-sdk/client-sagemaker";
import {
  DynamoDBClient,
  GetItemCommand,
} from "@aws-sdk/client-dynamodb";

import { handler, reconcile, type SchedulerConfig, type WallClock } from "../apply.js";
import { makePk, makeSk } from "../../lib/working-hours.js";
import { BOOTH_SCHEDULE_NAME } from "../../lib/current-config.js";

/**
 * Integration tests for the Scheduler_Function reconcile logic.
 *
 * These mock the SageMaker and DynamoDB SDK v3 clients with `aws-sdk-client-mock`
 * and exercise `reconcile()` (and `handler()`) over fixed wall-clocks, asserting
 * the create/delete/no-op decisions and the exact command shapes.
 *
 * Validates: Requirements 21.2 (read today's Working_Hours), 21.3 (inside
 * window + NOT_DEPLOYED -> CreateEndpoint), 21.4 (outside window + IN_SERVICE ->
 * DeleteEndpoint), 21.5 (desired == actual -> no change).
 */

const sagemakerMock = mockClient(SageMakerClient);
const dynamodbMock = mockClient(DynamoDBClient);

const CONFIG: SchedulerConfig = {
  endpointName: "flux2-klein-9b-g6e2",
  scheduleTable: "Schedule_Store",
  timezone: "America/Los_Angeles",
  endpointConfigName: "flux2-klein-9b-g6e2-config",
};

/** A wall-clock instant whose `time` falls inside the 09:00–17:00 fixture window. */
const NOW_INSIDE: WallClock = { day: "2025-01-15", time: "12:00" };
/** A wall-clock instant whose `time` falls outside the 09:00–17:00 fixture window. */
const NOW_OUTSIDE: WallClock = { day: "2025-01-15", time: "20:00" };

/** Build a DynamoDB `GetItem` response containing a day's Working_Hours item. */
function scheduleItemResponse(day: string, startTime: string, endTime: string) {
  return {
    Item: {
      pk: { S: makePk(BOOTH_SCHEDULE_NAME) },
      sk: { S: makeSk(day) },
      day: { S: day },
      startTime: { S: startTime },
      endTime: { S: endTime },
      updatedBy: { S: "admin" },
      updatedAt: { S: "2025-01-15T00:00:00.000Z" },
    },
  };
}

/** A SageMaker "endpoint does not exist" error (maps to NOT_DEPLOYED). */
function endpointNotFoundError(): Error {
  const err = new Error(
    `Could not find endpoint "${CONFIG.endpointName}".`,
  );
  err.name = "ValidationException";
  return err;
}

/** Fresh, mocked SDK clients for a test. */
function makeClients() {
  return {
    sagemaker: new SageMakerClient({}),
    dynamodb: new DynamoDBClient({}),
  };
}

beforeEach(() => {
  sagemakerMock.reset();
  dynamodbMock.reset();
  // Silence the reconcile decision log so test output stays clean.
  vi.spyOn(console, "info").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("reconcile — inside window + NOT_DEPLOYED → CreateEndpoint (Req 21.3)", () => {
  it("creates the endpoint and does not delete it", async () => {
    dynamodbMock
      .on(GetItemCommand)
      .resolves(scheduleItemResponse(NOW_INSIDE.day, "09:00", "17:00"));
    sagemakerMock.on(DescribeEndpointCommand).rejects(endpointNotFoundError());
    sagemakerMock.on(CreateEndpointCommand).resolves({
      EndpointArn: "arn:aws:sagemaker:us-west-2:111122223333:endpoint/flux2-klein-9b-g6e2",
    });

    const result = await reconcile(makeClients(), CONFIG, NOW_INSIDE);

    expect(result.action).toBe("CREATE");
    expect(result.desiredRunning).toBe(true);
    expect(result.status).toBe("NOT_DEPLOYED");
    expect(result.hasSchedule).toBe(true);

    const createCalls = sagemakerMock.commandCalls(CreateEndpointCommand);
    expect(createCalls).toHaveLength(1);
    expect(createCalls[0].args[0].input).toEqual({
      EndpointName: CONFIG.endpointName,
      EndpointConfigName: CONFIG.endpointConfigName,
      Tags: [{ Key: "AiPhoto", Value: "true" }],
    });

    expect(sagemakerMock.commandCalls(DeleteEndpointCommand)).toHaveLength(0);
  });

  it("reads today's Working_Hours from the Schedule_Store by pk/sk (Req 21.2)", async () => {
    dynamodbMock
      .on(GetItemCommand)
      .resolves(scheduleItemResponse(NOW_INSIDE.day, "09:00", "17:00"));
    sagemakerMock.on(DescribeEndpointCommand).rejects(endpointNotFoundError());
    sagemakerMock.on(CreateEndpointCommand).resolves({});

    await reconcile(makeClients(), CONFIG, NOW_INSIDE);

    const getCalls = dynamodbMock.commandCalls(GetItemCommand);
    // Two GetItems now: the booth-wide schedule for today + the current-config
    // pointer. Assert the schedule read uses the stable booth key.
    expect(getCalls.length).toBeGreaterThanOrEqual(1);
    const scheduleGet = getCalls.find(
      (c) =>
        (c.args[0].input as { Key?: { sk?: { S?: string } } }).Key?.sk?.S ===
        makeSk(NOW_INSIDE.day),
    );
    expect(scheduleGet?.args[0].input).toEqual({
      TableName: CONFIG.scheduleTable,
      Key: {
        pk: { S: makePk(BOOTH_SCHEDULE_NAME) },
        sk: { S: makeSk(NOW_INSIDE.day) },
      },
    });
  });
});

describe("reconcile — outside window + IN_SERVICE → DeleteEndpoint (Req 21.4)", () => {
  it("deletes the endpoint and does not create it", async () => {
    dynamodbMock
      .on(GetItemCommand)
      .resolves(scheduleItemResponse(NOW_OUTSIDE.day, "09:00", "17:00"));
    sagemakerMock
      .on(DescribeEndpointCommand)
      .resolves({ EndpointStatus: "InService" });
    sagemakerMock.on(DeleteEndpointCommand).resolves({});

    const result = await reconcile(makeClients(), CONFIG, NOW_OUTSIDE);

    expect(result.action).toBe("DELETE");
    expect(result.desiredRunning).toBe(false);
    expect(result.status).toBe("IN_SERVICE");

    const deleteCalls = sagemakerMock.commandCalls(DeleteEndpointCommand);
    expect(deleteCalls).toHaveLength(1);
    expect(deleteCalls[0].args[0].input).toEqual({
      EndpointName: CONFIG.endpointName,
    });

    expect(sagemakerMock.commandCalls(CreateEndpointCommand)).toHaveLength(0);
  });

  it("also deletes when there is no schedule item but the endpoint is IN_SERVICE", async () => {
    // No item for today -> desired stopped; an IN_SERVICE endpoint must be torn down.
    dynamodbMock.on(GetItemCommand).resolves({});
    sagemakerMock
      .on(DescribeEndpointCommand)
      .resolves({ EndpointStatus: "InService" });
    sagemakerMock.on(DeleteEndpointCommand).resolves({});

    const result = await reconcile(makeClients(), CONFIG, NOW_INSIDE);

    expect(result.action).toBe("DELETE");
    expect(result.desiredRunning).toBe(false);
    expect(result.hasSchedule).toBe(false);
    expect(sagemakerMock.commandCalls(DeleteEndpointCommand)).toHaveLength(1);
    expect(sagemakerMock.commandCalls(CreateEndpointCommand)).toHaveLength(0);
  });
});

describe("reconcile — idempotent no-ops when desired == actual (Req 21.5)", () => {
  it("does nothing when desired running and the endpoint is already IN_SERVICE", async () => {
    dynamodbMock
      .on(GetItemCommand)
      .resolves(scheduleItemResponse(NOW_INSIDE.day, "09:00", "17:00"));
    sagemakerMock
      .on(DescribeEndpointCommand)
      .resolves({ EndpointStatus: "InService" });

    const result = await reconcile(makeClients(), CONFIG, NOW_INSIDE);

    expect(result.action).toBe("NONE");
    expect(result.desiredRunning).toBe(true);
    expect(result.status).toBe("IN_SERVICE");
    expect(sagemakerMock.commandCalls(CreateEndpointCommand)).toHaveLength(0);
    expect(sagemakerMock.commandCalls(DeleteEndpointCommand)).toHaveLength(0);
  });

  it("does nothing when desired stopped and the endpoint is already NOT_DEPLOYED", async () => {
    dynamodbMock
      .on(GetItemCommand)
      .resolves(scheduleItemResponse(NOW_OUTSIDE.day, "09:00", "17:00"));
    sagemakerMock.on(DescribeEndpointCommand).rejects(endpointNotFoundError());

    const result = await reconcile(makeClients(), CONFIG, NOW_OUTSIDE);

    expect(result.action).toBe("NONE");
    expect(result.desiredRunning).toBe(false);
    expect(result.status).toBe("NOT_DEPLOYED");
    expect(sagemakerMock.commandCalls(CreateEndpointCommand)).toHaveLength(0);
    expect(sagemakerMock.commandCalls(DeleteEndpointCommand)).toHaveLength(0);
  });

  it("does not delete a transitional CREATING endpoint when desired stopped", async () => {
    // Outside the window, but the endpoint is mid-creation. The reconcile only
    // acts on IN_SERVICE for deletes, leaving transitional states untouched so
    // it never thrashes.
    dynamodbMock
      .on(GetItemCommand)
      .resolves(scheduleItemResponse(NOW_OUTSIDE.day, "09:00", "17:00"));
    sagemakerMock
      .on(DescribeEndpointCommand)
      .resolves({ EndpointStatus: "Creating" });

    const result = await reconcile(makeClients(), CONFIG, NOW_OUTSIDE);

    expect(result.action).toBe("NONE");
    expect(result.desiredRunning).toBe(false);
    expect(result.status).toBe("CREATING");
    expect(sagemakerMock.commandCalls(DeleteEndpointCommand)).toHaveLength(0);
    expect(sagemakerMock.commandCalls(CreateEndpointCommand)).toHaveLength(0);
  });
});

describe("reconcile — no schedule item for today (Req 21.2)", () => {
  it("treats a missing item as stopped and is a no-op when NOT_DEPLOYED (never throws)", async () => {
    dynamodbMock.on(GetItemCommand).resolves({});
    sagemakerMock.on(DescribeEndpointCommand).rejects(endpointNotFoundError());

    const result = await reconcile(makeClients(), CONFIG, NOW_INSIDE);

    expect(result.action).toBe("NONE");
    expect(result.hasSchedule).toBe(false);
    expect(result.desiredRunning).toBe(false);
    expect(result.status).toBe("NOT_DEPLOYED");
    expect(sagemakerMock.commandCalls(CreateEndpointCommand)).toHaveLength(0);
    expect(sagemakerMock.commandCalls(DeleteEndpointCommand)).toHaveLength(0);
  });
});

describe("reconcile — acts only on the CURRENT endpoint (config pointer)", () => {
  /** GetItem responses: schedule item for the booth + a current-config pointer. */
  function withCurrentConfig(currentName: string, day: string, start: string, end: string) {
    dynamodbMock.on(GetItemCommand).callsFake((input) => {
      const sk = input.Key?.sk?.S as string | undefined;
      if (sk === "CONFIG#CURRENT") {
        return {
          Item: {
            pk: { S: "CONFIG#CURRENT" },
            sk: { S: "CONFIG#CURRENT" },
            configName: { S: currentName },
          },
        };
      }
      // Otherwise it's the booth schedule lookup.
      return scheduleItemResponse(day, start, end);
    });
  }

  it("creates the CURRENT endpoint (from the pointer), not the env default", async () => {
    withCurrentConfig("other-current-cfg", NOW_INSIDE.day, "09:00", "17:00");
    sagemakerMock.on(DescribeEndpointCommand).rejects(endpointNotFoundError());
    sagemakerMock.on(CreateEndpointCommand).resolves({});

    const result = await reconcile(makeClients(), CONFIG, NOW_INSIDE);

    expect(result.action).toBe("CREATE");
    const createCalls = sagemakerMock.commandCalls(CreateEndpointCommand);
    expect(createCalls).toHaveLength(1);
    // Both the endpoint AND its config are the CURRENT one, not CONFIG.endpointName.
    expect(createCalls[0].args[0].input).toEqual({
      EndpointName: "other-current-cfg",
      EndpointConfigName: "other-current-cfg",
      Tags: [{ Key: "AiPhoto", Value: "true" }],
    });
    // The endpoint that was described is also the current one.
    const describeCalls = sagemakerMock.commandCalls(DescribeEndpointCommand);
    expect(describeCalls[0].args[0].input).toEqual({
      EndpointName: "other-current-cfg",
    });
  });

  it("deletes the CURRENT endpoint (from the pointer) when outside the window", async () => {
    withCurrentConfig("other-current-cfg", NOW_OUTSIDE.day, "09:00", "17:00");
    sagemakerMock.on(DescribeEndpointCommand).resolves({ EndpointStatus: "InService" });
    sagemakerMock.on(DeleteEndpointCommand).resolves({});

    const result = await reconcile(makeClients(), CONFIG, NOW_OUTSIDE);

    expect(result.action).toBe("DELETE");
    const deleteCalls = sagemakerMock.commandCalls(DeleteEndpointCommand);
    expect(deleteCalls).toHaveLength(1);
    expect(deleteCalls[0].args[0].input).toEqual({
      EndpointName: "other-current-cfg",
    });
  });
});

describe("handler — honors the delaySeconds offset before reconciling", () => {
  it("awaits the injected sleep for the offset, then runs reconcile", async () => {
    // Fix the clock to 2025-01-15T12:00:00Z and use UTC so the wall-clock lands
    // inside the 09:00–17:00 window deterministically.
    const fixedNow = new Date("2025-01-15T12:00:00.000Z");
    const utcConfig: SchedulerConfig = { ...CONFIG, timezone: "UTC" };

    dynamodbMock
      .on(GetItemCommand)
      .resolves(scheduleItemResponse("2025-01-15", "09:00", "17:00"));
    sagemakerMock.on(DescribeEndpointCommand).rejects(endpointNotFoundError());
    sagemakerMock.on(CreateEndpointCommand).resolves({});

    const sleep = vi.fn<(ms: number) => Promise<void>>().mockResolvedValue(undefined);

    const result = await handler(
      { delaySeconds: 30 },
      {
        clients: makeClients(),
        config: utcConfig,
        now: () => fixedNow,
        sleep,
      },
    );

    expect(sleep).toHaveBeenCalledTimes(1);
    expect(sleep).toHaveBeenCalledWith(30000);
    expect(result.action).toBe("CREATE");
    expect(result.day).toBe("2025-01-15");
    expect(result.time).toBe("12:00");
    expect(sagemakerMock.commandCalls(CreateEndpointCommand)).toHaveLength(1);
  });

  it("does not sleep when no delaySeconds is provided", async () => {
    const fixedNow = new Date("2025-01-15T20:00:00.000Z");
    const utcConfig: SchedulerConfig = { ...CONFIG, timezone: "UTC" };

    dynamodbMock.on(GetItemCommand).resolves({});
    sagemakerMock.on(DescribeEndpointCommand).rejects(endpointNotFoundError());

    const sleep = vi.fn<(ms: number) => Promise<void>>().mockResolvedValue(undefined);

    const result = await handler(
      {},
      {
        clients: makeClients(),
        config: utcConfig,
        now: () => fixedNow,
        sleep,
      },
    );

    expect(sleep).not.toHaveBeenCalled();
    expect(result.action).toBe("NONE");
  });
});
