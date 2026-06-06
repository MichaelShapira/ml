/**
 * Integration tests for the browser schedule CRUD (Requirements 20.1-20.5).
 *
 * Mocks the DynamoDB Document client + config and verifies:
 *   - an invalid entry (end <= start) is rejected BEFORE any PutItem (Req 20.2);
 *   - a valid entry is persisted with the correct key shape (Req 20.1);
 *   - delete removes the day's item (Req 20.4);
 *   - reopening (list/Query) returns the persisted hours (Req 20.5).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryCommand, PutCommand, DeleteCommand } from "@aws-sdk/lib-dynamodb";

const docSend = vi.fn();

vi.mock("./awsClients", () => ({
  getDynamoClient: () => ({ send: docSend }),
  withAuthRetry: <T>(run: () => Promise<T>) => run(),
}));

vi.mock("../config", () => ({
  getConfig: () => ({
    region: "us-east-1",
    endpointName: "flux2-klein-9b-g6e2",
    endpointConfigName: "cfg",
    ioBucket: "io",
    scheduleTable: "Schedule",
    userPoolId: "p",
    userPoolClientId: "c",
    identityPoolId: "i",
  }),
}));

import {
  listWorkingHours,
  putWorkingHours,
  deleteWorkingHours,
  InvalidWorkingHoursError,
} from "./schedule";

beforeEach(() => docSend.mockReset());

describe("putWorkingHours (Req 20.1, 20.2)", () => {
  it("rejects an invalid entry (end <= start) before any PutItem", async () => {
    await expect(
      putWorkingHours({ day: "2025-06-14", startTime: "17:00", endTime: "09:00" }),
    ).rejects.toBeInstanceOf(InvalidWorkingHoursError);
    expect(docSend).not.toHaveBeenCalled();
  });

  it("persists a valid entry with the ENDPOINT#/DAY# key shape", async () => {
    docSend.mockResolvedValue({});
    await putWorkingHours({ day: "2025-06-14", startTime: "09:00", endTime: "17:30" });

    expect(docSend).toHaveBeenCalledTimes(1);
    const cmd = docSend.mock.calls[0][0];
    expect(cmd).toBeInstanceOf(PutCommand);
    expect(cmd.input.TableName).toBe("Schedule");
    expect(cmd.input.Item.pk).toBe("ENDPOINT#booth");
    expect(cmd.input.Item.sk).toBe("DAY#2025-06-14");
    expect(cmd.input.Item.startTime).toBe("09:00");
    expect(cmd.input.Item.endTime).toBe("17:30");
  });
});

describe("deleteWorkingHours (Req 20.4)", () => {
  it("deletes the day's item by key", async () => {
    docSend.mockResolvedValue({});
    await deleteWorkingHours("2025-06-14");
    const cmd = docSend.mock.calls[0][0];
    expect(cmd).toBeInstanceOf(DeleteCommand);
    expect(cmd.input.Key).toEqual({
      pk: "ENDPOINT#booth",
      sk: "DAY#2025-06-14",
    });
  });
});

describe("listWorkingHours (Req 20.5)", () => {
  it("queries by pk and returns the persisted hours", async () => {
    docSend.mockResolvedValue({
      Items: [
        {
          pk: "ENDPOINT#flux2-klein-9b-g6e2",
          sk: "DAY#2025-06-14",
          day: "2025-06-14",
          startTime: "09:00",
          endTime: "17:30",
          updatedBy: "admin",
          updatedAt: "2025-06-01T12:00:00Z",
        },
      ],
    });

    const result = await listWorkingHours();
    const cmd = docSend.mock.calls[0][0];
    expect(cmd).toBeInstanceOf(QueryCommand);
    expect(cmd.input.ExpressionAttributeValues).toEqual({
      ":pk": "ENDPOINT#booth",
    });
    expect(result).toEqual([
      {
        endpointName: "booth",
        day: "2025-06-14",
        startTime: "09:00",
        endTime: "17:30",
        updatedBy: "admin",
        updatedAt: "2025-06-01T12:00:00Z",
      },
    ]);
  });
});
