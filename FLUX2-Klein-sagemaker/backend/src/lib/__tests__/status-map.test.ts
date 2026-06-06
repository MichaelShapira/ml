import { describe, expect, it } from "vitest";
import {
  EndpointStatus,
  canStart,
  canStop,
  mapStatus,
} from "../status-map.js";

describe("mapStatus", () => {
  it("maps the not-found condition (null/undefined) to NOT_DEPLOYED", () => {
    expect(mapStatus(null)).toBe(EndpointStatus.NOT_DEPLOYED);
    expect(mapStatus(undefined)).toBe(EndpointStatus.NOT_DEPLOYED);
  });

  it("maps only the exact 'InService' status to IN_SERVICE", () => {
    expect(mapStatus("InService")).toBe(EndpointStatus.IN_SERVICE);
  });

  it("never reports availability for differently-cased variants", () => {
    expect(mapStatus("inservice")).not.toBe(EndpointStatus.IN_SERVICE);
    expect(mapStatus("INSERVICE")).not.toBe(EndpointStatus.IN_SERVICE);
    expect(mapStatus(" InService ")).not.toBe(EndpointStatus.IN_SERVICE);
  });

  it("maps creating/updating statuses to CREATING", () => {
    expect(mapStatus("Creating")).toBe(EndpointStatus.CREATING);
    expect(mapStatus("Updating")).toBe(EndpointStatus.CREATING);
    expect(mapStatus("SystemUpdating")).toBe(EndpointStatus.CREATING);
  });

  it("maps Deleting to DELETING", () => {
    expect(mapStatus("Deleting")).toBe(EndpointStatus.DELETING);
  });

  it("maps known failure states to FAILED", () => {
    expect(mapStatus("Failed")).toBe(EndpointStatus.FAILED);
    expect(mapStatus("OutOfService")).toBe(EndpointStatus.FAILED);
    expect(mapStatus("RollbackFailed")).toBe(EndpointStatus.FAILED);
  });

  it("is total: unrecognized strings default to FAILED (never IN_SERVICE)", () => {
    expect(mapStatus("")).toBe(EndpointStatus.FAILED);
    expect(mapStatus("SomethingBrandNew")).toBe(EndpointStatus.FAILED);
  });
});

describe("canStart", () => {
  it("permits start only when NOT_DEPLOYED", () => {
    expect(canStart(EndpointStatus.NOT_DEPLOYED)).toBe(true);
    expect(canStart(EndpointStatus.CREATING)).toBe(false);
    expect(canStart(EndpointStatus.IN_SERVICE)).toBe(false);
    expect(canStart(EndpointStatus.DELETING)).toBe(false);
    expect(canStart(EndpointStatus.FAILED)).toBe(false);
  });
});

describe("canStop", () => {
  it("permits stop for any status except NOT_DEPLOYED", () => {
    expect(canStop(EndpointStatus.NOT_DEPLOYED)).toBe(false);
    expect(canStop(EndpointStatus.CREATING)).toBe(true);
    expect(canStop(EndpointStatus.IN_SERVICE)).toBe(true);
    expect(canStop(EndpointStatus.DELETING)).toBe(true);
    expect(canStop(EndpointStatus.FAILED)).toBe(true);
  });

  it("is the exact complement of canStart across all statuses", () => {
    for (const status of Object.values(EndpointStatus)) {
      expect(canStop(status)).toBe(!canStart(status));
    }
  });
});
