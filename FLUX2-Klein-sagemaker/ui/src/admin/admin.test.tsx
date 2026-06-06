/**
 * Unit tests for the admin UI (Requirements 13.1/13.2, 14.1/14.2, 15.5, 18.2,
 * 19.2/19.3, 20.2/20.3).
 *
 *   - AdminTab renders only when isAdmin (Req 14.1, 14.2);
 *   - EndpointPanel shows "no endpoints available" when empty (Req 15.5) and
 *     requires an explicit confirm before stop (Req 18.2);
 *   - ScheduleCalendar marks days with hours (Req 19.2/20.3) and shows a
 *     validation message for end <= start (Req 19.2/20.2).
 */
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

// --- Mock the api modules consumed by the admin components -------------------
const listAllConfigNames = vi.fn();
const listManagedConfigsWithStatus = vi.fn();
const describeEndpointStatus = vi.fn();
const startEndpoint = vi.fn();
const stopEndpoint = vi.fn();
const makeConfigCurrent = vi.fn();
const addConfig = vi.fn();
const removeConfig = vi.fn();

vi.mock("../api/endpoints", () => ({
  EndpointStatus: {
    NOT_DEPLOYED: "NOT_DEPLOYED",
    CREATING: "CREATING",
    IN_SERVICE: "IN_SERVICE",
    DELETING: "DELETING",
    FAILED: "FAILED",
  },
  EndpointConfigError: class extends Error {},
  statusLabel: (s: string) =>
    ({
      NOT_DEPLOYED: "Not deployed",
      CREATING: "Starting…",
      IN_SERVICE: "Running",
      DELETING: "Stopping…",
      FAILED: "Failed",
    })[s] ?? "Unknown",
  listAllConfigNames: (...a: unknown[]) => listAllConfigNames(...a),
  listManagedConfigsWithStatus: (...a: unknown[]) =>
    listManagedConfigsWithStatus(...a),
  describeEndpointStatus: (...a: unknown[]) => describeEndpointStatus(...a),
  startEndpoint: (...a: unknown[]) => startEndpoint(...a),
  stopEndpoint: (...a: unknown[]) => stopEndpoint(...a),
  makeConfigCurrent: (...a: unknown[]) => makeConfigCurrent(...a),
  addConfig: (...a: unknown[]) => addConfig(...a),
  removeConfig: (...a: unknown[]) => removeConfig(...a),
}));

const listWorkingHours = vi.fn();
const putWorkingHours = vi.fn();
const deleteWorkingHours = vi.fn();

vi.mock("../api/schedule", () => ({
  InvalidWorkingHoursError: class extends Error {},
  listWorkingHours: (...a: unknown[]) => listWorkingHours(...a),
  putWorkingHours: (...a: unknown[]) => putWorkingHours(...a),
  deleteWorkingHours: (...a: unknown[]) => deleteWorkingHours(...a),
}));

// ScheduleCalendar imports validateWorkingHours from api/workingHours (pure).
vi.mock("../api/workingHours", () => ({
  validateWorkingHours: (start: string, end: string) => {
    const m = (t: string) => {
      const x = /^([01]\d|2[0-3]):([0-5]\d)$/.exec(t);
      return x ? Number(x[1]) * 60 + Number(x[2]) : null;
    };
    const s = m(start);
    const e = m(end);
    return s !== null && e !== null && e > s;
  },
}));

// ScheduleCalendar imports the cost API (Cost Explorer). Mock it so tests don't
// instantiate the AWS client chain; return a simple breakdown + month total.
vi.mock("../api/cost", () => ({
  getDailyCostByService: vi.fn().mockResolvedValue({
    day: "2026-01-10",
    components: [{ service: "Amazon SageMaker", amount: 1.23 }],
    total: 1.23,
    currency: "USD",
  }),
  getMonthToDateCost: vi.fn().mockResolvedValue({
    month: "2026-01",
    total: 4.56,
    currency: "USD",
  }),
  formatCost: (amount: number, currency: string) => `${currency} ${amount.toFixed(2)}`,
}));

// EndpointPanel reads the managed endpoint name from runtime config.
vi.mock("../config", () => ({
  getConfig: () => ({
    region: "us-east-1",
    userPoolId: "us-east-1_test",
    userPoolClientId: "c",
    identityPoolId: "i",
    endpointName: "flux2-klein-9b-g6e2",
    endpointConfigName: "flux2-klein-9b-g6e2",
    ioBucket: "io",
    scheduleTable: "Schedule",
    senderEmail: "booth@example.com",
    timezone: "Asia/Jerusalem",
  }),
}));

const listUsers = vi.fn();
const signOutUser = vi.fn();

vi.mock("../api/users", () => ({
  listUsers: (...a: unknown[]) => listUsers(...a),
  signOutUser: (...a: unknown[]) => signOutUser(...a),
}));

// UsersPanel reads the current admin's username to exclude it from the list.
vi.mock("../auth/authService", () => ({
  authService: {
    getUsername: () => Promise.resolve("admin"),
  },
}));

import { AdminTab } from "./AdminTab";
import { EndpointPanel } from "./EndpointPanel";
import { ScheduleCalendar } from "./ScheduleCalendar";
import { UsersPanel } from "./UsersPanel";
import { EndpointStatus } from "../api/endpoints";

beforeEach(() => {
  listAllConfigNames.mockReset().mockResolvedValue(["cfg-a", "cfg-b", "cfg-c"]);
  listManagedConfigsWithStatus.mockReset().mockResolvedValue([]);
  describeEndpointStatus.mockReset().mockResolvedValue(EndpointStatus.IN_SERVICE);
  startEndpoint.mockReset();
  stopEndpoint.mockReset().mockResolvedValue({ ok: true, message: "deleting" });
  makeConfigCurrent.mockReset().mockResolvedValue(undefined);
  addConfig.mockReset().mockResolvedValue(undefined);
  removeConfig.mockReset().mockResolvedValue(undefined);
  listWorkingHours.mockReset().mockResolvedValue([]);
  putWorkingHours.mockReset().mockResolvedValue(undefined);
  deleteWorkingHours.mockReset().mockResolvedValue(undefined);
  listUsers.mockReset().mockResolvedValue([]);
  signOutUser.mockReset().mockResolvedValue(undefined);
});

describe("AdminTab visibility (Req 14.1, 14.2)", () => {
  it("renders nothing for a non-admin", () => {
    const { container } = render(<AdminTab isAdmin={false} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the admin tab for an admin", () => {
    render(<AdminTab isAdmin />);
    expect(screen.getByTestId("admin-tab")).toBeInTheDocument();
  });
});

describe("EndpointPanel (managed-config list + picker)", () => {
  it("adds a config from the picker to the managed list", async () => {
    listManagedConfigsWithStatus.mockResolvedValue([]);
    render(<EndpointPanel />);

    // Empty managed list initially.
    expect(await screen.findByTestId("no-endpoints")).toBeInTheDocument();

    // Pick a config and click Add.
    fireEvent.change(screen.getByTestId("config-picker-input"), {
      target: { value: "cfg-a" },
    });
    fireEvent.click(screen.getByTestId("add-config-button"));

    await waitFor(() => expect(addConfig).toHaveBeenCalledWith("cfg-a"));
  });

  it("rejects adding a config that is not an existing account config", async () => {
    listManagedConfigsWithStatus.mockResolvedValue([]);
    render(<EndpointPanel />);
    await screen.findByTestId("no-endpoints");

    fireEvent.change(screen.getByTestId("config-picker-input"), {
      target: { value: "does-not-exist" },
    });
    fireEvent.click(screen.getByTestId("add-config-button"));

    expect(addConfig).not.toHaveBeenCalled();
    expect(await screen.findByTestId("endpoint-action-message")).toHaveTextContent(
      /not an existing/i,
    );
  });

  it("lists managed configs, badges the current, and offers per-row actions", async () => {
    listManagedConfigsWithStatus.mockResolvedValue([
      { name: "cfg-a", status: EndpointStatus.NOT_DEPLOYED, isCurrent: true },
      { name: "cfg-b", status: EndpointStatus.NOT_DEPLOYED, isCurrent: false },
    ]);
    render(<EndpointPanel />);

    expect(await screen.findByTestId("endpoint-config-cfg-a")).toBeInTheDocument();
    expect(screen.getByTestId("endpoint-config-cfg-b")).toBeInTheDocument();
    expect(screen.getByTestId("current-badge-cfg-a")).toBeInTheDocument();
    expect(screen.queryByTestId("current-badge-cfg-b")).not.toBeInTheDocument();
    // Non-current offers Make current; current does not.
    expect(screen.getByTestId("make-current-cfg-b")).toBeInTheDocument();
    expect(screen.queryByTestId("make-current-cfg-a")).not.toBeInTheDocument();
    // NOT_DEPLOYED → Start offered; each row has Remove.
    expect(screen.getByTestId("start-cfg-a")).toBeInTheDocument();
    expect(screen.getByTestId("remove-cfg-a")).toBeInTheDocument();
  });

  it("makes another config current (only one current at a time)", async () => {
    listManagedConfigsWithStatus.mockResolvedValue([
      { name: "cfg-a", status: EndpointStatus.NOT_DEPLOYED, isCurrent: true },
      { name: "cfg-b", status: EndpointStatus.NOT_DEPLOYED, isCurrent: false },
    ]);
    render(<EndpointPanel />);

    fireEvent.click(await screen.findByTestId("make-current-cfg-b"));
    await waitFor(() => expect(makeConfigCurrent).toHaveBeenCalledWith("cfg-b"));
    expect(await screen.findByTestId("current-badge-cfg-b")).toBeInTheDocument();
    expect(screen.queryByTestId("current-badge-cfg-a")).not.toBeInTheDocument();
  });

  it("requires an explicit confirm before stopping a deployed config's endpoint", async () => {
    listManagedConfigsWithStatus.mockResolvedValue([
      { name: "cfg-a", status: EndpointStatus.IN_SERVICE, isCurrent: true },
    ]);
    render(<EndpointPanel />);

    fireEvent.click(await screen.findByTestId("stop-cfg-a"));
    expect(stopEndpoint).not.toHaveBeenCalled();
    expect(screen.getByTestId("stop-confirm-cfg-a")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("confirm-stop-cfg-a"));
    await waitFor(() => expect(stopEndpoint).toHaveBeenCalledWith("cfg-a"));
  });

  it("removes a config from the managed list", async () => {
    listManagedConfigsWithStatus.mockResolvedValue([
      { name: "cfg-a", status: EndpointStatus.NOT_DEPLOYED, isCurrent: true },
    ]);
    render(<EndpointPanel />);

    fireEvent.click(await screen.findByTestId("remove-cfg-a"));
    await waitFor(() => expect(removeConfig).toHaveBeenCalledWith("cfg-a"));
  });
});

describe("ScheduleCalendar (Req 19.2/19.3, 20.2/20.3)", () => {
  it("marks days that have defined Working_Hours", async () => {
    const today = new Date();
    const day = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-15`;
    listWorkingHours.mockResolvedValue([
      {
        endpointName: "flux2-klein-9b-g6e2",
        day,
        startTime: "09:00",
        endTime: "17:00",
        updatedBy: "admin",
        updatedAt: "2025-01-01T00:00:00Z",
      },
    ]);
    render(<ScheduleCalendar />);
    const cell = await screen.findByTestId(`calendar-day-${day}`);
    expect(cell).toHaveAttribute("data-has-hours", "true");
  });

  it("shows a validation message when end <= start and does not persist", async () => {
    const today = new Date();
    const day = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-10`;
    render(<ScheduleCalendar />);

    fireEvent.click(await screen.findByTestId(`calendar-day-${day}`));
    fireEvent.change(screen.getByTestId("start-time-input"), { target: { value: "17:00" } });
    fireEvent.change(screen.getByTestId("end-time-input"), { target: { value: "09:00" } });
    fireEvent.click(screen.getByTestId("schedule-save-button"));

    expect(await screen.findByTestId("schedule-validation-error")).toBeInTheDocument();
    expect(putWorkingHours).not.toHaveBeenCalled();
  });
});

describe("UsersPanel (admin user management)", () => {
  it("lists other users (excluding the current admin) and force-signs-out behind a confirm", async () => {
    listUsers.mockResolvedValue([
      { username: "admin", status: "CONFIRMED", enabled: true },
      { username: "visitor", status: "CONFIRMED", enabled: true },
    ]);
    render(<UsersPanel />);

    // The current admin is excluded; only other users render.
    expect(await screen.findByText("visitor")).toBeInTheDocument();
    expect(screen.queryByText("admin")).not.toBeInTheDocument();

    // Sign-out requires a confirm step before calling the API.
    fireEvent.click(screen.getByTestId("signout-visitor"));
    expect(signOutUser).not.toHaveBeenCalled();
    expect(screen.getByTestId("signout-confirm-visitor")).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("confirm-signout-visitor"));
    await waitFor(() =>
      expect(signOutUser).toHaveBeenCalledWith("us-east-1_test", "visitor"),
    );
  });

  it("shows an empty message when there are no other users", async () => {
    // Only the current admin exists → after exclusion the list is empty.
    listUsers.mockResolvedValue([
      { username: "admin", status: "CONFIRMED", enabled: true },
    ]);
    render(<UsersPanel />);
    expect(await screen.findByTestId("no-users")).toBeInTheDocument();
  });
});
