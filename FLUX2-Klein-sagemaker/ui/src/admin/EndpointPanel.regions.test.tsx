/**
 * EndpointPanel with extra endpoint regions: the admin picks a region, adds a
 * config from it, and every row action targets the endpoint in the row's
 * region. The same config name in two regions is two separate rows.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

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
  statusLabel: (s: string) => s,
  EndpointConfigError: class EndpointConfigError extends Error {},
  listAllConfigNames: (...a: unknown[]) => listAllConfigNames(...a),
  listManagedConfigsWithStatus: (...a: unknown[]) => listManagedConfigsWithStatus(...a),
  describeEndpointStatus: (...a: unknown[]) => describeEndpointStatus(...a),
  startEndpoint: (...a: unknown[]) => startEndpoint(...a),
  stopEndpoint: (...a: unknown[]) => stopEndpoint(...a),
  makeConfigCurrent: (...a: unknown[]) => makeConfigCurrent(...a),
  addConfig: (...a: unknown[]) => addConfig(...a),
  removeConfig: (...a: unknown[]) => removeConfig(...a),
}));

vi.mock("../config", () => ({
  getConfig: () => ({
    region: "us-east-1",
    ioBucket: "io-use1",
    scheduleTable: "Schedule",
    endpointRegions: [{ region: "us-east-2", ioBucket: "io-use2" }],
  }),
}));

vi.mock("../auth/authService", () => ({ authService: {} }));

import { EndpointPanel } from "./EndpointPanel";

const NAME = "FluxAiBoot-flux3-g7e";

beforeEach(() => {
  listAllConfigNames.mockReset().mockImplementation(async (region?: string) =>
    region === "us-east-2" ? [NAME] : ["FluxAiBoot-g6e", NAME],
  );
  listManagedConfigsWithStatus.mockReset().mockResolvedValue([]);
  describeEndpointStatus.mockReset().mockResolvedValue("NOT_DEPLOYED");
  startEndpoint.mockReset().mockResolvedValue({ ok: true, message: "creating" });
  stopEndpoint.mockReset().mockResolvedValue({ ok: true, message: "deleting" });
  makeConfigCurrent.mockReset().mockResolvedValue(undefined);
  addConfig.mockReset().mockResolvedValue(undefined);
  removeConfig.mockReset().mockResolvedValue(undefined);
});

describe("EndpointPanel with extra endpoint regions", () => {
  it("lists the region's configs and adds the config in the chosen region", async () => {
    render(<EndpointPanel />);
    await screen.findByTestId("no-endpoints");
    await waitFor(() => expect(listAllConfigNames).toHaveBeenCalledWith("us-east-1"));

    const select = screen.getByTestId("config-region-select") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.value)).toEqual(["us-east-1", "us-east-2"]);
    fireEvent.change(select, { target: { value: "us-east-2" } });
    await waitFor(() => expect(listAllConfigNames).toHaveBeenCalledWith("us-east-2"));

    fireEvent.change(screen.getByTestId("config-picker-input"), { target: { value: NAME } });
    fireEvent.click(screen.getByTestId("add-config-button"));
    await waitFor(() => expect(addConfig).toHaveBeenCalledWith(NAME, "us-east-2"));
  });

  it("refuses a name that does not exist in the chosen region", async () => {
    render(<EndpointPanel />);
    await screen.findByTestId("no-endpoints");
    fireEvent.change(screen.getByTestId("config-region-select"), { target: { value: "us-east-2" } });
    await waitFor(() => expect(listAllConfigNames).toHaveBeenCalledWith("us-east-2"));

    fireEvent.change(screen.getByTestId("config-picker-input"), { target: { value: "FluxAiBoot-g6e" } });
    fireEvent.click(screen.getByTestId("add-config-button"));
    expect(await screen.findByTestId("endpoint-action-message")).toHaveTextContent(
      "“FluxAiBoot-g6e (us-east-2)” is not an existing endpoint configuration.",
    );
    expect(addConfig).not.toHaveBeenCalled();
  });

  it("keeps the same name in two regions apart and acts in each row's region", async () => {
    listManagedConfigsWithStatus.mockResolvedValue([
      { name: NAME, region: "us-east-1", id: NAME, status: "NOT_DEPLOYED", isCurrent: true },
      { name: NAME, region: "us-east-2", id: `${NAME}@us-east-2`, status: "IN_SERVICE", isCurrent: false },
    ]);
    render(<EndpointPanel />);

    const home = await screen.findByTestId(`endpoint-config-${NAME}`);
    const remote = screen.getByTestId(`endpoint-config-${NAME}@us-east-2`);
    expect(within(home).getByTestId(`region-${NAME}`)).toHaveTextContent("Region: us-east-1 (booth)");
    expect(within(remote).getByTestId(`region-${NAME}@us-east-2`)).toHaveTextContent("Region: us-east-2");

    fireEvent.click(screen.getByTestId(`start-${NAME}`));
    await waitFor(() => expect(startEndpoint).toHaveBeenCalledWith(NAME, "us-east-1"));

    fireEvent.click(screen.getByTestId(`make-current-${NAME}@us-east-2`));
    await waitFor(() => expect(makeConfigCurrent).toHaveBeenCalledWith(NAME, "us-east-2"));
    expect(await screen.findByTestId(`current-badge-${NAME}@us-east-2`)).toBeInTheDocument();
    expect(screen.queryByTestId(`current-badge-${NAME}`)).not.toBeInTheDocument();
    expect(await screen.findByTestId("endpoint-action-message")).toHaveTextContent(
      `“${NAME} (us-east-2)” is now the current endpoint.`,
    );

    fireEvent.click(screen.getByTestId(`stop-${NAME}@us-east-2`));
    fireEvent.click(screen.getByTestId(`confirm-stop-${NAME}@us-east-2`));
    await waitFor(() => expect(stopEndpoint).toHaveBeenCalledWith(NAME, "us-east-2"));
    expect(stopEndpoint).toHaveBeenCalledTimes(1);
  });

  it("refuses to add a config twice in the same region", async () => {
    listManagedConfigsWithStatus.mockResolvedValue([
      { name: NAME, region: "us-east-2", id: `${NAME}@us-east-2`, status: "NOT_DEPLOYED", isCurrent: true },
    ]);
    render(<EndpointPanel />);
    await screen.findByTestId(`endpoint-config-${NAME}@us-east-2`);
    fireEvent.change(screen.getByTestId("config-region-select"), { target: { value: "us-east-2" } });
    await waitFor(() => expect(listAllConfigNames).toHaveBeenCalledWith("us-east-2"));
    fireEvent.change(screen.getByTestId("config-picker-input"), { target: { value: NAME } });
    fireEvent.click(screen.getByTestId("add-config-button"));
    expect(await screen.findByTestId("endpoint-action-message")).toHaveTextContent("is already added");
    expect(addConfig).not.toHaveBeenCalled();
  });
});
