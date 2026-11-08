/**
 * EndpointPanel — admin endpoint-config management UI.
 *
 * Two parts:
 *  1. An autocomplete picker listing ALL the account's SageMaker endpoint
 *     configurations. The admin types/selects one and clicks "Add" to add it to
 *     the booth's curated managed list (persisted in DynamoDB).
 *  2. The curated managed list: each added config is shown as a row with its
 *     live status and per-row actions — Start, Stop (confirm), Make current,
 *     and Remove. Exactly ONE managed config is "current" (used by generation +
 *     the scheduler); selecting another switches it (only one current at a time).
 *
 * When the stack was deployed with extra endpoint regions, a region selector
 * next to the picker lists the configs of the chosen region, and each managed
 * row shows its region. Start/Stop/Status then act on the endpoint in that
 * region, and making it current points the booth (and the scheduler) there.
 *
 * All AWS access is via `api/endpoints.ts` under the Admin_Role; IAM is the
 * real boundary.
 */
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { KioskScreen, TouchButton, PrimaryButton } from "../theme";
import {
  listAllConfigNames,
  listManagedConfigsWithStatus,
  describeEndpointStatus,
  startEndpoint,
  stopEndpoint,
  makeConfigCurrent,
  addConfig,
  removeConfig,
  EndpointConfigError,
  EndpointStatus,
  statusLabel,
  type EndpointConfigSummary,
} from "../api/endpoints";
import { modelFamilyForConfig, modelFamilyLabel } from "../api/modelFamily";
import { configId, getEndpointRegions, homeRegion } from "../api/regions";

/** Display name of a config: the bare name at home, `name (region)` elsewhere. */
function displayName(c: { name: string; region: string }): string {
  return c.region === homeRegion() ? c.name : `${c.name} (${c.region})`;
}

type LoadState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "loaded"; configs: EndpointConfigSummary[] };

export function EndpointPanel() {
  const [load, setLoad] = useState<LoadState>({ kind: "loading" });
  const [allConfigs, setAllConfigs] = useState<string[]>([]);
  const [pick, setPick] = useState("");
  const [regions] = useState(getEndpointRegions);
  const [pickRegion, setPickRegion] = useState<string>(() => homeRegion());
  const [busyName, setBusyName] = useState<string | null>(null);
  const [confirmingStop, setConfirmingStop] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const datalistId = useId();

  /** Load the curated managed list (with status + current). */
  const refresh = useCallback(async () => {
    setLoad({ kind: "loading" });
    try {
      const configs = await listManagedConfigsWithStatus();
      setLoad({ kind: "loaded", configs });
    } catch (err) {
      const m =
        err instanceof EndpointConfigError
          ? "Configuration error: AWS credentials or region are invalid."
          : "Failed to load managed configurations.";
      setLoad({ kind: "error", message: m });
    }
  }, []);

  /** Load all config names of the picker's region for the autocomplete (best-effort). */
  const refreshAll = useCallback(async () => {
    try {
      setAllConfigs(await listAllConfigNames(pickRegion));
    } catch {
      setAllConfigs([]);
    }
  }, [pickRegion]);

  useEffect(() => {
    void refresh();
    void refreshAll();
  }, [refresh, refreshAll]);

  /**
   * Silent background poll: every 10 s, re-fetch the live status of each managed
   * config and patch it into the list IN PLACE — no loading spinner, no status
   * reset, no message changes. Re-arms whenever the set of managed configs
   * changes; cleared on unmount (leaving the panel). A re-entrancy guard skips a
   * tick if the previous one is still in flight.
   */
  // Rows are keyed by `id` (the name at home, `name@region` elsewhere), so the
  // same config name in two regions stays two rows.
  const loadedKey =
    load.kind === "loaded"
      ? JSON.stringify(
          load.configs
            .map((c) => ({ id: c.id, name: c.name, region: c.region }))
            .sort((a, b) => a.id.localeCompare(b.id)),
        )
      : "";
  const silentPollingRef = useRef(false);
  useEffect(() => {
    if (!loadedKey) {
      return;
    }
    const rows = JSON.parse(loadedKey) as Array<{ id: string; name: string; region: string }>;
    const tick = async () => {
      if (silentPollingRef.current) {
        return;
      }
      silentPollingRef.current = true;
      try {
        const updates = await Promise.all(
          rows.map(async (row) => {
            try {
              return { id: row.id, status: await describeEndpointStatus(row.name, row.region) };
            } catch {
              // Leave this row's status unchanged on a transient failure.
              return null;
            }
          }),
        );
        setLoad((prev) =>
          prev.kind === "loaded"
            ? {
                kind: "loaded",
                configs: prev.configs.map((c) => {
                  const u = updates.find((x) => x && x.id === c.id);
                  return u ? { ...c, status: u.status } : c;
                }),
              }
            : prev,
        );
      } finally {
        silentPollingRef.current = false;
      }
    };
    const intervalId = setInterval(() => void tick(), 10_000);
    return () => clearInterval(intervalId);
  }, [loadedKey]);

  const refreshOne = useCallback(async (c: EndpointConfigSummary) => {
    try {
      const status = await describeEndpointStatus(c.name, c.region);
      setLoad((prev) =>
        prev.kind === "loaded"
          ? {
              kind: "loaded",
              configs: prev.configs.map((x) =>
                x.id === c.id ? { ...x, status } : x,
              ),
            }
          : prev,
      );
    } catch {
      /* leave status as-is */
    }
  }, []);

  const onAdd = useCallback(async () => {
    const name = pick.trim();
    if (!name) return;
    const ref = { name, region: pickRegion };
    const label = displayName(ref);
    // Only allow adding a config that actually exists in that region.
    if (allConfigs.length > 0 && !allConfigs.includes(name)) {
      setMessage(`“${label}” is not an existing endpoint configuration.`);
      return;
    }
    const id = configId(ref);
    const already =
      load.kind === "loaded" && load.configs.some((c) => c.id === id);
    if (already) {
      setMessage(`“${label}” is already added.`);
      return;
    }
    setBusyName(id);
    setMessage(null);
    try {
      await addConfig(name, pickRegion);
      setPick("");
      await refresh();
      setMessage(`Added “${label}”.`);
    } finally {
      setBusyName(null);
    }
  }, [pick, pickRegion, allConfigs, load, refresh]);

  const onRemove = useCallback(
    async (c: EndpointConfigSummary) => {
      setBusyName(c.id);
      setMessage(null);
      try {
        await removeConfig(c.name, c.region);
        await refresh();
        setMessage(`Removed “${displayName(c)}”.`);
      } finally {
        setBusyName(null);
      }
    },
    [refresh],
  );

  const onStart = useCallback(
    async (c: EndpointConfigSummary) => {
      setBusyName(c.id);
      setMessage(null);
      try {
        const result = await startEndpoint(c.name, c.region);
        setMessage(result.message);
        await refreshOne(c);
      } finally {
        setBusyName(null);
      }
    },
    [refreshOne],
  );

  const onConfirmStop = useCallback(
    async (c: EndpointConfigSummary) => {
      setBusyName(c.id);
      setMessage(null);
      setConfirmingStop(null);
      try {
        const result = await stopEndpoint(c.name, c.region);
        setMessage(result.message);
        await refreshOne(c);
      } finally {
        setBusyName(null);
      }
    },
    [refreshOne],
  );

  const onMakeCurrent = useCallback(async (c: EndpointConfigSummary) => {
    setBusyName(c.id);
    setMessage(null);
    try {
      await makeConfigCurrent(c.name, c.region);
      setLoad((prev) =>
        prev.kind === "loaded"
          ? {
              kind: "loaded",
              configs: prev.configs.map((x) => ({
                ...x,
                isCurrent: x.id === c.id,
              })),
            }
          : prev,
      );
      setMessage(`“${displayName(c)}” is now the current endpoint.`);
    } finally {
      setBusyName(null);
    }
  }, []);

  return (
    <KioskScreen label="Endpoints" testId="endpoint-panel">
      {/* --- Add-config picker (autocomplete over all account configs) --- */}
      <div className="endpoint-add" data-testid="endpoint-add">
        <label className="endpoint-add__label" htmlFor={`${datalistId}-input`}>
          Add an endpoint configuration
        </label>
        <div className="endpoint-add__row">
          {regions.length > 1 && (
            <select
              className="endpoint-add__region"
              aria-label="Region"
              data-testid="config-region-select"
              value={pickRegion}
              onChange={(e) => {
                setPickRegion(e.target.value);
                setPick("");
                if (message) setMessage(null);
              }}
            >
              {regions.map((r) => (
                <option key={r.region} value={r.region}>
                  {r.home ? `${r.region} (booth)` : r.region}
                </option>
              ))}
            </select>
          )}
          <input
            id={`${datalistId}-input`}
            className="endpoint-add__input"
            list={datalistId}
            placeholder="Select or type a config name…"
            value={pick}
            data-testid="config-picker-input"
            onChange={(e) => {
              setPick(e.target.value);
              if (message) setMessage(null);
            }}
          />
          <datalist id={datalistId} data-testid="config-picker-options">
            {allConfigs.map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
          <PrimaryButton
            testId="add-config-button"
            block={false}
            disabled={!pick.trim() || busyName !== null}
            onClick={() => void onAdd()}
          >
            Add
          </PrimaryButton>
        </div>
      </div>

      {load.kind === "loading" && (
        <p data-testid="endpoints-loading">Loading managed configurations…</p>
      )}

      {load.kind === "error" && (
        <p role="alert" data-testid="endpoints-config-error">
          {load.message}
        </p>
      )}

      {load.kind === "loaded" && load.configs.length === 0 && (
        <p data-testid="no-endpoints">
          No configurations added yet. Pick one above and click Add.
        </p>
      )}

      {load.kind === "loaded" && load.configs.length > 0 && (
        <ul className="endpoint-panel__list" data-testid="endpoint-config-list">
          {load.configs.map((c) => {
            const busy = busyName === c.id;
            const canStart = c.status === EndpointStatus.NOT_DEPLOYED;
            const canStop =
              c.status !== EndpointStatus.NOT_DEPLOYED && c.status !== null;
            return (
              <li
                key={c.id}
                className={`endpoint-config${c.isCurrent ? " endpoint-config--current" : ""}`}
                data-testid={`endpoint-config-${c.id}`}
                data-current={c.isCurrent ? "true" : "false"}
              >
                <div className="endpoint-config__header">
                  <span className="endpoint-config__name">{c.name}</span>
                  {c.isCurrent && (
                    <span
                      className="endpoint-config__badge"
                      data-testid={`current-badge-${c.id}`}
                    >
                      Current
                    </span>
                  )}
                </div>
                <div className="endpoint-config__status">
                  Status: {statusLabel(c.status)}
                </div>
                <div
                  className="endpoint-config__model"
                  data-testid={`model-${c.id}`}
                >
                  Model: {modelFamilyLabel(modelFamilyForConfig(c.name))}
                </div>
                {regions.length > 1 && (
                  <div
                    className="endpoint-config__region"
                    data-testid={`region-${c.id}`}
                  >
                    Region: {c.region}
                    {c.region === homeRegion() ? " (booth)" : ""}
                  </div>
                )}
                <div className="endpoint-config__actions">
                  {!c.isCurrent && (
                    <TouchButton
                      variant="secondary"
                      className="endpoint-config__make-current"
                      testId={`make-current-${c.id}`}
                      disabled={busy}
                      onClick={() => void onMakeCurrent(c)}
                    >
                      Make current
                    </TouchButton>
                  )}
                  <div className="endpoint-config__action-row">
                    <TouchButton
                      variant="secondary"
                      testId={`refresh-${c.id}`}
                      disabled={busy}
                      onClick={() => void refreshOne(c)}
                    >
                      Refresh
                    </TouchButton>
                    {canStart && (
                      <PrimaryButton
                        testId={`start-${c.id}`}
                        block={false}
                        disabled={busy}
                        onClick={() => void onStart(c)}
                      >
                        Start
                      </PrimaryButton>
                    )}
                    {canStop && confirmingStop !== c.id && (
                      <TouchButton
                        variant="danger"
                        testId={`stop-${c.id}`}
                        disabled={busy}
                        onClick={() => setConfirmingStop(c.id)}
                      >
                        Stop
                      </TouchButton>
                    )}
                    <TouchButton
                      variant="secondary"
                      testId={`remove-${c.id}`}
                      disabled={busy}
                      onClick={() => void onRemove(c)}
                    >
                      Remove
                    </TouchButton>
                  </div>
                </div>

                {confirmingStop === c.id && (
                  <div
                    className="endpoint-panel__confirm"
                    data-testid={`stop-confirm-${c.id}`}
                  >
                    <p>Delete this endpoint? This stops serving transformations.</p>
                    <TouchButton
                      variant="secondary"
                      testId={`cancel-stop-${c.id}`}
                      onClick={() => setConfirmingStop(null)}
                    >
                      Cancel
                    </TouchButton>
                    <TouchButton
                      variant="danger"
                      testId={`confirm-stop-${c.id}`}
                      disabled={busy}
                      onClick={() => void onConfirmStop(c)}
                    >
                      Confirm Stop
                    </TouchButton>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {message && <p data-testid="endpoint-action-message">{message}</p>}

      {load.kind !== "loading" && (
        <TouchButton
          variant="secondary"
          testId="endpoints-reload"
          onClick={() => {
            void refresh();
            void refreshAll();
          }}
        >
          Reload
        </TouchButton>
      )}
    </KioskScreen>
  );
}

export default EndpointPanel;
