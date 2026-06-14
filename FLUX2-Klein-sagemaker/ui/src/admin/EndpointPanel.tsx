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

type LoadState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "loaded"; configs: EndpointConfigSummary[] };

export function EndpointPanel() {
  const [load, setLoad] = useState<LoadState>({ kind: "loading" });
  const [allConfigs, setAllConfigs] = useState<string[]>([]);
  const [pick, setPick] = useState("");
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

  /** Load all account config names for the autocomplete picker (best-effort). */
  const refreshAll = useCallback(async () => {
    try {
      setAllConfigs(await listAllConfigNames());
    } catch {
      setAllConfigs([]);
    }
  }, []);

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
  const loadedNamesKey =
    load.kind === "loaded"
      ? load.configs.map((c) => c.name).sort().join("|")
      : "";
  const silentPollingRef = useRef(false);
  useEffect(() => {
    if (!loadedNamesKey) {
      return;
    }
    const names = loadedNamesKey.split("|");
    const tick = async () => {
      if (silentPollingRef.current) {
        return;
      }
      silentPollingRef.current = true;
      try {
        const updates = await Promise.all(
          names.map(async (name) => {
            try {
              return { name, status: await describeEndpointStatus(name) };
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
                  const u = updates.find((x) => x && x.name === c.name);
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
  }, [loadedNamesKey]);

  const refreshOne = useCallback(async (name: string) => {
    try {
      const status = await describeEndpointStatus(name);
      setLoad((prev) =>
        prev.kind === "loaded"
          ? {
              kind: "loaded",
              configs: prev.configs.map((c) =>
                c.name === name ? { ...c, status } : c,
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
    // Only allow adding a config that actually exists in the account.
    if (allConfigs.length > 0 && !allConfigs.includes(name)) {
      setMessage(`“${name}” is not an existing endpoint configuration.`);
      return;
    }
    const already =
      load.kind === "loaded" && load.configs.some((c) => c.name === name);
    if (already) {
      setMessage(`“${name}” is already added.`);
      return;
    }
    setBusyName(name);
    setMessage(null);
    try {
      await addConfig(name);
      setPick("");
      await refresh();
      setMessage(`Added “${name}”.`);
    } finally {
      setBusyName(null);
    }
  }, [pick, allConfigs, load, refresh]);

  const onRemove = useCallback(
    async (name: string) => {
      setBusyName(name);
      setMessage(null);
      try {
        await removeConfig(name);
        await refresh();
        setMessage(`Removed “${name}”.`);
      } finally {
        setBusyName(null);
      }
    },
    [refresh],
  );

  const onStart = useCallback(
    async (name: string) => {
      setBusyName(name);
      setMessage(null);
      try {
        const result = await startEndpoint(name);
        setMessage(result.message);
        await refreshOne(name);
      } finally {
        setBusyName(null);
      }
    },
    [refreshOne],
  );

  const onConfirmStop = useCallback(
    async (name: string) => {
      setBusyName(name);
      setMessage(null);
      setConfirmingStop(null);
      try {
        const result = await stopEndpoint(name);
        setMessage(result.message);
        await refreshOne(name);
      } finally {
        setBusyName(null);
      }
    },
    [refreshOne],
  );

  const onMakeCurrent = useCallback(async (name: string) => {
    setBusyName(name);
    setMessage(null);
    try {
      await makeConfigCurrent(name);
      setLoad((prev) =>
        prev.kind === "loaded"
          ? {
              kind: "loaded",
              configs: prev.configs.map((c) => ({
                ...c,
                isCurrent: c.name === name,
              })),
            }
          : prev,
      );
      setMessage(`“${name}” is now the current endpoint.`);
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
            const busy = busyName === c.name;
            const canStart = c.status === EndpointStatus.NOT_DEPLOYED;
            const canStop =
              c.status !== EndpointStatus.NOT_DEPLOYED && c.status !== null;
            return (
              <li
                key={c.name}
                className={`endpoint-config${c.isCurrent ? " endpoint-config--current" : ""}`}
                data-testid={`endpoint-config-${c.name}`}
                data-current={c.isCurrent ? "true" : "false"}
              >
                <div className="endpoint-config__header">
                  <span className="endpoint-config__name">{c.name}</span>
                  {c.isCurrent && (
                    <span
                      className="endpoint-config__badge"
                      data-testid={`current-badge-${c.name}`}
                    >
                      Current
                    </span>
                  )}
                </div>
                <div className="endpoint-config__status">
                  Status: {statusLabel(c.status)}
                </div>
                <div className="endpoint-config__actions">
                  {!c.isCurrent && (
                    <TouchButton
                      variant="secondary"
                      className="endpoint-config__make-current"
                      testId={`make-current-${c.name}`}
                      disabled={busy}
                      onClick={() => void onMakeCurrent(c.name)}
                    >
                      Make current
                    </TouchButton>
                  )}
                  <div className="endpoint-config__action-row">
                    <TouchButton
                      variant="secondary"
                      testId={`refresh-${c.name}`}
                      disabled={busy}
                      onClick={() => void refreshOne(c.name)}
                    >
                      Refresh
                    </TouchButton>
                    {canStart && (
                      <PrimaryButton
                        testId={`start-${c.name}`}
                        block={false}
                        disabled={busy}
                        onClick={() => void onStart(c.name)}
                      >
                        Start
                      </PrimaryButton>
                    )}
                    {canStop && confirmingStop !== c.name && (
                      <TouchButton
                        variant="danger"
                        testId={`stop-${c.name}`}
                        disabled={busy}
                        onClick={() => setConfirmingStop(c.name)}
                      >
                        Stop
                      </TouchButton>
                    )}
                    <TouchButton
                      variant="secondary"
                      testId={`remove-${c.name}`}
                      disabled={busy}
                      onClick={() => void onRemove(c.name)}
                    >
                      Remove
                    </TouchButton>
                  </div>
                </div>

                {confirmingStop === c.name && (
                  <div
                    className="endpoint-panel__confirm"
                    data-testid={`stop-confirm-${c.name}`}
                  >
                    <p>Delete this endpoint? This stops serving transformations.</p>
                    <TouchButton
                      variant="secondary"
                      testId={`cancel-stop-${c.name}`}
                      onClick={() => setConfirmingStop(null)}
                    >
                      Cancel
                    </TouchButton>
                    <TouchButton
                      variant="danger"
                      testId={`confirm-stop-${c.name}`}
                      disabled={busy}
                      onClick={() => void onConfirmStop(c.name)}
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
