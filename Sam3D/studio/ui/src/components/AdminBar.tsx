import { useEffect, useState, useCallback } from "react";
import { api, type EndpointStatus } from "../api";
import type { Identity } from "../auth";

const LIVE = new Set(["InService"]);
const BUSY = new Set(["Creating", "Updating", "SystemUpdating", "Stopping"]);

export function AdminBar({ identity }: { identity: Identity }) {
  const [status, setStatus] = useState<string>("…");
  const [msg, setMsg] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState(false);

  const refresh = useCallback(async () => {
    setChecking(true);
    try {
      const s: EndpointStatus = await api.endpointStatus();
      setStatus(s.status);
    } catch (e: any) {
      setStatus("unknown");
      setMsg(e?.message || "");
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 15000); // light polling
    return () => clearInterval(t);
  }, [refresh]);

  async function act(fn: () => Promise<EndpointStatus>) {
    setBusy(true); setMsg("");
    try {
      const r = await fn();
      setStatus(r.status);
      setMsg(r.message || "");
    } catch (e: any) {
      setMsg(e?.message || "action failed");
    } finally {
      setBusy(false);
      setTimeout(refresh, 2000);
    }
  }

  const dot =
    LIVE.has(status) ? "bg-green-500" : BUSY.has(status) ? "bg-amber-500" : "bg-zinc-500";

  return (
    <div className="flex items-center gap-3 text-sm bg-panel border border-edge rounded-lg px-3 py-2 flex-wrap">
      <span className="flex items-center gap-2">
        <span className={`inline-block w-2.5 h-2.5 rounded-full ${dot}`} />
        Endpoint: <b>{status}</b>
      </span>
      <button
        className="px-2.5 py-1 rounded-md border border-edge hover:bg-ink disabled:opacity-50"
        disabled={checking}
        onClick={refresh}
        title="Check the endpoint status now"
      >
        {checking ? "Checking…" : "↻ Check status"}
      </button>
      {identity.isAdmin && (
        <>
          <button
            className="px-2.5 py-1 rounded-md border border-edge hover:bg-ink disabled:opacity-50"
            disabled={busy || LIVE.has(status) || BUSY.has(status)}
            onClick={() => act(api.startEndpoint)}
          >
            Start
          </button>
          <button
            className="px-2.5 py-1 rounded-md border border-edge hover:bg-ink disabled:opacity-50"
            disabled={busy || status === "Stopped" || BUSY.has(status)}
            onClick={() => act(api.stopEndpoint)}
          >
            Stop
          </button>
        </>
      )}
      {msg && <span className="text-zinc-400 truncate max-w-[40ch]">{msg}</span>}
    </div>
  );
}
