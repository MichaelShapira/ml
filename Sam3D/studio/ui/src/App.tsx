import { useEffect, useState } from "react";
import { getCurrentIdentity, signOut, type Identity } from "./auth";
import { Login } from "./components/Login";
import { Studio } from "./components/Studio";
import { StudioMobile } from "./components/StudioMobile";
import { config } from "./config";

// True on phone-sized viewports; updates live on resize/rotate.
function useIsMobile(): boolean {
  const [mobile, setMobile] = useState(() =>
    typeof window !== "undefined" && window.matchMedia("(max-width: 768px)").matches
  );
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 768px)");
    const on = () => setMobile(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return mobile;
}

export function App() {
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [loading, setLoading] = useState(true);
  const isMobile = useIsMobile();

  useEffect(() => {
    getCurrentIdentity().then((id) => { setIdentity(id); setLoading(false); });
  }, []);

  if (!config.userPoolId || !config.apiUrl) {
    return (
      <div className="h-full grid place-items-center p-8 text-center">
        <div>
          <h1 className="text-lg font-semibold">Configuration missing</h1>
          <p className="text-sm text-zinc-400 mt-2">
            config.js was not loaded. Deploy with <code>deploy.sh</code>, or set VITE_* env vars for local dev.
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return <div className="h-full grid place-items-center text-zinc-400">Loading…</div>;
  }

  if (!identity) {
    return <Login onSignedIn={setIdentity} />;
  }

  if (isMobile) {
    return <StudioMobile identity={identity} />;
  }

  return (
    <div className="h-full flex flex-col">
      <header className="flex items-center justify-between px-5 py-3 border-b border-edge">
        <div>
          <h1 className="text-base font-semibold">SAM3D Studio</h1>
          <p className="text-xs text-zinc-400">Click an object → 3D model. WebGPU segmentation + SageMaker.</p>
        </div>
        <div className="flex items-center gap-3 text-sm">
          <span className="text-zinc-400">
            {identity.username}{" "}
            <span className={identity.isAdmin ? "text-accent" : "text-zinc-500"}>
              ({identity.role})
            </span>
          </span>
          <button
            className="px-3 py-1.5 rounded-md border border-edge hover:bg-panel"
            onClick={() => { signOut(); setIdentity(null); }}
          >
            Sign out
          </button>
        </div>
      </header>
      <Studio identity={identity} />
    </div>
  );
}
