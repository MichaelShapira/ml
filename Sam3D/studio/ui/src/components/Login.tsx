import { useState } from "react";
import { signIn, type Identity } from "../auth";

export function Login({ onSignedIn }: { onSignedIn: (id: Identity) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setError("");
    try {
      onSignedIn(await signIn(username.trim(), password));
    } catch (err: any) {
      setError(err?.message || "Sign-in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="h-full grid place-items-center p-6">
      <form onSubmit={submit} className="w-full max-w-sm bg-panel border border-edge rounded-xl p-6 space-y-4">
        <div>
          <h1 className="text-lg font-semibold">SAM3D Studio</h1>
          <p className="text-sm text-zinc-400">Sign in to continue.</p>
        </div>
        <input
          className="w-full bg-ink border border-edge rounded-md px-3 py-2 text-sm"
          placeholder="Username" autoComplete="username"
          value={username} onChange={(e) => setUsername(e.target.value)}
        />
        <input
          className="w-full bg-ink border border-edge rounded-md px-3 py-2 text-sm"
          placeholder="Password" type="password" autoComplete="current-password"
          value={password} onChange={(e) => setPassword(e.target.value)}
        />
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button
          className="w-full bg-accent text-white font-medium rounded-md px-3 py-2 disabled:opacity-50"
          disabled={busy || !username || !password}
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
