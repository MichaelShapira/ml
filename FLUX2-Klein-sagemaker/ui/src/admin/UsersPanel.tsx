/**
 * UsersPanel — admin view of user-pool members with a force-sign-out action.
 *
 * Lists users (username + status) and lets an admin globally sign a user out
 * (revokes their refresh tokens). All AWS access is via `api/users.ts` under
 * the Admin_Role; IAM is the real boundary. This is an admin console, so
 * usernames are shown here (distinct from the visitor UI, which never reveals
 * identity).
 */
import { useCallback, useEffect, useState } from "react";
import { KioskScreen, TouchButton } from "../theme";
import { getConfig } from "../config";
import { authService } from "../auth/authService";
import { listUsers, signOutUser, type BoothUser } from "../api/users";

type LoadState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "loaded"; users: BoothUser[] };

/** Friendly labels for the common Cognito user statuses. */
function statusLabel(status: string): string {
  switch (status) {
    case "CONFIRMED":
      return "Active";
    case "FORCE_CHANGE_PASSWORD":
      return "Must set password";
    case "RESET_REQUIRED":
      return "Reset required";
    case "UNCONFIRMED":
      return "Unconfirmed";
    default:
      return status;
  }
}

export function UsersPanel() {
  const [load, setLoad] = useState<LoadState>({ kind: "loading" });
  const [busy, setBusy] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoad({ kind: "loading" });
    setMessage(null);
    try {
      const [users, currentUsername] = await Promise.all([
        listUsers(getConfig().userPoolId),
        authService.getUsername(),
      ]);
      // Exclude the currently signed-in admin: you can't sign yourself out here,
      // and showing yourself is noise.
      const others = currentUsername
        ? users.filter((u) => u.username !== currentUsername)
        : users;
      setLoad({ kind: "loaded", users: others });
    } catch {
      setLoad({ kind: "error", message: "Failed to load users." });
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const onSignOut = useCallback(
    async (username: string) => {
      setBusy(username);
      setConfirming(null);
      setMessage(null);
      try {
        await signOutUser(getConfig().userPoolId, username);
        setMessage(`Signed out ${username}.`);
      } catch {
        setMessage(`Couldn't sign out ${username}.`);
      } finally {
        setBusy(null);
      }
    },
    [],
  );

  return (
    <KioskScreen label="Users" testId="users-panel">
      <div className="users-panel__toolbar">
        <TouchButton
          variant="secondary"
          testId="refresh-users-button"
          onClick={() => void refresh()}
        >
          Refresh
        </TouchButton>
      </div>

      {load.kind === "loading" && <p data-testid="users-loading">Loading users…</p>}

      {load.kind === "error" && (
        <p role="alert" data-testid="users-error">
          {load.message}
        </p>
      )}

      {load.kind === "loaded" && load.users.length === 0 && (
        <p data-testid="no-users">No users found.</p>
      )}

      {load.kind === "loaded" && load.users.length > 0 && (
        <ul className="users-panel__list" data-testid="users-list">
          {load.users.map((u) => (
            <li key={u.username} className="users-panel__row">
              <div className="users-panel__info">
                <span className="users-panel__name">{u.username}</span>
                <span className="users-panel__status">{statusLabel(u.status)}</span>
              </div>
              {confirming === u.username ? (
                <div className="users-panel__confirm" data-testid={`signout-confirm-${u.username}`}>
                  <TouchButton
                    variant="secondary"
                    testId={`cancel-signout-${u.username}`}
                    onClick={() => setConfirming(null)}
                  >
                    Cancel
                  </TouchButton>
                  <TouchButton
                    variant="danger"
                    testId={`confirm-signout-${u.username}`}
                    disabled={busy === u.username}
                    onClick={() => void onSignOut(u.username)}
                  >
                    Confirm
                  </TouchButton>
                </div>
              ) : (
                <TouchButton
                  variant="danger"
                  testId={`signout-${u.username}`}
                  disabled={busy === u.username}
                  onClick={() => setConfirming(u.username)}
                >
                  Sign out
                </TouchButton>
              )}
            </li>
          ))}
        </ul>
      )}

      {message && <p data-testid="users-message">{message}</p>}
    </KioskScreen>
  );
}

export default UsersPanel;
