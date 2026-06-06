/**
 * App — root component: auth gate, admin chrome, booth/admin tabs.
 *
 * Responsibilities (Requirements 11, 13, 14):
 *   - gate the capture flow behind sign-in (`useAuth`); show the sign-in screen
 *     with an auth-error message on invalid credentials (Req 11.1, 11.3);
 *   - support sign-out returning to the sign-in interface (Req 11.4);
 *   - NEVER show a username or identity anywhere (Req 13.1-13.4).
 *
 * Chrome visibility:
 *   - Standard_User → a clean kiosk view: just the capture flow, no top bar,
 *     no tabs, no connection indicator, no sign-out (it's a public kiosk).
 *   - Admin_User → a top bar with the connection indicator, Booth/Admin tabs,
 *     and Sign Out.
 */
import { useState } from "react";
import { useAuth } from "./auth/useAuth";
import { SignInScreen } from "./auth/SignInScreen";
import { BoothFlow } from "./booth/BoothFlow";
import { AdminTab } from "./admin/AdminTab";
import { ConnectionIndicator, TouchButton } from "./theme";

type Tab = "booth" | "admin";

export default function App() {
  const {
    isAuthenticated,
    isAdmin,
    connectionStatus,
    authError,
    signIn,
    signOut,
  } = useAuth();
  const [tab, setTab] = useState<Tab>("booth");

  if (!isAuthenticated) {
    return <SignInScreen onSignIn={signIn} authError={authError} />;
  }

  // Standard visitor: bare kiosk capture flow, no chrome at all.
  if (!isAdmin) {
    return (
      <div className="app-shell" data-testid="app-shell">
        <main className="app-shell__content">
          <BoothFlow isAuthenticated={isAuthenticated} isAdmin={false} />
        </main>
      </div>
    );
  }

  // Admin: full chrome with connection indicator + tabs + sign out.
  return (
    <div className="app-shell" data-testid="app-shell">
      <header className="app-shell__bar">
        <ConnectionIndicator status={connectionStatus} testId="connection-indicator" />
        <nav className="app-shell__tabs">
          <TouchButton
            variant={tab === "booth" ? "primary" : "secondary"}
            testId="booth-tab-button"
            onClick={() => setTab("booth")}
          >
            Booth
          </TouchButton>
          <TouchButton
            variant={tab === "admin" ? "primary" : "secondary"}
            testId="admin-tab-button"
            onClick={() => setTab("admin")}
          >
            Admin
          </TouchButton>
          <TouchButton variant="secondary" testId="sign-out-button" onClick={signOut}>
            Sign Out
          </TouchButton>
        </nav>
      </header>

      <main className="app-shell__content">
        {tab === "admin" ? (
          <AdminTab isAdmin />
        ) : (
          <BoothFlow isAuthenticated={isAuthenticated} isAdmin />
        )}
      </main>
    </div>
  );
}
