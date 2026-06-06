/**
 * AdminTab — the admin-only tab (Requirements 13.1, 13.2, 14.1, 14.2).
 *
 * Renders the endpoint panel, the schedule calendar, and the users panel. The
 * component renders `null` unless `isAdmin` is true, so it is hidden for
 * standard/unauthenticated users. This hiding is cosmetic — IAM (via the
 * Admin_Role mapping) is the real authorization boundary; a non-admin reaching
 * an admin AWS call is denied by IAM regardless of the UI.
 */
import { useState } from "react";
import { EndpointPanel } from "./EndpointPanel";
import { ScheduleCalendar } from "./ScheduleCalendar";
import { UsersPanel } from "./UsersPanel";
import { TouchButton } from "../theme";

export interface AdminTabProps {
  /** Whether the signed-in user is an admin (cosmetic gate). */
  isAdmin: boolean;
}

type AdminView = "endpoints" | "schedule" | "users";

export function AdminTab({ isAdmin }: AdminTabProps) {
  const [view, setView] = useState<AdminView>("endpoints");

  if (!isAdmin) {
    return null;
  }

  return (
    <section className="admin-tab" data-testid="admin-tab" aria-label="Admin">
      <nav className="admin-tab__nav">
        <TouchButton
          variant={view === "endpoints" ? "primary" : "secondary"}
          testId="admin-endpoints-tab"
          onClick={() => setView("endpoints")}
        >
          Endpoints
        </TouchButton>
        <TouchButton
          variant={view === "schedule" ? "primary" : "secondary"}
          testId="admin-schedule-tab"
          onClick={() => setView("schedule")}
        >
          Schedule
        </TouchButton>
        <TouchButton
          variant={view === "users" ? "primary" : "secondary"}
          testId="admin-users-tab"
          onClick={() => setView("users")}
        >
          Users
        </TouchButton>
      </nav>
      {view === "endpoints" && <EndpointPanel />}
      {view === "schedule" && <ScheduleCalendar />}
      {view === "users" && <UsersPanel />}
    </section>
  );
}

export default AdminTab;
