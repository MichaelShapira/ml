/**
 * ConnectionIndicator — the two-state connection badge (Requirement 13.1-13.3).
 *
 * A presentational, identity-free indicator: it renders only a coloured dot and
 * a "Connected"/"Disconnected" label, never the signed-in username or any other
 * identity (Requirement 13.4). The capture-flow shell (task 9.1) supplies the
 * status from `useAuth().connectionStatus`.
 *
 * It lives in the theme layer because it is shared chrome consumed by multiple
 * screens and is styled entirely by the theme stylesheet.
 */

/** The two allowed connection states (Requirement 13.1). */
export type ConnectionStatus = "connected" | "disconnected";

export interface ConnectionIndicatorProps {
  /** Current connection status. */
  status: ConnectionStatus;
  /** Optional test id for component tests. */
  testId?: string;
}

export function ConnectionIndicator({
  status,
  testId,
}: ConnectionIndicatorProps) {
  const connected = status === "connected";
  return (
    <span
      className={
        connected
          ? "connection-indicator connection-indicator--connected"
          : "connection-indicator"
      }
      role="status"
      aria-label={connected ? "Connected" : "Disconnected"}
      data-testid={testId}
    >
      <span className="connection-indicator__dot" aria-hidden="true" />
      {connected ? "Connected" : "Disconnected"}
    </span>
  );
}

export default ConnectionIndicator;
