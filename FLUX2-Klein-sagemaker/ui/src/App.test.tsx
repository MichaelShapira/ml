/**
 * App gate + connection-indicator tests (Requirements 11.1, 13.1-13.4, 14.1, 14.2).
 *
 * The booth flow and admin tab are heavy (camera, AWS clients), so this test
 * mocks `useAuth` and the child flows to focus on the gate behaviour:
 *   - unauthenticated → sign-in screen;
 *   - authenticated → connection indicator (connected/disconnected), no username;
 *   - Admin tab visible iff isAdmin.
 */
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

// --- Mocks (declared before importing App) ----------------------------------
const mockUseAuth = vi.fn();

vi.mock("./auth/useAuth", () => ({
  useAuth: () => mockUseAuth(),
}));

// Stub the heavy child flows so the gate renders without camera/AWS wiring.
vi.mock("./booth/BoothFlow", () => ({
  BoothFlow: () => <div data-testid="booth-flow" />,
}));
vi.mock("./admin/AdminTab", () => ({
  AdminTab: ({ isAdmin }: { isAdmin: boolean }) =>
    isAdmin ? <div data-testid="admin-tab" /> : null,
}));

import App from "./App";

function authState(overrides: Record<string, unknown> = {}) {
  return {
    isAuthenticated: true,
    isAdmin: false,
    connectionStatus: "connected" as const,
    authError: null,
    signIn: vi.fn(),
    signOut: vi.fn(),
    clearAuthError: vi.fn(),
    ...overrides,
  };
}

describe("App auth gate", () => {
  beforeEach(() => mockUseAuth.mockReset());

  it("shows the sign-in screen when unauthenticated", () => {
    mockUseAuth.mockReturnValue(authState({ isAuthenticated: false }));
    render(<App />);
    expect(screen.getByTestId("sign-in-screen")).toBeInTheDocument();
    expect(screen.queryByTestId("app-shell")).not.toBeInTheDocument();
  });

  it("surfaces an auth error on the sign-in screen", () => {
    mockUseAuth.mockReturnValue(
      authState({ isAuthenticated: false, authError: "Invalid credentials" }),
    );
    render(<App />);
    expect(screen.getByTestId("auth-error")).toHaveTextContent("Invalid credentials");
  });

  it("shows a bare kiosk (no chrome) for a standard visitor", () => {
    mockUseAuth.mockReturnValue(authState({ isAdmin: false, connectionStatus: "connected" }));
    render(<App />);
    // Non-admin: capture flow only — no top bar, no indicator, no tabs.
    expect(screen.getByTestId("booth-flow")).toBeInTheDocument();
    expect(screen.queryByTestId("connection-indicator")).not.toBeInTheDocument();
    expect(screen.queryByTestId("booth-tab-button")).not.toBeInTheDocument();
    expect(screen.queryByTestId("sign-out-button")).not.toBeInTheDocument();
  });

  it("shows the connected indicator and the booth flow for an admin", () => {
    mockUseAuth.mockReturnValue(authState({ isAdmin: true, connectionStatus: "connected" }));
    render(<App />);
    expect(screen.getByTestId("connection-indicator")).toHaveTextContent("Connected");
    expect(screen.getByTestId("booth-flow")).toBeInTheDocument();
  });

  it("shows the disconnected indicator (admin) when credentials are unavailable", () => {
    mockUseAuth.mockReturnValue(authState({ isAdmin: true, connectionStatus: "disconnected" }));
    render(<App />);
    expect(screen.getByTestId("connection-indicator")).toHaveTextContent("Disconnected");
  });

  it("hides the Admin tab for non-admins and shows it for admins", () => {
    mockUseAuth.mockReturnValue(authState({ isAdmin: false }));
    const { rerender } = render(<App />);
    expect(screen.queryByTestId("admin-tab-button")).not.toBeInTheDocument();

    mockUseAuth.mockReturnValue(authState({ isAdmin: true }));
    rerender(<App />);
    expect(screen.getByTestId("admin-tab-button")).toBeInTheDocument();
  });

  it("never renders a username anywhere in the shell", () => {
    mockUseAuth.mockReturnValue(authState({ isAdmin: true }));
    const { container } = render(<App />);
    // The indicator only ever reads "Connected"/"Disconnected"; assert no
    // identity leaks into the chrome.
    expect(container.textContent).not.toMatch(/welcome|signed in as|@/i);
  });
});
