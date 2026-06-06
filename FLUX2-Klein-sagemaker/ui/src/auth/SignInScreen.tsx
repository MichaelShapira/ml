/**
 * SignInScreen — the always-on sign-in gate (Requirements 11.1, 11.3).
 *
 * Collects username + password and calls `useAuth().signIn`. On invalid
 * credentials the hook surfaces `authError`, which is rendered here (never an
 * unhandled throw). The booth flow is not reachable until sign-in succeeds.
 *
 * Note: this is the one screen where a username is typed by the visitor; it is
 * never displayed back anywhere in the app (Requirement 13.4).
 */
import { useState, type FormEvent } from "react";
import { KioskScreen, PrimaryButton } from "../theme";

export interface SignInScreenProps {
  onSignIn: (username: string, password: string) => void | Promise<void>;
  authError: string | null;
}

export function SignInScreen({ onSignIn, authError }: SignInScreenProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void onSignIn(username, password);
  };

  return (
    <KioskScreen label="Sign in" testId="sign-in-screen">
      <form className="sign-in" onSubmit={submit}>
        <h1>AI Photo Booth</h1>
        <label>
          Username
          <input
            type="text"
            autoComplete="username"
            value={username}
            data-testid="username-input"
            onChange={(e) => setUsername(e.target.value)}
          />
        </label>
        <label>
          Password
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            data-testid="password-input"
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        {authError && (
          <p role="alert" data-testid="auth-error">
            {authError}
          </p>
        )}
        <PrimaryButton type="submit" testId="sign-in-button">
          Sign In
        </PrimaryButton>
      </form>
    </KioskScreen>
  );
}

export default SignInScreen;
