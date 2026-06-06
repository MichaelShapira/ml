/**
 * Async poll decision for the SPA (Requirements 8.2, 8.4, 8.5).
 *
 * Mirrors `backend/src/lib/poll-decision.ts`. Strict precedence:
 *   output present → READY; else failure present → FAILED;
 *   else elapsed > timeoutMs → TIMEOUT; else PENDING.
 * There is no BLOCKED status — output presence yields READY directly.
 */

/** The async timeout deadline, in milliseconds (120 seconds). */
export const POLL_TIMEOUT_MS = 120_000;

/** A poll outcome. */
export type PollDecision = "READY" | "FAILED" | "TIMEOUT" | "PENDING";

/** Inputs to the poll decision, all observed by the poller before calling. */
export interface PollInput {
  /** Whether the result object exists at the Output_Location. */
  outputPresent: boolean;
  /** Whether the endpoint wrote a failure object for this request. */
  failurePresent: boolean;
  /** Milliseconds elapsed since the request was submitted. */
  elapsedMs: number;
  /** Timeout deadline in ms; defaults to {@link POLL_TIMEOUT_MS}. */
  timeoutMs?: number;
}

/** Decide the current poll outcome under strict precedence. */
export function decidePoll({
  outputPresent,
  failurePresent,
  elapsedMs,
  timeoutMs = POLL_TIMEOUT_MS,
}: PollInput): PollDecision {
  if (outputPresent) {
    return "READY";
  }
  if (failurePresent) {
    return "FAILED";
  }
  if (elapsedMs > timeoutMs) {
    return "TIMEOUT";
  }
  return "PENDING";
}
