/**
 * Async generation poll decision (pure logic — no AWS SDK, no I/O).
 *
 * The Generation_Service poll handler (`generation/poll.ts`) probes the async
 * endpoint's Output_Location and failure key, then asks this module which
 * terminal/non-terminal state the job is in. Keeping the precedence logic pure
 * lets it be exhaustively property-tested (design Property 4) before the S3 /
 * Rekognition I/O is wired in.
 *
 * Note on `READY`: this module returns a `READY`-*candidate* — it only means the
 * output object exists. The poll handler still runs the Content_Filter
 * (Rekognition) on the downloaded bytes and may downgrade a `READY` candidate to
 * `BLOCKED`. `BLOCKED` is therefore decided in the handler, not here.
 *
 * Requirements: 8.2, 8.4, 8.5 — design Property 4.
 */

/**
 * The async timeout deadline, in milliseconds (120 seconds).
 *
 * A job is only declared `TIMEOUT` once elapsed time is **strictly greater**
 * than this value; exactly `POLL_TIMEOUT_MS` is still `PENDING` (Requirement
 * 8.5: "within 120 seconds").
 */
export const POLL_TIMEOUT_MS = 120_000;

/**
 * The possible outcomes of a single poll evaluation.
 *
 * Declared as a `const` object plus a same-named union type so the poll handler
 * can reference values (`PollDecision.READY`) and the type (`PollDecision`)
 * interchangeably. `BLOCKED` is intentionally absent here — see the module note.
 */
export const PollDecision = {
  /** Output object present — candidate result, pending content moderation. */
  READY: "READY",
  /** No output, but a failure object is present. */
  FAILED: "FAILED",
  /** No output or failure, and the elapsed time exceeded the deadline. */
  TIMEOUT: "TIMEOUT",
  /** No output or failure yet, and still within the deadline. */
  PENDING: "PENDING",
} as const;

/** A poll outcome: one of `READY | FAILED | TIMEOUT | PENDING`. */
export type PollDecision = (typeof PollDecision)[keyof typeof PollDecision];

/** Inputs to the poll decision, all observed by the handler before calling. */
export interface PollInput {
  /** Whether the result object exists at the Output_Location. */
  outputPresent: boolean;
  /** Whether the endpoint wrote a failure object for this request. */
  failurePresent: boolean;
  /** Milliseconds elapsed since the request was submitted. */
  elapsedMs: number;
}

/**
 * Decide the current poll outcome under strict precedence (Property 4):
 *
 *   1. output present            → `READY`   (candidate; see module note)
 *   2. else failure present      → `FAILED`
 *   3. else elapsed > 120000 ms  → `TIMEOUT` (strictly greater)
 *   4. else                      → `PENDING`
 *
 * The order is significant: a present output wins even if a failure is also
 * present or the deadline has passed, and a present failure wins over a timeout.
 */
export function decidePoll({
  outputPresent,
  failurePresent,
  elapsedMs,
}: PollInput): PollDecision {
  if (outputPresent) {
    return PollDecision.READY;
  }
  if (failurePresent) {
    return PollDecision.FAILED;
  }
  if (elapsedMs > POLL_TIMEOUT_MS) {
    return PollDecision.TIMEOUT;
  }
  return PollDecision.PENDING;
}
