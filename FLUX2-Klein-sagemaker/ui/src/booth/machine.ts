/**
 * Capture-flow finite state machine (pure logic — no React, no I/O).
 *
 * The Photo_Booth_App models its visitor capture flow as an explicit, pure
 * state machine so the transitions can be property-tested (design Properties
 * 5–8) independently of React, and so the React screens can derive exactly one
 * active screen from a single source of truth.
 *
 * The flow is:
 *
 *   SignedOut → Start → Camera → Review → Effects → Loading → Result | Error
 *
 * with two cross-cutting transitions:
 *   - SIGN_OUT from any state returns to SignedOut and ends the session.
 *   - IDLE_TIMEOUT from any visitor-facing state returns to a fresh Start.
 *
 * Results accumulate by slot
 * ---------------------------
 * A single session can hold up to three generated images at once
 * (design: co-present background + character + merge):
 *   - "background" — the result of a background effect,
 *   - "person"     — the result of a character / dress effect,
 *   - "merged"     — the result of merging the two above into one image.
 * Generating a background effect fills the "background" slot, a character
 * effect fills the "person" slot, and they no longer overwrite each other.
 * Re-running the same category replaces only that slot. MERGE is offered only
 * once BOTH the background and person slots are filled, and writes "merged".
 *
 * Invariants enforced here:
 *   - Selection lock: once a generation begins (Loading), further SELECT/MERGE
 *     events are ignored and the in-flight job is retained (Requirement 6.6).
 *   - Loading / Result mutual exclusion: Loading and Result are distinct
 *     states, so the derived `activeScreen` is never both at once
 *     (Requirement 9.3).
 *   - A failed/timed-out generation preserves the results gathered so far, so a
 *     visitor never loses an earlier image because a later one failed.
 *   - Session data is cleared whenever the machine returns to Start by any
 *     path (Requirements 10.4, 10.5): the `Start` state carries no captured
 *     photo and no generated results.
 *   - Terminal poll statuses are `READY | FAILED | TIMEOUT | PENDING` only —
 *     there is no content-moderation / `BLOCKED` path (matches
 *     `backend/src/lib/poll-decision.ts`).
 *
 * Requirements: 1.2, 3.2, 4.1, 4.2, 5.5, 6.5, 6.6, 9.1, 9.2, 9.3, 9.4, 10.3,
 * 10.4, 10.5 — design Properties 5–8.
 */

/**
 * The id of a selected Effect_Option.
 *
 * Ideally this would be imported from the shared backend catalog
 * (`backend/src/lib/effects.ts`, `EffectOption["id"]`) so the booth has a
 * single source of truth for effect ids. However, the UI package is
 * type-checked in isolation: `ui/tsconfig.app.json` uses `include: ["src"]`
 * and is built with project-reference build mode (`tsc -b`), which cannot
 * resolve a `.ts` source file outside `ui/src` without composite-project
 * (TS6307) errors. Effect ids are opaque strings to this pure state machine,
 * which never inspects the catalog, so the type is aliased locally. The React
 * `EffectSelector` and the `Generation_Service` resolve the id against the
 * shared catalog at the edges of the app, where the import path is available.
 */
export type EffectId = string;

/**
 * A base64-encoded PNG/JPEG still captured from the webcam (Requirement 3.4),
 * carried through the flow until it is submitted for transformation.
 */
export type CapturedPhoto = string;

/**
 * A reference to a generated result image (an object URL produced by the
 * Generation_Service from the PNG bytes read out of S3). Opaque to the machine.
 */
export type TransformedImage = string;

/**
 * Which generated image a result belongs to. Background and character effects
 * fill distinct slots so they can be shown together; "merged" is the combined
 * image produced by the merge action.
 */
export type ResultSlot = "background" | "person" | "merged";

/** A single generated image plus the effect that produced it (if any). */
export interface SlotResult {
  /** Renderable reference to the generated image (object URL). */
  readonly image: TransformedImage;
  /**
   * The effect id that produced this image, or `null` for the merged image
   * (which is produced from a synthesized merge prompt, not a catalog effect).
   */
  readonly effectId: EffectId | null;
}

/**
 * The set of generated images accumulated in the current session, keyed by
 * slot. Any subset may be present; an empty object means nothing generated yet.
 */
export interface GeneratedResults {
  readonly background?: SlotResult;
  readonly person?: SlotResult;
  readonly merged?: SlotResult;
}

/**
 * Describes the in-flight generation while `Loading`, so the I/O layer
 * (`BoothFlow`) knows what to submit and the reducer knows which slot a READY
 * result fills.
 *
 *   - "effect" — a catalog effect applied to the captured photo; the result
 *     fills `slot` ("background" or "person").
 *   - "merge"  — combine the existing background + person images into one; the
 *     result fills the "merged" slot.
 */
export type GenerationJob =
  | { readonly kind: "effect"; readonly effectId: EffectId; readonly slot: "background" | "person" }
  | { readonly kind: "merge" };

/** Why a generation attempt ended in the `Error` state (never a success). */
export interface ErrorReason {
  /** Failure record was written, or the 120 s client-side deadline elapsed. */
  kind: "FAILED" | "TIMEOUT";
  /** Optional human-readable detail surfaced by the Generation_Service. */
  message?: string;
}

/**
 * The result of a single async poll, applied to the machine while `Loading`.
 *
 * Mirrors the terminal/non-terminal statuses of
 * `backend/src/lib/poll-decision.ts` (`READY | FAILED | TIMEOUT | PENDING`).
 * `image` on a `READY` result is the renderable generated image reference.
 */
export type PollResult =
  | { status: "READY"; image: TransformedImage }
  | { status: "FAILED"; reason?: string }
  | { status: "TIMEOUT" }
  | { status: "PENDING" };

/** Visitor is not authenticated; the sign-in interface is shown. */
export interface SignedOutState {
  readonly name: "SignedOut";
}
/** Authenticated, idle start screen; no session data is held. */
export interface StartState {
  readonly name: "Start";
}
/** Live webcam view; camera errors are handled in-screen (self-loop). */
export interface CameraState {
  readonly name: "Camera";
}
/** A still has been captured and is shown with Reset / Continue controls. */
export interface ReviewState {
  readonly name: "Review";
  readonly capturedPhoto: CapturedPhoto;
}
/** Effect selection; the captured photo is shown alongside the 12 options. */
export interface EffectsState {
  readonly name: "Effects";
  readonly capturedPhoto: CapturedPhoto;
  /** Results gathered so far this session (empty on first entry). */
  readonly results: GeneratedResults;
}
/** A generation is in progress; the selection is locked. */
export interface LoadingState {
  readonly name: "Loading";
  readonly capturedPhoto: CapturedPhoto;
  /** The in-flight job (effect or merge) — drives submission + slotting. */
  readonly job: GenerationJob;
  /** Results gathered before this job started (preserved on failure). */
  readonly results: GeneratedResults;
}
/**
 * One or more generated images are shown (never simultaneously with Loading).
 * `results` holds at least one slot.
 */
export interface ResultState {
  readonly name: "Result";
  /**
   * The original captured photo, retained so the visitor can pick another
   * effect from the result view and regenerate in place (the studio layout
   * keeps the options visible alongside the results).
   */
  readonly capturedPhoto: CapturedPhoto;
  /** The generated images accumulated so far, keyed by slot. */
  readonly results: GeneratedResults;
}
/**
 * Generation failed or timed out. Holds the error reason and preserves any
 * results gathered before the failure. The captured photo is retained so the
 * visitor can return to the effect selection view (Requirement 9.4).
 */
export interface ErrorState {
  readonly name: "Error";
  readonly reason: ErrorReason;
  readonly capturedPhoto: CapturedPhoto;
  /** Results gathered before the failed attempt (may be empty). */
  readonly results: GeneratedResults;
}

/** The full capture-flow state union. */
export type BoothState =
  | SignedOutState
  | StartState
  | CameraState
  | ReviewState
  | EffectsState
  | LoadingState
  | ResultState
  | ErrorState;

/** The events that drive the capture-flow machine. */
export type BoothEvent =
  /** Cognito sign-in succeeded; enter the booth (Requirement 11.2). */
  | { type: "AUTHENTICATED" }
  /** Sign out from anywhere; discard the session (Requirement 11.4). */
  | { type: "SIGN_OUT" }
  /** Tap Start; go to the live camera view (Requirements 1.2). */
  | { type: "START" }
  /** Tap Take Photo; capture a still (Requirements 3.1, 3.2). */
  | { type: "CAPTURE"; photo: CapturedPhoto }
  /** Tap Reset; discard the photo and return to camera (Requirement 4.1). */
  | { type: "RESET" }
  /** Tap Continue; keep the photo and go to effects (Requirement 4.2). */
  | { type: "CONTINUE" }
  /**
   * Select an effect; lock it in and begin loading (Requirements 5.5, 6.5).
   * `slot` ("background" | "person") is resolved from the catalog category by
   * the caller, so the result lands in the correct slot.
   */
  | { type: "SELECT"; effectId: EffectId; slot: "background" | "person" }
  /**
   * Merge the existing background + person images into one (the "merged" slot).
   * Ignored unless both source slots are filled and no job is in flight.
   */
  | { type: "MERGE" }
  /** Apply an async poll result while loading (Requirements 9.2–9.4). */
  | { type: "POLL"; result: PollResult }
  /** Tap New Session from the result screen (Requirement 10.4). */
  | { type: "NEW_SESSION" }
  /** Inactivity timer fired; auto-reset to Start (Requirement 10.5). */
  | { type: "IDLE_TIMEOUT" }
  /** Webcam unavailable / access denied (Requirement 2.4). */
  | { type: "CAMERA_ERROR"; reason?: string }
  /** Retry control: re-attempt camera access, or leave the error screen. */
  | { type: "RETRY" };

/** The set of distinct screens the renderer can show, one per state. */
export type ActiveScreen =
  | "SignIn"
  | "Start"
  | "Camera"
  | "Review"
  | "Effects"
  | "Loading"
  | "Result"
  | "Error";

/** Singleton SignedOut state (no data). */
const SIGNED_OUT: SignedOutState = { name: "SignedOut" };

/**
 * Construct a fresh `Start` state.
 *
 * Every path back to Start funnels through here, which guarantees no captured
 * photo or generated results are ever carried into a new session
 * (Requirements 10.4, 10.5; design Property 7). `Start` simply has no such
 * fields to carry.
 */
function freshStart(): StartState {
  return { name: "Start" };
}

/** The machine's entry state: not yet authenticated. */
export const initialState: BoothState = SIGNED_OUT;

/** True when both source slots needed for a merge are present. */
export function canMerge(results: GeneratedResults): boolean {
  return Boolean(results.background) && Boolean(results.person);
}

/**
 * Begin a generation: enter `Loading` with the given job, preserving the
 * results gathered so far. Shared by the SELECT (effect) and MERGE paths.
 */
function startJob(
  capturedPhoto: CapturedPhoto,
  results: GeneratedResults,
  job: GenerationJob,
): LoadingState {
  return { name: "Loading", capturedPhoto, results, job };
}

/**
 * Apply a poll result to a `Loading` state (Requirements 9.2–9.4; Property 5).
 *
 *   - `READY`   → `Result`, storing the image in the job's slot (merging it
 *                 into any results already gathered).
 *   - `FAILED`  → `Error` (preserving prior results), retaining the photo.
 *   - `TIMEOUT` → `Error` (preserving prior results), retaining the photo.
 *   - `PENDING` → stay in `Loading` unchanged.
 */
function applyPoll(state: LoadingState, result: PollResult): BoothState {
  switch (result.status) {
    case "READY": {
      const slot: ResultSlot = state.job.kind === "merge" ? "merged" : state.job.slot;
      const effectId: EffectId | null =
        state.job.kind === "effect" ? state.job.effectId : null;
      return {
        name: "Result",
        capturedPhoto: state.capturedPhoto,
        results: { ...state.results, [slot]: { image: result.image, effectId } },
      };
    }
    case "FAILED":
      return {
        name: "Error",
        reason: { kind: "FAILED", ...(result.reason ? { message: result.reason } : {}) },
        capturedPhoto: state.capturedPhoto,
        results: state.results,
      };
    case "TIMEOUT":
      return {
        name: "Error",
        reason: { kind: "TIMEOUT" },
        capturedPhoto: state.capturedPhoto,
        results: state.results,
      };
    case "PENDING":
      return state;
    default:
      return assertNever(result);
  }
}

/**
 * The pure capture-flow reducer.
 *
 * Total over every (state, event) pair: any event that is not meaningful in
 * the current state is a no-op that returns the same state, so callers can
 * dispatch freely without guarding. The two cross-cutting transitions
 * (`SIGN_OUT`, `IDLE_TIMEOUT`) are handled first and apply from any state.
 */
export function transition(state: BoothState, event: BoothEvent): BoothState {
  // Cross-cutting: sign out from any state ends the session (Requirement 11.4).
  if (event.type === "SIGN_OUT") {
    return SIGNED_OUT;
  }

  // Cross-cutting: idle auto-reset returns any visitor-facing state to a fresh
  // Start, clearing all session data (Requirement 10.5). SignedOut is the
  // sign-in screen, not a visitor-facing capture state, so it is unaffected.
  if (event.type === "IDLE_TIMEOUT") {
    return state.name === "SignedOut" ? state : freshStart();
  }

  // Cross-cutting: "New Session" returns any visitor-facing state to a fresh
  // Start, clearing all session data (Requirement 10.4). The studio renders the
  // New Session control across the Effects/Loading/Result/Error phases, so the
  // event must be honoured from all of them — not only from Result. SignedOut
  // is the sign-in screen, so it is unaffected.
  if (event.type === "NEW_SESSION") {
    return state.name === "SignedOut" ? state : freshStart();
  }

  switch (state.name) {
    case "SignedOut":
      // Only authentication advances out of the sign-in screen.
      return event.type === "AUTHENTICATED" ? freshStart() : state;

    case "Start":
      return event.type === "START" ? { name: "Camera" } : state;

    case "Camera":
      switch (event.type) {
        case "CAPTURE":
          return { name: "Review", capturedPhoto: event.photo };
        case "CAMERA_ERROR":
        case "RETRY":
          // Camera errors and retries are handled within the camera screen
          // (diagram self-loop); the machine stays in Camera.
          return state;
        default:
          return state;
      }

    case "Review":
      switch (event.type) {
        case "RESET":
          // Discard the captured photo and return to the live camera view.
          return { name: "Camera" };
        case "CONTINUE":
          // Retain the captured photo into effect selection (no results yet).
          return { name: "Effects", capturedPhoto: state.capturedPhoto, results: {} };
        default:
          return state;
      }

    case "Effects":
      // Selecting an effect records the choice and begins loading into its slot.
      if (event.type === "SELECT") {
        return startJob(state.capturedPhoto, state.results, {
          kind: "effect",
          effectId: event.effectId,
          slot: event.slot,
        });
      }
      // Merge is possible directly from Effects if both slots are already filled.
      if (event.type === "MERGE" && canMerge(state.results)) {
        return startJob(state.capturedPhoto, state.results, { kind: "merge" });
      }
      return state;

    case "Loading":
      switch (event.type) {
        case "SELECT":
        case "MERGE":
          // Selection lock: ignore further job requests, retain the in-flight one.
          return state;
        case "POLL":
          return applyPoll(state, event.result);
        default:
          return state;
      }

    case "Result":
      // Regenerate in place: pick another effect (replaces only that slot) or
      // merge the two source images. Other slots are preserved.
      if (event.type === "SELECT") {
        return startJob(state.capturedPhoto, state.results, {
          kind: "effect",
          effectId: event.effectId,
          slot: event.slot,
        });
      }
      if (event.type === "MERGE" && canMerge(state.results)) {
        return startJob(state.capturedPhoto, state.results, { kind: "merge" });
      }
      return state;

    case "Error":
      // Retry / back returns to effect selection (results preserved); selecting
      // an effect or merging regenerates in place.
      switch (event.type) {
        case "RETRY":
          return { name: "Effects", capturedPhoto: state.capturedPhoto, results: state.results };
        case "SELECT":
          return startJob(state.capturedPhoto, state.results, {
            kind: "effect",
            effectId: event.effectId,
            slot: event.slot,
          });
        case "MERGE":
          return canMerge(state.results)
            ? startJob(state.capturedPhoto, state.results, { kind: "merge" })
            : state;
        default:
          return state;
      }

    default:
      return assertNever(state);
  }
}

/**
 * Derive the single active screen for a state (Requirement 9.3).
 *
 * Exactly one screen id is returned per state. Because `Loading` and `Result`
 * are distinct states, this derivation can never report both as active, which
 * is what guarantees the loading state and the generated images are never
 * shown at the same time.
 */
export function activeScreen(state: BoothState): ActiveScreen {
  switch (state.name) {
    case "SignedOut":
      return "SignIn";
    case "Start":
      return "Start";
    case "Camera":
      return "Camera";
    case "Review":
      return "Review";
    case "Effects":
      return "Effects";
    case "Loading":
      return "Loading";
    case "Result":
      return "Result";
    case "Error":
      return "Error";
    default:
      return assertNever(state);
  }
}

/**
 * Exhaustiveness guard: makes the compiler flag any unhandled state/event
 * variant, and throws if an impossible value is reached at runtime.
 */
function assertNever(value: never): never {
  throw new Error(`Unhandled capture-flow variant: ${JSON.stringify(value)}`);
}
