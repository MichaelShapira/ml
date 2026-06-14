/**
 * BoothFlow — wires the capture-flow state machine to the screens (Section 9).
 *
 * Owns the `machine.ts` state via a reducer, renders exactly one screen from
 * `activeScreen(state)` (so Loading and Result are never shown together,
 * Req 9.3), and bridges the screens to the browser-side Generation_Service:
 *
 *   - on SELECT it submits via `api/generation.submitGeneration` and then polls
 *     every ~2 s, dispatching POLL(result) into the machine until a terminal
 *     status (READY | FAILED | TIMEOUT);
 *   - on return to Start (New Session or idle auto-reset) it revokes any result
 *     object URL and clears the submission (Req 10.4, 10.5);
 *   - an inactivity timer dispatches IDLE_TIMEOUT from any visitor-facing state.
 *
 * The machine remains the single source of truth for transitions; this
 * component only performs the I/O the pure machine cannot.
 */
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import {
  transition,
  activeScreen,
  initialState,
  type BoothState,
  type BoothEvent,
  type GeneratedResults,
} from "./machine";
import { StartScreen } from "./StartScreen";
import { CameraView } from "./CameraView";
import { ReviewScreen } from "./ReviewScreen";
import { StudioView, type StudioPhase, type StudioResults } from "./StudioView";
import { buildMergePrompt } from "./effects";
import { useLayoutMode } from "./useLayoutMode";
import { LOADING_MESSAGES, messageForElapsed } from "./loadingMessages";
import {
  submitGeneration,
  submitMerge,
  pollGeneration,
  revokeResult,
  type SubmitResult,
} from "../api/generation";

/** Map the machine's per-slot results to the studio's renderable URL map. */
function toStudioResults(results: GeneratedResults): StudioResults {
  return {
    background: results.background?.image ?? null,
    person: results.person?.image ?? null,
    merged: results.merged?.image ?? null,
  };
}

/** Poll cadence while a generation is in flight. */
const POLL_INTERVAL_MS = 2000;

/** How long each loading message stays on screen before rotating. */
const LOADING_ROTATE_MS = 2500;

/** Idle inactivity timeout that auto-resets the booth to Start (kiosk reset). */
const IDLE_TIMEOUT_MS = 180_000;

export interface BoothFlowProps {
  /** True once the visitor is authenticated (drives the initial AUTHENTICATED). */
  isAuthenticated: boolean;
  /** When true (admin), the camera step also offers an Upload Photo control. */
  isAdmin?: boolean;
}

export function BoothFlow({ isAuthenticated, isAdmin = false }: BoothFlowProps) {
  const [state, dispatch] = useReducer(transition, initialState);
  const layout = useLayoutMode();

  // Rotating loading message while a generation is in flight.
  const [loadingMessage, setLoadingMessage] = useState<string>(() =>
    messageForElapsed(0, LOADING_ROTATE_MS, LOADING_MESSAGES),
  );

  // Mutable refs for the in-flight generation so effects can cancel cleanly.
  const submissionRef = useRef<SubmitResult | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // All generated object URLs produced this session. A session can hold several
  // at once (background, character, merged) and regenerating a slot creates a
  // new one, so we track them as a set and revoke them all when the session
  // ends rather than juggling a single URL.
  const resultUrlsRef = useRef<Set<string>>(new Set());
  const cancelledRef = useRef(false);

  const send = useCallback((event: BoothEvent) => dispatch(event), []);

  // Enter the booth once authenticated; sign-out elsewhere resets to SignedOut.
  useEffect(() => {
    if (isAuthenticated && state.name === "SignedOut") {
      send({ type: "AUTHENTICATED" });
    }
    if (!isAuthenticated && state.name !== "SignedOut") {
      send({ type: "SIGN_OUT" });
    }
  }, [isAuthenticated, state.name, send]);

  const clearPollTimer = useCallback(() => {
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  // Revoke every generated object URL produced this session and clear the set.
  const revokeAllResults = useCallback(() => {
    for (const url of resultUrlsRef.current) {
      revokeResult(url);
    }
    resultUrlsRef.current.clear();
  }, []);

  // Drive submission + polling when entering Loading with a selected effect.
  useEffect(() => {
    if (state.name !== "Loading") {
      return;
    }
    cancelledRef.current = false;
    let stopped = false;

    const poll = async (handle: SubmitResult) => {
      if (stopped || cancelledRef.current) {
        return;
      }
      try {
        const result = await pollGeneration(handle);
        if (stopped || cancelledRef.current) {
          if (result.status === "READY") {
            revokeResult(result.imageUrl);
          }
          return;
        }
        if (result.status === "READY") {
          resultUrlsRef.current.add(result.imageUrl);
          send({ type: "POLL", result: { status: "READY", image: result.imageUrl } });
        } else if (result.status === "FAILED") {
          send({ type: "POLL", result: { status: "FAILED" } });
        } else if (result.status === "TIMEOUT") {
          send({ type: "POLL", result: { status: "TIMEOUT" } });
        } else {
          pollTimerRef.current = setTimeout(() => void poll(handle), POLL_INTERVAL_MS);
        }
      } catch {
        if (!stopped && !cancelledRef.current) {
          send({ type: "POLL", result: { status: "FAILED" } });
        }
      }
    };

    const start = async () => {
      try {
        // Branch on the in-flight job: a single-effect generation submits the
        // captured photo with the effect's prompt; a merge submits the two
        // existing generated images with a synthesized multi-reference prompt.
        const job = state.job;
        let handle: SubmitResult;
        if (job.kind === "merge") {
          const bg = state.results.background;
          const person = state.results.person;
          if (!bg || !person || bg.effectId === null || person.effectId === null) {
            // Should never happen — the machine only starts a merge when both
            // source slots are filled — but guard so we fail gracefully.
            throw new Error("merge requested without both source images");
          }
          // By convention the background result is image 1 and the character
          // result is image 2 (the merge prompt references them in that order).
          handle = await submitMerge({
            prompt: buildMergePrompt(bg.effectId, person.effectId),
            images: [bg.image, person.image],
          });
        } else {
          handle = await submitGeneration({
            effectId: job.effectId,
            photo: state.capturedPhoto,
          });
        }
        submissionRef.current = handle;
        if (!stopped && !cancelledRef.current) {
          void poll(handle);
        }
      } catch {
        if (!stopped && !cancelledRef.current) {
          send({ type: "POLL", result: { status: "FAILED" } });
        }
      }
    };

    void start();

    return () => {
      stopped = true;
      clearPollTimer();
    };
    // Re-run only when entering a NEW Loading (effect/photo identify the job).
  }, [state.name, send, clearPollTimer]); // eslint-disable-line react-hooks/exhaustive-deps

  // Clean up the result URLs + submission whenever we return to Start.
  useEffect(() => {
    if (state.name === "Start") {
      cancelledRef.current = true;
      clearPollTimer();
      submissionRef.current = null;
      revokeAllResults();
    }
  }, [state.name, clearPollTimer, revokeAllResults]);

  // Idle auto-reset for the kiosk: after a period of NO user interaction, return
  // to Start so the booth is ready for the next visitor. Important details:
  //   - never auto-reset while a generation is in flight ("Loading") — that
  //     would abandon a photo the visitor is actively waiting for;
  //   - reset the countdown on real user activity (tap/click/key/touch), so a
  //     visitor reading the result or typing an email is never kicked out;
  //   - only applies to visitor-facing states (not SignedOut/Start).
  useEffect(() => {
    const idleStates: BoothState["name"][] = [
      "Camera",
      "Review",
      "Effects",
      "Result",
      "Error",
    ];
    if (!idleStates.includes(state.name)) {
      return;
    }

    let timer: ReturnType<typeof setTimeout>;
    const reset = () => {
      clearTimeout(timer);
      timer = setTimeout(() => send({ type: "IDLE_TIMEOUT" }), IDLE_TIMEOUT_MS);
    };
    const activity = ["pointerdown", "keydown", "touchstart", "mousemove"] as const;
    for (const evt of activity) {
      window.addEventListener(evt, reset, { passive: true });
    }
    reset();

    return () => {
      clearTimeout(timer);
      for (const evt of activity) {
        window.removeEventListener(evt, reset);
      }
    };
  }, [state.name, send]);

  // Revoke result URLs on unmount.
  useEffect(() => () => revokeAllResults(), [revokeAllResults]);

  // Rotate the loading message while a generation is in flight.
  useEffect(() => {
    if (state.name !== "Loading") {
      return;
    }
    const startedAt = Date.now();
    setLoadingMessage(messageForElapsed(0, LOADING_ROTATE_MS, LOADING_MESSAGES));
    const id = setInterval(() => {
      const elapsed = Date.now() - startedAt;
      setLoadingMessage(messageForElapsed(elapsed, LOADING_ROTATE_MS, LOADING_MESSAGES));
    }, LOADING_ROTATE_MS);
    return () => clearInterval(id);
  }, [state.name]);

  const screen = activeScreen(state);

  switch (screen) {
    case "Start":
      return <StartScreen onStart={() => send({ type: "START" })} />;
    case "Camera":
      return (
        <CameraView
          allowUpload={isAdmin}
          onCapture={(photo) => send({ type: "CAPTURE", photo })}
        />
      );
    case "Review":
      return state.name === "Review" ? (
        <ReviewScreen
          photo={state.capturedPhoto}
          onReset={() => send({ type: "RESET" })}
          onContinue={() => send({ type: "CONTINUE" })}
        />
      ) : null;
    case "Effects":
    case "Loading":
    case "Result":
    case "Error": {
      // The studio co-presents the photo, the effect options, and any generated
      // results. Derive the photo + phase + results from whichever state we're in.
      let photo: string;
      let phase: StudioPhase;
      let errorMessage: string | undefined;

      switch (state.name) {
        case "Effects":
          photo = state.capturedPhoto;
          phase = "idle";
          break;
        case "Loading":
          photo = state.capturedPhoto;
          phase = "loading";
          break;
        case "Result":
          photo = state.capturedPhoto;
          phase = "result";
          break;
        case "Error":
          photo = state.capturedPhoto;
          phase = "error";
          errorMessage =
            state.reason.kind === "TIMEOUT"
              ? "That took too long. Pick an effect to try again."
              : state.reason.message ??
                "Generation failed. Pick an effect to try again.";
          break;
        default:
          return null;
      }

      const results = toStudioResults(state.results);

      return (
        <StudioView
          layout={layout}
          photo={photo}
          phase={phase}
          results={results}
          loadingMessage={loadingMessage}
          errorMessage={errorMessage}
          onSelect={(effectId, slot) => send({ type: "SELECT", effectId, slot })}
          onMerge={() => send({ type: "MERGE" })}
          onNewSession={() => send({ type: "NEW_SESSION" })}
          isAdmin={isAdmin}
        />
      );
    }
    case "SignIn":
    default:
      return null;
  }
}

export default BoothFlow;
