/**
 * StudioView — the combined capture studio (Effects + Loading + Result + Error).
 *
 * Replaces the separate EffectSelector / LoadingScreen / ResultScreen as three
 * full-screen steps with ONE view where the original photo, the effect options,
 * and the generated result are co-present. The capture-flow machine is still
 * the source of truth; this component just renders whichever phase the machine
 * is in, keeping the options visible so the visitor can pick another effect and
 * regenerate in place.
 *
 * Two responsive layouts (see {@link useLayoutMode}):
 *   - "monitor": background options stacked on the LEFT, character options on
 *     the RIGHT, original photo centered, generated image directly BELOW it.
 *   - "mobile": image area on top with an original/generated carousel; the two
 *     option groups stacked BELOW it. A new generation replaces the previous.
 */
import { useCallback, useState, type FormEvent, type ReactNode } from "react";
import {
  BACKGROUND_EFFECTS,
  PERSON_EFFECTS,
  effectImageUrl,
  type EffectOption,
} from "./effects";
import { ImageCarousel } from "./ImageCarousel";
import { EmailKeyboard } from "./EmailKeyboard";
import { TouchButton, PrimaryButton } from "../theme";
import { sendPhotoEmail, isValidEmail, InvalidEmailError, EmailDeliveryError } from "../api/email";
import type { LayoutMode } from "./useLayoutMode";

/** The generation phase the studio is showing. */
export type StudioPhase = "idle" | "loading" | "result" | "error";

export interface StudioViewProps {
  /** Layout to render (monitor vs mobile). */
  layout: LayoutMode;
  /** The original captured photo. */
  photo: string;
  /** Current generation phase derived from the machine state. */
  phase: StudioPhase;
  /** The generated image URL when phase is "result", else null. */
  generatedUrl?: string | null;
  /** Rotating loading message shown while phase is "loading". */
  loadingMessage?: string;
  /** Error copy shown when phase is "error". */
  errorMessage?: string;
  /** Pick an effect (starts/restarts generation). Ignored while loading. */
  onSelect: (effectId: string) => void;
  /** Start a fresh session (discard everything, back to Start). */
  onNewSession: () => void;
}

/** A vertical list of effect option buttons for one category. */
function OptionColumn({
  title,
  options,
  locked,
  layout,
  onSelect,
}: {
  title: string;
  options: readonly EffectOption[];
  locked: boolean;
  layout: LayoutMode;
  onSelect: (effectId: string) => void;
}) {
  return (
    <div className="studio__option-group">
      <h2 className="studio__option-title">{title}</h2>
      <div className="studio__option-list" role="list">
        {options.map((option) => {
          const imageUrl = effectImageUrl(option, layout);
          return (
            <TouchButton
              key={option.id}
              role="listitem"
              variant="secondary"
              disabled={locked}
              testId={`effect-${option.id}`}
              className={imageUrl ? "studio__option--with-image" : undefined}
              onClick={() => onSelect(option.id)}
            >
              {imageUrl && (
                <img
                  className="studio__option-thumb"
                  src={imageUrl}
                  alt=""
                  aria-hidden="true"
                  loading="lazy"
                />
              )}
              <span className="studio__option-label">{option.label}</span>
            </TouchButton>
          );
        })}
      </div>
    </div>
  );
}

/** The spinner + rotating message overlay shown while generating. */
function GeneratingOverlay({ message }: { message?: string }) {
  return (
    <div className="studio__generating" role="status" aria-live="polite">
      <div className="loading-screen__spinner" aria-hidden="true" />
      {message && (
        <p className="studio__generating-text" data-testid="loading-message">
          {message}
        </p>
      )}
    </div>
  );
}

/** Characters allowed anywhere in an email address (used to sanitize input). */
const EMAIL_CHAR_RE = /[^a-zA-Z0-9@._+\-]/g;

/** Strip any character that cannot appear in an email address. */
function sanitizeEmail(value: string): string {
  return value.replace(EMAIL_CHAR_RE, "").toLowerCase();
}

/** The email-my-photo form (shown once a result exists). */
function EmailForm({ imageUrl, layout }: { imageUrl: string; layout: LayoutMode }) {
  const [email, setEmail] = useState("");
  const [keyboardOpen, setKeyboardOpen] = useState(false);
  const [state, setState] = useState<
    | { kind: "idle" }
    | { kind: "sending" }
    | { kind: "sent" }
    | { kind: "error"; message: string }
  >({ kind: "idle" });

  // On the kiosk monitor there is no physical/native keyboard, so we render our
  // own on-screen email keyboard. On mobile the native OS keyboard is used.
  const useOnScreenKeyboard = layout === "monitor";

  const clearError = useCallback(() => {
    setState((s) => (s.kind === "error" ? { kind: "idle" } : s));
  }, []);

  const submit = useCallback(async () => {
    if (!isValidEmail(email)) {
      setState({ kind: "error", message: "Please enter a valid email address." });
      return;
    }
    setState({ kind: "sending" });
    try {
      await sendPhotoEmail({ to: email, imageSrc: imageUrl });
      setState({ kind: "sent" });
    } catch (err) {
      const message =
        err instanceof InvalidEmailError || err instanceof EmailDeliveryError
          ? err.message
          : "Couldn't send the email. Please check the address and try again.";
      setState({ kind: "error", message });
    }
  }, [email, imageUrl]);

  const onSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      void submit();
    },
    [submit],
  );

  if (state.kind === "sent") {
    return (
      <div className="result-screen__email">
        <p className="result-screen__email-sent" data-testid="email-sent" role="status">
          Sent! Check your inbox.
        </p>
      </div>
    );
  }

  return (
    <form className="result-screen__email" onSubmit={onSubmit}>
      <label className="result-screen__email-label">
        Email me my photo
        <input
          type="text"
          inputMode="email"
          autoComplete="email"
          // On the monitor kiosk, suppress any native keyboard and drive entry
          // from our on-screen keyboard instead.
          readOnly={useOnScreenKeyboard}
          placeholder="you@example.com"
          value={email}
          data-testid="email-input"
          onFocus={() => {
            if (useOnScreenKeyboard) setKeyboardOpen(true);
          }}
          onClick={() => {
            if (useOnScreenKeyboard) setKeyboardOpen(true);
          }}
          onChange={(ev) => {
            // Regex-sanitize so only valid email characters can ever be entered
            // (covers the native/mobile keyboard path).
            setEmail(sanitizeEmail(ev.target.value));
            clearError();
          }}
        />
      </label>

      {useOnScreenKeyboard && keyboardOpen && (
        <EmailKeyboard
          onInput={(ch) => {
            setEmail((cur) => sanitizeEmail(cur + ch));
            clearError();
          }}
          onBackspace={() => {
            setEmail((cur) => cur.slice(0, -1));
            clearError();
          }}
          onClear={() => {
            setEmail("");
            clearError();
          }}
          onDone={() => setKeyboardOpen(false)}
        />
      )}

      {state.kind === "error" && (
        <p role="alert" data-testid="email-error">
          {state.message}
        </p>
      )}
      <TouchButton
        type="submit"
        variant="secondary"
        testId="send-email-button"
        disabled={state.kind === "sending"}
      >
        {state.kind === "sending" ? "Sending…" : "Send Email"}
      </TouchButton>
    </form>
  );
}

export function StudioView({
  layout,
  photo,
  phase,
  generatedUrl,
  loadingMessage,
  errorMessage,
  onSelect,
  onNewSession,
}: StudioViewProps) {
  const locked = phase === "loading";
  const hasResult = phase === "result" && Boolean(generatedUrl);

  const overlay =
    phase === "loading" ? <GeneratingOverlay message={loadingMessage} /> : null;

  const backgrounds = (
    <OptionColumn
      title="Backgrounds"
      options={BACKGROUND_EFFECTS}
      locked={locked}
      layout={layout}
      onSelect={onSelect}
    />
  );
  const characters = (
    <OptionColumn
      title="Characters"
      options={PERSON_EFFECTS}
      locked={locked}
      layout={layout}
      onSelect={onSelect}
    />
  );

  // The result/error meta shown under the image area.
  const resultMeta: ReactNode = (
    <>
      {hasResult && (
        <p className="result-screen__notice" data-testid="ai-generated-notice">
          This image is AI-generated.
        </p>
      )}
      {phase === "error" && (
        <p className="kiosk-error" data-testid="studio-error" role="alert">
          {errorMessage ?? "Something went wrong. Pick an effect to try again."}
        </p>
      )}
      {hasResult && <EmailForm imageUrl={generatedUrl as string} layout={layout} />}
    </>
  );

  if (layout === "monitor") {
    // Monitor: [left options] [ center: original over generated ] [right options]
    return (
      <section
        className="studio studio--monitor"
        aria-label="Choose an effect"
        data-testid="studio-view"
        data-layout="monitor"
      >
        <aside className="studio__side studio__side--left">{backgrounds}</aside>

        <div className="studio__center">
          <figure className="studio__image-block">
            <figcaption className="studio__image-label">Original</figcaption>
            <img
              className="studio__image"
              src={photo}
              alt="Your original photo"
              data-testid="studio-original"
            />
          </figure>

          <figure className="studio__image-block studio__image-block--result">
            <figcaption className="studio__image-label">Generated</figcaption>
            <div className="studio__result-frame">
              {hasResult ? (
                <img
                  className="studio__image"
                  src={generatedUrl as string}
                  alt="Your AI-generated photo"
                  data-testid="result-image"
                />
              ) : (
                <div className="studio__result-placeholder">
                  {overlay ?? <span>Pick an effect to generate</span>}
                </div>
              )}
            </div>
          </figure>

          <div className="studio__meta">{resultMeta}</div>

          <div className="studio__actions">
            <PrimaryButton testId="new-session-button" onClick={onNewSession}>
              New Session
            </PrimaryButton>
          </div>
        </div>

        <aside className="studio__side studio__side--right">{characters}</aside>
      </section>
    );
  }

  // Mobile: image (carousel) on top, options stacked below.
  return (
    <section
      className="studio studio--mobile"
      aria-label="Choose an effect"
      data-testid="studio-view"
      data-layout="mobile"
    >
      <div className="studio__image-area">
        <ImageCarousel
          originalUrl={photo}
          generatedUrl={hasResult ? generatedUrl : null}
          generatingOverlay={overlay}
        />
        <div className="studio__meta">{resultMeta}</div>
      </div>

      <div className="studio__options">
        {backgrounds}
        {characters}
      </div>

      <div className="studio__actions">
        <PrimaryButton testId="new-session-button" onClick={onNewSession}>
          New Session
        </PrimaryButton>
      </div>
    </section>
  );
}

export default StudioView;
