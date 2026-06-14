/**
 * StudioView — the combined capture studio (Effects + Loading + Result + Error).
 *
 * Renders the original photo, the effect options, and any generated results
 * co-present. The capture-flow machine is the source of truth; this component
 * renders whichever phase the machine is in and keeps the options visible so
 * the visitor can pick another effect and regenerate in place.
 *
 * A session can hold up to three generated images at once — a background result,
 * a character (dress) result, and a merged image combining the two:
 *   - "monitor": background options on the LEFT, character options on the RIGHT,
 *     original photo centered, and each generated image stacked directly BELOW
 *     it (background, then character, then merged).
 *   - "mobile": the image area is a carousel with one node per image (Original,
 *     Background, Character, Merged); the two option groups stack BELOW it.
 *
 * Once BOTH a background and a character image exist, a "Merge" button appears
 * that combines them into a single image.
 *
 * Sharing: each generated image has a "Share with me" button. Tapping it uploads
 * that image to the short-lived Share_Bucket, presigns a 15-minute download URL,
 * and shows it as a QR code (see {@link ShareQrModal}). There is no email.
 */
import { useCallback, useState, type ReactNode } from "react";
import {
  BACKGROUND_EFFECTS,
  PERSON_EFFECTS,
  effectImageUrl,
  type EffectOption,
} from "./effects";
import { ImageCarousel, type CarouselSlide } from "./ImageCarousel";
import { ShareQrModal, type ShareState } from "./ShareQrModal";
import { TouchButton, PrimaryButton } from "../theme";
import { shareImage } from "../api/share";
import type { LayoutMode } from "./useLayoutMode";

/** The generation phase the studio is showing. */
export type StudioPhase = "idle" | "loading" | "result" | "error";

/** The generated images available this session, keyed by slot. */
export interface StudioResults {
  /** Background-effect result (object URL), if generated. */
  background?: string | null;
  /** Character/dress-effect result (object URL), if generated. */
  person?: string | null;
  /** Merged image (object URL), if produced. */
  merged?: string | null;
}

/** The slot column option group binds each column to the slot it fills. */
type Slot = "background" | "person";

export interface StudioViewProps {
  /** Layout to render (monitor vs mobile). */
  layout: LayoutMode;
  /** The original captured photo. */
  photo: string;
  /** Current generation phase derived from the machine state. */
  phase: StudioPhase;
  /** Generated images by slot. */
  results?: StudioResults;
  /** Rotating loading message shown while phase is "loading". */
  loadingMessage?: string;
  /** Error copy shown when phase is "error". */
  errorMessage?: string;
  /** Pick an effect (starts/restarts generation). Ignored while loading. */
  onSelect: (effectId: string, slot: Slot) => void;
  /** Merge the background + character images into one. Shown only when allowed. */
  onMerge: () => void;
  /** Start a fresh session (discard everything, back to Start). */
  onNewSession: () => void;
}

/** Human-readable caption for each result slot. */
const SLOT_LABEL: Record<"background" | "person" | "merged", string> = {
  background: "Background",
  person: "Character",
  merged: "Merged",
};

/** A vertical list of effect option buttons for one category/slot. */
function OptionColumn({
  title,
  options,
  slot,
  locked,
  layout,
  onSelect,
}: {
  title: string;
  options: readonly EffectOption[];
  slot: Slot;
  locked: boolean;
  layout: LayoutMode;
  onSelect: (effectId: string, slot: Slot) => void;
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
              onClick={() => onSelect(option.id, slot)}
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

export function StudioView({
  layout,
  photo,
  phase,
  results,
  loadingMessage,
  errorMessage,
  onSelect,
  onMerge,
  onNewSession,
}: StudioViewProps) {
  const locked = phase === "loading";

  // "Share with me" state for the QR overlay.
  const [share, setShare] = useState<ShareState>({ status: "idle" });

  const handleShare = useCallback(async (imageUrl: string) => {
    setShare({ status: "loading" });
    try {
      const result = await shareImage(imageUrl);
      setShare({
        status: "ready",
        url: result.url,
        expiresInSeconds: result.expiresInSeconds,
      });
    } catch {
      setShare({
        status: "error",
        message: "Couldn't prepare the download. Please try again.",
      });
    }
  }, []);

  const closeShare = useCallback(() => setShare({ status: "idle" }), []);

  /** A "Share with me" button for a specific generated image. */
  const shareButton = (url: string, slot: "background" | "person" | "merged") => (
    <TouchButton
      variant="secondary"
      testId={`share-button-${slot}`}
      disabled={share.status === "loading"}
      onClick={() => void handleShare(url)}
    >
      Share with me
    </TouchButton>
  );

  // Generated images present this session, in display order.
  const present: { slot: "background" | "person" | "merged"; url: string }[] = [];
  if (results?.background) present.push({ slot: "background", url: results.background });
  if (results?.person) present.push({ slot: "person", url: results.person });
  if (results?.merged) present.push({ slot: "merged", url: results.merged });

  const hasAnyResult = present.length > 0;
  // Merge is offered only once BOTH source images exist.
  const mergeAvailable = Boolean(results?.background) && Boolean(results?.person);

  const overlay =
    phase === "loading" ? <GeneratingOverlay message={loadingMessage} /> : null;

  const backgrounds = (
    <OptionColumn
      title="Backgrounds"
      options={BACKGROUND_EFFECTS}
      slot="background"
      locked={locked}
      layout={layout}
      onSelect={onSelect}
    />
  );
  const characters = (
    <OptionColumn
      title="Characters"
      options={PERSON_EFFECTS}
      slot="person"
      locked={locked}
      layout={layout}
      onSelect={onSelect}
    />
  );

  // The result/error meta shown under the image area.
  const resultMeta: ReactNode = (
    <>
      {hasAnyResult && (
        <p className="result-screen__notice" data-testid="ai-generated-notice">
          This image is AI-generated.
        </p>
      )}
      {phase === "error" && (
        <p className="kiosk-error" data-testid="studio-error" role="alert">
          {errorMessage ?? "Something went wrong. Pick an effect to try again."}
        </p>
      )}
    </>
  );

  // Shared action controls: Merge (when available) + New Session.
  const actions: ReactNode = (
    <div className="studio__actions">
      {mergeAvailable && (
        <TouchButton
          variant="primary"
          testId="merge-button"
          disabled={locked}
          onClick={onMerge}
        >
          {results?.merged ? "Merge again" : "Merge background + character"}
        </TouchButton>
      )}
      <PrimaryButton testId="new-session-button" onClick={onNewSession}>
        New Session
      </PrimaryButton>
    </div>
  );

  const shareModal = <ShareQrModal state={share} layout={layout} onClose={closeShare} />;

  if (layout === "monitor") {
    // Monitor: [left options] [ center: original then each result stacked ] [right options]
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

          {/* Each generated image stacked below the original, with its own Share button. */}
          {present.map(({ slot, url }) => (
            <figure key={slot} className="studio__image-block studio__image-block--result">
              <figcaption className="studio__image-label">{SLOT_LABEL[slot]}</figcaption>
              <div className="studio__result-frame">
                <img
                  className="studio__image"
                  src={url}
                  alt={`Your ${SLOT_LABEL[slot]} photo`}
                  data-testid={`result-image-${slot}`}
                />
              </div>
              <div className="studio__image-action">{shareButton(url, slot)}</div>
            </figure>
          ))}

          {/* Loading placeholder block, or an empty prompt when nothing yet. */}
          {phase === "loading" && (
            <figure className="studio__image-block studio__image-block--result">
              <div className="studio__result-frame">
                <div className="studio__result-placeholder">{overlay}</div>
              </div>
            </figure>
          )}
          {phase !== "loading" && !hasAnyResult && (
            <figure className="studio__image-block studio__image-block--result">
              <div className="studio__result-frame">
                <div className="studio__result-placeholder">
                  <span>Pick an effect to generate</span>
                </div>
              </div>
            </figure>
          )}

          <div className="studio__meta">{resultMeta}</div>

          {actions}
        </div>

        <aside className="studio__side studio__side--right">{characters}</aside>

        {shareModal}
      </section>
    );
  }

  // Mobile: image (carousel) on top, options stacked below.
  const slides: CarouselSlide[] = [
    { key: "original", label: "Original", url: photo },
    ...present.map(({ slot, url }) => ({ key: slot, label: SLOT_LABEL[slot], url })),
  ];

  return (
    <section
      className="studio studio--mobile"
      aria-label="Choose an effect"
      data-testid="studio-view"
      data-layout="mobile"
    >
      <div className="studio__image-area">
        <ImageCarousel
          slides={slides}
          generatingOverlay={overlay}
          // Render a Share button beneath the active slide when it's a generated
          // one (never for the original).
          renderSlideAction={(slide) =>
            slide.key === "original"
              ? null
              : shareButton(slide.url, slide.key as "background" | "person" | "merged")
          }
        />
        <div className="studio__meta">{resultMeta}</div>
      </div>

      <div className="studio__options">
        {backgrounds}
        {characters}
      </div>

      {actions}

      {shareModal}
    </section>
  );
}

export default StudioView;
