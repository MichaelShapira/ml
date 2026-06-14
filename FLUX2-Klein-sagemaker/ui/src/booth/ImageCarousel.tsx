/**
 * ImageCarousel — mobile multi-image switcher.
 *
 * On phones the image area shows one image at a time with a tab per available
 * image so the visitor can flip between them. The booth feeds it the original
 * photo plus whichever generated results exist this session (background,
 * character, and the merged image), so the carousel grows a node each time a
 * new result lands. A single slide (just the original) shows no controls.
 *
 * The carousel snaps to the newest slide whenever the slide set grows, so a
 * freshly generated image is shown as soon as it arrives.
 */
import { useEffect, useRef, useState } from "react";

/** One selectable image in the carousel. */
export interface CarouselSlide {
  /** Stable key identifying the slide (e.g. "original", "background"). */
  key: string;
  /** Short tab caption (e.g. "Original", "Background"). */
  label: string;
  /** Image URL (data URI or object URL). */
  url: string;
}

export interface ImageCarouselProps {
  /** Slides to show, in display order. The first is typically the original. */
  slides: CarouselSlide[];
  /** Optional overlay shown over the active slide (e.g. a spinner). */
  generatingOverlay?: React.ReactNode;
  /**
   * Optional action rendered beneath the frame for the ACTIVE slide (e.g. a
   * "Share with me" button for a generated slide). Receives the active slide so
   * the caller can decide whether/what to render (e.g. nothing for the original).
   */
  renderSlideAction?: (slide: CarouselSlide) => React.ReactNode;
}

export function ImageCarousel({ slides, generatingOverlay, renderSlideAction }: ImageCarouselProps) {
  const keys = slides.map((s) => s.key);
  const [activeKey, setActiveKey] = useState<string>(() => keys[keys.length - 1] ?? "");
  const prevCountRef = useRef(slides.length);

  // Snap to the newest slide whenever the slide set grows (a new result landed).
  // Also keep the active key valid if the active slide disappears.
  useEffect(() => {
    const currentKeys = slides.map((s) => s.key);
    const grew = slides.length > prevCountRef.current;
    prevCountRef.current = slides.length;
    if (grew) {
      setActiveKey(currentKeys[currentKeys.length - 1] ?? "");
      return;
    }
    if (!currentKeys.includes(activeKey)) {
      setActiveKey(currentKeys[currentKeys.length - 1] ?? "");
    }
  }, [slides, activeKey]);

  const active = slides.find((s) => s.key === activeKey) ?? slides[slides.length - 1];
  if (!active) {
    return null;
  }

  return (
    <div className="image-carousel" data-testid="image-carousel">
      <div className="image-carousel__frame">
        <img
          className="image-carousel__image"
          src={active.url}
          alt={active.key === "original" ? "Your original photo" : `Your ${active.label} photo`}
          data-testid="carousel-image"
        />
        {generatingOverlay && (
          <div className="image-carousel__overlay">{generatingOverlay}</div>
        )}
      </div>

      {renderSlideAction && (
        <div className="image-carousel__action">{renderSlideAction(active)}</div>
      )}

      {slides.length > 1 && (
        <div className="image-carousel__controls" role="tablist" aria-label="Compare images">
          {slides.map((slide) => {
            const isActive = slide.key === active.key;
            return (
              <button
                key={slide.key}
                type="button"
                role="tab"
                aria-selected={isActive}
                className={`image-carousel__tab${isActive ? " is-active" : ""}`}
                data-testid={`carousel-tab-${slide.key}`}
                onClick={() => setActiveKey(slide.key)}
              >
                {slide.label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default ImageCarousel;
