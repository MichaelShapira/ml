/**
 * ImageCarousel — mobile original/generated image switcher.
 *
 * On phones the generated image replaces the original in the same slot, with a
 * two-item carousel so the visitor can flip between "Original" and "Generated".
 * When no generated image exists yet, only the original is shown (no controls).
 * A new generation replaces the previous generated image (the parent simply
 * passes the latest `generatedUrl`).
 */
import { useEffect, useState } from "react";

export interface ImageCarouselProps {
  /** The original captured photo (always present). */
  originalUrl: string;
  /** The latest generated image, or null/undefined before/while generating. */
  generatedUrl?: string | null;
  /** Optional overlay shown over the generated slot (e.g. a spinner). */
  generatingOverlay?: React.ReactNode;
}

type Slide = "generated" | "original";

export function ImageCarousel({
  originalUrl,
  generatedUrl,
  generatingOverlay,
}: ImageCarouselProps) {
  const hasGenerated = Boolean(generatedUrl);
  // Default to showing the generated image when it's available (the visitor
  // most wants to see the result); otherwise the original.
  const [slide, setSlide] = useState<Slide>(hasGenerated ? "generated" : "original");

  // When a (new) generated image arrives, snap to it.
  useEffect(() => {
    if (hasGenerated) {
      setSlide("generated");
    } else {
      setSlide("original");
    }
  }, [generatedUrl, hasGenerated]);

  const showingGenerated = slide === "generated" && hasGenerated;
  const activeUrl = showingGenerated ? (generatedUrl as string) : originalUrl;

  return (
    <div className="image-carousel" data-testid="image-carousel">
      <div className="image-carousel__frame">
        <img
          className="image-carousel__image"
          src={activeUrl}
          alt={showingGenerated ? "Your AI-generated photo" : "Your original photo"}
          data-testid="carousel-image"
        />
        {generatingOverlay && (
          <div className="image-carousel__overlay">{generatingOverlay}</div>
        )}
      </div>

      {hasGenerated && (
        <div className="image-carousel__controls" role="tablist" aria-label="Compare images">
          <button
            type="button"
            role="tab"
            aria-selected={!showingGenerated}
            className={`image-carousel__tab${!showingGenerated ? " is-active" : ""}`}
            data-testid="carousel-original-tab"
            onClick={() => setSlide("original")}
          >
            Original
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={showingGenerated}
            className={`image-carousel__tab${showingGenerated ? " is-active" : ""}`}
            data-testid="carousel-generated-tab"
            onClick={() => setSlide("generated")}
          >
            Generated
          </button>
        </div>
      )}
    </div>
  );
}

export default ImageCarousel;
