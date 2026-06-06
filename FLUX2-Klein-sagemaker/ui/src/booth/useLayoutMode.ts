/**
 * useLayoutMode — pick the booth's presentation layout from the viewport.
 *
 * The capture studio renders two structurally different layouts:
 *   - `"monitor"`: a large portrait monitor. Background options stack on the
 *     left, character options on the right, the original photo is centered and
 *     the generated image sits directly below it.
 *   - `"mobile"`: a phone. The image area is on top (with an original/generated
 *     carousel), and the option groups stack below it.
 *
 * The split is by viewport width: phones and narrow windows get `"mobile"`,
 * wide screens get `"monitor"`. A `matchMedia` listener keeps it live across
 * rotation / resize. SSR-safe (defaults to `"mobile"` when `window` is absent).
 */
import { useEffect, useState } from "react";

/** The two booth layouts. */
export type LayoutMode = "monitor" | "mobile";

/**
 * Minimum viewport width (CSS px) at which the wide three-column monitor layout
 * is used. Below this we use the stacked mobile layout with the carousel.
 * Tuned so tablets in portrait and phones use mobile; a large portrait monitor
 * (typically >= 900px wide even in portrait) uses the monitor layout.
 */
export const MONITOR_MIN_WIDTH_PX = 820;

/** The media query that selects the monitor layout. */
export const MONITOR_MEDIA_QUERY = `(min-width: ${MONITOR_MIN_WIDTH_PX}px)`;

/** Resolve the current layout mode from `window` (SSR-safe). */
function readLayoutMode(): LayoutMode {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return "mobile";
  }
  return window.matchMedia(MONITOR_MEDIA_QUERY).matches ? "monitor" : "mobile";
}

/**
 * React hook returning the live {@link LayoutMode}. Re-renders when the viewport
 * crosses the {@link MONITOR_MIN_WIDTH_PX} breakpoint.
 */
export function useLayoutMode(): LayoutMode {
  const [mode, setMode] = useState<LayoutMode>(readLayoutMode);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }
    const mql = window.matchMedia(MONITOR_MEDIA_QUERY);
    const onChange = () => setMode(mql.matches ? "monitor" : "mobile");
    onChange();
    // addEventListener is the modern API; fall back to addListener for older
    // engines (kept for safety, harmless on current browsers).
    if (typeof mql.addEventListener === "function") {
      mql.addEventListener("change", onChange);
      return () => mql.removeEventListener("change", onChange);
    }
    mql.addListener(onChange);
    return () => mql.removeListener(onChange);
  }, []);

  return mode;
}

export default useLayoutMode;
