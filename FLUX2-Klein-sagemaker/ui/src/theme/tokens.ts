/**
 * Portrait/touch design tokens (Requirement 25).
 *
 * These are the single source of truth for the kiosk's spacing, sizing,
 * radius, typography, and colour scales. The same values are mirrored as CSS
 * custom properties in `theme/index.css` (the `:root` block) so styling can
 * happen in plain CSS *and* the rare inline style or computed value in a React
 * component can reference the exact same number without drift.
 *
 * The most important token is {@link TOUCH_TARGET_MIN_PX}: every interactive
 * control must be at least 44x44 CSS pixels (Requirement 25.2). It is exported
 * both as a number (for measurement/derivation) and as a CSS length string
 * (for inline styles).
 *
 * Nothing here imports React or the DOM, so the tokens are safe to use from
 * pure modules and tests as well as components.
 */

/**
 * Minimum touch target size in CSS pixels (Requirement 25.2).
 *
 * Apple HIG and the WCAG 2.5.5 (Target Size) guidance both land on ~44px as
 * the comfortable minimum for finger taps on a kiosk; this is the value the
 * base control styles enforce as `min-width`/`min-height`.
 */
export const TOUCH_TARGET_MIN_PX = 44 as const;

/** {@link TOUCH_TARGET_MIN_PX} as a CSS length string (e.g. for inline styles). */
export const touchTargetMin = `${TOUCH_TARGET_MIN_PX}px` as const;

/**
 * Spacing scale (CSS length strings). A small, rhythmic set keeps the
 * per-step screens uncluttered (Requirement 25.3) rather than offering an
 * unbounded range of paddings.
 */
export const space = {
  xs: "0.5rem",
  sm: "0.75rem",
  md: "1rem",
  lg: "1.5rem",
  xl: "2rem",
  xxl: "3rem",
} as const;

/** Corner radii for surfaces and touch controls. */
export const radius = {
  sm: "0.5rem",
  md: "0.875rem",
  lg: "1.25rem",
  pill: "999px",
} as const;

/**
 * Type scale. Tap-friendly kiosks use larger base text than a desktop app so
 * labels stay legible at arm's length and controls stay easy to hit.
 */
export const fontSize = {
  sm: "1rem",
  md: "1.125rem",
  lg: "1.375rem",
  xl: "1.875rem",
  display: "2.5rem",
} as const;

/** Font weights used across the theme. */
export const fontWeight = {
  regular: 400,
  medium: 600,
  bold: 700,
} as const;

/**
 * Colour palette. Values are duplicated into the `:root` CSS variables; the
 * kiosk uses a single dark, high-contrast theme so it reads well under lobby
 * lighting and does not depend on the OS light/dark preference.
 */
export const color = {
  /** App backdrop behind the portrait frame. */
  backdrop: "#0b0d12",
  /** Portrait frame / screen background. */
  background: "#13161f",
  /** Raised surface (cards, option tiles). */
  surface: "#1d2230",
  /** Primary body text. */
  text: "#f5f7fa",
  /** De-emphasised text (hints, captions, AI-generated notice). */
  muted: "#aab2c5",
  /** Primary action colour. */
  primary: "#4c8dff",
  /** Pressed primary action colour. */
  primaryActive: "#2f6fe0",
  /** Text/icon colour rendered on top of the primary colour. */
  primaryText: "#0b0d12",
  /** Secondary (outline) control border. */
  border: "#39415a",
  /** Destructive / error colour (stop endpoint, failure messages). */
  danger: "#ff6b6b",
  /** Focus ring colour for keyboard and switch-access focus. */
  focus: "#9ec2ff",
  /** Connection indicator: connected. */
  connected: "#3ddc84",
  /** Connection indicator: disconnected. */
  disconnected: "#ff6b6b",
} as const;

/**
 * The maximum width of the portrait content frame. On a true portrait kiosk
 * the frame fills the height; this cap keeps line lengths and control rows
 * comfortable and letterboxes the content on wide/landscape dev screens so the
 * layout always reads as portrait (height > width — Requirement 25.1).
 */
export const portraitMaxWidth = "30rem" as const;

/** The complete token object, handy for spreading or snapshotting. */
export const theme = {
  touchTargetMin,
  touchTargetMinPx: TOUCH_TARGET_MIN_PX,
  space,
  radius,
  fontSize,
  fontWeight,
  color,
  portraitMaxWidth,
} as const;

/** The shape of {@link theme}. */
export type Theme = typeof theme;
