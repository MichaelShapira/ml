/**
 * Portrait/touch theme barrel (Requirement 25).
 *
 * Single import surface for the kiosk theme so capture-flow (tasks 9.1-9.4)
 * and admin (tasks 10.x) screens can pull layout primitives, touch controls,
 * and design tokens from one place:
 *
 *   import { KioskScreen, PrimaryButton, TouchButton, theme } from "../theme";
 *
 * The global stylesheet (`theme/index.css`) is imported once at the app entry
 * point (`main.tsx`); it is intentionally not re-exported here because CSS is a
 * side-effecting import, not a value.
 */

export { KioskScreen } from "./KioskScreen";
export type { KioskScreenProps } from "./KioskScreen";

export { TouchButton } from "./TouchButton";
export type { TouchButtonProps, TouchButtonVariant } from "./TouchButton";

export { PrimaryButton } from "./PrimaryButton";
export type { PrimaryButtonProps } from "./PrimaryButton";

export { ConnectionIndicator } from "./ConnectionIndicator";
export type {
  ConnectionIndicatorProps,
  ConnectionStatus,
} from "./ConnectionIndicator";

export {
  theme,
  touchTargetMin,
  TOUCH_TARGET_MIN_PX,
  space,
  radius,
  fontSize,
  fontWeight,
  color,
  portraitMaxWidth,
} from "./tokens";
export type { Theme } from "./tokens";
