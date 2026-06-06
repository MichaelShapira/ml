/**
 * TouchButton — the base interactive control for the kiosk (Requirement 25.2,
 * 25.4).
 *
 * Wraps a native `<button>` so it inherits real button semantics (keyboard
 * and switch-access operability, correct focus handling) while guaranteeing
 * the touch contract:
 *
 *   - At least a 44x44 CSS-pixel target, enforced by the `.touch-button` class
 *     (`min-width`/`min-height: var(--touch-target-min)`).
 *   - Press feedback keyed to `:active`/`:focus-visible` rather than `:hover`,
 *     so there are no hover-only affordances and the control works by touch.
 *
 * `type` defaults to `"button"` so a TouchButton never accidentally submits a
 * form. All other native button props pass straight through.
 */
import type { ButtonHTMLAttributes, ReactNode } from "react";

/** Visual emphasis of a touch control. */
export type TouchButtonVariant = "primary" | "secondary" | "danger";

export interface TouchButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual emphasis. Defaults to `"secondary"`. */
  variant?: TouchButtonVariant;
  /** When true, the control fills its container and uses the hero sizing. */
  block?: boolean;
  /** Button content (label, and optionally an icon). */
  children: ReactNode;
  /** Optional test id for component tests. */
  testId?: string;
}

const VARIANT_CLASS: Record<TouchButtonVariant, string> = {
  primary: "touch-button--primary",
  secondary: "touch-button--secondary",
  danger: "touch-button--danger",
};

/** Joins truthy class names into a single `className` string. */
function cx(...names: Array<string | false | undefined>): string {
  return names.filter(Boolean).join(" ");
}

export function TouchButton({
  variant = "secondary",
  block = false,
  className,
  children,
  testId,
  type,
  ...rest
}: TouchButtonProps) {
  return (
    <button
      // Default to a non-submitting button so it is safe inside forms.
      type={type ?? "button"}
      className={cx(
        "touch-button",
        VARIANT_CLASS[variant],
        block && "touch-button--block",
        className
      )}
      data-testid={testId}
      {...rest}
    >
      {children}
    </button>
  );
}

export default TouchButton;
