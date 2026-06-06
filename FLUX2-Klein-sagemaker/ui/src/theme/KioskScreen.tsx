/**
 * KioskScreen — the portrait content frame for a single capture-flow or admin
 * step (Requirement 25.1, 25.3).
 *
 * Every screen of the app renders inside a `KioskScreen` so the whole UI shares
 * one portrait frame (a centered column, capped at the portrait max-width, that
 * always reads taller than it is wide). The component exposes three optional
 * regions:
 *
 *   - `header`  — fixed top region (e.g. title + connection indicator).
 *   - children  — the step body; grows to fill the frame and is centered.
 *   - `footer`  — fixed bottom action region. Per Requirement 25.3 a step
 *                 should place only the controls it needs here (typically a
 *                 single primary action).
 *
 * The component is presentational only: it owns no state and performs no I/O,
 * so it is trivially reusable by the capture-flow screens (tasks 9.1-9.4) and
 * the admin screens (tasks 10.x).
 */
import type { ReactNode } from "react";

export interface KioskScreenProps {
  /** Optional fixed header region (e.g. step title, connection indicator). */
  header?: ReactNode;
  /** The step body; centered and allowed to grow within the portrait frame. */
  children?: ReactNode;
  /** Optional fixed footer region for the step's action control(s). */
  footer?: ReactNode;
  /** Extra class names appended to the screen root. */
  className?: string;
  /**
   * Accessible label for the screen region, announced to assistive tech so the
   * booth is navigable without a mouse (Requirement 25.4).
   */
  label?: string;
  /** Optional test id for component tests (tasks 9.6 / 10.4). */
  testId?: string;
}

/** Joins truthy class names into a single `className` string. */
function cx(...names: Array<string | false | undefined>): string {
  return names.filter(Boolean).join(" ");
}

export function KioskScreen({
  header,
  children,
  footer,
  className,
  label,
  testId,
}: KioskScreenProps) {
  return (
    <section
      className={cx("kiosk-screen", className)}
      aria-label={label}
      data-testid={testId}
    >
      {header !== undefined && (
        <header className="kiosk-screen__header">{header}</header>
      )}
      <div className="kiosk-screen__body">{children}</div>
      {footer !== undefined && (
        <footer className="kiosk-screen__footer">{footer}</footer>
      )}
    </section>
  );
}

export default KioskScreen;
