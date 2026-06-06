# theme/

Portrait-mode touch theme for the kiosk SPA (Requirement 25).

The capture-flow screens (tasks 9.1-9.4) and admin screens (tasks 10.x) build
on these primitives instead of writing bespoke layout/control CSS.

## Files

- `index.css` — global stylesheet: design tokens as CSS custom properties, the
  portrait content frame (height > width, Req 25.1), base touch-control styles
  enforcing ≥44×44 CSS-pixel targets (Req 25.2), per-step layout utilities
  (Req 25.3), and touch-first focus/active states with no hover-only
  affordances (Req 25.4). Imported once in `src/main.tsx`.
- `tokens.ts` — typed design tokens (spacing, radius, type scale, colours,
  `TOUCH_TARGET_MIN_PX`, portrait max-width). Mirrors the CSS custom properties
  so components can reference the same values from TypeScript.
- `KioskScreen.tsx` — the portrait frame for a single step: optional `header`,
  centered `children` body, and a `footer` action area.
- `TouchButton.tsx` — base touch control (`primary` / `secondary` / `danger`
  variants, optional `block` hero sizing) over a native `<button>`.
- `PrimaryButton.tsx` — the single prominent primary action per step.
- `ConnectionIndicator.tsx` — identity-free two-state connection badge
  (Req 13.1-13.4), shared chrome styled by the theme.
- `index.ts` — barrel exporting the primitives, the `ConnectionIndicator`, and
  the tokens.

## Usage

```tsx
import { KioskScreen, PrimaryButton, ConnectionIndicator } from "../theme";

function StartScreen({ onStart }: { onStart: () => void }) {
  return (
    <KioskScreen
      label="Start"
      header={<ConnectionIndicator status="connected" />}
      footer={<PrimaryButton onClick={onStart}>Start</PrimaryButton>}
    >
      <h1 className="kiosk-title">AI Photo Booth</h1>
      <p className="kiosk-subtitle">Tap Start to take your photo.</p>
    </KioskScreen>
  );
}
```

The global stylesheet is a side-effecting import and is loaded once at the app
entry (`src/main.tsx`); screens import only the components/tokens from `../theme`.
