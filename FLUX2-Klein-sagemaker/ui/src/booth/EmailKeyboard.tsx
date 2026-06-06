/**
 * EmailKeyboard — a minimal on-screen keyboard for entering an email address.
 *
 * The booth runs on a touch monitor with NO physical keyboard, so on the
 * monitor layout we render our own keyboard instead of relying on a native
 * one (which only appears on touch devices like phones). It only exposes the
 * characters that can appear in an email address — lowercase letters, digits,
 * and the handful of allowed symbols (`@ . _ - +`) — plus a few common-domain
 * shortcuts, backspace, space-free by design, and Done.
 *
 * It is a controlled component: it never owns the value. Each key calls
 * `onInput(char)` / `onBackspace()` / `onDone()` and the parent applies the
 * edit, so the same regex validation path is used regardless of input source.
 */
import { TouchButton } from "../theme";

export interface EmailKeyboardProps {
  /** Append a single character to the email value. */
  onInput: (char: string) => void;
  /** Delete the last character. */
  onBackspace: () => void;
  /** Clear the whole field. */
  onClear: () => void;
  /** Finish editing (e.g. close the keyboard). */
  onDone: () => void;
}

/** Letter rows (lowercase — email local parts are case-insensitive in practice). */
const LETTER_ROWS: readonly string[][] = [
  ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
  ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
  ["z", "x", "c", "v", "b", "n", "m"],
];

/** Digits row. */
const DIGITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"];

/** Symbols valid in an email address (no spaces). */
const SYMBOLS = ["@", ".", "_", "-", "+"];

/** Common domain shortcuts to speed up entry on a touch screen. */
const DOMAIN_SHORTCUTS = ["@gmail.com", "@outlook.com", "@yahoo.com", ".com"];

export function EmailKeyboard({
  onInput,
  onBackspace,
  onClear,
  onDone,
}: EmailKeyboardProps) {
  return (
    <div className="email-keyboard" data-testid="email-keyboard" role="group" aria-label="On-screen keyboard">
      <div className="email-keyboard__row">
        {DIGITS.map((d) => (
          <button
            key={d}
            type="button"
            className="email-keyboard__key"
            data-testid={`key-${d}`}
            onClick={() => onInput(d)}
          >
            {d}
          </button>
        ))}
      </div>

      {LETTER_ROWS.map((row, i) => (
        <div className="email-keyboard__row" key={`letters-${i}`}>
          {row.map((ch) => (
            <button
              key={ch}
              type="button"
              className="email-keyboard__key"
              data-testid={`key-${ch}`}
              onClick={() => onInput(ch)}
            >
              {ch}
            </button>
          ))}
        </div>
      ))}

      <div className="email-keyboard__row">
        {SYMBOLS.map((s) => (
          <button
            key={s}
            type="button"
            className="email-keyboard__key email-keyboard__key--symbol"
            data-testid={`key-symbol-${s}`}
            onClick={() => onInput(s)}
          >
            {s}
          </button>
        ))}
        <button
          type="button"
          className="email-keyboard__key email-keyboard__key--wide"
          data-testid="key-backspace"
          aria-label="Backspace"
          onClick={onBackspace}
        >
          ⌫
        </button>
      </div>

      <div className="email-keyboard__row email-keyboard__row--shortcuts">
        {DOMAIN_SHORTCUTS.map((s) => (
          <button
            key={s}
            type="button"
            className="email-keyboard__key email-keyboard__key--shortcut"
            data-testid={`key-shortcut-${s}`}
            onClick={() => onInput(s)}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="email-keyboard__row email-keyboard__row--actions">
        <TouchButton
          variant="secondary"
          testId="keyboard-clear"
          onClick={onClear}
        >
          Clear
        </TouchButton>
        <TouchButton variant="primary" testId="keyboard-done" onClick={onDone}>
          Done
        </TouchButton>
      </div>
    </div>
  );
}

export default EmailKeyboard;
