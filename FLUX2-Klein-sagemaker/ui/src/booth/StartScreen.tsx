/**
 * StartScreen — the single-tap entry screen (Requirement 1).
 *
 * Renders one primary Start control in the portrait kiosk frame. Tapping it
 * asks the parent to begin the capture flow (the parent dispatches START to the
 * state machine). Per Requirement 1.3 / 25.3 the screen shows only this one
 * control.
 */
import { KioskScreen, PrimaryButton } from "../theme";

export interface StartScreenProps {
  /** Begin the capture flow (parent dispatches START). */
  onStart: () => void;
}

export function StartScreen({ onStart }: StartScreenProps) {
  return (
    <KioskScreen
      label="Start"
      testId="start-screen"
      footer={
        <PrimaryButton testId="start-button" onClick={onStart}>
          Start
        </PrimaryButton>
      }
    >
      <div className="start-screen__hero">
        <h1 className="start-screen__title">AI Photo Booth</h1>
        <p className="start-screen__subtitle">Tap Start to take your photo</p>
      </div>
    </KioskScreen>
  );
}

export default StartScreen;
