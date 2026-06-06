/**
 * ReviewScreen — review the captured photo (Requirements 3.3, 4).
 *
 * Shows the captured still with a Reset control (discard → back to camera,
 * Req 4.1) and a Continue control (retain → effect selection, Req 4.2). The
 * parent dispatches RESET / CONTINUE to the state machine.
 */
import { KioskScreen, PrimaryButton, TouchButton } from "../theme";

export interface ReviewScreenProps {
  /** The captured photo as a base64 data URL. */
  photo: string;
  /** Discard the photo and return to the camera (RESET). */
  onReset: () => void;
  /** Keep the photo and continue to effect selection (CONTINUE). */
  onContinue: () => void;
}

export function ReviewScreen({ photo, onReset, onContinue }: ReviewScreenProps) {
  return (
    <KioskScreen
      label="Review your photo"
      testId="review-screen"
      footer={
        <div className="review-screen__actions">
          <TouchButton variant="secondary" testId="reset-button" onClick={onReset}>
            Reset
          </TouchButton>
          <PrimaryButton testId="continue-button" onClick={onContinue}>
            Continue
          </PrimaryButton>
        </div>
      }
    >
      <img
        className="review-screen__photo"
        src={photo}
        alt="Your captured photo"
        data-testid="review-photo"
      />
    </KioskScreen>
  );
}

export default ReviewScreen;
