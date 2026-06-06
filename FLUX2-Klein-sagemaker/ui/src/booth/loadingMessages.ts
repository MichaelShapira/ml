/**
 * Playful rotating loading messages shown while a transformation is generated
 * (Requirement 9.1/9.2). The booth's endpoint takes several seconds per image,
 * so instead of a static spinner we cycle through light-hearted status lines.
 *
 * Pure data + a small helper so it can be unit-tested independently of React.
 */

/** At least 30 rotating messages for the loading state. */
export const LOADING_MESSAGES: readonly string[] = [
  "Brewing the paints…",
  "Teaching pixels to pose…",
  "Mixing a fresh batch of imagination…",
  "Warming up the magic brushes…",
  "Convincing the photons to cooperate…",
  "Summoning your alter ego…",
  "Sketching the impossible…",
  "Negotiating with the color wheel…",
  "Polishing every last pixel…",
  "Consulting the muses…",
  "Untangling the creativity cables…",
  "Sprinkling a little extra sparkle…",
  "Rendering your moment of fame…",
  "Aligning the artistic chakras…",
  "Whispering sweet prompts to the model…",
  "Stretching the canvas…",
  "Calibrating the awesome levels…",
  "Letting the GPUs flex a little…",
  "Painting outside the lines (on purpose)…",
  "Bribing the pixels with extra detail…",
  "Dusting off the digital easel…",
  "Tuning the dream frequency…",
  "Reticulating the splines…",
  "Coaxing colors out of hiding…",
  "Composing your masterpiece…",
  "Adding a dramatic flair…",
  "Asking the AI to try its best angle…",
  "Buffing the highlights…",
  "Loading creativity… please hold the pose…",
  "Almost there, adding the finishing touches…",
  "Double-checking your good side…",
  "Making it gallery-worthy…",
  "Herding the rogue pixels back into place…",
  "Giving the background a glow-up…",
];

/**
 * Pick the message for a given elapsed time, cycling every `intervalMs`.
 * Deterministic given the inputs, so it's trivially testable.
 */
export function messageForElapsed(
  elapsedMs: number,
  intervalMs = 2500,
  messages: readonly string[] = LOADING_MESSAGES,
): string {
  if (messages.length === 0) {
    return "";
  }
  const safeElapsed = Number.isFinite(elapsedMs) && elapsedMs > 0 ? elapsedMs : 0;
  const index = Math.floor(safeElapsed / intervalMs) % messages.length;
  return messages[index];
}
