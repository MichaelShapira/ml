/**
 * Effect_Option catalog and prompt mapping (pure logic).
 *
 * This module is the single source of truth for the photo booth's predefined
 * transformation choices. It is imported by both the backend request-builder
 * (to resolve a selected effect to its prompt) and the UI effect selector /
 * capture-flow state machine (to render the options and record a selection).
 *
 * Pure logic only: no AWS SDK calls, no I/O.
 *
 * Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 7.1
 */

/** The two categories an Effect_Option can belong to. */
export type EffectCategory = "background" | "person";

/**
 * A predefined transformation choice.
 *
 * The captured photo is always supplied to the endpoint as the reference
 * image, so each `prompt` is written as an edit instruction.
 */
export interface EffectOption {
  /** Stable lookup key, unique across the whole catalog (e.g. "bg_spaceship"). */
  id: string;
  /** Whether this option replaces the background or transforms the person. */
  category: EffectCategory;
  /** Human-readable, touch-control label. Non-empty. */
  label: string;
  /** Prompt string mapped to Async_Request.inputs. Non-empty. */
  prompt: string;
}

/**
 * The predefined effect catalog: exactly 6 background options and exactly 6
 * person options (Requirements 5 and 6). Every option has a unique id and a
 * non-empty label and prompt.
 */
export const EFFECTS: readonly EffectOption[] = [
  // --- Background (6) ---
  // FLUX.2 is instruction-based and ignores negative prompts, so each prompt
  // states the change first, then positively describes what to preserve. The
  // person(s) are kept exactly; only the background is replaced.
  {
    id: "bg_spaceship",
    category: "background",
    label: "Spaceship interior",
    prompt:
      "Replace only the background with the interior of a futuristic spaceship: sleek metal corridors, glowing control panels, soft blue lighting. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic.",
  },
  {
    id: "bg_colosseum",
    category: "background",
    label: "Roman colosseum",
    prompt:
      "Replace only the background with the interior of the ancient Roman Colosseum under a dramatic sky. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic, cinematic lighting.",
  },
  {
    id: "bg_tropical",
    category: "background",
    label: "Tropical background",
    prompt:
      "Replace only the background with a sunlit tropical beach: palm trees, turquoise water, white sand. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic.",
  },
  {
    id: "bg_snowy_peak",
    category: "background",
    label: "Snowy mountain peak",
    prompt:
      "Replace only the background with a snowy mountain summit under a crisp blue sky, distant peaks and drifting snow. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic.",
  },
  {
    id: "bg_neon_city",
    category: "background",
    label: "Neon city street at night",
    prompt:
      "Replace only the background with a neon-lit city street at night, rain-slick pavement reflecting pink and blue signage. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Cinematic, photorealistic.",
  },
  {
    id: "bg_enchanted_forest",
    category: "background",
    label: "Enchanted forest",
    prompt:
      "Replace only the background with an enchanted forest: glowing fireflies, mossy ancient trees, shafts of magical light. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic, fantasy.",
  },

  // --- Person (6) ---
  // Composition lock comes FIRST (FLUX.2 weights earliest tokens most), so the
  // model treats this as "restyle the people that are here" rather than
  // "compose a themed scene" (which invents extra people). Then the costume,
  // then identity preservation. Gender-neutral wording keeps each person's
  // gender/age/skin tone.
  {
    id: "person_viking",
    category: "person",
    label: "Viking warrior",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a Viking warrior in a fur cloak with leather and iron armor and subtle war paint, with hair in Norse braids that suit them. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_roman_emperor",
    category: "person",
    label: "Roman royalty",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as ancient Roman royalty in a white and gold toga with a golden laurel wreath. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_astronaut",
    category: "person",
    label: "Astronaut",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person in a detailed white astronaut spacesuit with the helmet open so their face stays fully visible. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_renaissance",
    category: "person",
    label: "Renaissance noble",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a Renaissance noble in rich velvet and lace period attire, like a classical oil painting. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting.",
  },
  {
    id: "person_cyberpunk",
    category: "person",
    label: "Cyberpunk hacker",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a cyberpunk hacker in a neon-accented jacket with an augmented-reality visor and glowing tattoos. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_knight",
    category: "person",
    label: "Medieval knight",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a medieval knight in polished plate armor with a surcoat. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
];

/** Internal index from effect id to its option for O(1) lookups. */
const EFFECTS_BY_ID: ReadonlyMap<string, EffectOption> = new Map(
  EFFECTS.map((effect) => [effect.id, effect]),
);

/** Error thrown when an effect id does not match any catalog entry. */
export class UnknownEffectError extends Error {
  constructor(public readonly effectId: string) {
    super(`Unknown effectId: ${JSON.stringify(effectId)}`);
    this.name = "UnknownEffectError";
  }
}

/**
 * Look up an Effect_Option by its id.
 *
 * @returns the matching option, or `undefined` if the id is unknown.
 */
export function findEffect(effectId: string): EffectOption | undefined {
  return EFFECTS_BY_ID.get(effectId);
}

/**
 * Resolve an effect id to its option, rejecting unknown ids.
 *
 * @throws {UnknownEffectError} when `effectId` matches no catalog entry.
 */
export function getEffect(effectId: string): EffectOption {
  const effect = EFFECTS_BY_ID.get(effectId);
  if (effect === undefined) {
    throw new UnknownEffectError(effectId);
  }
  return effect;
}

/**
 * Resolve an effect id to its predefined prompt string, rejecting unknown ids.
 *
 * This is the lookup helper used by the Generation_Service to set the
 * Async_Request `inputs` field (Requirement 7.1/7.2).
 *
 * @throws {UnknownEffectError} when `effectId` matches no catalog entry.
 */
export function getPromptForEffect(effectId: string): string {
  return getEffect(effectId).prompt;
}
