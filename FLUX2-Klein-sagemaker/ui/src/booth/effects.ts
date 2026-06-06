/**
 * Effect_Option catalog for the SPA (Requirements 5, 6, 7.1).
 *
 * This mirrors the shared `backend/src/lib/effects.ts` catalog. The UI package
 * is type-checked in isolation (`tsconfig.app.json` `include: ["src"]`, built
 * with `tsc -b`), so it cannot import a `.ts` source from `backend/src`
 * (TS6307). The catalog is therefore duplicated here as the UI's source of
 * truth for rendering the 12 options and mapping a selected effect id to its
 * prompt. The values are kept identical to the backend catalog so the
 * Async_Request the browser builds matches what any server-side caller would
 * build.
 */

/** The two categories an Effect_Option can belong to. */
export type EffectCategory = "background" | "person";

/** A predefined transformation choice. */
export interface EffectOption {
  /** Stable lookup key, unique across the whole catalog (e.g. "bg_spaceship"). */
  id: string;
  /** Whether this option replaces the background or transforms the person. */
  category: EffectCategory;
  /** Human-readable, touch-control label. Non-empty. */
  label: string;
  /** Prompt string mapped to Async_Request.inputs. Non-empty. */
  prompt: string;
  /**
   * Optional thumbnail basename (e.g. "viking.jpeg") for the option button.
   * Resized variants are served from `/effects/monitor/<image>` and
   * `/effects/smartphone/<image>` (see {@link effectImageUrl}). UI-only; not
   * part of the Async_Request the backend builds.
   */
  image?: string;
}

/**
 * The predefined effect catalog: exactly 6 background options and exactly 6
 * person options. Identical to `backend/src/lib/effects.ts`.
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
    image: "spaceship.jpeg",
    prompt:
      "Replace only the background with the interior of a futuristic spaceship: sleek metal corridors, glowing control panels, soft blue lighting. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic.",
  },
  {
    id: "bg_colosseum",
    category: "background",
    label: "Roman colosseum",
    image: "coloseum.jpeg",
    prompt:
      "Replace only the background with the interior of the ancient Roman Colosseum under a dramatic sky. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic, cinematic lighting.",
  },
  {
    id: "bg_tropical",
    category: "background",
    label: "Tropical background",
    image: "forest.jpeg",
    prompt:
      "Replace only the background with a sunlit tropical beach: palm trees, turquoise water, white sand. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic.",
  },
  {
    id: "bg_snowy_peak",
    category: "background",
    label: "Snowy mountain peak",
    image: "mountain.jpeg",
    prompt:
      "Replace only the background with a snowy mountain summit under a crisp blue sky, distant peaks and drifting snow. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic.",
  },
  {
    id: "bg_neon_city",
    category: "background",
    label: "Neon city street at night",
    image: "neon.jpeg",
    prompt:
      "Replace only the background with a neon-lit city street at night, rain-slick pavement reflecting pink and blue signage. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Cinematic, photorealistic.",
  },
  {
    id: "bg_enchanted_forest",
    category: "background",
    label: "Enchanted forest",
    image: "enchanted.jpeg",
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
    image: "viking.jpeg",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a Viking warrior in a fur cloak with leather and iron armor and subtle war paint, with hair in Norse braids that suit them. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_roman_emperor",
    category: "person",
    label: "Roman royalty",
    image: "emperor.jpeg",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as ancient Roman royalty in a white and gold toga with a golden laurel wreath. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_astronaut",
    category: "person",
    label: "Astronaut",
    image: "astronaut.jpeg",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person in a detailed white astronaut spacesuit with the helmet open so their face stays fully visible. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_renaissance",
    category: "person",
    label: "Renaissance noble",
    image: "noble.jpeg",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a Renaissance noble in rich velvet and lace period attire, like a classical oil painting. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting.",
  },
  {
    id: "person_cyberpunk",
    category: "person",
    label: "Cyberpunk hacker",
    image: "hacker.jpeg",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a cyberpunk hacker in a neon-accented jacket with an augmented-reality visor and glowing tattoos. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_knight",
    category: "person",
    label: "Medieval knight",
    image: "knight.jpeg",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a medieval knight in polished plate armor with a surcoat. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
];

/** Background options, in catalog order. */
export const BACKGROUND_EFFECTS: readonly EffectOption[] = EFFECTS.filter(
  (e) => e.category === "background",
);

/** Person options, in catalog order. */
export const PERSON_EFFECTS: readonly EffectOption[] = EFFECTS.filter(
  (e) => e.category === "person",
);

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

/** Look up an Effect_Option by id, or `undefined` if unknown. */
export function findEffect(effectId: string): EffectOption | undefined {
  return EFFECTS_BY_ID.get(effectId);
}

/** Resolve an effect id to its prompt, rejecting unknown ids. */
export function getPromptForEffect(effectId: string): string {
  const effect = EFFECTS_BY_ID.get(effectId);
  if (effect === undefined) {
    throw new UnknownEffectError(effectId);
  }
  return effect.prompt;
}

/**
 * Resolve the button thumbnail URL for an effect in the given layout, or
 * `undefined` when the effect has no image. Monitor and phone use different
 * pre-resized folders (served from `ui/public/effects/<size>/`) so each layout
 * downloads only the pixels it needs.
 */
export function effectImageUrl(
  effect: EffectOption,
  layout: "monitor" | "mobile",
): string | undefined {
  if (!effect.image) {
    return undefined;
  }
  const folder = layout === "monitor" ? "monitor" : "smartphone";
  return `${import.meta.env.BASE_URL}effects/${folder}/${effect.image}`;
}
