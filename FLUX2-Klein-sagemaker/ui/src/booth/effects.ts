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

/**
 * Optional per-effect overrides for the inference request. Most effects derive
 * everything from the captured photo and the booth defaults, but some need a
 * specific output shape or sampling. For example, a waist-up half-body portrait
 * needs a tall (portrait) output canvas — the endpoint otherwise derives the
 * output size from the (near-square) reference image, leaving no vertical room
 * for a half-body crop. `width`/`height` are passed to the endpoint, which
 * clamps them to [256, 1536] and snaps to a multiple of 16.
 */
export interface EffectRequestOverrides {
  /** Output width in px (endpoint clamps to [256,1536] and snaps to /16). */
  width?: number;
  /** Output height in px (endpoint clamps to [256,1536] and snaps to /16). */
  height?: number;
  /** Override the booth's default denoising steps for this effect. */
  steps?: number;
  /** Override the booth's default guidance scale for this effect. */
  guidance?: number;
}

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
   * Short noun phrase describing this effect's contribution, used when building
   * a multi-reference MERGE prompt (see {@link buildMergePrompt}). For a
   * background effect it names the scene ("the futuristic spaceship interior");
   * for a person effect it names the outfit ("a Viking warrior outfit"). Kept
   * separate from {@link label} (a terse button caption) so the merge sentence
   * reads naturally. Non-empty.
   */
  mergePhrase: string;
  /**
   * Optional per-effect inference-request overrides (output size / sampling).
   * Omitted for effects that should use the booth defaults and derive the
   * output size from the reference photo. See {@link EffectRequestOverrides}.
   */
  request?: EffectRequestOverrides;
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
    mergePhrase: "the interior of a futuristic spaceship",
    prompt:
      "Replace only the background with the interior of a futuristic spaceship: sleek metal corridors, glowing control panels, soft blue lighting. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic.",
  },
  {
    id: "bg_colosseum",
    category: "background",
    label: "Roman colosseum",
    image: "coloseum.jpeg",
    mergePhrase: "the interior of the ancient Roman Colosseum under a dramatic sky",
    prompt:
      "Replace only the background with the interior of the ancient Roman Colosseum under a dramatic sky. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic, cinematic lighting.",
  },
  {
    id: "bg_tropical",
    category: "background",
    label: "Tropical background",
    image: "forest.jpeg",
    mergePhrase: "a sunlit tropical beach with palm trees and turquoise water",
    prompt:
      "Replace only the background with a sunlit tropical beach: palm trees, turquoise water, white sand. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic.",
  },
  {
    id: "bg_snowy_peak",
    category: "background",
    label: "Snowy mountain peak",
    image: "mountain.jpeg",
    mergePhrase: "a snowy mountain summit under a crisp blue sky",
    prompt:
      "Replace only the background with a snowy mountain summit under a crisp blue sky, distant peaks and drifting snow. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic.",
  },
  {
    id: "bg_neon_city",
    category: "background",
    label: "Neon city street at night",
    image: "neon.jpeg",
    mergePhrase: "a neon-lit city street at night with rain-slick pavement",
    prompt:
      "Replace only the background with a neon-lit city street at night, rain-slick pavement reflecting pink and blue signage. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Cinematic, photorealistic.",
  },
  {
    id: "bg_enchanted_forest",
    category: "background",
    label: "Enchanted forest",
    image: "enchanted.jpeg",
    mergePhrase: "an enchanted forest with glowing fireflies and shafts of magical light",
    prompt:
      "Replace only the background with an enchanted forest: glowing fireflies, mossy ancient trees, shafts of magical light. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic, fantasy.",
  },
  {
    id: "bg_medieval_castle",
    category: "background",
    label: "Medieval castle",
    image: "castle.jpeg",
    mergePhrase: "the interior of a grand medieval castle with stone walls and torch-lit halls",
    prompt:
      "Replace only the background with the interior of a grand medieval castle: stone walls, torch-lit halls, hanging banners, and a high vaulted ceiling. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic, cinematic lighting.",
  },
  {
    id: "bg_lunar",
    category: "background",
    label: "Lunar landscape",
    image: "lunar.jpeg",
    mergePhrase: "a barren lunar landscape on the surface of the Moon with grey cratered terrain and Earth on the horizon",
    prompt:
      "Replace only the background with a barren lunar landscape on the surface of the Moon: grey cratered terrain, a black star-filled sky, and the distant Earth rising on the horizon. Keep the existing people unchanged — same faces, expressions, hair, clothing, poses, and positions — and keep exactly the same number of people as in the original photo. Do not add or remove anyone. Blend the new background naturally behind them with matching light and shadow. Photorealistic.",
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
    mergePhrase: "a Viking warrior outfit — a fur cloak with leather and iron armor and Norse braids",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a Viking warrior in a fur cloak with leather and iron armor and subtle war paint, with hair in Norse braids that suit them. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_roman_emperor",
    category: "person",
    label: "Roman royalty",
    image: "emperor.jpeg",
    mergePhrase: "ancient Roman royal attire — a white and gold toga with a golden laurel wreath",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as ancient Roman royalty in a white and gold toga with a golden laurel wreath. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_astronaut",
    category: "person",
    label: "Astronaut",
    image: "astronaut.jpeg",
    mergePhrase: "a detailed white astronaut spacesuit with the helmet open so the face stays visible",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person in a detailed white astronaut spacesuit with the helmet open so their face stays fully visible. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_renaissance",
    category: "person",
    label: "Renaissance noble",
    image: "noble.jpeg",
    mergePhrase: "Renaissance noble attire in rich velvet and lace period clothing",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a Renaissance noble in rich velvet and lace period attire, like a classical oil painting. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting.",
  },
  {
    id: "person_cyberpunk",
    category: "person",
    label: "Cyberpunk hacker",
    image: "hacker.jpeg",
    mergePhrase: "a cyberpunk hacker outfit — a neon-accented jacket with an augmented-reality visor and glowing tattoos",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a cyberpunk hacker in a neon-accented jacket with an augmented-reality visor and glowing tattoos. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  {
    id: "person_knight",
    category: "person",
    label: "Medieval knight",
    image: "knight.jpeg",
    mergePhrase: "a medieval knight's polished plate armor with a surcoat",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person as a medieval knight in polished plate armor with a surcoat. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
  },
  // Official evening: the SAME working template as the costume effects above —
  // identical opening ("keep the exact same people … do not add, remove, or
  // duplicate anyone") and identical closing identity lock (which keeps each
  // person's GENDER unchanged and keeps the ORIGINAL background). The only
  // differences from a plain costume effect are a single neutral wardrobe clause
  // ("attire suitable for an official evening event") and a light skin retouch.
  //
  // IMPORTANT — two hard-won rules:
  //   1) Do NOT enumerate garments by gender ("a tuxedo for a masculine look AND
  //      a gown for a feminine look") — FLUX reads it as a two-person cast list
  //      and invents a second person. The closing clause preserves each person's
  //      gender, so a neutral "formal evening attire" instruction lets the model
  //      dress a woman in a gown and a man in a suit on its own.
  //   2) Do NOT replace the background or add a bokeh blur — both recompose the
  //      scene and FLUX clones the subject into a group. Keep the original
  //      background completely unchanged.
  {
    id: "person_official_event",
    category: "person",
    label: "Official evening",
    image: "official_event.jpeg",
    mergePhrase:
      "elegant, formal attire suitable for an official evening event",
    prompt:
      "Keep the exact same people from the original photo — the same number of people, same faces, same positions; do not add, remove, or duplicate anyone. Change only their clothing: dress each existing person in elegant, formal attire suitable for an official evening event. Lightly retouch the skin to gently soften temporary blemishes, spots, and shine while keeping natural skin texture. Keep each person's face, gender, age, skin tone, expression, and pose unchanged, and keep the original background unchanged. Match the original lighting. Photorealistic.",
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

/**
 * Admin-set prompt overrides, keyed by effect id. Populated at app startup from
 * the Schedule_Store (see `api/prompts.ts` → {@link applyPromptOverrides}) and
 * holds ONLY effects an admin has explicitly customized (`isCustom` rows). When
 * an effect has no override here, {@link getPromptForEffect} falls back to the
 * catalog default below — so changing a default in this file is always honored
 * and never shadowed by a stale seed in DynamoDB.
 */
let promptOverrides: ReadonlyMap<string, string> = new Map();

/**
 * Replace the in-memory admin prompt-override registry. Called once after the
 * SPA loads the effect prompts from DynamoDB, and again after an admin saves or
 * restores a prompt, so the very next generation uses the new value.
 */
export function applyPromptOverrides(overrides: ReadonlyMap<string, string>): void {
  promptOverrides = new Map(overrides);
}

/** The current admin prompt overrides (read-only view, for the admin UI). */
export function getPromptOverrides(): ReadonlyMap<string, string> {
  return promptOverrides;
}

/**
 * The built-in catalog default prompt for an effect (ignores any admin
 * override). This is the single source of truth for "restore to default" and
 * for the booth fallback. Rejects unknown ids.
 */
export function getDefaultPromptForEffect(effectId: string): string {
  const effect = EFFECTS_BY_ID.get(effectId);
  if (effect === undefined) {
    throw new UnknownEffectError(effectId);
  }
  return effect.prompt;
}

/**
 * Resolve an effect id to its effective prompt, rejecting unknown ids. Uses the
 * admin override when one is set (see {@link applyPromptOverrides}); otherwise
 * the catalog default.
 */
export function getPromptForEffect(effectId: string): string {
  const effect = EFFECTS_BY_ID.get(effectId);
  if (effect === undefined) {
    throw new UnknownEffectError(effectId);
  }
  return promptOverrides.get(effectId) ?? effect.prompt;
}

/**
 * Resolve an effect id to its inference-request overrides, or an empty object
 * when the effect uses the booth defaults. Rejects unknown ids.
 */
export function getRequestOverridesForEffect(effectId: string): EffectRequestOverrides {
  const effect = EFFECTS_BY_ID.get(effectId);
  if (effect === undefined) {
    throw new UnknownEffectError(effectId);
  }
  return effect.request ?? {};
}

/** Resolve an effect id to its category ("background" | "person"). */
export function getEffectCategory(effectId: string): EffectCategory {
  const effect = EFFECTS_BY_ID.get(effectId);
  if (effect === undefined) {
    throw new UnknownEffectError(effectId);
  }
  return effect.category;
}

/**
 * Map a category to the result slot it fills. Mirrors the `ResultSlot` union in
 * `machine.ts` (kept as a string union here to avoid a cross-module type import
 * in this otherwise self-contained catalog).
 */
export function slotForCategory(category: EffectCategory): "background" | "person" {
  return category === "background" ? "background" : "person";
}

/**
 * Build the multi-reference MERGE prompt that combines a background result with
 * a character result into a single image.
 *
 * Follows Black Forest Labs' multi-reference prompting guidance for FLUX.2:
 * reference each image explicitly ("image 1", "image 2") and describe each
 * image's role. By convention the caller submits the background result as image
 * 1 and the character result as image 2.
 *
 * CRITICAL: both reference images contain the SAME person (image 1 is the
 * person on a new background; image 2 is the person in a new costume). If the
 * prompt only says "use the background from image 1", the model still sees a
 * person in image 1 and keeps them too, producing TWO copies of the subject.
 * So the prompt must (a) hard-lock the output to exactly one person, (b) take
 * that person from image 2, and (c) use image 1 for its background ONLY,
 * explicitly ignoring any person in image 1.
 *
 * @throws {UnknownEffectError} if either id is not in the catalog.
 */
export function buildMergePrompt(
  backgroundEffectId: string,
  personEffectId: string,
): string {
  const bg = EFFECTS_BY_ID.get(backgroundEffectId);
  const person = EFFECTS_BY_ID.get(personEffectId);
  if (bg === undefined) {
    throw new UnknownEffectError(backgroundEffectId);
  }
  if (person === undefined) {
    throw new UnknownEffectError(personEffectId);
  }
  return (
    // One-person lock FIRST (earliest tokens weighted most).
    `Create a single photorealistic photograph that contains EXACTLY ONE person. ` +
    `Take that one person only from image 2 — keep their exact same face, identity, hair, and pose, still wearing ${person.mergePhrase}. ` +
    // Image 1 = background ONLY; explicitly discard its person to avoid a clone.
    `Use image 1 ONLY as the background scene (${bg.mergePhrase}); completely ignore and do not include any person, face, body, or silhouette from image 1 — image 1 provides the background and nothing else. ` +
    `Place that single person from image 2 into this background. ` +
    // Anti-duplication, stated explicitly.
    `The final image must show only that one person — do not add, duplicate, clone, mirror, repeat, or invent any additional people, and never show two versions of the person. ` +
    `Keep the person's face and identity unchanged and clearly recognizable. ` +
    `Match the scale, lighting, shadows, and perspective of the background so the person sits naturally in the scene. Photorealistic, cinematic lighting.`
  );
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
