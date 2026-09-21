// Types for surface.mjs — form features the encoder cannot see (it is uncased, mean-pooled, and
// trained to ignore punctuation).

/** How many surface features are appended to every embedding (8). */
export const SURFACE_DIM: number;

/** Every surface feature is multiplied by this (0.2) so it sits near the embedding's own per-dimension magnitude. */
export const SCALE: number;

/**
 * Eight numbers about how a text is written, each in `[0, SCALE]`, in this order: contains a
 * question mark; ends on one; exclamation marks; share of capital letters; share of all-caps
 * words; length in words (log-scaled); repeated `!!` or `??`; emoji. `null` / `undefined` read as
 * the empty string.
 */
export function surfaceFeatures(text: string | null | undefined): number[];

/** Embedding ++ surface features — the vector every head is trained on. Returns a new `Float32Array` of `vector.length + SURFACE_DIM`. */
export function withSurface(vector: ArrayLike<number>, text: string | null | undefined): Float32Array;
