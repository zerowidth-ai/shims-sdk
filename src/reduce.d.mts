// Types for reduce.mjs — a document read in pieces has to become one decision. Internal module;
// `reduceChunks` is re-exported from the package root.

import type { ReduceOptions } from './format.mjs';

/** The part of a per-piece decision that reduction reads. */
export interface ChunkResult {
  familiarity?: number | null;
  confidence?: number | null;
  /** Read only under `'pooled'`. */
  probs?: Record<string, number>;
}

/**
 * Combine per-piece decisions into one. The result starts from the piece with the highest
 * familiarity and adds `from` (that piece's text) and `chunks` (how many pieces there were).
 * Under `'pooled'`, `answer`, `probs` and `confidence` are replaced by the familiarity-weighted
 * pool while `familiarity` stays the best piece's; every other field — including `action` — is
 * still the best piece's, so re-derive the action from the combined numbers (`Shim.decideChunks` does).
 *
 * @param results one `decideVector()` result per piece, in order; must not be empty
 * @param chunks the piece texts, same order
 * @param labels the shim's answers
 */
export function reduceChunks<R extends ChunkResult>(
  results: readonly R[],
  chunks: readonly string[],
  labels: readonly string[],
  opts?: ReduceOptions,
): R & { from: string; chunks: number };
