// Types for familiarity.mjs — "have I seen anything like this before?" — and the nearest-neighbour
// head that reads the same shipped examples.

import type { Vector, PackedReference } from './format.mjs';
import type { Prediction } from './math.mjs';

/** A shim's shipped training set in memory: int8 vectors at full `DIM` width, one scale per vector. */
export interface Reference {
  /** `count × DIM` quantised values. */
  q: Int8Array;
  /** One dequantisation scale per vector. */
  scales: Float32Array;
  /** 21 quantiles (0, 0.05 … 1) of each example's cosine to its nearest OTHER example. The yardstick familiarity is scored against. */
  spread: number[];
  /** Number of vectors. */
  count: number;
  /** Answer index per vector, or `null` when none were given — in which case `knnPredict` cannot vote. */
  labels: number[] | null;
}

/**
 * Cosine similarity over the first `width` dimensions, −1 to 1. `width` defaults to `EMBED_DIM`
 * (the semantic half, for "is this familiar"); pass `DIM` to include the surface features.
 */
export function cosine(a: Vector, b: Vector, width?: number): number;

/**
 * Quantise vectors to int8 with one scale each, and measure how close the examples sit to each
 * other (leave-one-out nearest-neighbour cosine). Vectors must be `DIM` wide. `labels` is kept by
 * reference, not copied.
 */
export function packReference(vectors: readonly Vector[], labels?: number[] | null): Reference;

/** A weights file's base64 reference → typed arrays. `null` when there is nothing to unpack. */
export function unpackReference(ref: PackedReference | null | undefined): Reference | null;

/**
 * Where this input sits in the training set's own neighbourhood distribution, as a 0–1 percentile
 * to three places. 0.5 means as typical as the median example; below ~0.2 means the shim has no
 * examples near it. Reads the embedding half of the vector only. `null` when the reference is
 * missing or empty.
 */
export function familiarityScore(vec: Vector, ref: Reference | null | undefined): number | null;

/** How sharply the kNN vote favours closer neighbours: a neighbour at cosine `c` votes `exp((c − 1) / KNN_SHARPNESS)` (0.1). */
export const KNN_SHARPNESS: number;

/**
 * Nearest-neighbour head: a distance-weighted vote of the `k` closest shipped examples (default 5)
 * over the full `DIM` width, surface features included. `null` when the reference is missing,
 * empty or unlabelled.
 *
 * @param K number of answers, which sizes `probs`
 */
export function knnPredict(vec: Vector, ref: Reference | null | undefined, K: number, k?: number): Prediction | null;
