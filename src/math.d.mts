// Types for math.mjs — head training and inference. Isomorphic and deterministic: the same
// examples always produce the same weights.

import type { Vector } from './format.mjs';

/**
 * A linear head in memory. `W` is `K × dim` float32, row per answer; `b` is `K` biases.
 * A centroid head has the same shape: `W` holds unit class means and `b` is zero.
 */
export interface Head {
  W: Float32Array;
  b: Float32Array;
  /** Number of answers. */
  K: number;
  /** Width of the vectors it reads. */
  dim: number;
}

/** What every predictor returns. */
export interface Prediction {
  /** Probability per answer index, 0–1, summing to 1. */
  probs: number[];
  /** Index of the most probable answer. */
  index: number;
  /** `probs[index]`, 0–1. */
  confidence: number;
}

/** Per-row weights: anything with a length, numeric indexing and `reduce`. */
export type RowWeights = readonly number[] | Float32Array | Float64Array;

/** Options for `fitHead`. Any other key throws — the old optimiser's `epochs` / `lr` belong to `fitHeadGD`. */
export interface FitHeadOptions {
  /** L2 penalty on weights and biases. Defaults to `defaultL2(K)`. */
  l2?: number;
  /** Per-row weights. Defaults to 1 for every row. */
  weights?: RowWeights | null;
  /** L-BFGS iteration cap. Defaults to 500. */
  maxIter?: number;
  /** Stop when the gradient norm falls below this. Defaults to 1e-5. */
  tol?: number;
}

/** The default L2 penalty for a `K`-answer head: `min(3e-3, 0.012 / K)`. The compiler cross-validates its own instead. */
export const defaultL2: (K: number) => number;

/**
 * Multinomial logistic regression over frozen embeddings, fitted to convergence with L-BFGS.
 * Convex, zero-initialised and evaluated in a fixed order, so it is deterministic.
 * Throws on an empty `X` and on any unknown option.
 *
 * @param X one vector per row, each exactly `dim` wide
 * @param y answer index per row, 0 … K−1
 */
export function fitHead(X: readonly Vector[], y: ArrayLike<number>, K: number, dim: number, opts?: FitHeadOptions): Head;

/**
 * The head trainer this SDK shipped until 2026-09-19: fixed-count full-batch gradient descent.
 * Kept so older experiments reproduce. Do not use it for anything new.
 */
export function fitHeadGD(
  X: ArrayLike<Vector>, y: ArrayLike<number>, K: number, dim: number,
  opts?: { epochs?: number; lr?: number; l2?: number; weights?: RowWeights | null },
): Head;

/** The scale applied to cosines before the centroid head's softmax (20). */
export const CENTROID_SCALE: number;

/**
 * Nearest class mean: average each answer's vectors and normalise. Closed-form. An answer with no
 * rows keeps a zero vector and is never chosen over one that has evidence. Throws on an empty `X`.
 */
export function fitCentroids(
  X: ArrayLike<Vector>, y: ArrayLike<number>, K: number, dim: number,
  opts?: { weights?: ArrayLike<number> | null },
): Head;

/** What a predictor reads of a head: the same shape as `Head`, but any indexable numbers will do. */
export interface HeadLike {
  W: ArrayLike<number>;
  b: ArrayLike<number>;
  K: number;
  dim: number;
}

/** Softmax over scaled cosines to each centroid. Same return shape as `softmax`. Reads `W`, `K` and `dim` only. */
export function centroidPredict(head: Omit<HeadLike, 'b'>, x: Vector): Prediction;

/**
 * Per-tag independent binary heads, for multi-label (`tags`). Each tag is balanced against its own
 * absence, then scaled by `rowWeight`. Returns `K` two-class heads; class 1 is "tag applies".
 *
 * @param yMulti the set of tag indices on each row
 */
export function fitTagHeads(
  X: readonly Vector[], yMulti: readonly ReadonlySet<number>[], K: number, dim: number,
  opts?: Omit<FitHeadOptions, 'weights'> & { rowWeight?: ArrayLike<number> | null },
): Head[];

/** A linear head's prediction: softmax over `W·x + b`. */
export function softmax(head: HeadLike, x: Vector): Prediction;

/** Class-balanced row weights, so a rare answer is not simply ignored. Mean weight is 1. */
export function balanceWeights(y: readonly number[], K: number): number[];

/** Deterministic PRNG (a 32-bit LCG), so holdout splits are reproducible across machines. Each call returns a number in [0, 1). */
export function rng(seed: number): () => number;

/** A shuffled copy of `arr`, determined entirely by `seed`. */
export function shuffled<T>(arr: Iterable<T>, seed: number): T[];

/**
 * Raise probabilities to the power `1/T` and renormalise. `T < 1` sharpens, `T > 1` softens; the
 * winner never changes. Returns the INPUT array itself, not a copy, when `T` is 1 or falsy.
 */
export function applyTemperature(probs: number[], T: number): number[];

/** A held-out prediction with its true answer index. */
export interface CalibrationSample {
  probs: number[];
  y: number;
}

/**
 * Fit a temperature on held-out predictions by minimising negative log-likelihood, to three places.
 * Returns 1 when there is nothing to fit — no samples, or no held-out mistakes.
 */
export function fitTemperature(samples: readonly CalibrationSample[] | null | undefined): number;

/**
 * Expected calibration error, 0–1: the mean gap between claimed confidence and how often it is
 * right, over `bins` equal-width confidence bins (default 10). `null` with no samples.
 */
export function calibrationError(
  samples: readonly { confidence: number; ok: boolean }[] | null | undefined,
  bins?: number,
): number | null;

/**
 * How common each answer is in a stream of UNLABELLED input, from the shim's own calibrated
 * probabilities on it (EM re-estimation of the class prior). Returns one share per answer, summing to 1.
 *
 * @param probRows one probability row per input
 * @param trainedPrior the prior the head was trained under — uniform for a class-balanced head
 */
export function estimatePrior(
  probRows: readonly ArrayLike<number>[], trainedPrior: ArrayLike<number>,
  opts?: { iters?: number; tol?: number },
): number[];

/** One held-out row for `weightedGate`: confidence 0–1, whether it was right, and its weight. */
export interface GateRow {
  c: number;
  ok: boolean | number;
  w: number;
}

/**
 * The lowest confidence at which the rows kept are `target` accurate (default 0.9), rows weighted
 * by `w`. The threshold is rounded DOWN to three places; `1` with coverage `0` when no confidence qualifies.
 */
export function weightedGate(rows: Iterable<GateRow>, target?: number): {
  /** The gate, 0–1. */
  threshold: number;
  /** Weighted share of rows at or above it, 0–1. */
  coverage: number;
};
