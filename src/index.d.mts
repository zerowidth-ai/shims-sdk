// Types for @zerowidth/shims-sdk — tiny task-specific decision models. One encoder is shared by every
// shim in an app; each shim is a few thousand numbers on top of it.
//
// Dependency-free on purpose: nothing here imports types from @huggingface/transformers.

import type {
  Action, Decision, DecisionFor, DraftWeightsInput, EmbedOptions, Embedder,
  EncoderOptions, Gates, HeadKind, ReduceOptions, ShimReport, ShimType, ShimWeightsInput, Vector,
} from './format.mjs';
import type { Head } from './math.mjs';
import type { Reference } from './familiarity.mjs';
import type { Tree } from './tree.mjs';

export { DIM, EMBED_DIM, ENCODER_ID, headBytes } from './format.mjs';
export { surfaceFeatures, withSurface, SURFACE_DIM } from './surface.mjs';
export { reduceChunks } from './reduce.mjs';
export { observe, observing, outcome, ring } from './observe.mjs';

export type {
  Action, CalibrationReport, ClassifyDecision, ClassifyExample, ClassifyReport, ClassifySource,
  ClassifyWeights, Decision, DecisionFor, DraftReport, DraftWeights, DraftWeightsInput, EmbedOptions,
  Embedder, EncoderOptions, Gates, HeadChoice, HeadKind, PackedHead, PackedReference, PackedTree,
  Readiness, ReduceOptions, ShimNote, ShimReport, ShimReportBase, ShimSource, ShimSourceInput,
  ShimType, ShimWeights, ShimWeightsInput, StructureReport, TagReport, TagsDecision, TagsExample,
  TagsReport, TagsSource, TagsWeights, Vector,
} from './format.mjs';
export type { Head, HeadLike, Prediction } from './math.mjs';
export type { Reference } from './familiarity.mjs';
export type { Tree, TreePrediction } from './tree.mjs';
export type { ChunkResult } from './reduce.mjs';
export type {
  DecisionRow, ObserveEntry, Outcome, OutcomeRow, Ring, RingFilter, RingRow, RingSummary, Sink,
} from './observe.mjs';

/** What the encoder reads in one pass, in wordpiece tokens (512). Longer pieces are truncated. */
export const ENCODER_MAX_TOKENS: number;
/** Roughly where `ENCODER_MAX_TOKENS` lands in characters (1700). Pieces longer than this are cut on whitespace. */
export const ENCODER_MAX_CHARS: number;
/**
 * Input longer than this many characters (420) is split and decided in pieces. Deliberately far
 * below what the encoder can read: reading a long message in pieces measured better than reading it whole.
 */
export const LONG_INPUT_CHARS: number;
/** Target size of one piece, in characters (160). */
export const CHUNK_CHARS: number;

/**
 * Split on sentence ends and newlines, then regroup so pieces approach `target` characters
 * (default `CHUNK_CHARS`). A piece still over `ENCODER_MAX_CHARS` is cut on whitespace. Always
 * returns at least one piece.
 */
export function splitForDecision(text: string, target?: number): string[];

/**
 * Load (once) the shared encoder. Every shim in the process reuses it, and only the first call's
 * options are read. The tokenizer and model are transformers.js objects, left untyped here.
 */
export function loadEncoder(opts?: EncoderOptions): Promise<{ tokenizer: unknown; model: unknown }>;

/**
 * Text → one vector per input, in-process: mean-pool over real tokens, L2-normalise, append the
 * surface features. Each vector is a `Float32Array` of `DIM`. The compiler calls this exact
 * function, so build-time and runtime vectors cannot drift apart.
 */
export function embed(texts: string, opts?: EmbedOptions): Promise<Float32Array>;
export function embed(texts: string[], opts?: EmbedOptions): Promise<Float32Array[]>;

/**
 * Take over encoding — for a server, a test that wants vectors without a model, or an app that
 * already runs a worker and would rather share it. Pass `null` to go back to the SDK's own: a
 * module worker in a browser, in-process in Node.
 */
export function setEmbedder(fn: Embedder | null | undefined): void;

/** The embedder in play: the one set with `setEmbedder`, else the SDK's worker (browser) or in-process `embed` (elsewhere). */
export const currentEmbedder: () => Embedder;

/** Encode text with whatever embedder is in play. A string gives one vector; an array gives one per text. */
export const encode: {
  (texts: string, opts?: EmbedOptions): Promise<Vector>;
  (texts: string[], opts?: EmbedOptions): Promise<Vector[]>;
};

/**
 * Download and warm the encoder before the first decision, so the first one is not the slow one.
 * Optional: everything works without it. Calls the embedder's `warm` when it has one, else loads in-process.
 */
export function preload(opts?: EncoderOptions): Promise<void>;

/** Options for `Shim.decide` and `Bank.decide`. The whole object is also handed to the embedder. */
export interface DecideOptions extends EmbedOptions {
  /** `false` decides on the whole input in one read however long it is. */
  long?: boolean;
  /** How a split input's pieces are combined. Defaults to `'best'`. */
  reduce?: ReduceOptions['mode'];
  /** Exponent on familiarity under `'pooled'`. Defaults to 3. */
  reducePower?: number;
  /** `false` records `text: null` for this decision when something is observing. */
  keepText?: boolean;
}

/** A row for `recalibrate`: the vector a decision was made on, and the answer that turned out right. */
export interface OutcomeSample {
  vector?: Vector | null;
  label?: string | string[] | null;
}

/** What `recalibrate` did. */
export interface RecalibrateResult {
  /** Whether the refitted temperature was kept: only when it lowered calibration error on these same rows. */
  applied: boolean;
  /** The temperature now in force. */
  temperature: number;
  /** Expected calibration error (0–1) under the old temperature; `null` when there were too few rows. */
  before: number | null;
  /** Expected calibration error (0–1) under the fitted temperature; `null` when there were too few rows. */
  after: number | null;
  /** Usable rows: those with a vector and a label that is one of this shim's answers. */
  samples: number;
  /** The temperature that was fitted, kept or not. Absent when there were too few rows. */
  fitted?: number;
  /** Why nothing was fitted. Present only when there were too few rows. */
  why?: string;
}

/** What `refitFloor` did. `before` / `after` are familiarity floors, 0–1 percentiles. */
export type RefitFloorResult =
  | { applied: false; before: number; after: number; samples: number; why: string }
  | {
      applied: true; before: number; after: number; samples: number;
      /** Share of the given vectors, 0–1, the old floor refused. */
      refusedBefore: number;
      /** The share the new floor aims to refuse — the `target` passed in. */
      refusedAfter: number;
    };

/** What `refitGate` did. `before` / `after` are confidence gates, 0–1. */
export type RefitGateResult =
  | { applied: false; before: number; after: number; samples: number; why: string }
  | {
      applied: true; before: number; after: number; samples: number;
      /** Share of the given vectors, 0–1, at or above the old gate. */
      actsOnBefore: number;
      /** Share at or above the new gate. */
      actsOnAfter: number;
      /** The answer mix estimated from the vectors: a 0–1 share per answer. */
      mix: Record<string, number>;
    };

/**
 * One compiled shim: a task, its answers, and the head that decides between them.
 *
 * The type parameter is an assertion, not an inference: weights imported as JSON widen `type` to
 * `string`, so say `new Shim<'classify'>(weights)` to get `answer: string` rather than the union.
 */
export class Shim<T extends ShimType = ShimType> {
  /**
   * Throws when the weights are a draft, are not format 2, or were compiled for a different encoder.
   *
   * @param opts `temperature` overrides the build's calibration temperature; normally unused
   */
  constructor(compiled: ShimWeightsInput, opts?: { temperature?: number });

  readonly name: string;
  readonly type: T;
  readonly question: string | null | undefined;
  /** The answers. */
  readonly labels: string[];
  /** The build report, exactly as it sits in the weights. Weights from `compileShim` always carry one. */
  readonly report: ShimReport;
  /** The shipped examples, unpacked. `null` when the weights carried none — then familiarity is `null` and nothing is refused. */
  readonly reference: Reference | null;
  /** The structure in use. Falls back to `'linear'` when the weights name a tree or centroid they do not carry. */
  readonly head: HeadKind;
  /** The L2 penalty the build chose for the linear head, or `null`. */
  readonly l2: number | null;
  /** The linear head — for a tags shim, one two-class head per tag. */
  readonly heads: T extends 'tags' ? Head[] : Head;
  readonly tree: Tree | null;
  readonly centroid: Head | null;

  /** The calibration temperature in force: fitted at build, replaceable by `recalibrate()`. It changes what a confidence means, never which answer wins. */
  get temperature(): number;

  /** The two numbers in force: when to act, and when to say nothing. `refitGate` / `refitFloor` results win over the build's. */
  get gates(): Gates;

  /** The groups, by answer name, when this shim is a tree; otherwise `null`. */
  get groups(): string[][] | null;

  /**
   * Refit the temperature on decisions this shim has made, once you know how they turned out.
   * Needs `minSamples` usable rows (default 30); rows without a vector, or whose label is not one of
   * this shim's answers, are skipped. Kept only if it improves calibration on those same rows.
   * Declines, and says why, on a tags shim: tags apply no temperature.
   */
  recalibrate(rows: readonly (OutcomeSample | null | undefined)[] | null | undefined, opts?: { minSamples?: number }): RecalibrateResult;

  /**
   * Refit the familiarity floor from real IN-SCOPE input, unlabelled. The build's floor is a
   * percentile of the shim's own examples and does not transfer to real traffic. Handing this raw
   * traffic that is half out-of-scope drags the floor toward zero and disables refusal.
   *
   * @param opts `target`: share of these inputs the new floor should refuse (default 0.05).
   *   `minSamples`: fewer than this and nothing changes (default 20). `floor`: the lowest floor it
   *   will set (default 0.005).
   */
  refitFloor(vectors: readonly Vector[] | null | undefined, opts?: { target?: number; minSamples?: number; floor?: number }): RefitFloorResult;

  /**
   * Re-read the confidence gate for the traffic this shim actually gets, from unlabelled in-scope
   * vectors: it estimates its own answer mix and reweights the build's held-out rows to it. It
   * cannot see an answer the shim gets confidently wrong. Declines, and says why, on a tags shim,
   * on a build whose gate was withheld, and on weights compiled before `gateRows` shipped.
   *
   * @param opts `target`: accuracy the gate should promise, 0–1 (default 0.9). `minSamples`:
   *   fewer vectors than this and nothing changes (default 100).
   */
  refitGate(vectors: readonly Vector[] | null | undefined, opts?: { target?: number; minSamples?: number }): RefitGateResult;

  /**
   * Uncalibrated probability per answer index, from whichever structure the build chose. Classify
   * shims only: throws on a tags shim, whose per-tag scores are on `decideVector()`.
   */
  rawProbs(vec: Vector): number[];

  /** The action for a familiarity and a confidence, under the gates in force. A `null` familiarity never refuses. */
  actionFor(familiarity: number | null | undefined, confidence: number): Action;

  /** Decide from an already-computed vector. Use when several shims share one encode. Never recorded. */
  decideVector(vec: Vector): DecisionFor<T>;

  /**
   * Encode and decide. Input over `LONG_INPUT_CHARS` is split and decided in pieces unless
   * `{ long: false }`. Recorded when something is observing, and then the result carries an `id`.
   */
  decide(text: string, opts?: DecideOptions): Promise<DecisionFor<T>>;

  /** Split, decide on every piece, and combine them — whatever the input's length. Not recorded. */
  decideLong(text: string, opts?: DecideOptions): Promise<DecisionFor<T>>;

  /** One decision from a vector per piece. The action is re-derived from the combined confidence and familiarity. */
  decideChunks(vectors: readonly Vector[], chunks: readonly string[], opts?: ReduceOptions): DecisionFor<T> & { from: string; chunks: number };

  /**
   * Safe to act on without escalating: confidence at or above `threshold` (default: the gate in
   * force) AND familiarity above the floor in force. A `null` familiarity counts as familiar.
   */
  isConfident(result: { confidence: number; familiarity?: number | null }, threshold?: number): boolean;

  /** Same as `new Shim(compiled)`. */
  static load(compiled: ShimWeightsInput): Shim;

  private _temperature;
  private _gate;
  private _floor;
  private _predict;
  private _observed;
  private _decideLong;
}

/** What a `Bank` accepts per key: a loaded shim, compiled weights, or a draft (skipped). */
export type BankEntry = Shim | ShimWeightsInput | DraftWeightsInput;

/** One decision per key of the bank. A key whose entry was a draft is absent at runtime. */
export type BankDecisions<S> = { [K in keyof S]: S[K] extends Shim<infer T extends ShimType> ? DecisionFor<T> : Decision };

/**
 * Several shims over the same input, sharing a single forward pass: the encode happens once no
 * matter how many shims the bank holds.
 */
export class Bank<S extends Record<string, BankEntry> = Record<string, BankEntry>> {
  /** Weights are loaded into shims. Drafts are skipped rather than rejected, so a half-written shim does not stop the finished ones. */
  constructor(shims: S);

  /** The loaded shims, by key. Drafts are absent. */
  readonly shims: { [K in keyof S]: Shim };

  /**
   * One encode, one decision per shim. A long input is split once and each shim weighs the pieces
   * for itself. When something is observing, each shim's decision is recorded with its own `id`,
   * `via: 'bank'` and its key as `field`.
   */
  decide(text: string, opts?: DecideOptions): Promise<BankDecisions<S>>;

  /** Runtime bytes of the shims' linear heads (float32), excluding centroids, tree leaves, the shipped examples and the shared encoder. */
  get bytes(): number;
}
