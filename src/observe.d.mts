// Types for observe.mjs — what a shim saw, what it said, and what happened next. Nothing is sent
// anywhere: there is no default sink, and an app that never calls `observe()` records nothing.

import type { Action, Gates, Vector } from './format.mjs';

export type { Action } from './format.mjs';

/**
 * What the person did with a decision. `accepted` and `ignored` are deliberately different: a
 * suggestion scrolled past is not a suggestion refused.
 */
export type Outcome = 'accepted' | 'corrected' | 'rejected' | 'ignored';

/** One recorded decision, as handed to the sink. */
export interface DecisionRow {
  kind: 'decision';
  /** The id the decision's result carries; report an outcome against it. */
  id: string;
  /** Name of the shim that decided. */
  shim: string;
  /** When, in ms since the Unix epoch. */
  at: number;
  /** Wall time from the start of the call to the record, in ms to two places, encoding included. Rows from one `Bank` or `System` call share a start. */
  ms: number;
  /** The whole input, or `null` when the call passed `{ keepText: false }`. */
  text: string | null;
  /** Whitespace-separated words in the input. */
  words: number;
  /** Characters in the input. */
  chars: number;
  /**
   * The vector the decision was made on — for a long input, the chosen piece's. Handed over BY
   * REFERENCE: a sink that keeps it past the next decision should copy it (`ring` does). `null`
   * when a long decision's piece could not be matched.
   */
  vector: Vector | null;
  /** What was said: a label, or for a tags shim the tags raised. */
  answer: string | string[];
  /** Calibrated confidence, 0–1. */
  confidence: number;
  /** Familiarity, 0–1 percentile; `null` for weights with no reference set. */
  familiarity: number | null;
  action: Action;
  /** The gates in force when it decided. */
  gates: Gates;
  /** How the shim was asked. Absent for a plain `shim.decide()`. */
  via?: 'bank' | 'system' | 'adaptive';
  /** With `via: 'bank'` or `'adaptive'`: the key the shim sits under in the bank. */
  field?: string | null;
  /** With `via: 'system'`: the system's name. */
  system?: string;
  /** With `via: 'adaptive'`: `'yes'` / `'no'` when a remembered verdict spoke, otherwise `null`. */
  memory?: 'yes' | 'no' | null;
}

/** One reported outcome, as handed to the sink. */
export interface OutcomeRow {
  kind: 'outcome';
  /** The decision it refers to. */
  id: string | undefined;
  outcome: Outcome;
  /** The right answer, when the outcome is `corrected`; otherwise `null`. */
  label: string | null;
  /** When, in ms since the Unix epoch. */
  at: number;
}

/** Everything a sink receives. */
export type ObserveEntry = DecisionRow | OutcomeRow;

/** Where decisions go. */
export type Sink = (entry: ObserveEntry) => void;

/**
 * Set where decisions go. The sink is called synchronously on every decide, so it must be cheap —
 * push to an array, not a network call. An exception it throws is swallowed. Pass `null` (or
 * anything that is not a function) to stop. There is one sink per process; a second call replaces the first.
 */
export function observe(fn: Sink | null | undefined): void;

/** Whether a sink is set. While false, decisions carry no `id` and nothing is allocated. */
export const observing: () => boolean;

/** Hand one entry to the sink. Called by the runtime; an app should not need to. A no-op with no sink. */
export function record(entry: ObserveEntry): void;

/**
 * Report the outcome of a decision, when the app can see one. A no-op when nothing is observing —
 * which is also the only time `id` is `undefined`, so a result's optional `id` can be passed straight in.
 *
 * @param id the `id` from the decision this refers to
 * @param opts `label`: the right answer, when the outcome is `corrected`
 */
export function outcome(id: string | undefined, outcome: Outcome, opts?: { label?: string | null }): void;

/** A fresh decision id: `d` + a base-36 timestamp + a base-36 counter. Called by the runtime. */
export const nextId: () => string;

/** A decision as `ring` keeps it: the vector copied, and the outcome attached once reported. */
export interface RingRow extends Omit<DecisionRow, 'vector'> {
  /** The ring's own copy. */
  vector: Float32Array | null;
  /** Set once `outcome()` has been called for this `id`. */
  outcome?: Outcome;
  /** The label passed with the outcome, or `null` when none was. */
  correctedTo?: string | null;
}

/** Filters for `Ring.for`. `null` / omitted means "any". */
export interface RingFilter {
  action?: Action | null;
  /** `true`: only `'act'` decisions; `false`: only the rest. */
  acted?: boolean | null;
  outcome?: Outcome | null;
}

/** What a first week wants to know, none of which needs a label. */
export interface RingSummary {
  /** Decisions held. */
  decisions: number;
  /** Share that were `'act'`, 0–1 to three places. */
  acted: number;
  /** Share that were `'suggest'`, 0–1. */
  suggested: number;
  /** Share that were `'refuse'`, 0–1. A high value means the floor is wrong rather than the model. */
  refused: number;
  /** Median input length in words; `null` with no rows. */
  medianWords: number | null;
  /** Median decision time in ms, to one place; `null` with no rows. */
  medianMs: number | null;
  /** Decisions with an outcome reported. Zero means `recalibrate` can never become usable. */
  withOutcome: number;
}

/** A bounded in-memory sink. */
export interface Ring {
  /** The sink itself: pass `log.push` to `observe()`. Safe to detach. An outcome whose decision has already been dropped is ignored. */
  push(entry: ObserveEntry): void;
  /** The live row array, oldest first. */
  readonly rows: RingRow[];
  /** Drop everything. */
  clear(): void;
  /** Decisions for one shim, filtered. */
  for(name: string, filter?: RingFilter): RingRow[];
  /**
   * Vectors to hand `refitFloor` / `refitGate`. With no filter: decisions a person accepted or
   * corrected — the strongest evidence available that the input was in scope. Pass a filter (even
   * `{}`, for everything) to choose otherwise. Never feed a refit raw traffic that may be out of scope.
   */
  vectorsFor(name: string, filter?: RingFilter | null): Float32Array[];
  /** Rows to hand `recalibrate`: accepted and corrected decisions, labelled with the correction when there was one, else the shim's own answer. */
  outcomesFor(name: string): { vector: Float32Array; label: string | string[] }[];
  /** Counts and medians for one shim, or for everything when `name` is omitted. */
  summary(name?: string | null): RingSummary;
}

/**
 * A bounded in-memory sink, which is what most apps want to start with. Keeps the most recent
 * `limit` decisions (default 1000) and copies each vector on the way in.
 */
export function ring(limit?: number): Ring;
