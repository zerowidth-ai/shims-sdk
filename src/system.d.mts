// Types for system.mjs — several shims wired into one decision: rules that run before any model,
// a router handing off to specialists, a bank of independent fields, a person at the end.
// A system is that wiring as data (`*.system.json`); it ships next to the shims it names.

import type { Decision, ReduceOptions, Vector } from './format.mjs';
import type { Shim } from './index.mjs';

/**
 * A step's gates. Each defaults to what the shim earned — its live confidence gate and familiarity
 * floor, refits included. A number overrides it; `false` lets everything through.
 */
export interface StepGate {
  /** Confidence, 0–1. */
  confidence?: number | false;
  /** Familiarity, 0–1 percentile. */
  familiarity?: number | false;
}

/** One pattern in a rules step. */
export interface Rule {
  /** Becomes the system's answer when this rule matches. */
  name: string;
  /** Regular-expression source, tested against the whole input. */
  match: string;
  /** Regular-expression flags. Defaults to `'i'`. */
  flags?: string;
}

/** Fields every step carries. */
export interface StepBase {
  /** Unique within the system. */
  id: string;
  /**
   * Run on every input, even after an earlier step has decided the outcome. An `always` shim step
   * is an annotation — urgency, tone — so its answer never becomes the system's and it never sets
   * the outcome.
   */
  always?: boolean;
  /** For people reading the spec. Not read by the runtime. */
  note?: string;
}

/** Patterns that run before any model. The first rule to match stops the run. */
export interface RulesStep extends StepBase {
  kind: 'rules';
  rules: Rule[];
  /** The outcome when a rule matches. Defaults to `'stopped'`. */
  outcome?: string;
}

/** Fields the two gated step kinds, `shim` and `route`, share. */
export interface GatedStep extends StepBase {
  gate?: StepGate;
  /** Familiar but under the confidence gate: the outcome to stop with, or `'continue'`. Defaults to `'escalate'`. */
  onUnsure?: string;
  /** Under the familiarity floor: the outcome to stop with, or `'continue'`. Defaults to `'escalate'`. */
  onUnfamiliar?: string;
}

/** Ask one shim. */
export interface ShimStep extends GatedStep {
  kind: 'shim';
  /** Name of the shim to ask. */
  shim: string;
}

/** Hand an earlier step's answer to the specialist for it. */
export interface RouteStep extends GatedStep {
  kind: 'route';
  /** The `id` of an EARLIER step whose answer picks the route. */
  on: string;
  /** Answer → the shim to ask next. `null` is a deliberate dead end: the run ends `'routed'` with the parent's answer. */
  routes: Record<string, string | null>;
  /** An answer with no entry in `routes`: `'continue'` carries on with what is already decided; anything else is the outcome to stop with. Defaults to `'escalate'`. */
  onMissing?: string;
  /** Where a `null` route ends, for the trace. Defaults to `'no specialist'`. */
  endsAt?: string;
  /**
   * When the step this routes on is not confident, ask every specialist and hand to the one that
   * recognises the input: its familiarity must clear its own floor AND beat the runner-up by
   * `margin` (default 0.25). Needs at least two routes with a shim.
   */
  rescue?: true | 'siblings' | { margin?: number };
}

/** Ask several shims as independent fields. Never decides the outcome. */
export interface BankStep extends StepBase {
  kind: 'bank';
  /** Field name → shim name. */
  shims: Record<string, string>;
  gate?: StepGate;
}

export type SystemStep = RulesStep | ShimStep | RouteStep | BankStep;

/** A `*.system.json`: which shim asks first, what each answer hands to, where the gates sit, and what happens when nothing is confident. */
export interface SystemSpec {
  name: string;
  type?: 'system';
  /** At least one. Run in order. */
  steps: SystemStep[];
  /** How this system's long inputs are combined. Defaults to `'best'`. */
  reduce?: ReduceOptions['mode'];
  reducePower?: number;
  /**
   * Confidence along a path multiplies. When two or more routing steps decided and the product
   * of their confidences is under this (0–1), the path is weak: the answer is withheld. Defaults to 0.
   */
  minPathConfidence?: number;
  /** The outcome for a weak path, when nothing else set one. Defaults to `'escalate'`. */
  onWeakPath?: string;
  /** What the input is, in words. Carried into `compileSystem`'s summary; not read by the runtime. */
  input?: string;
  /** What each outcome means, in words. Carried into `compileSystem`'s summary; not read by the runtime. */
  outcomes?: Record<string, string>;
  /** For people reading the spec. */
  note?: string;
}

/** A step as TypeScript sees it in imported JSON, where `kind` has widened to `string`. `validateSystem` says what is wrong with it. */
export interface SystemStepInput {
  id: string;
  kind: string;
  always?: boolean;
  note?: string;
  rules?: Rule[];
  outcome?: string;
  shim?: string;
  shims?: Record<string, string>;
  on?: string;
  routes?: Record<string, string | null>;
  onMissing?: string;
  endsAt?: string;
  rescue?: boolean | string | { margin?: number };
  gate?: { confidence?: number | boolean; familiarity?: number | boolean };
  onUnsure?: string;
  onUnfamiliar?: string;
}

/** A system spec as imported JSON. Every `SystemSpec` is a valid `SystemSpecInput`. */
export interface SystemSpecInput {
  name: string;
  type?: string;
  steps: SystemStepInput[];
  reduce?: string;
  reducePower?: number;
  minPathConfidence?: number;
  onWeakPath?: string;
  input?: string;
  outcomes?: Record<string, string>;
  note?: string;
}

/**
 * Check a spec's shape. Returns one message per problem; empty means valid. Accepts anything.
 *
 * @param have names of the shims that exist; when non-empty, a step naming any other shim is an error
 */
export function validateSystem(spec: unknown, have?: readonly string[]): string[];

/** Every shim a system can reach, so a caller knows what to load. */
export const shimsUsed: (spec: { steps?: readonly SystemStepInput[] }) => string[];

/** How a gated step's decision stood against its gates. `rescued`: not confident, but a sibling vote picked a specialist. */
export type StepState = 'confident' | 'unsure' | 'unfamiliar' | 'rescued';

/** A shim or route step's result: the shim's decision, plus how it stood. */
export type StepResult = Decision & {
  /** Name of the shim that was asked. */
  shim: string;
  state: StepState;
  /** When `state` is `'rescued'`: the specialist that won the vote — and `answer` is then the route that leads to it, not what the shim said. Otherwise `null`. */
  rescuedBy: string | null;
};

/** One field of a bank step: `applied` cleared the confidence gate, `offered` did not, `silent` was under the familiarity floor. */
export type BankFieldResult = Decision & { state: 'applied' | 'offered' | 'silent' };

/** The ballot from a sibling rescue. */
export interface RescueVote {
  /** Every specialist asked, most familiar first; familiarity 0–1 to three places. */
  votes: { key: string; familiarity: number }[];
  /** The route key that won, or `null` when nobody recognised the input clearly enough. */
  took: string | null;
  /** The shim behind the winning route, or `null`. */
  by: string | null;
  /** Familiarity gap between the top two, to three places. */
  margin: number;
}

/** One line of a run's trace, in the order things happened. Narrow with `kind`, then with `in`. */
export type TraceEntry =
  | { id: string; kind: 'rules'; /** Name of the rule that matched, or `null`. */ matched: string | null }
  | { id: string; kind: 'bank'; fields: Record<string, { answer: string | string[]; state: BankFieldResult['state'] }> }
  | { id: string; kind: 'route'; /** The step it routes on never decided. */ skipped: string }
  | { id: string; kind: 'route'; /** The answer routed on. */ on: string | string[]; to: null; note: string }
  | {
      id: string; kind: 'shim' | 'route'; shim: string; answer: string | string[];
      /** 0–1. */ confidence: number;
      /** 0–1 percentile. */ familiarity: number | null;
      state: StepState;
    }
  | { id: string; kind: 'rescue'; /** The step that was unsure. */ on: string; /** What that step said. */ was: string | string[]; votes: RescueVote['votes']; took: string | null; margin: number };

/** Options for `System.run`. */
export interface SystemRunOptions {
  /** Pieces already split. Defaults to splitting input over `LONG_INPUT_CHARS`, else the whole text. */
  chunks?: string[];
  /** Vectors already encoded, ONE PER ENTRY of the chunks in play, so a replay can skip the encoder. */
  vectors?: Vector[];
  /** Overrides the spec's `reduce`. */
  reduce?: ReduceOptions['mode'];
  /** Overrides the spec's `reducePower`. */
  reducePower?: number;
  /** `false` records `text: null` when something is observing. */
  keepText?: boolean;
}

/** What one run of a system decided. */
export interface SystemRunResult {
  /**
   * What should happen next: `'routed'` when an answer was reached, `'escalate'` when not, a rules
   * step's `outcome` (default `'stopped'`), or whatever string the spec named in `onUnsure`,
   * `onUnfamiliar`, `onMissing` or `onWeakPath`.
   */
  outcome: string;
  /**
   * The system's answer: the matched rule's name, or the answer of the last non-`always` shim or
   * route step that was confident or rescued (for a `null` route, the answer routed on). `null`
   * when nothing decided or the path was weak.
   */
  answer: string | string[] | null;
  /** While something is observing: the `id` of the decision whose answer became the system's — the one to report an `outcome()` against. Otherwise `null`. */
  id: string | null;
  /** Per step `id`: a shim or route step's result, or a bank step's results by field. Steps that did not run are absent. */
  results: Record<string, StepResult | Record<string, BankFieldResult>>;
  trace: TraceEntry[];
  /** Product of the routing steps' confidences, 0–1 to four places; `1` when none decided. `always` steps are left out. */
  pathConfidence: number;
  /** True when two or more routing steps decided and `pathConfidence` is under the spec's `minPathConfidence`. */
  weakPath: boolean;
  /** Distinct shims asked. Each is evaluated once per run however many steps name it. */
  heads: number;
  /** Texts encoded (or supplied): the number of pieces. */
  encodes: number;
  /** The pieces, when the input was split; otherwise `null`. */
  chunks: string[] | null;
}

/** A system ready to run: a validated spec plus the loaded shims it names. */
export class System {
  /**
   * Throws, listing every problem, when the spec is invalid or names a shim that was not given.
   *
   * @param shims loaded shims by NAME — the names the spec's steps use
   */
  constructor(spec: SystemSpecInput, shims: Record<string, Shim>);

  readonly spec: SystemSpecInput;
  readonly shims: Record<string, Shim>;
  readonly name: string;

  /** The gates a step applies to a shim: the step's overrides, else the shim's live gates. `false` becomes 0. */
  gatesFor(step: { gate?: SystemStepInput['gate'] }, shim: Shim): {
    /** Confidence gate, 0–1. */ conf: number;
    /** Familiarity floor, 0–1. */ fam: number;
  };

  /**
   * A confident child can rescue an unsure parent: ask every specialist the route on `stepId` can
   * hand to, and see which one RECOGNISES the input. Familiarity is the ballot, not confidence.
   * `null` when no route on that step has `rescue`, or fewer than two specialists stand.
   *
   * @param ask decides (once) with the named shim
   */
  askSiblings(stepId: string, ask: (shimName: string) => { familiarity: number }): RescueVote | null;

  /**
   * Run the input through the steps. One encode for the whole system: every shim reads the same
   * vector, or — for long input — every piece, each shim weighing them for itself. When something
   * is observing, each shim asked is recorded once, with `via: 'system'` and this system's name.
   */
  run(text: string, opts?: SystemRunOptions): Promise<SystemRunResult>;
}
