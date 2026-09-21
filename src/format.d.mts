// Types for format.mjs — the on-disk format for a compiled shim — plus the shared domain types
// (source, weights, report, decision, embedder) the rest of the SDK's declarations import.
// Hand-written; keep in step with format.mjs, index.mjs and compile/compile.mjs.

import type { Head } from './math.mjs';

/* ---------- vectors and the encoder ---------- */

/**
 * Anything indexable by number with a length: a `Float32Array`, a plain `number[]`, an `Int8Array`.
 * Functions that only read a vector accept this; functions that produce one say exactly what they return.
 */
export type Vector = ArrayLike<number>;

/**
 * Options for loading the shared encoder. Everything not named here is passed through to
 * transformers.js `from_pretrained` — `{ device: 'webgpu' }` and so on. Only the FIRST load in a
 * process reads them — the encoder is loaded once and reused — so pass them to `preload()` at
 * startup, which also reaches the SDK's worker.
 */
export interface EncoderOptions {
  /** Hugging Face model id. Defaults to `ENCODER_ID`. Weights are only valid for the encoder they were compiled with. */
  modelId?: string;
  /** Quantisation passed to transformers.js. Defaults to `'q8'`. */
  dtype?: string;
  /** Host to fetch the model from instead of the Hugging Face hub. Sets transformers.js `env.remoteHost`. */
  remoteHost?: string;
  /** Path template under `remoteHost`, e.g. `'models/{model}/'`. Sets transformers.js `env.remotePathTemplate`. */
  remotePathTemplate?: string;
  /** Where the ONNX runtime's `.wasm` files are served from, instead of jsDelivr: a URL prefix, or a map of file name to URL. */
  wasmPaths?: string | Record<string, string>;
  /** Passed through to transformers.js untouched. */
  [option: string]: unknown;
}

/**
 * Options for one embed call: the encoder options, and nothing else. There is no batch size. Every
 * text gets a forward pass of its own, because a quantised encoder's vector for a text shifts with
 * whatever shares its batch, and a shim must see the same vector at build and at runtime.
 */
export interface EmbedOptions extends EncoderOptions {}

/**
 * Text in, vectors out. The contract every embedder keeps: a `string` in gives ONE vector back,
 * a `string[]` in gives an array with one vector per text, in order. Vectors are `DIM` wide
 * (the embedding plus the surface features) and the embedding half is L2-normalised.
 * The SDK always awaits the result, so a synchronous embedder works too.
 */
export interface Embedder {
  (texts: string | string[], opts?: EmbedOptions): Vector | Vector[] | Promise<Vector | Vector[]>;
  /** Optional: load and warm the model ahead of the first decision. `preload()` calls it when present. */
  warm?: (opts?: EncoderOptions) => unknown;
}

/* ---------- the source file: *.shim.json ---------- */

/** What kind of decision a shim makes. `classify`: one answer of K. `tags`: any of K, independent yes/no heads. */
export type ShimType = 'classify' | 'tags';

/** One hand-written example for a `classify` shim. */
export interface ClassifyExample {
  text: string;
  /** Must be one of the shim's `labels`. */
  label: string;
}

/** One hand-written example for a `tags` shim. */
export interface TagsExample {
  text: string;
  /** Every tag that applies; each must be one of the shim's `labels`. May be empty. */
  labels: string[];
}

/** Fields every shim source carries, whatever its type. */
export interface ShimSourceBase {
  /** The shim's name. Decisions are recorded under it and a `System` refers to the shim by it. */
  name: string;
  /** The question this shim answers, for people reading the file and the report. */
  question?: string | null;
  /**
   * The real setting the input comes from. The compiler only checks that it is present (40+
   * characters) when every answer comes from generated prototypes, and warns `no-context` if not.
   */
  context?: string;
  /** The answers. At least two. */
  labels: string[];
  /**
   * Generated texts per answer, keyed by answer. They train alongside real examples at a weight of
   * `prototypeWeight(real examples per answer)`: 1 with none, fading to a floor of 0.25. Three or
   * more for every answer is enough to compile with no examples at all.
   */
  prototypes?: Record<string, string[]> | null;
  /**
   * Unlabelled text the app believes is IN SCOPE. With 20 or more, the familiarity floor is fitted
   * on these instead of on the shim's own examples. Read for `classify` shims only.
   */
  realSample?: string[];
  /** Per-answer descriptions written by authoring tools. Not read by the compiler or the runtime. */
  definitions?: Record<string, string>;
}

/** Source for a `classify` shim — one answer of K. */
export interface ClassifySource extends ShimSourceBase {
  /** Defaults to `'classify'` when omitted. */
  type?: 'classify';
  /** Hand-written examples. At least 3 per answer (or prototypes for every answer) before it compiles; fewer gives a draft. */
  examples: ClassifyExample[];
  /**
   * Grouping. `'auto'` (the default) lets the compiler try a tree when there are 8+ answers and keep
   * it only if it beats a flat head; `'off'` never tries; an object of `group name → answers` pins
   * a grouping, which must place every answer in exactly one group and is always used.
   */
  tree?: 'auto' | 'off' | Record<string, string[]>;
}

/** Source for a `tags` shim — any of K, independent binary heads. */
export interface TagsSource extends ShimSourceBase {
  type: 'tags';
  examples: TagsExample[];
  /** `tree` applies to classify shims only; anything other than `'auto'` / `'off'` fails validation here. */
  tree?: 'auto' | 'off';
}

/** A `*.shim.json` source file. */
export type ShimSource = ClassifySource | TagsSource;

/**
 * A source file as TypeScript sees it when it arrives as imported JSON: literal unions have widened
 * to `string`. Accepted wherever a source is read; `validateSource` reports what is wrong with it.
 */
export interface ShimSourceInput {
  name: string;
  type?: string;
  question?: string | null;
  context?: string;
  labels: string[];
  examples: { text: string; label?: string; labels?: string[] }[];
  prototypes?: Record<string, string[]> | null;
  tree?: string | Record<string, string[]>;
  realSample?: string[];
  definitions?: Record<string, string>;
}

/* ---------- the weights file: *.weights.json ---------- */

/** A head as it sits in a weights file: `W` and `b` are base64 of little-endian float32. */
export interface PackedHead {
  /** Number of answers this head chooses between. */
  K: number;
  /** base64 float32, `K × dim` values, row per answer. */
  W: string;
  /** base64 float32, `K` biases. */
  b: string;
}

/** The shipped training set as it sits in a weights file: int8 vectors, one float32 scale each. */
export interface PackedReference {
  /** base64 int8, `count × DIM` values. */
  q: string;
  /** base64 float32, one dequantisation scale per vector. */
  scales: string;
  /** 21 quantiles (0, 0.05 … 1) of each example's cosine to its nearest other example. The familiarity yardstick. */
  spread: number[];
  /** Number of vectors. */
  count: number;
  /** Answer index per vector. For a tags shim, the first tag of each row (0 when it has none). */
  labels?: number[] | null;
}

/** A tree as it sits in a weights file. The root reads the shim's reference set, so only leaves ship. */
export interface PackedTree {
  /** Answer indices per group. */
  groups: number[][];
  /** One linear head per group; `null` for a group with a single answer or a single answer present. */
  leaves: (PackedHead | null)[];
}

/** Which structure decides: a fitted linear head, nearest neighbours over the shipped examples, nearest class mean, or a grouped tree. */
export type HeadKind = 'linear' | 'knn' | 'centroid' | 'tree';

/** One line of the build report's advice. */
export interface ShimNote {
  /** `error` fails `shim-compile --check`. */
  level: 'info' | 'warn' | 'error';
  /**
   * Stable identifier. Emitted today: `draft`, `too-early`, `not-learnable`, `weak`, `thin-answer`,
   * `low-recall`, `incoherent-answer`, `length-gap`, `few-examples`, `multi-modal`, `head-by-coverage`,
   * `no-context`, `tree`, `tree-not-used`, `weak-tag`; and from `compileSystem`: `partial-ballot`,
   * `provisional-step`, `no-threshold`.
   */
  code: string;
  /** A sentence or two for the person reviewing the build. */
  text: string;
}

/** Fields every finished build report carries, whatever the shim's type. */
export interface ShimReportBase {
  /** Balanced accuracy, 0–1: mean per-answer recall on held-out rows. For tags, the mean over tags of each tag's balanced accuracy. */
  macroRecall: number;
  /** What guessing scores, 0–1: `1 / K` for classify, `0.5` for tags. */
  chance: number;
  /** True when there are too few real examples for the score to settle (under 8 per answer for classify, 4 per tag for tags). */
  provisional: boolean;
  /** The opposite of `provisional`. */
  scorable: boolean;
  /**
   * The confidence gate, 0–1: at or above it, held-out decisions were 90% accurate. Rounded down to
   * three places. `1` — never act — when the build is provisional or no confidence reaches 90%.
   */
  suggestedThreshold: number;
  /** Share of held-out decisions, 0–1, at or above `suggestedThreshold`. `0` when the gate is withheld. */
  coverageAtThreshold: number;
  /** On a provisional build, the gate that WAS computed (optimistic, from too few rows); otherwise `null`. Never applied by the runtime. */
  provisionalThreshold: number | null;
  /** Below this familiarity (0–1 percentile) the runtime refuses. Fitted to refuse about 5% of held-out in-scope rows; `0.25` when there were too few to fit. */
  familiarityFloor: number;
  /** How many held-out familiarity scores the floor was fitted on. `0` means the default was kept. */
  familiarityFittedOn: number;
  /** `'examples'`: fitted on the shim's own examples, which does not transfer to real traffic. `'realSample'`: fitted on the source's in-scope sample. */
  floorFittedOn: 'examples' | 'realSample';
  /** Number of generated prototype rows trained on. */
  prototypes: number;
  /** Weight each prototype row carried, 0.25–1. */
  prototypeWeight: number;
  /** What the held-out score was measured on. `'prototypes'` means there were too few real examples to score on. */
  scoredOn: 'examples' | 'prototypes';
  /** Held-out recall per answer, 0–1; `null` for an answer with no scored rows. For tags, each tag's balanced accuracy. */
  recall: Record<string, number | null>;
  /** Runtime bytes of the heads (float32), excluding the reference set and the shared encoder. */
  bytes: number;
  /** Runtime bytes of the shipped int8 examples and their scales. */
  referenceBytes: number;
  /** Advice for a reviewer. */
  notes: ShimNote[];
}

/** What the build measured when it fitted a calibration temperature. ECE values are 0–1. */
export interface CalibrationReport {
  /** The temperature the runtime applies. `1` when the fitted one did not improve calibration. */
  temperature: number;
  /** Expected calibration error of the raw confidences; `null` with no held-out rows. */
  eceBefore: number | null;
  /** Expected calibration error with `temperature` applied. */
  eceAfter: number | null;
  /** The temperature that maximised likelihood, kept or not. */
  fittedTemperature: number;
  /** Expected calibration error had `fittedTemperature` been applied. */
  eceIfScaled: number | null;
  /** Whether `fittedTemperature` was kept. */
  applied: boolean;
  /** Held-out predictions it was fitted on. */
  samples: number;
}

/** What the compiler decided about grouping, and what it tried. */
export interface StructureReport {
  chosen: 'tree' | 'flat';
  /** Group counts tried, each with its held-out balanced accuracy (0–1). Empty when no tree was tried. */
  tried: { groups: number; macroRecall: number }[];
  /** Present when a tree was tried: whether the grouping came from the source. */
  pinned?: boolean;
  /** Present when a tree was tried: the best flat head's balanced accuracy, 0–1. */
  flat?: number;
  /** Present when a tree was tried: the groups by answer name when a tree ships, otherwise `null`. */
  groups?: string[][] | null;
  /** Present when no tree was tried: why. */
  reason?: string;
}

/** The build report for a `classify` shim. Every score is cross-validated: each row predicted by a head that never saw it. */
export interface ClassifyReport extends ShimReportBase {
  /** Held-out accuracy, 0–1, to four places. Read `accuracyInterval` rather than this. */
  accuracy: number;
  /** 95% Wilson interval for `accuracy`, `[low, high]`, each 0–1. `null` when nothing was evaluated. */
  accuracyInterval: [number, number] | null;
  /** `confusion[true][predicted]`: counts of held-out rows, indexed by answer. */
  confusion: number[][];
  /** Number of held-out predictions behind the scores. */
  evaluated: number;
  /** One held-out verdict per scored row, in source order: predicted answer index, confidence 0–1, right or not. `null` for a row no fold scored. */
  perExample: ({ pred: number; conf: number; ok: boolean } | null)[];
  calibration: CalibrationReport;
  /**
   * The held-out rows the gate was read off: `[true answer index, calibrated confidence 0–1, 1 if right else 0]`.
   * `refitGate()` reweights these. `null` on a provisional build; absent on weights compiled before it shipped.
   */
  gateRows: [number, number, 0 | 1][] | null;
  /** The answers, repeated from the weights. */
  labels: string[];
  /** The head that ships. */
  head: HeadKind;
  structure: StructureReport;
}

/** Per-tag cross-validation for a `tags` shim. */
export interface TagReport {
  label: string;
  /** Held-out accuracy for this tag as a yes/no question, 0–1. `null` when the tag is on every row or none. */
  accuracy: number | null;
  /** Held-out balanced accuracy for this tag, 0–1, against a 0.5 coin flip. */
  macroRecall: number | null;
  /** Rows that carry this tag. */
  n: number;
}

/** The build report for a `tags` shim. The gate is fitted on whole tag sets, not per tag. */
export interface TagsReport extends ShimReportBase {
  perTag: TagReport[];
}

/** A finished build's report: the part of a weights file a person reads in a pull request. */
export type ShimReport = ClassifyReport | TagsReport;

/** The report inside a draft: what is still missing. */
export interface DraftReport {
  draft: true;
  /** Hand-written examples per answer so far. */
  counts: Record<string, number>;
  /** Examples still needed to reach `minPerAnswer` for every answer. */
  needed: number;
  /** Examples each answer needs before the shim compiles. */
  minPerAnswer: number;
  /** Bytes one linear head will cost once it compiles. */
  bytes: number;
  notes: ShimNote[];
}

/** How the three flat heads measured, and why one was picked. Scores are held-out balanced accuracy, 0–1. */
export interface HeadChoice {
  linear: number;
  knn: number;
  centroid: number;
  /** Share of held-out input, 0–1, each head could act on at its own calibrated 90% gate. */
  coverage: Record<'centroid' | 'linear' | 'knn', number>;
  /** Set when a rival within one standard error on accuracy could act on 10+ points more input and took the pick. */
  coverageOverride: null | {
    from: string;
    to: string;
    /** Balanced-accuracy gap, 0–1. */
    accuracyGap: number;
    standardError: number;
    /** `[from, to]` coverage, each 0–1. */
    coverage: [number, number];
  };
  /** Balanced accuracy per L2 penalty tried, keyed by the penalty as a string. */
  l2: Record<string, number>;
  /** The penalty the linear head was fitted with. */
  l2Chosen: number;
  /** Held-out recall per answer under the linear and kNN heads, 0–1; `null` with no scored rows. */
  perAnswer: { label: string; linear: number | null; knn: number | null }[];
}

/** Fields every weights file carries, draft or finished. */
export interface WeightsBase {
  /** Format version. This runtime loads `2` only. */
  format: typeof FORMAT_VERSION;
  name: string;
  question: string | null;
  labels: string[];
  /** The encoder the vectors came from. The runtime refuses weights compiled for a different one. */
  encoder: string;
  /** Width of the vectors the heads were trained on: embedding plus surface features. */
  dim: number;
  /** ISO 8601 timestamp of the build. */
  compiledAt: string;
  /** Number of hand-written examples in the source. */
  examples: number;
}

/** Fields every finished (non-draft) weights file carries. */
export interface FinishedWeightsBase extends WeightsBase {
  draft?: undefined;
  /** Texts that had to be embedded for this build; the rest came from the vector cache. */
  embedded: number;
  /** The shipped training set, quantised. Familiarity, refusal, long-input routing, the kNN head and the tree root all read it. */
  reference: PackedReference;
}

/** Compiled weights for a `classify` shim. */
export interface ClassifyWeights extends FinishedWeightsBase {
  type: 'classify';
  /** Which structure the runtime decides with. */
  head: HeadKind;
  /** How the flat heads measured against each other. */
  headChoice: HeadChoice;
  /** One packed linear head. It always ships, whichever `head` was chosen, as the fallback. */
  heads: [PackedHead];
  /** The L2 penalty the linear head was fitted with. */
  l2: number;
  /** Unit class means packed as a head. Present only when `head` is `'centroid'`. */
  centroid?: PackedHead;
  /** Present only when `head` is `'tree'`. */
  tree?: PackedTree;
  report: ClassifyReport;
}

/** Compiled weights for a `tags` shim: one binary head per tag. */
export interface TagsWeights extends FinishedWeightsBase {
  type: 'tags';
  head: 'linear';
  /** One packed two-class head per tag, in `labels` order; class 1 is "tag applies". */
  heads: PackedHead[];
  report: TagsReport;
}

/** A compiled `*.weights.json`: what `compileShim` emits for a finished shim and what `new Shim()` loads. */
export type ShimWeights = ClassifyWeights | TagsWeights;

/** What `compileShim` emits for a shim that cannot compile yet. It has no head; `new Shim()` refuses it and `Bank` skips it. */
export interface DraftWeights extends WeightsBase {
  type: ShimType;
  draft: true;
  report: DraftReport;
}

/**
 * What `new Shim()` actually reads, typed loosely enough that a weights file imported as JSON
 * (where `type`, `head` and `format` have widened to `string` / `number`) is accepted as it is.
 * Every `ShimWeights` is a valid `ShimWeightsInput`.
 */
export interface ShimWeightsInput {
  name: string;
  /** Must be `2`, or the constructor throws. */
  format: number;
  /** `'tags'` selects the multi-label path; anything else is treated as classify. */
  type: string;
  labels: string[];
  heads: PackedHead[];
  /** Truthy makes the constructor throw. */
  draft?: boolean;
  /** When present and not this runtime's `ENCODER_ID`, the constructor throws. */
  encoder?: string;
  question?: string | null;
  /** Defaults to `'linear'`. `'tree'` and `'centroid'` fall back to `'linear'` when their data is missing. */
  head?: string;
  l2?: number | null;
  reference?: PackedReference | null;
  tree?: PackedTree | null;
  centroid?: PackedHead | null;
  /** The runtime reads the gates, the temperature and the gate rows; the rest is carried for display. */
  report?: {
    suggestedThreshold?: number;
    familiarityFloor?: number;
    calibration?: { temperature?: number };
    gateRows?: ArrayLike<number>[] | null;
  };
}

/** A draft as a `Bank` sees it: anything with a truthy `draft` is skipped rather than loaded. */
export interface DraftWeightsInput {
  draft: boolean;
  name?: string;
}

/* ---------- decisions ---------- */

/**
 * What the shim thinks you should do with its answer, from the gate and floor it earned at build:
 * `'act'` — confidence cleared the gate; `'suggest'` — it did not; `'refuse'` — familiarity is
 * under the floor, so say nothing.
 */
export type Action = 'act' | 'suggest' | 'refuse';

/** The two numbers a shim gates on. */
export interface Gates {
  /** Confidence (0–1) at or above which a decision is `'act'`. `0.7` when the build shipped none. */
  confidence: number;
  /** Familiarity (0–1 percentile) below which a decision is `'refuse'`. `0.25` when the build shipped none. */
  familiarity: number;
}

/** Fields every decision carries, whatever the shim's type. */
export interface DecisionBase {
  /** How sure the shim is of the answer it gave, 0–1, after the calibration temperature. */
  confidence: number;
  /**
   * Where the input sits among the shim's own examples, as a 0–1 percentile: 0.5 is as typical as the
   * median example, under ~0.2 means nothing nearby. `null` only for weights with no reference set.
   */
  familiarity: number | null;
  action: Action;
  /** Present only while something is observing: the id to report an `outcome()` against. */
  id?: string;
  /** Present when the input was split: the text of the piece this shim recognised best. */
  from?: string;
  /** Present when the input went through chunk reduction: how many pieces it was read in. */
  chunks?: number;
}

/** A `classify` shim's decision. */
export interface ClassifyDecision extends DecisionBase {
  /** One of the shim's `labels`. */
  answer: string;
  /** Calibrated probability per answer, 0–1, summing to 1. */
  probs: Record<string, number>;
  /** The winning answer's probability before the temperature was applied, 0–1. */
  rawConfidence: number;
  /** Present when the shim is a tree: the group the answer came through. */
  path?: {
    /** Index of the group. */
    group: number;
    /** The root's probability for that group, 0–1. */
    groupConfidence: number;
    /** The answers in that group. */
    siblings: string[];
  };
}

/** A `tags` shim's decision. */
export interface TagsDecision extends DecisionBase {
  /** Every tag whose probability is 0.5 or more. May be empty. */
  answer: string[];
  /** Probability per tag, 0–1, independent of each other. */
  scores: Record<string, number>;
}

/** The result of asking a shim. */
export type Decision = ClassifyDecision | TagsDecision;

/** The decision shape for a shim type; the union when the type is not known statically. */
export type DecisionFor<T extends ShimType> = T extends 'tags' ? TagsDecision : ClassifyDecision;

/** How a long input's per-piece decisions become one. */
export interface ReduceOptions {
  /**
   * `'best'` (default) decides on the piece the shim recognises best. `'pooled'` averages every
   * piece's probabilities weighted by familiarity — for documents whose evidence is spread out.
   * `'pooled'` reads `probs`, so it is meaningful for classify shims only.
   */
  mode?: 'best' | 'pooled';
  /** Exponent on familiarity when pooling. Higher behaves more like `'best'`. Defaults to 3. */
  power?: number;
}

/* ---------- format.mjs itself ---------- */

/** The encoder every weights file is compiled against. Changing it invalidates every weights file. */
export const ENCODER_ID: string;
/** Width of the sentence embedding (384). */
export const EMBED_DIM: number;
/** Width of what a head sees: `EMBED_DIM` plus the surface features (392). */
export const DIM: number;
/** The weights-file format this runtime reads and the compiler writes. */
export const FORMAT_VERSION: 2;

/** File name of the compiler's on-disk vector cache for an encoder. One file per encoder, so vectors cannot mix. */
export const vectorCacheFile: (id?: string) => string;

/** Float32 head → its base64 form for a weights file. */
export const packHead: (head: Pick<Head, 'K' | 'W' | 'b'>) => PackedHead;

/** Base64 head → float32 arrays. `dim` is not stored in the file; it defaults to `DIM`. */
export const unpackHead: (h: PackedHead, dim?: number) => Head;

/** Bytes one `K`-answer linear head costs at runtime: `K × (dim + 1) × 4`. */
export const headBytes: (K: number, dim?: number) => number;

/**
 * Minimal validation with messages aimed at whoever is editing the JSON by hand.
 * Returns one message per problem; empty means valid. Accepts anything, including `null`.
 * (`type: "schema"` is recognised here as a reserved format, but nothing compiles or runs it.)
 */
export function validateSource(src: unknown): string[];

/** Examples each answer needs before a shim compiles (3). */
export const MIN_PER_ANSWER: number;

/** Generated prototypes are worth roughly this many real examples per answer (8); their weight fades to the floor as real examples reach it. */
export const PROTOTYPE_PARITY: number;

/** Prototypes never decay below this weight (0.25). */
export const PROTOTYPE_FLOOR: number;

/** Weight of a prototype row given the mean number of real examples per answer: `max(PROTOTYPE_FLOOR, 1 − n / PROTOTYPE_PARITY)`. */
export const prototypeWeight: (realPerAnswer: number) => number;

/** Whether a source can compile yet, and what it is short of. */
export interface Readiness {
  /** Hand-written examples per answer. */
  counts: Record<string, number>;
  /** Answers with fewer than `MIN_PER_ANSWER` examples. */
  short: string[];
  /** Total examples still needed across `short`. */
  needed: number;
  /** True when every answer has 3+ prototypes — enough to compile with no examples. */
  described: boolean;
  /** Two or more answers, and either described or no answer short. */
  ready: boolean;
}

/** Can this compile yet? An unfinished shim is a draft, not a failure. Tolerates a partial or missing source. */
export function readiness(src: Partial<ShimSourceInput> | null | undefined): Readiness;
