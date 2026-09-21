// Types for compile.mjs — compile a *.shim.json (task + examples) into committed weights plus a
// report. The report is the part a person reads in a pull request: how good is this shim, which
// answers does it get wrong, and is the task even the right shape for a shim.

import type {
  ClassifySource, ClassifyWeights, DraftWeights, Embedder, ShimSourceInput, ShimWeights,
  TagsSource, TagsWeights,
} from '../format.mjs';

export type {
  ClassifySource, ClassifyWeights, DraftWeights, ShimReport, ShimSource, ShimSourceInput,
  ShimWeights, TagsSource, TagsWeights,
} from '../format.mjs';

/** Options for `compileShim`. */
export interface CompileOptions {
  /**
   * Directory for the on-disk vector cache, created if missing, so a recompile only embeds texts
   * it has not seen. Omit for no cache at all: nothing is read from or written to disk.
   */
  cacheDir?: string;
  /**
   * Replaces the encoder. It exists for tests that need vectors they can reason about; a real build
   * never passes it. Always called with an array of texts and must return one `DIM`-wide vector per text.
   */
  embed?: Embedder;
  /**
   * `false` turns off the coverage tiebreak, where a flat head within one standard error of the
   * most accurate one takes the pick if it can act on 10+ points more input. On by default.
   */
  coverageTiebreak?: boolean;
}

/** What `compileShim` resolves to: finished weights, or a draft stub when there are too few examples to compile. */
export type CompiledShim = ShimWeights | DraftWeights;

/**
 * Compile a shim source into weights. Embeds every example and prototype, cross-validates the
 * linear (over a grid of L2 penalties), nearest-neighbour and class-mean heads — and a tree when
 * there are 8+ answers or the source pins one — ships whichever measured best, and fits the
 * calibration temperature, the confidence gate and the familiarity floor on held-out rows.
 *
 * Rejects when `validateSource` finds a problem, with every message joined. A source that is valid
 * but not ready (an answer with under 3 examples and no prototypes) resolves to a draft — check
 * `draft` before reading the report's scores. The heads are deterministic for the same source;
 * `compiledAt` is the time of the build.
 */
export function compileShim(src: ClassifySource, opts?: CompileOptions): Promise<ClassifyWeights | DraftWeights>;
export function compileShim(src: TagsSource, opts?: CompileOptions): Promise<TagsWeights | DraftWeights>;
export function compileShim(src: ShimSourceInput, opts?: CompileOptions): Promise<CompiledShim>;
