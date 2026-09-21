// Types for the compiler's system.mjs — check the wiring. A system names shims and routes answers
// to them; this catches a renamed answer with no route, a shim that no longer exists, and a route
// to a shim that is still a draft, at build time rather than in front of a user.

import type { HeadKind, ShimNote } from '../format.mjs';
import type { SystemSpecInput } from '../system.mjs';

/** The part of a compiled shim (finished or draft) that `compileSystem` reads. Every `ShimWeights` and `DraftWeights` qualifies. */
export interface CompiledShimSummary {
  name: string;
  labels?: string[];
  draft?: boolean;
  head?: HeadKind | string;
  report?: {
    provisional?: boolean;
    suggestedThreshold?: number;
    familiarityFloor?: number;
    bytes?: number;
    referenceBytes?: number;
  };
}

/** What a checked system comes to. Systems compile to nothing — the apps read the spec; this is the summary and the verdict. */
export interface CompiledSystem {
  name: string;
  type: 'system';
  /** The spec's `input`, or `null`. */
  input: string | null;
  /** Every shim the system can reach that was among `compiled`. */
  shims: {
    name: string;
    /** Number of answers. */
    answers: number;
    head: string | undefined;
    /** The shim's confidence gate, 0–1. */
    threshold: number | undefined;
    /** The shim's familiarity floor, 0–1. */
    floor: number | undefined;
    /** Runtime bytes of its heads. */
    bytes: number;
  }[];
  steps: {
    id: string;
    kind: string;
    shim: string | null;
    routes: Record<string, string | null> | null;
    /** Names of the step's rules; empty for other kinds. */
    rules: string[];
  }[];
  /** The spec's `outcomes`, or `null`. */
  outcomes: Record<string, string> | null;
  /** Shims named across the steps: one per shim step and per route step, one per field of a bank step. An upper bound — a run evaluates each distinct shim once. */
  headsPerInput: number;
  /** Runtime bytes of every reachable shim's heads plus its shipped examples, excluding the shared encoder. */
  bytes: number;
  /** Problems that make the system wrong. Not thrown — check this. Empty means it passed. */
  errors: string[];
  /** Warnings: `partial-ballot`, `provisional-step`, `no-threshold`. */
  notes: ShimNote[];
}

/**
 * Check a system spec against the shims it names. On top of `validateSystem`: every answer a
 * routed-on step can give has a route (unless `onMissing: "continue"`), no route names an answer
 * that step never gives, no step uses a draft, and a `rescue` has at least two specialists to vote.
 *
 * @param compiled the compiled shims the spec may name, finished or draft
 */
export function compileSystem(spec: SystemSpecInput, compiled: readonly CompiledShimSummary[]): CompiledSystem;
