// Types for tree.mjs — a shim whose answers are grouped: a nearest-neighbour root picks the group,
// a small linear head per group picks the answer. The compiler fits it; the runtime decides with it.

import type { Vector, PackedHead, PackedTree } from './format.mjs';
import type { Head } from './math.mjs';
import type { Reference } from './familiarity.mjs';

/** A fitted (or unpacked) tree. */
export interface Tree {
  /** Number of answers. */
  K: number;
  /** Answer indices per group. */
  groups: number[][];
  /** Group index per answer. */
  groupOf: number[];
  /** The shim's examples relabelled by group: what the kNN root votes over. */
  rootRef: Reference;
  /** One linear head per group, over that group's answers in `groups` order. `null` where there is nothing to separate. */
  leaves: (Head | null)[];
}

/** A tree's prediction: `p(answer) = p(group) · p(answer | group)`, over every group. */
export interface TreePrediction {
  /** Index of the most probable answer. */
  index: number;
  /** Joint probability of that answer, 0–1. */
  confidence: number;
  /** Joint probability per answer index. */
  probs: number[];
  /** The group the winning answer belongs to. */
  group: number;
  /** The root's probability for that group, 0–1. */
  groupConfidence: number;
  /** The group the root liked best — not necessarily `group`. */
  rootTop: number;
}

/** Group counts worth trying for `K` answers: around √K and K/6, each clamped to 3 … K/2. Empty when `K < 8`. */
export function groupCandidates(K: number): number[];

/**
 * Learn a grouping from labelled vectors: answers a nearest-neighbour shim confuses with each other
 * go together (average-linkage clustering, groups capped at twice the mean size). With fewer than 4
 * examples on some answer it groups by mean example instead. Returns sorted answer indices per group;
 * may return more than `G` groups when the size cap blocks a merge.
 */
export function learnGroups(X: readonly Vector[], y: readonly number[], K: number, G: number, seed?: number): number[][];

/** Fit the root's reference and a class-balanced linear head per group. */
export function fitTree(
  X: readonly Vector[], y: readonly number[], K: number, dim: number, groups: number[][],
  opts?: { weights?: ArrayLike<number> | null },
): Tree;

/** Decide with a tree. A group with no leaf spreads its probability evenly over its answers. */
export function predictTree(tree: Tree, vec: Vector): TreePrediction;

/** What ships: the grouping and each leaf's head, packed with the `packHead` you pass (normally format.mjs's). */
export function packTree<P = PackedHead>(tree: Pick<Tree, 'groups' | 'leaves'>, packHead: (head: Head) => P): {
  groups: number[][];
  leaves: (P | null)[];
};

/**
 * Rebuild a tree from a weights file. The root reads the shim's own reference set, relabelled by
 * group, so `reference.labels` must be present.
 *
 * @param K number of answers
 */
export function unpackTree(
  packed: PackedTree,
  reference: Reference & { labels: number[] },
  K: number,
  unpackHead: (h: PackedHead) => Head,
): Tree;
