// A shim whose answers are grouped: a root picks the group, a small head per group picks the
// answer. Isomorphic — the compiler fits and cross-validates it, the runtime decides with it.
//
// Measured before it was built, on public benchmarks:
//   - kNN ROOT + linear LEAVES. A group spans several unrelated answers, which a single linear
//     direction cannot cover (CLINC150 domains: linear root 86.9%, kNN root 96.8%); a leaf
//     separates a few close answers, where linear wins. Linear-only trees lost to flat.
//   - Groups learned from the shim's own confusions matched hand-made hierarchies (±0.6).
//   - Trees gained +2.9 to +4.3 at 8 examples per answer, up to +3.0 at 32 — and nothing on a
//     narrow single-topic set with plenty of data. So the compiler decides, per shim.
//   - Committing to the root's best group, taking the top two, or weighing all of them landed
//     within half a point; this uses the full joint because every leaf is tiny.
//
// A leaf label implies its group, so anything that teaches an answer teaches the root too.
import { fitHead, softmax, balanceWeights, shuffled } from './math.mjs';
import { packReference, knnPredict } from './familiarity.mjs';
import { EMBED_DIM } from './format.mjs';

const K_NEIGHBOURS = 5;
const cosine = (a, b) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < EMBED_DIM; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / (Math.sqrt(na * nb) || 1);
};

/** Group counts worth trying for K answers. Empty when K is too small for a tree to matter. */
export function groupCandidates(K) {
  if (K < 8) return [];
  const clamp = g => Math.max(3, Math.min(Math.floor(K / 2), g));
  return [...new Set([clamp(Math.round(Math.sqrt(K))), clamp(Math.round(K / 6))])].sort((a, b) => a - b);
}

/** Average-linkage agglomerative clustering on a K×K similarity, into G groups, size-capped at 2× mean. */
function cluster(S, K, G) {
  const cap = Math.ceil((2 * K) / G);
  const groups = Array.from({ length: K }, (_, i) => [i]);
  const link = (a, b) => { let s = 0; for (const i of a) for (const j of b) s += S[i][j]; return s / (a.length * b.length); };
  while (groups.length > G) {
    let best = -Infinity, bi = -1, bj = -1;
    for (let i = 0; i < groups.length; i++) for (let j = i + 1; j < groups.length; j++) {
      if (groups[i].length + groups[j].length > cap) continue;
      const l = link(groups[i], groups[j]);
      if (l > best) { best = l; bi = i; bj = j; }
    }
    if (bi < 0) break;
    groups[bi] = [...groups[bi], ...groups[bj]];
    groups.splice(bj, 1);
  }
  return groups.map(g => g.sort((a, b) => a - b)).sort((a, b) => a[0] - b[0]);
}

/**
 * Learn a grouping from labelled vectors. Answers a nearest-neighbour shim confuses with each
 * other go together, so the root's decision is between things that are actually different.
 * With too few examples to cross-validate, falls back to grouping by mean example (measured
 * within a point of confusion-based).
 */
export function learnGroups(X, y, K, G, seed = 1) {
  const perLabel = new Array(K).fill(0); y.forEach(l => perLabel[l]++);
  // How alike two answers' examples are, as centroids. The whole signal when there are too few rows
  // to measure confusion — and the tie-break when there are enough.
  const dim = X[0].length, cent = Array.from({ length: K }, () => new Float32Array(dim));
  X.forEach((v, i) => { for (let d = 0; d < dim; d++) cent[y[i]][d] += v[d]; });
  const alike = Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, j) => i === j ? 0 : cosine(cent[i], cent[j])));
  if (Math.min(...perLabel) < 4) return cluster(alike, K, G);

  const C = Array.from({ length: K }, () => new Float64Array(K));
  const folds = 4, order = shuffled(X.map((_, i) => i), seed * 17);
  for (let f = 0; f < folds; f++) {
    const held = new Set(order.filter((_, i) => i % folds === f));
    const tr = X.map((_, i) => i).filter(i => !held.has(i));
    const ref = packReference(tr.map(i => X[i]), tr.map(i => y[i]));
    for (const i of held) knnPredict(X[i], ref, K, K_NEIGHBOURS).probs.forEach((p, j) => { C[y[i]][j] += p; });
  }
  // Confusion decides; geometry only breaks ties. A shim that confuses siblings a little and
  // strangers never leaves every cross-family link at exactly zero, and the clustering used to take
  // the first zero it met — merging two whole families into one group and stranding a singleton
  // (found by the test suite, 2026-09-19). 1e-3 is far below any measured confusion and far above
  // floating-point noise.
  const S = Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, j) => i === j ? 0 : C[i][j] + C[j][i] + 1e-3 * alike[i][j]));
  return cluster(S, K, G);
}

export function fitTree(X, y, K, dim, groups, { weights = null } = {}) {
  const groupOf = new Array(K);
  groups.forEach((g, gi) => g.forEach(l => { groupOf[l] = gi; }));
  const rootRef = packReference(X, y.map(l => groupOf[l]));
  const leaves = groups.map(members => {
    if (members.length < 2) return null;
    const idx = y.map((l, i) => members.includes(l) ? i : -1).filter(i => i >= 0);
    const ly = idx.map(i => members.indexOf(y[i]));
    const present = new Set(ly);
    if (present.size < 2) return null;
    const bw = balanceWeights(ly, members.length);
    const w = weights ? idx.map((i, k) => bw[k] * weights[i]) : bw;
    return fitHead(idx.map(i => X[i]), ly, members.length, dim, { weights: w });
  });
  return { K, groups, groupOf, rootRef, leaves };
}

/** Decide: p(answer) = p(group) · p(answer | group), over every group. */
export function predictTree(tree, vec) {
  const G = tree.groups.length;
  const rp = knnPredict(vec, tree.rootRef, G, K_NEIGHBOURS)?.probs ?? new Array(G).fill(1 / G);
  const probs = new Array(tree.K).fill(0);
  tree.groups.forEach((members, gi) => {
    const leaf = tree.leaves[gi];
    const lp = leaf ? softmax(leaf, vec).probs : members.map(() => 1 / members.length);
    members.forEach((l, k) => { probs[l] = rp[gi] * lp[k]; });
  });
  let index = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[index]) index = i;
  let group = 0;
  for (let i = 1; i < G; i++) if (rp[i] > rp[group]) group = i;
  return { index, confidence: probs[index], probs, group: tree.groupOf[index], groupConfidence: rp[tree.groupOf[index]], rootTop: group };
}

/** What ships: the grouping and each leaf's head. The root reads the shim's existing reference set. */
export function packTree(tree, packHead) {
  return { groups: tree.groups, leaves: tree.leaves.map(h => h ? packHead(h) : null) };
}

export function unpackTree(packed, reference, K, unpackHead) {
  const groupOf = new Array(K);
  packed.groups.forEach((g, gi) => g.forEach(l => { groupOf[l] = gi; }));
  const rootRef = { ...reference, labels: reference.labels.map(l => groupOf[l]) };
  return { K, groups: packed.groups, groupOf, rootRef, leaves: packed.leaves.map(h => h ? unpackHead(h) : null) };
}
