// "Have I seen anything like this before?"
//
// A linear head always answers. Given input from a region its examples never covered
// it extrapolates the boundary into empty space and returns something arbitrary —
// which is how "what the fuck are you doing" came back `warm` at 0.41 confidence.
// Confidence alone cannot catch this: a point far from everything still lands on one
// side of a plane, sometimes emphatically.
//
// Two summaries were tried and discarded. A single centroid per answer compresses the
// range to nothing (the median example sits only 0.39 from its own class mean, so
// unfamiliar input scores HIGHER than real examples). Farthest-point representatives
// are worse still — they select outliers, so gibberish scores above real text.
//
// What works is the direct question: how close is this to the nearest actual example,
// scored against how close the examples sit to each other. Costs 384 bytes per
// example — 56KB for 150, against a 22MB encoder.
import { EMBED_DIM, DIM } from './format.mjs';

/** Over `width` dimensions: EMBED_DIM for "is this familiar", DIM for classification. */
export function cosine(a, b, width = EMBED_DIM) {
  let dot = 0, na = 0, nb = 0;
  for (let d = 0; d < width; d++) { dot += a[d] * b[d]; na += a[d] * a[d]; nb += b[d] * b[d]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

const QUANTILES = 21;   // 0, 0.05, … 1.00

/**
 * int8 + one scale per vector, FULL width. Familiarity reads the embedding half;
 * the kNN head reads all of it, surface features included. Same bytes either way,
 * so a nearest-neighbour head costs nothing beyond what familiarity already ships.
 */
export function packReference(vectors, labels = null) {
  const n = vectors.length;
  const q = new Int8Array(n * DIM);
  const scales = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let mx = 0;
    for (let d = 0; d < DIM; d++) mx = Math.max(mx, Math.abs(vectors[i][d]));
    scales[i] = mx / 127 || 1e-6;
    for (let d = 0; d < DIM; d++) q[i * DIM + d] = Math.round(vectors[i][d] / scales[i]);
  }
  // How close the examples sit to each other, leave-one-out. This is the yardstick.
  const nn = vectors.map((v, i) => {
    let best = -2;
    for (let j = 0; j < n; j++) if (j !== i) best = Math.max(best, cosine(v, vectors[j]));
    return best;
  }).sort((a, b) => a - b);
  const spread = Array.from({ length: QUANTILES }, (_, k) =>
    +nn[Math.min(n - 1, Math.floor((k / (QUANTILES - 1)) * n))].toFixed(4));
  return { q, scales, spread, count: n, labels };
}

export function unpackReference(ref) {
  if (!ref?.q) return null;
  const toBytes = s => (typeof Buffer !== 'undefined'
    ? (b => new Uint8Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)))(Buffer.from(s, 'base64'))
    : Uint8Array.from(atob(s), c => c.charCodeAt(0)));
  const qb = toBytes(ref.q), sb = toBytes(ref.scales);
  return {
    q: new Int8Array(qb.buffer, qb.byteOffset, qb.byteLength),
    scales: new Float32Array(sb.buffer, sb.byteOffset, sb.byteLength / 4),
    spread: ref.spread, count: ref.count, labels: ref.labels ?? null,
  };
}

/**
 * Where this input sits in the training set's own neighbourhood distribution.
 * 0.5 means as typical as the median example; below ~0.2 means the shim is
 * answering about something it has no examples near.
 */
export function familiarityScore(vec, ref) {
  if (!ref?.count) return null;
  let best = -2;
  for (let i = 0; i < ref.count; i++) {
    const s = ref.scales[i];
    let dot = 0, nb = 0, na = 0;
    for (let d = 0; d < EMBED_DIM; d++) {          // semantic half only
      const w = ref.q[i * DIM + d] * s;
      dot += vec[d] * w; na += vec[d] * vec[d]; nb += w * w;
    }
    const c = dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
    if (c > best) best = c;
  }
  const sp = ref.spread, last = sp.length - 1;
  if (best <= sp[0]) return 0;
  if (best >= sp[last]) return 1;
  for (let k = 1; k <= last; k++) {
    if (best <= sp[k]) {
      const t = (best - sp[k - 1]) / ((sp[k] - sp[k - 1]) || 1);
      return +(((k - 1 + t) / last)).toFixed(3);
    }
  }
  return 1;
}

/**
 * Nearest-neighbour head. A linear head gives each answer ONE direction, so its
 * decision region is convex — it cannot hold an answer whose examples sit in several
 * unrelated places ("red" = blood, traffic lights, fruit) without also swallowing
 * whatever lies between them. kNN never needs a single region, so multi-modal answers
 * cost it nothing. Measured on exactly that case: kNN 77.8%, linear 72.2%, single
 * prototype 64.8%.
 */
export const KNN_SHARPNESS = 0.1;
export function knnPredict(vec, ref, K, k = 5) {
  if (!ref?.count || !ref.labels) return null;
  const kk = Math.max(1, Math.min(k, ref.count));
  const best = [];
  for (let i = 0; i < ref.count; i++) {
    const s = ref.scales[i];
    let dot = 0, na = 0, nb = 0;
    for (let d = 0; d < DIM; d++) {                // full width, surface features included
      const w = ref.q[i * DIM + d] * s;
      dot += vec[d] * w; na += vec[d] * vec[d]; nb += w * w;
    }
    const c = dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
    if (best.length < kk) { best.push({ c, y: ref.labels[i] }); best.sort((a, b) => a.c - b.c); }
    else if (c > best[0].c) { best[0] = { c, y: ref.labels[i] }; best.sort((a, b) => a.c - b.c); }
  }
  // Distance-weighted vote, softened into something usable as a confidence.
  //
  // The weight was raw cosine until 2026-09-19, which in this space is nearly a flat vote: unrelated
  // text already sits at 0.5-0.6, so an EXACT match (1.00) was outvoted by two strangers (0.61 +
  // 0.57). That is how "anorak" — present, verbatim, in the training rows as outerwear — came back
  // as accessories. exp((c-1)/0.1) gives an exact match 1, a near-paraphrase at 0.9 about a third,
  // and a stranger at 0.6 about a fiftieth. Of the sharpenings tried it is the conservative one: no
  // shim it was tried on cross-validates worse with it (mean 66.7 -> 68.3), benchmarks at 8 per answer
  // gain 1-2 points, and harder settings won more on some shims and lost on others.
  const votes = new Array(K).fill(0);
  let total = 0;
  for (const b of best) { const w = Math.exp((b.c - 1) / KNN_SHARPNESS); votes[b.y] += w; total += w; }
  if (!total) return { index: 0, probs: new Array(K).fill(1 / K), confidence: 1 / K };
  const probs = votes.map(v => v / total);
  let index = 0;
  for (let i = 1; i < K; i++) if (probs[i] > probs[index]) index = i;
  return { index, probs, confidence: probs[index] };
}
