// A fake encoder the tests can reason about.
//
// The real encoder is 32MB and its geometry is whatever it is. Tests need the opposite: vectors whose
// relationships are known by construction, so that an assertion like "the exact match must win" is a
// statement about the code and not about bge-small. Every text is a small spec:
//
//   "billing#3"        the centre named `billing`, plus noise drawn from seed 3
//   "billing~0.2#3"    the same with noise scale 0.2 (default 0.35); "~0" is the centre itself
//   "billing+refund#1" halfway between two centres — a genuinely ambiguous input
//   anything else      a stranger: a direction of its own, derived from the text
//
// One thing is copied from the real space on purpose: every vector shares a common component, so
// unrelated text sits at cosine ~0.5 rather than ~0. Sentence embeddings are like that (strangers at
// 0.5-0.6), and the kNN bug fixed on 2026-09-19 — an exact match outvoted by two strangers — cannot
// be reproduced in a space where strangers score zero.
import { EMBED_DIM, DIM } from '../src/format.mjs';

function rng(seed) { let s = seed >>> 0 || 1; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
const hash = str => { let h = 2166136261; for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; };
function gaussianUnit(seed) {
  const r = rng(seed), v = new Float64Array(EMBED_DIM); let sq = 0;
  for (let d = 0; d < EMBED_DIM; d++) { const u = Math.max(r(), 1e-12), w = r(); v[d] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w); sq += v[d] * v[d]; }
  const n = Math.sqrt(sq); for (let d = 0; d < EMBED_DIM; d++) v[d] /= n; return v;
}
const COMMON = gaussianUnit(hash('the-common-component'));
export const centre = name => gaussianUnit(hash('centre:' + name));

/** text -> Float32Array(DIM). Surface features are left at zero: these tests are about geometry. */
export function place(text) {
  // A sentence split out of a longer message keeps its full stop; "billing#3." is still billing#3.
  const m = /^([a-z0-9_+-]+?)(?:~([0-9.]+))?#(\d+)$/i.exec(String(text).trim().replace(/[.!?]+$/, ''));
  const v = new Float64Array(EMBED_DIM);
  if (m) {
    const names = m[1].split('+'), scale = m[2] === undefined ? 0.35 : +m[2], noise = gaussianUnit(hash(m[0]) ^ 0x9e3779b9);
    for (const n of names) { const c = centre(n); for (let d = 0; d < EMBED_DIM; d++) v[d] += c[d] / names.length; }
    for (let d = 0; d < EMBED_DIM; d++) v[d] += scale * noise[d];
  } else { const c = gaussianUnit(hash('stranger:' + text)); for (let d = 0; d < EMBED_DIM; d++) v[d] = c[d]; }
  let sq = 0; for (let d = 0; d < EMBED_DIM; d++) { v[d] += COMMON[d]; sq += v[d] * v[d]; }
  const out = new Float32Array(DIM), n = Math.sqrt(sq); for (let d = 0; d < EMBED_DIM; d++) out[d] = v[d] / n; return out;
}
/** Drop-in for the SDK's embed(): array in, array out; string in, vector out. */
export const embed = async texts => Array.isArray(texts) ? texts.map(place) : place(texts);

/** `n` examples of each label, as shim source rows. `from` offsets the seeds, so train and test never share a row. */
export const examples = (labels, n, { from = 0, scale } = {}) =>
  labels.flatMap(l => Array.from({ length: n }, (_, i) => ({ text: `${l}${scale === undefined ? '' : '~' + scale}#${from + i}`, label: l })));
export const cosine = (a, b) => { let d = 0, na = 0, nb = 0; for (let i = 0; i < EMBED_DIM; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return d / Math.sqrt(na * nb); };
