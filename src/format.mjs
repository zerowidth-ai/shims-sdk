// The on-disk format for a compiled shim.
//
// A weights file has two halves with different audiences. `report` is plain JSON a
// person reads in a pull request — accuracy, per-answer recall, whether the task
// even looks learnable. `heads` is an opaque base64 blob nobody reads, the way
// nobody reads a lockfile hash. Both are committed; only the first one is diffed.

import { SURFACE_DIM } from './surface.mjs';

// Swapped from all-MiniLM-L6-v2 after comparing encoders: same 384 dims (a drop-in —
// head and reference sizes unchanged), +10.5MB, and zero-shot on a hand-written set of
// contract clauses went 75.0% -> 96.7%. Mean pooling, as measured; BGE's own recipe is
// CLS pooling, which was not tested. Changing this invalidates every weights file.
export const ENCODER_ID = 'Xenova/bge-small-en-v1.5';
export const EMBED_DIM = 384;

/**
 * Vectors from different encoders cannot be mixed, and two encoders of the same width
 * would mix silently — the width check cannot catch it. So the cache is per encoder.
 *
 * And per way of embedding: a quantised encoder's vector for a text shifts with its batch-mates,
 * so vectors embedded in batches cannot be mixed with vectors embedded alone. `.single` names the
 * regime — one text per forward pass. Change how texts are embedded and this name changes with
 * it, so a cache from before is ignored and rebuilt rather than mixed in.
 */
export const vectorCacheFile = (id = ENCODER_ID) => `vectors.${id.replace(/[^a-z0-9]+/gi, '_')}.single.json`;
/** What a head actually sees: the embedding plus the form features it cannot encode. */
export const DIM = EMBED_DIM + SURFACE_DIM;
export const FORMAT_VERSION = 2;

const toB64 = f32 => {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
};

const fromB64 = str => {
  if (typeof Buffer !== 'undefined') {
    const b = Buffer.from(str, 'base64');
    // Buffer pools small allocations, so byteOffset is not always zero.
    return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
  }
  const bin = atob(str), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new Float32Array(u.buffer);
};

export const packHead = head => ({ K: head.K, W: toB64(head.W), b: toB64(head.b) });
export const unpackHead = (h, dim = DIM) =>
  ({ K: h.K, dim, W: fromB64(h.W), b: fromB64(h.b) });

/** Bytes a compiled shim's linear head costs at runtime. */
export const headBytes = (K, dim = DIM) => K * (dim + 1) * 4;

/** Minimal validation with messages aimed at whoever is editing the JSON by hand. */
export function validateSource(src) {
  const errs = [];
  if (!src?.name) errs.push('missing "name"');
  const type = src?.type ?? 'classify';
  if (!['classify', 'tags', 'schema'].includes(type)) errs.push(`unknown type "${type}"`);

  if (type === 'schema') {
    if (!src.fields || !Object.keys(src.fields).length) errs.push('schema needs at least one field');
    for (const [k, f] of Object.entries(src.fields ?? {})) {
      if (!Array.isArray(f.labels) || f.labels.length < 2) errs.push(`field "${k}" needs 2+ labels`);
    }
    for (const [i, ex] of (src.examples ?? []).entries()) {
      if (typeof ex.text !== 'string') errs.push(`example ${i}: missing "text"`);
      if (!ex.values || typeof ex.values !== 'object') errs.push(`example ${i}: missing "values"`);
    }
  } else {
    if (!Array.isArray(src.labels) || src.labels.length < 2) errs.push('needs 2+ "labels"');
    for (const [i, ex] of (src.examples ?? []).entries()) {
      if (typeof ex.text !== 'string') errs.push(`example ${i}: missing "text"`);
      const got = type === 'tags' ? ex.labels : ex.label;
      if (got === undefined) errs.push(`example ${i}: missing "${type === 'tags' ? 'labels' : 'label'}"`);
      for (const l of (type === 'tags' ? ex.labels ?? [] : [ex.label]))
        if (l !== undefined && !src.labels.includes(l)) errs.push(`example ${i}: "${l}" is not in labels`);
    }
    // Prototypes are keyed by answer. A key that is not an answer used to compile anyway: its rows got
    // the index -1, which crashed cross-validation with a TypeError when it was lucky and quietly
    // trained on a nonexistent class when it was not.
    if (src.prototypes !== undefined && src.prototypes !== null) {
      if (typeof src.prototypes !== 'object' || Array.isArray(src.prototypes)) errs.push('"prototypes" must be an object of answer → [texts]');
      else for (const [l, list] of Object.entries(src.prototypes)) {
        if (!src.labels?.includes(l)) errs.push(`prototypes: "${l}" is not in labels`);
        if (!Array.isArray(list) || list.some(t => typeof t !== 'string')) errs.push(`prototypes: "${l}" must be a list of texts`);
      }
    }
    // "tree": "auto" (default) | "off" | { "group name": ["answer", …], … }
    if (src.tree !== undefined && !['auto', 'off'].includes(src.tree)) {
      if (type !== 'classify') errs.push('"tree" applies to classify shims only');
      else if (typeof src.tree !== 'object' || Array.isArray(src.tree)) errs.push('"tree" must be "auto", "off", or an object of groups');
      else {
        const placed = Object.values(src.tree).flat();
        const missing = src.labels.filter(l => !placed.includes(l));
        const unknown = placed.filter(l => !src.labels.includes(l));
        if (unknown.length) errs.push(`"tree" names answers not in labels: ${unknown.join(', ')}`);
        if (missing.length) errs.push(`"tree" leaves answers out of every group: ${missing.join(', ')}`);
        if (new Set(placed).size !== placed.length) errs.push('"tree" places an answer in more than one group');
      }
    }
  }
  return errs;
}

export const MIN_PER_ANSWER = 3;

/**
 * Generated prototypes are worth roughly this many real examples per answer. Measured
 * on a 16-answer shim against an independent holdout: with no real examples prototypes
 * alone score 45%, and they keep beating real data until about 8 per answer, at which
 * point real examples overtake them (47.5% vs 42.5%). So they carry a shim early and
 * fade out rather than being switched off.
 */
export const PROTOTYPE_PARITY = 8;

/**
 * Prototypes never decay all the way out.
 *
 * Measured at ~8 real examples per answer, where real data should in theory have taken
 * over: on a shim whose examples were written by hand (so roughly representative) a
 * floor of 0 is fine — 47.5% vs 42.5% for full weight. On a shim whose examples came
 * from gap-filling — the hard tail by construction — a floor of 0 costs FIFTEEN points
 * (46.3% vs 61.3%). Real examples cover where a shim was already struggling; the
 * prototypes are the only thing still describing the easy middle.
 *
 * Any floor between 0.15 and 0.5 performs the same on both within the noise of an
 * 80-item holdout. 0.25 is the middle of that band, not a tuned optimum.
 */
export const PROTOTYPE_FLOOR = 0.25;

export const prototypeWeight = realPerAnswer =>
  Math.max(PROTOTYPE_FLOOR, 1 - realPerAnswer / PROTOTYPE_PARITY);

/**
 * Can this compile yet? A brand new shim is not broken, it is unfinished — the
 * studio needs to tell those apart so an empty shim reads as "keep going" rather
 * than as a failure.
 */
export function readiness(src) {
  const labels = src?.labels ?? [];
  // Prototypes are enough to compile. A shim with described answers is not a draft —
  // it is a working shim with no hand-written examples yet.
  const protos = src?.prototypes ?? {};
  const described = labels.length >= 2 && labels.every(l => (protos[l]?.length ?? 0) >= 3);
  const counts = Object.fromEntries(labels.map(l => [l, 0]));
  for (const ex of src?.examples ?? []) {
    const ls = src.type === 'tags' ? (ex.labels ?? []) : [ex.label];
    for (const l of ls) if (l in counts) counts[l]++;
  }
  const short = labels.filter(l => counts[l] < MIN_PER_ANSWER);
  const needed = short.reduce((n, l) => n + (MIN_PER_ANSWER - counts[l]), 0);
  return { counts, short, needed, described,
           ready: labels.length >= 2 && (described || short.length === 0) };
}
