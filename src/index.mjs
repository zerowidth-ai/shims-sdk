// shim — tiny task-specific decision models.
//
// One encoder is shared by every shim in an app; each shim is a few thousand
// numbers on top of it. Loading five shims costs one encoder plus ~30KB, and
// deciding five things about the same text costs ONE forward pass.

import { AutoTokenizer, AutoModel, env } from '@huggingface/transformers';
import { softmax, applyTemperature, fitTemperature, calibrationError, centroidPredict, estimatePrior, weightedGate } from './math.mjs';
import { unpackHead, DIM, EMBED_DIM, ENCODER_ID } from './format.mjs';
import { withSurface } from './surface.mjs';
import { familiarityScore, unpackReference, knnPredict } from './familiarity.mjs';
import { unpackTree, predictTree } from './tree.mjs';

export { DIM, EMBED_DIM, ENCODER_ID, headBytes } from './format.mjs';
export { surfaceFeatures, withSurface, SURFACE_DIM } from './surface.mjs';
export { reduceChunks } from './reduce.mjs';
export { observe, observing, outcome, ring } from './observe.mjs';
import { reduceChunks } from './reduce.mjs';
import { observing, record, nextId } from './observe.mjs';

/** `{ reduce: 'pooled', reducePower: 2 }` on any decide() call, in the shape reduce.mjs wants. */
const reduceOpts = (opts = {}) => ({ mode: opts.reduce, power: opts.reducePower });

/**
 * What the encoder will actually read in one pass. bge-small is a BERT with 512 positions, and we
 * ask for all of them. We asked for 128 for a long time, which was a silent data loss: a paragraph
 * with no full stops in it arrives as one piece, and two thirds of it was being dropped with
 * nothing in the vector to say so.
 */
export const ENCODER_MAX_TOKENS = 512;
/** Roughly where ENCODER_MAX_TOKENS lands in characters, at ~3.3 characters per wordpiece. */
export const ENCODER_MAX_CHARS = 1700;

/**
 * Where a decision starts splitting its input — deliberately far below what the encoder can read.
 *
 * Splitting is not about the ceiling. Measured on 60 documents per shape,
 * reading a long message in pieces beats reading it whole even when the whole thing fits: 35 right
 * against 23 on a message with one decisive line in it. Mean-pooling over a document's tokens
 * drowns the sentence that matters in the same way that mean-pooling over its chunks does.
 *
 * It is not free, though: on a mid-length message whose signal is spread about, one read wins
 * (53 against 45). The split threshold is the dial between those, and 420 is where it sits because
 * that mix came out ahead overall — 80 against 76 — not because the encoder cannot manage more.
 */
export const LONG_INPUT_CHARS = 420;

/**
 * Target size for one chunk. Deliberately well under LONG_INPUT_CHARS: a chunk as
 * large as the ceiling would just reproduce the problem, since the whole point is that
 * one sentence somewhere in the message carries the answer and averaging buries it.
 */
export const CHUNK_CHARS = 160;

/** Sentence-ish split, then regroup so no chunk is too small to carry meaning. */
export function splitForDecision(text, target = CHUNK_CHARS) {
  const parts = String(text).split(/(?<=[.!?])\s+|\n+/).map(s => s.trim()).filter(Boolean);
  const out = [];
  let buf = '';
  for (const p of parts) {
    if (buf && (buf.length + p.length + 1) > target) { out.push(buf); buf = p; }
    else buf = buf ? `${buf} ${p}` : p;
  }
  if (buf) out.push(buf);
  // Sentence splitting assumes sentences. People write paragraphs with no full stops in them, and
  // those arrive here as a single piece of unbounded length — which the encoder then truncates
  // without saying so. Anything still over the ceiling gets cut on whitespace.
  const safe = [];
  for (const c of out) {
    if (c.length <= ENCODER_MAX_CHARS) { safe.push(c); continue; }
    const words = c.split(/\s+/);
    let line = '';
    for (const w of words) {
      if (line && (line.length + w.length + 1) > target) { safe.push(line); line = w; }
      else line = line ? `${line} ${w}` : w;
    }
    if (line) safe.push(line);
  }
  return safe.length ? safe : [String(text)];
}

let encoderPromise = null;

/**
 * Load (once) the shared encoder. Every shim in the process reuses it.
 * `opts` is passed through to transformers.js — `{ device: 'webgpu' }`, a local
 * `modelPath`, etc.
 */
export function loadEncoder(opts = {}) {
  if (!encoderPromise) {
    const { modelId = ENCODER_ID, dtype = 'q8', remoteHost, remotePathTemplate, wasmPaths, ...rest } = opts;
    // Where the bytes come from. By default the model is fetched from the Hugging Face hub; the
    // ONNX runtime's .wasm ships with the app when the bundler follows its import.meta.url, and
    // comes from jsDelivr when it does not. An app that must not call a third party at load hosts
    // them itself and names them here. They are read once, by whichever call loads the
    // encoder first — so pass them to `preload()` at startup, which also reaches the SDK's worker.
    if (remoteHost) env.remoteHost = remoteHost;
    if (remotePathTemplate) env.remotePathTemplate = remotePathTemplate;
    if (wasmPaths && env.backends?.onnx?.wasm) env.backends.onnx.wasm.wasmPaths = wasmPaths;
    encoderPromise = (async () => {
      const [tokenizer, model] = await Promise.all([
        AutoTokenizer.from_pretrained(modelId),
        AutoModel.from_pretrained(modelId, { dtype, ...rest }),
      ]);
      return { tokenizer, model };
    })();
  }
  return encoderPromise;
}

/**
 * Text -> one vector per input. Mean-pool over real tokens, then L2 normalise.
 * The compiler imports this exact function, so build-time and runtime vectors
 * cannot drift apart.
 *
 * Sharing the function is not enough on its own. A quantised encoder does not return the same
 * vector for a text whatever shares its batch: cosine 0.9995 against the same text embedded alone,
 * where two identical calls agree to the bit. A compiler that embeds 32 at a time and a browser
 * that embeds one would never see the same vector, and a cached vector would depend on whatever
 * its text first arrived with. So every text gets a forward pass of its own, and a vector depends
 * on its text and nothing else.
 */
export async function embed(texts, opts = {}) {
  const list = Array.isArray(texts) ? texts : [texts];
  const { tokenizer, model } = await loadEncoder(opts);
  const out = [];
  const BATCH = 1;   // not a tuning knob — see above; a larger batch changes the vectors

  for (let i = 0; i < list.length; i += BATCH) {
    const chunk = list.slice(i, i + BATCH);
    const enc = await tokenizer(chunk, { padding: true, truncation: true, max_length: ENCODER_MAX_TOKENS });
    const h = (await model(enc)).last_hidden_state;   // [B, T, EMBED_DIM]
    const [B, T] = h.dims;
    const mask = enc.attention_mask.data;
    for (let b = 0; b < B; b++) {
      const v = new Float32Array(EMBED_DIM);
      let n = 0;
      for (let t = 0; t < T; t++) {
        if (!Number(mask[b * T + t])) continue;
        n++;
        const off = (b * T + t) * EMBED_DIM;   // model stride, NOT the head's DIM
        for (let d = 0; d < EMBED_DIM; d++) v[d] += h.data[off + d];
      }
      let sq = 0;
      for (let d = 0; d < EMBED_DIM; d++) { v[d] /= (n || 1); sq += v[d] * v[d]; }
      const norm = Math.sqrt(sq) || 1;
      for (let d = 0; d < EMBED_DIM; d++) v[d] /= norm;
      out.push(withSurface(v, list[i + b]));
    }
  }
  return Array.isArray(texts) ? out : out[0];
}

/**
 * Where the encoding happens.
 *
 * In a browser the encoder must not run on the main thread — one embed is ~20–30ms and a batch
 * is a visibly frozen page — so the SDK starts its own module worker the first time anything
 * needs a vector. No app has to write that worker, wire a message protocol, or call anything
 * before deciding. Node (the compiler, a server, a test) embeds in-process.
 *
 * `setEmbedder` remains for the cases that want control: an app that already runs a worker and
 * would rather share it, a server, a test that wants vectors without a model.
 */
let custom = null;
let auto = null;

function workerEmbedder() {
  const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
  const pending = new Map();
  let seq = 0;
  worker.onmessage = ({ data }) => {
    const p = pending.get(data.id);
    if (!p) return;
    pending.delete(data.id);
    data.ok ? p.resolve(data.vectors) : p.reject(new Error(data.error));
  };
  const call = (type, texts, opts) => new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, texts, opts });
  });
  const fn = async (texts, opts = {}) => {
    const list = Array.isArray(texts) ? texts : [texts];
    if (!list.length) return Array.isArray(texts) ? [] : null;
    const out = (await call('embed', list, opts)).map(v => v instanceof Float32Array ? v : new Float32Array(v));
    return Array.isArray(texts) ? out : out[0];
  };
  fn.warm = opts => call('warm', null, opts);
  return fn;
}

function resolveEmbedder() {
  if (custom) return custom;
  if (auto) return auto;
  // A worker needs a bundler that can resolve `new URL(..., import.meta.url)`; anywhere it
  // cannot (Node, a test runner, an unusual build) fall back to embedding in-process.
  if (typeof Worker !== 'undefined' && typeof document !== 'undefined') {
    try { auto = workerEmbedder(); } catch { auto = embed; }
  } else auto = embed;
  return auto;
}

export function setEmbedder(fn) { custom = fn ?? null; }
export const currentEmbedder = () => resolveEmbedder();

/**
 * Download and warm the encoder before the first decision, so the first one is not the slow one.
 * Optional: everything works without it, the first decide just waits for the model.
 */
/** Encode text with whatever embedder is in play — the worker in a browser, in-process in Node. */
export const encode = (texts, opts = {}) => resolveEmbedder()(texts, opts);

export async function preload(opts = {}) {
  const e = resolveEmbedder();
  if (e.warm) return e.warm(opts);
  await loadEncoder(opts);
}

/** One compiled shim: a task, its answers, and the head that decides between them. */
export class Shim {
  /** @param {object} compiled · @param {{temperature?: number}} [opts] an override, normally unused */
  constructor(compiled, opts = {}) {
    if (compiled.draft) throw new Error(
      `shim "${compiled.name}" is still a draft — it has no head yet`);
    if (compiled.format !== 2) throw new Error(
      `shim "${compiled.name}" is format ${compiled.format}; this runtime needs 2 — rebuild with shim-compile`);
    // Same-width encoders make a stale weights file dangerous rather than broken: it
    // loads, it answers, and every answer is noise. Refuse instead.
    if (compiled.encoder && compiled.encoder !== ENCODER_ID) throw new Error(
      `shim "${compiled.name}" was compiled for ${compiled.encoder}; this runtime uses ${ENCODER_ID} — rebuild with shim-compile`);
    this.name = compiled.name;
    this.type = compiled.type;
    // The calibration temperature: fitted at build, replaceable by recalibrate() once real
    // outcomes exist. It changes what the confidence MEANS, never which answer wins — so it is
    // not a risk dial. To act less often, raise the gate; that is a policy, and lives there.
    this._temperature = opts.temperature ?? compiled.report?.calibration?.temperature ?? 1;
    this.question = compiled.question;
    this.labels = compiled.labels;
    this.report = compiled.report;
    this.reference = unpackReference(compiled.reference);
    this.head = compiled.head ?? 'linear';
    // The penalty the build chose for the linear head, so a live refit uses the same one.
    this.l2 = compiled.l2 ?? null;
    this.heads = compiled.type === 'tags'
      ? compiled.heads.map(h => unpackHead(h))
      : unpackHead(compiled.heads[0]);
    // A tree needs its reference set: the root votes over the same shipped examples.
    this.tree = compiled.head === 'tree' && compiled.tree && this.reference?.labels
      ? unpackTree(compiled.tree, this.reference, this.labels.length, h => unpackHead(h))
      : null;
    if (compiled.head === 'tree' && !this.tree) this.head = 'linear';
    // The nearest-centroid head, when the build measured it as the best of the three. The linear
    // head still ships beside it, so a weights file with the centroid stripped out still answers.
    this.centroid = compiled.head === 'centroid' && compiled.centroid ? unpackHead(compiled.centroid) : null;
    if (compiled.head === 'centroid' && !this.centroid) this.head = 'linear';
  }

  get temperature() { return this._temperature; }

  /**
   * Refit the temperature on decisions this shim has actually made, once you know how they turned
   * out — the accepts and corrections a deployed shim collects are exactly what the compiler had.
   * Kept only if it improves calibration on those same rows, because fitting on a few dozen points
   * made three of our shims measurably worse.
   *
   * @param {{vector: Float32Array, label: string}[]} rows
   * @returns {{applied: boolean, temperature: number, before: number, after: number, samples: number}}
   */
  recalibrate(rows, { minSamples = 30 } = {}) {
    // A tags shim scores each tag on its own and applies no temperature, so there is nothing to refit.
    if (this.type === 'tags') return { applied: false, temperature: this._temperature, before: null, after: null, samples: 0, why: 'tags shims apply no temperature; not supported yet' };
    const usable =(rows ?? []).filter(r => r?.vector && this.labels.includes(r.label));
    const samples = usable.map(r => ({ probs: this.rawProbs(r.vector), y: this.labels.indexOf(r.label) }));
    const score = T => calibrationError(samples.map(s => {
      const p = applyTemperature(s.probs, T);
      let i = 0; for (let j = 1; j < p.length; j++) if (p[j] > p[i]) i = j;
      return { confidence: p[i], ok: i === s.y };
    }));
    if (samples.length < minSamples) return { applied: false, temperature: this._temperature, before: null, after: null, samples: samples.length, why: `needs ${minSamples} outcomes, has ${samples.length}` };
    const before = score(this._temperature), fitted = fitTemperature(samples), after = score(fitted);
    const applied = after < before;
    if (applied) this._temperature = fitted;
    return { applied, temperature: this._temperature, fitted, before, after, samples: samples.length };
  }

  /**
   * Refit the familiarity floor from real in-scope input.
   *
   * The floor the build ships is a percentile of the shim's OWN examples, so it is calibrated to
   * the distribution the shim was built from. Real input sits lower, and measurably so: against
   * real search queries, floors aiming to wrongly refuse one request in twenty refused between one
   * in three and five in six, while still correctly turning away 96-98% of genuinely irrelevant
   * text. The mechanism is sound; the threshold was fitted on the wrong population.
   *
   * What this needs is much weaker than labels — vectors of text the app believes is IN SCOPE, not
   * what any of it means. "These are the kinds of things people ask us" is available on day one of
   * any deployment, long before anyone has labelled anything.
   *
   * It MUST be in-scope. Handing it raw traffic that is half out-of-scope drags the floor toward
   * zero and disables refusal, which is the failure this gate exists to prevent.
   *
   * @param {Float32Array[]} vectors · in-scope input, unlabelled
   * @param {{target?: number, minSamples?: number, floor?: number}} [opts]
   */
  refitFloor(vectors, { target = 0.05, minSamples = 20, floor = 0.005 } = {}) {
    const fams = (vectors ?? []).map(v => familiarityScore(v, this.reference)).filter(v => v != null);
    const before = this.gates.familiarity;
    if (fams.length < minSamples) {
      return { applied: false, before, after: before, samples: fams.length,
        why: `needs ${minSamples} in-scope samples, has ${fams.length}` };
    }
    fams.sort((a, b) => a - b);
    const after = +Math.max(floor, fams[Math.floor(target * fams.length)]).toFixed(3);
    const refusedBefore = fams.filter(v => v < before).length / fams.length;
    this._floor = after;
    return { applied: true, before, after, samples: fams.length,
      refusedBefore: +refusedBefore.toFixed(3), refusedAfter: +target.toFixed(3) };
  }

  /**
   * Re-read the confidence gate for the traffic this shim actually gets.
   *
   * The build's promise — "above X this shim is 90% accurate" — is read off held-out examples in
   * which every answer is about equally common. Skew alone does not break it (measured: a random
   * Zipf stream leaves it at 91-93%). What breaks it is the HARD answers being the busy ones: then
   * the same gate delivers 80-84%. No labels are needed to repair that. The shim's own calibrated
   * probabilities on unlabelled input estimate how common each answer is; the build's held-out
   * rows are reweighted to that mix and the gate is read off again. Measured on exactly that worst
   * case, from 100 unlabelled inputs: CLINC150 79.5 → 91.7, Banking77 82.9 → 89.8, MASSIVE
   * 83.5 → 89.1. The price is honest: it acts on less (coverage 100 → 71%, 81 → 64%, 62 → 40%).
   *
   * What it cannot repair: an answer the shim gets CONFIDENTLY wrong. The mix is estimated from the
   * shim's own predictions, so a busy answer it mislabels as its neighbour is invisible to it — on a
   * stream built from a CLINC shim's own blind spots it estimated 4.6% for an answer that was 16.8%
   * of traffic, and the gate went 66% → 74%, not to 90. Only outcomes fix that (`recalibrate`, or
   * corrections folded into the next build).
   *
   * @param {Float32Array[]} vectors · input this shim has seen in use, unlabelled and in scope
   * @param {{target?: number, minSamples?: number}} [opts]
   */
  refitGate(vectors, { target = 0.9, minSamples = 100 } = {}) {
    const before = this.gates.confidence, rows = this.report?.gateRows;
    const no = why => ({ applied: false, before, after: before, samples: vectors?.length ?? 0, why });
    if (this.type === 'tags') return no('tags shims gate on the whole tag set; not supported yet');
    if (!rows?.length) return no('this build shipped no held-out rows — a provisional gate, or weights compiled before refitGate existed');
    if ((vectors?.length ?? 0) < minSamples) return no(`needs ${minSamples} unlabelled inputs, has ${vectors?.length ?? 0}`);

    const K = this.labels.length, T = this._temperature;
    // Heads are fitted with class-balanced weights, so the prior they were trained under is uniform.
    const trained = new Array(K).fill(1 / K);
    const probs = vectors.map(v => applyTemperature(this.rawProbs(v), T));
    const prior = estimatePrior(probs, trained);
    const share = new Array(K).fill(0);
    for (const [y] of rows) share[y] += 1 / rows.length;
    const gate = weightedGate(rows.map(([y, c, ok]) => ({ c, ok, w: share[y] ? prior[y] / share[y] : 0 })), target);

    const covered = g => probs.filter(p => Math.max(...p) >= g).length / probs.length;
    this._gate = gate.threshold;
    return { applied: true, before, after: gate.threshold, samples: vectors.length,
      actsOnBefore: +covered(before).toFixed(3), actsOnAfter: +covered(gate.threshold).toFixed(3),
      mix: Object.fromEntries(this.labels.map((l, k) => [l, +prior[k].toFixed(4)])) };
  }

  /** Whichever structure the build chose: tree, nearest neighbours, nearest centroid, or linear. */
  _predict(vec) {
    if (this.tree) return predictTree(this.tree, vec);
    if (this.head === 'knn' && this.reference?.labels) return knnPredict(vec, this.reference, this.labels.length, 5);
    if (this.centroid) return centroidPredict(this.centroid, vec);
    return softmax(this.heads, vec);
  }

  rawProbs(vec) {
    // For a tags shim this used to hand back [] without complaint: `_predict` treats the list of
    // per-tag heads as one head. There is no distribution to return; the per-tag scores are on decideVector().
    if (this.type === 'tags') throw new Error(`${this.name}: rawProbs is for classify shims — a tags shim scores each tag on its own; read "scores" from decideVector()`);
    return Array.from(this._predict(vec).probs);
  }

  /** The two numbers the build fitted: when to act, and when to say nothing. */
  get gates() {
    return { confidence: this._gate ?? this.report?.suggestedThreshold ?? 0.7,   // `_gate`: set by refitGate()
             // `_floor` is set by refitFloor() from real in-scope traffic and wins when present:
             // the shipped number was fitted on the shim's own examples and does not transfer.
             familiarity: this._floor ?? this.report?.familiarityFloor ?? 0.25 };
  }

  actionFor(familiarity, confidence) {
    const { confidence: gate, familiarity: floor } = this.gates;
    if (familiarity != null && familiarity < floor) return 'refuse';
    return confidence >= gate ? 'act' : 'suggest';
  }

  /** The groups, by answer name, when this shim is a tree. */
  get groups() { return this.tree ? this.tree.groups.map(g => g.map(i => this.labels[i])) : null; }

  /** Decide from an already-computed vector. Use when several shims share one encode. */
  decideVector(vec) {
    if (this.type === 'tags') {
      const tags = [];
      const scores = {};
      this.heads.forEach((h, k) => {
        const p = softmax(h, vec).probs[1];
        scores[this.labels[k]] = p;
        if (p >= 0.5) tags.push(this.labels[k]);
      });
      // Confidence means "how sure am I of the answer I gave". For a clause that
      // raised flags that is the weakest flag raised; for one that raised none it is
      // how far the strongest near-miss sits below the line.
      const familiarity = familiarityScore(vec, this.reference);
      const confidence = tags.length
        ? Math.min(...tags.map(t => scores[t]))
        : 1 - Math.max(...Object.values(scores));
      // This passed `tags.length ? 1 : 0` until 2026-09-18, so a tag raised at 0.51 was treated
      // exactly like one at 0.99 and `suggest` was unreachable: any tag firing meant act, none
      // firing meant suggest, and the confidence computed two lines above was handed to the caller
      // and never used. On a 12-tag task that confidence separates cleanly — 0.92 when the tag set
      // is right against 0.58 when it is wrong — so the signal was there and thrown away.
      return { answer: tags, scores, confidence, familiarity,
        action: this.actionFor(familiarity, confidence) };
    }
    const r = this._predict(vec);
    // One fitted number turns raw scores into a confidence that means what it says; without it a
    // 150-answer shim reports 0.08 while being right nine times in ten.
    const T = this._temperature;
    const scaled = T === 1 ? r.probs : applyTemperature(r.probs, T);
    const probs = {};
    this.labels.forEach((l, i) => { probs[l] = scaled[i]; });
    const familiarity = familiarityScore(vec, this.reference);
    const out = { answer: this.labels[r.index], confidence: scaled[r.index], probs, familiarity,
                  rawConfidence: r.probs[r.index],
                  // What the shim thinks you should do with this, from the gate and floor it earned
                  // at build: 'act' on it, 'suggest' it, or 'refuse' — say nothing at all.
                  action: this.actionFor(familiarity, scaled[r.index]) };
    // Which group the answer came through, and how sure the root was of that group.
    if (this.tree) out.path = { group: r.group, groupConfidence: r.groupConfidence, siblings: this.groups[r.group] };
    return out;
  }

  async decide(text, opts = {}) {
    const text_ = String(text ?? '');
    const t0 = observing() ? performance.now() : 0;
    // Past the encoder's ceiling, truncation silently drops the end of the input —
    // and in a support message the deciding sentence is usually at the end. Measured:
    // a query buried in a 188-word message scores 41.7% truncated and 100% when the
    // message is split and the most FAMILIAR chunk is the one decided on.
    if (opts.long !== false && text_.length > LONG_INPUT_CHARS) {
      const { result, vector } = await this._decideLong(text_, opts);
      return this._observed(result, text_, vector, t0, opts);
    }
    const vec = await currentEmbedder()(text_, opts);
    return this._observed(this.decideVector(vec), text_, vec, t0, opts);
  }

  /**
   * Hand the decision to whatever is watching, and give it an id so an outcome can be attached
   * later. Costs nothing when nobody is watching, which is the default.
   *
   * The two things in this SDK that need real traffic — refitFloor and recalibrate — have never
   * had any, and both are blocked on capture rather than on cleverness. Capture only happens if it
   * is the default; an integration that has to remember to log will not.
   */
  _observed(r, text, vec, t0, opts = {}, extra = null) {
    if (!observing()) return r;
    const id = nextId();
    record({ kind: 'decision', id, shim: this.name, at: Date.now(),
      ms: +(performance.now() - t0).toFixed(2),
      text: opts.keepText === false ? null : text,
      words: text.trim() ? text.trim().split(/\s+/).length : 0,
      chars: text.length,
      vector: vec,
      answer: r.answer, confidence: r.confidence, familiarity: r.familiarity,
      action: r.action, gates: this.gates, ...extra });
    return { ...r, id };
  }

  /**
   * Split, decide on every piece, and combine them. The default reads the piece this shim
   * recognises best; `{ reduce: 'pooled' }` weighs every piece instead, which is worth about ten
   * points on a document whose evidence is spread across sentences rather than hidden in one —
   * and costs about as much on a document where it IS hidden in one. reduce.mjs has the numbers.
   */
  async decideLong(text, opts = {}) { return (await this._decideLong(text, opts)).result; }

  /** The decision AND the vector it was made on — the chosen piece's — so a recorded long decision can drive a refit. */
  async _decideLong(text, opts = {}) {
    const chunks = splitForDecision(text);
    if (chunks.length < 2) { const vector = await currentEmbedder()(text, opts); return { result: this.decideVector(vector), vector }; }
    const vectors = await currentEmbedder()(chunks, opts);
    const result = this.decideChunks(vectors, chunks, reduceOpts(opts));
    return { result, vector: vectors[chunks.indexOf(result.from)] ?? null };
  }

  /**
   * One decision from a vector per chunk. The action is re-derived from the COMBINED confidence and
   * familiarity: `reduceChunks` starts from the best chunk's result, and under `pooled` it replaces
   * that chunk's confidence with the pooled one — but it knows nothing about gates, so the action it
   * carried over was the best chunk's. A pooled confidence of 0.64 was being reported as "act" against
   * a gate of 0.999 (found by the test suite, 2026-09-19).
   */
  decideChunks(vectors, chunks, opts = {}) {
    const r = reduceChunks(vectors.map(v => this.decideVector(v)), chunks, this.labels, opts);
    return r.chunks > 1 ? { ...r, action: this.actionFor(r.familiarity, r.confidence) } : r;
  }

  /**
   * Safe to act on without escalating: confident enough AND looking at something
   * the examples actually cover. Either test alone misses a failure mode — a head
   * can be very sure about input it has no business judging.
   */
  isConfident(result, threshold = this.gates.confidence) {
    // Through `gates`, not the report: until 2026-09-19 this read the build's numbers directly, so a
    // floor moved by refitFloor() changed `action` and was ignored here.
    const familiar = result.familiarity == null || result.familiarity > this.gates.familiarity;
    return result.confidence >= threshold && familiar;
  }

  static load(compiled) { return new Shim(compiled); }
}

/**
 * Several shims over the same input, sharing a single forward pass.
 * This is the "four or five running at once" case — the encode is the expensive
 * part, and it happens once no matter how many shims are in the bank.
 */
export class Bank {
  constructor(shims) {
    // Drafts are skipped rather than rejected, so a half-written shim in the folder
    // does not stop the finished ones from running.
    this.shims = Object.fromEntries(
      Object.entries(shims)
        .filter(([, v]) => v instanceof Shim || !v.draft)
        .map(([k, v]) => [k, v instanceof Shim ? v : new Shim(v)]));
  }

  async decide(text, opts = {}) {
    const text_ = String(text ?? '');
    const t0 = observing() ? performance.now() : 0;
    const out = {};
    if (opts.long !== false && text_.length > LONG_INPUT_CHARS) {
      // One split, one set of encodes, shared across every shim — but each shim weighs the
      // chunks for itself, because different shims recognise different sentences.
      const chunks = splitForDecision(text_);
      const vectors = await currentEmbedder()(chunks, opts);
      for (const [key, shim] of Object.entries(this.shims)) {
        const r = shim.decideChunks(vectors, chunks, reduceOpts(opts));
        out[key] = shim._observed(r, text_, vectors[chunks.indexOf(r.from)] ?? null, t0, opts, { via: 'bank', field: key });
      }
      return out;
    }
    const vec = await currentEmbedder()(text_, opts);
    // Recorded per shim, each with its own id: a bank is how an app runs several shims at once, so
    // a bank that recorded nothing would leave most real decisions unobserved.
    for (const [key, shim] of Object.entries(this.shims)) out[key] = shim._observed(shim.decideVector(vec), text_, vec, t0, opts, { via: 'bank', field: key });
    return out;
  }

  /** Total runtime cost of the heads, excluding the shared encoder. */
  get bytes() {
    return Object.values(this.shims).reduce((sum, s) => {
      const hs = Array.isArray(s.heads) ? s.heads : [s.heads];
      return sum + hs.reduce((a, h) => a + h.K * (h.dim + 1) * 4, 0);
    }, 0);
  }
}
