// Learning from the people using it.
//
// Every accepted suggestion, every "this one instead", every filter removed is a label on
// real input from the one person who knows. This keeps them and uses them immediately —
// but as a MEMORY, not a retrain. Measured on three shims, same corrections, untouched
// test searches:
//
//   live fine-tune of the head    category +6.3   fit -4.2   material -38.7
//   memory, cosine >= 0.82        no change anywhere; every corrected search fixed
//   added as examples + rebuilt   category +3.1   fit -6.3   material +1.8
//
// A head moved to fit a few dozen hard cases damages everything near them; a memory
// cannot touch what it is not near. What it reaches is rephrasings — "warm outfit for a
// wedding in winter" sits at 0.94 from "something warm for a winter wedding", while no
// unrelated test search came within 0.79 of any correction. Generalising beyond that is
// the compiler's job, with prototypes and a holdout, as a step someone chooses.
//
// A "no" is usable here where it is not in training: it says "not this answer, for this
// search and ones like it", which a memory can honour exactly without guessing what the
// right answer was.
import { Shim, currentEmbedder, splitForDecision, LONG_INPUT_CHARS, EMBED_DIM } from './index.mjs';
import { cosine } from './familiarity.mjs';
import { reduceChunks } from './reduce.mjs';
import { observing } from './observe.mjs';

/** How close a search must be to a remembered one to inherit its verdict. */
export const MEMORY_MATCH = 0.82;

export class AdaptiveShim {
  constructor(compiled, { match = MEMORY_MATCH } = {}) {
    this.base = compiled instanceof Shim ? compiled : new Shim(compiled);
    if (this.base.type === 'tags') throw new Error('AdaptiveShim supports classify shims only');
    this.match = match;
    this.name = this.base.name;
    this.labels = this.base.labels;
    this.report = this.base.report;
    this.feedback = [];      // { id, text, label, verdict: 'yes'|'no', vector, at, committed? }
  }

  nearest(vec, verdict, label = null) {
    let best = null, sim = -2;
    for (const f of this.feedback) {
      if (f.verdict !== verdict || (label && f.label !== label) || !f.vector) continue;
      const s = cosine(vec, f.vector, EMBED_DIM);
      if (s > sim) { sim = s; best = f; }
    }
    return best && sim >= this.match ? { entry: best, similarity: +sim.toFixed(3) } : null;
  }

  decideVector(vec) {
    const r = this.base.decideVector(vec);
    const yes = this.nearest(vec, 'yes');
    if (yes) {
      // The person chose this answer for input like this, so the action is theirs too: `act`. Without
      // this the result said e.g. "answer: dresses, action: refuse" — the remembered answer beside the
      // base shim's verdict on a DIFFERENT answer — and every caller had to know to ignore `action`.
      return { ...r, answer: yes.entry.label, action: 'act', shipped: { answer: r.answer, confidence: r.confidence, action: r.action },
               remembered: { text: yes.entry.text, similarity: yes.similarity, id: yes.entry.id } };
    }
    const no = this.nearest(vec, 'no', r.answer);
    // And a remembered "no" means say nothing: the one thing known is that this answer is wrong here.
    return { ...r, action: no ? 'refuse' : r.action, shipped: { answer: r.answer, confidence: r.confidence, action: r.action },
             rejected: no ? { text: no.entry.text, similarity: no.similarity, id: no.entry.id } : null };
  }

  /**
   * Long input, with memory: each chunk is decided by THIS shim (so a remembered correction still
   * applies to the chunk it was about), and the action follows the combined result the same way
   * Shim.decideChunks does — unless memory has already spoken for the chunk that was chosen.
   */
  decideChunks(vectors, chunks, opts = {}) {
    const r = reduceChunks(vectors.map(v => this.decideVector(v)), chunks, this.labels, opts);
    if (r.chunks > 1 && !r.remembered && !r.rejected) return { ...r, action: this.base.actionFor(r.familiarity, r.confidence) };
    return r;
  }

  isConfident(result, threshold) { return this.base.isConfident(result, threshold); }

  /** Record what was actually SAID — the remembered answer, not the base shim's — and whether memory spoke. */
  observed(r, text, vector, t0, field = null) {
    return this.base._observed(r, text, vector, t0, {}, { via: 'adaptive', field, memory: r.remembered ? 'yes' : r.rejected ? 'no' : null });
  }

  record({ text, vector, label, verdict }) {
    // One verdict per (text, label); and a new "yes" for a search replaces any other
    // "yes" for that same search — you picked a different answer, not a second one.
    this.feedback = this.feedback.filter(f => !(f.text === text &&
      (f.label === label || (verdict === 'yes' && f.verdict === 'yes'))));
    const entry = { id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
                    text, label, verdict, vector, at: new Date().toISOString() };
    this.feedback.push(entry);
    return entry;
  }

  get temperature() { return this.base.temperature; }

  /**
   * Refit the confidence scale on what people actually chose. The build calibrated against the
   * shim's own examples; this calibrates against the traffic it really meets.
   *
   * One caveat worth knowing: these rows are not a random sample. People correct what they were
   * shown, so accepted suggestions are over-represented among the decisions the shim was already
   * sure about. Treat a temperature refitted this way as better than the build's, not as truth.
   */
  recalibrate(opts) {
    const rows = this.feedback.filter(f => f.verdict === 'yes' && f.vector)
      .map(f => ({ vector: f.vector, label: f.label }));
    return this.base.recalibrate(rows, opts);
  }

  forget(id) { this.feedback = this.feedback.filter(f => f.id !== id); }
  reset() { this.feedback = []; }

  /** Serialisable (vectors dropped; they are recomputed from text on restore). */
  export() { return this.feedback.map(({ vector, ...f }) => f); }

  async restore(entries, embedder = currentEmbedder()) {
    const ok = (entries ?? []).filter(e => e?.text && e?.label && e?.verdict);
    if (!ok.length) return;
    const vecs = await embedder(ok.map(e => e.text));
    this.feedback = ok.map((e, i) => ({ ...e, vector: vecs[i] }));
  }
}

/** Several adaptive shims over one input, one encode. Results carry what they were decided on. */
export class AdaptiveBank {
  constructor(shims, opts) {
    this.shims = Object.fromEntries(Object.entries(shims)
      .map(([k, v]) => [k, v instanceof AdaptiveShim ? v : new AdaptiveShim(v, opts)]));
  }

  async decide(text) {
    const text_ = String(text ?? '');
    const embedder = currentEmbedder();
    const t0 = observing() ? performance.now() : 0;
    const out = {};
    if (text_.length > LONG_INPUT_CHARS) {
      const chunks = splitForDecision(text_);
      const vectors = await embedder(chunks);
      for (const [key, shim] of Object.entries(this.shims)) {
        const r = shim.decideChunks(vectors, chunks);
        // The vector is kept so a correction can be attached to the chunk it was about, which is
        // the most familiar one — the same chunk `from` names.
        const vector = vectors[chunks.indexOf(r.from)] ?? vectors[0];
        out[key] = { ...shim.observed(r, text_, vector, t0, key), vector };
      }
      return out;
    }
    const vec = await embedder(text_);
    for (const [key, shim] of Object.entries(this.shims)) out[key] = { ...shim.observed(shim.decideVector(vec), text_, vec, t0, key), vector: vec };
    return out;
  }
}
