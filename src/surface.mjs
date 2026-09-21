// Form features the encoder cannot see.
//
// The encoder (bge-small-en-v1.5, like MiniLM before it) is UNCASED: "WTF" and "wtf" produce byte-identical token ids, so
// capitalisation is destroyed before the network runs. Mean-pooling then discards word
// order, and the contrastive training objective deliberately collapses punctuation,
// because "!" does not change what a sentence means.
//
// These eight numbers hand that back. Measured effect at SCALE below: is-question
// +30 points, tone +0.7, intensity -1.9 (about three rows — inside CV noise).
//
// Scaled to sit near the embedding's own per-dimension magnitude (RMS ~0.051). Larger
// values are effectively LESS regularised than the embedding dims, which flatters these
// features when they carry signal and punishes the model when they do not.

export const SURFACE_DIM = 8;
export const SCALE = 0.2;

export function surfaceFeatures(text) {
  const t = String(text ?? '');
  const letters = (t.match(/[A-Za-z]/g) || []).length || 1;
  const caps = (t.match(/[A-Z]/g) || []).length;
  const words = t.split(/\s+/).filter(Boolean);
  const shouty = words.filter(w => w.length > 1 && w === w.toUpperCase() && /[A-Z]/.test(w)).length;
  const clamp = v => Math.max(0, Math.min(1, v));

  return [
    t.includes('?') ? 1 : 0,                                    // asks something
    /\?\s*$/.test(t) ? 1 : 0,                                   // ends on a question
    clamp((t.match(/!/g) || []).length / 3),                    // emphasis
    clamp((caps / letters) * 2),                                // shouting, by letter
    clamp((shouty / Math.max(1, words.length)) * 3),            // shouting, by word
    clamp(Math.log(1 + words.length) / 4),                      // length
    /([!?])\1/.test(t) ? 1 : 0,                                 // !!! or ???
    clamp((t.match(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu) || []).length / 2),
  ].map(v => v * SCALE);
}

/** Embedding ++ surface features. The vector every head is actually trained on. */
export function withSurface(vector, text) {
  const f = surfaceFeatures(text);
  const out = new Float32Array(vector.length + SURFACE_DIM);
  out.set(vector, 0);
  out.set(f, vector.length);
  return out;
}
