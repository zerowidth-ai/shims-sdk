// A document read in pieces has to become one decision. Which reduction?
//
// Experiment 7-8 compared four on ONE document shape — a single decisive sentence buried in
// uninformative filler — and best-chunk-by-familiarity won it outright. That became the default.
// But a needle-in-filler document is the ideal case for winner-take-all, and it says nothing about
// a document whose evidence is SPREAD: a complaint, a review, a long clause, where every sentence
// carries a little and none carries it all. On those, reading one sentence throws the rest away.
//
// The measurement builds four document shapes from held-out rows and runs every reduction
// over the same chunk vectors, so they all cost the same. Two numbers per cell, because accuracy
// alone is not the whole story: a reduction that is RIGHT but no longer confident enough to clear
// the shim's own gate has not produced a decision, it has produced an escalation.
//
//   banking77, gate 0.398          needle    spread     mixed       two     acted
//   best chunk by familiarity       93/92     96/96     89/89     97/97       374
//   familiarity-weighted            93/78    100/100    94/93     98/96       367
//   familiarity^3-weighted          93/91     96/96     92/92     98/96       375
//
// PER SHAPE the difference is large and real: pooling is worth +10 accuracy on CLINC's spread-
// evidence documents, and winner-take-all is worth +52 on a needle. SUMMED OVER A MIX OF SHAPES
// THEY CANCEL. Acted-on decisions across both tasks, out of 800:
//
//   best chunk by familiarity  708      familiarity^2  710
//   familiarity-weighted       707      familiarity^3  711  ← best, by 3 of 800
//
// A 0.5% spread is not a result. So the default does NOT change: `best` is simpler, has no
// interaction between chunk count and confidence, and is what each shim's temperature was
// calibrated against.
//
// `pooled` is here because the per-shape numbers are worth having when you KNOW your document
// shape. A contract clause is distributed evidence by construction — a proviso and a main limb,
// both load-bearing — and there pooling is the right call and the benchmark says so. An app that
// knows this about its own input can ask for it; an app that does not should not guess.
//
// `power` slides along the tradeoff rather than escaping it: higher suppresses unfamiliar chunks
// harder, so it behaves more like `best` on a needle and less like pooling on spread evidence.

const argmax = (scores, labels) => labels.reduce((b, l) => scores[l] > scores[b] ? l : b, labels[0]);

/**
 * Combine per-chunk decisions into one.
 *
 * @param {object[]} results  one decideVector() result per chunk, in order
 * @param {string[]} chunks   the chunk texts, same order
 * @param {string[]} labels   the shim's answers
 * @param {{mode?: 'best'|'pooled', power?: number}} opts
 */
export function reduceChunks(results, chunks, labels, opts = {}) {
  const { mode = 'best', power = 3 } = opts;
  if (results.length === 1) return { ...results[0], from: chunks[0], chunks: 1 };

  // The chunk this shim recognises best. Under `best` it IS the decision; under `pooled` it is
  // still the one worth showing a person, so `from` means the same thing either way.
  let at = 0;
  results.forEach((r, i) => { if ((r.familiarity ?? 0) > (results[at].familiarity ?? 0)) at = i; });
  const pick = { ...results[at], from: chunks[at], chunks: chunks.length };
  if (mode !== 'pooled') return pick;
  // A tags decision is a set of tags with a score each, not a distribution over answers, so there
  // is nothing here to pool. Pooling it anyway read `probs` off results that have none and came
  // back with the first label as a string, at confidence 0. Read the best piece instead.
  if (!results[at].probs) return pick;

  const weights = results.map(r => Math.pow(r.familiarity ?? r.confidence ?? 0, power));
  const total = weights.reduce((a, b) => a + b, 0);
  // Nothing here is familiar at all, so there is no evidence to pool. Falling back to the single
  // best chunk keeps behaviour sane for input the shim has no business judging — which its gates
  // will refuse anyway.
  if (!(total > 0)) return pick;

  const probs = {};
  for (const l of labels) probs[l] = 0;
  results.forEach((r, i) => {
    const w = weights[i] / total;
    for (const l of labels) probs[l] += w * (r.probs?.[l] ?? 0);
  });

  const answer = argmax(probs, labels);
  return {
    ...pick,
    answer,
    probs,
    confidence: probs[answer],
    // Familiarity stays the BEST chunk's, not an average: the question it answers is "is any of
    // this mine to judge", and averaging would let filler talk a shim out of input it knows.
    familiarity: results[at].familiarity,
  };
}
