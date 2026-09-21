// Head training and inference. Isomorphic — identical code runs in the compiler
// (Node, at build time) and in the browser (for authoring tools that refit as you type).
// Deterministic by construction: zero init, a fixed evaluation order, and a convex objective
// solved to convergence. The same examples always produce the same weights, which is what
// makes committing them to git meaningful.

/**
 * Multinomial logistic regression over frozen embeddings, fitted to convergence.
 *
 * L-BFGS on the weighted mean cross-entropy plus (l2/2)·|W|². The objective is convex, so there is
 * one answer and the optimiser runs until it has found it rather than for a fixed count — no epochs,
 * no learning rate. Zero init and a fixed evaluation order keep it deterministic, which is what
 * makes committing the weights meaningful.
 *
 * The default l2 is the finding (2026-09-19). Until then this was 250 epochs of
 * gradient descent at l2 = 3e-3, and the suspicion was too few steps. It was not: eight times the
 * epochs moved Banking77 from 78.1 to 78.3. It was the penalty. Embeddings are unit vectors, so a
 * usable logit gap needs weights with a norm in the tens, and 3e-3 does not allow that on a
 * many-answer shim — at 77 answers the head's mean confidence was 0.03, barely off uniform. That is
 * also where the "ECE 89" heads came from; the temperature was compensating for a head that had not
 * been allowed to fit.
 *
 * Why it depends on K: the loss is a mean over examples and each answer sees about 1/K of them, so
 * a fixed penalty presses K times harder on each answer's weights as answers are added. 3e-3 was
 * chosen on 3-4 answer shims, where it is fine — across a set of small hand-built shims (K ≤ 16, mostly ≤ 6) any
 * value from 3e-3 to 1e-4 cross-validates within noise (66.9 / 66.5 / 66.3 mean balanced accuracy).
 * So the default keeps 3e-3 where it was chosen and scales it down as 0.012 / K beyond that: 1.6e-4
 * at 77 answers, 8e-5 at 150, which is the flat part of the measured curve for both. On
 * human-written test rows that is CLINC150 87.7 → 91.8, Banking77 78.1 → 85.4 (8/answer) and
 * 83.1 → 90.2 (32), MASSIVE 68.8 → 74.8 and 71.4 → 80.8.
 *
 * That default is for direct callers (tree leaves, tag heads, a live refit with no build to read).
 * The compiler does not rely on it: answer count turned out not to be the whole story — `food` has
 * 85 answers and still prefers 3e-3, because it is noisy rather than because it is small — so
 * compile.mjs cross-validates a grid of penalties per shim and records the one it chose as `l2`.
 */
export const defaultL2 = K => Math.min(3e-3, 0.012 / K);
export function fitHead(X, y, K, dim, { l2 = defaultL2(K), weights = null, maxIter = 500, tol = 1e-5, ...rest } = {}) {
  const n = X.length;
  if (!n) throw new Error('fitHead: no examples');
  // An option that is silently ignored reads as a result: an experiment once passed `prior`, which
  // nothing read, and concluded that a warm start does nothing. The old optimiser's knobs are gone; say so.
  const unknown = Object.keys(rest);
  if (unknown.length) throw new Error(`fitHead: unknown option ${unknown.join(', ')} — epochs/lr belong to fitHeadGD`);

  const P = K * (dim + 1);                                  // θ = [ W (K·dim) | b (K) ]
  const w = weights ?? new Array(n).fill(1);
  const wSum = w.reduce((a, v) => a + v, 0);
  const flat = new Float64Array(n * dim);
  X.forEach((x, i) => flat.set(x, i * dim));
  const p = new Float64Array(K);

  // Loss at θ; fills g with the gradient when given one.
  const evaluate = (th, g) => {
    let loss = 0;
    if (g) g.fill(0);
    for (let i = 0; i < n; i++) {
      const xo = i * dim; let max = -Infinity;
      for (let k = 0; k < K; k++) {
        let s = th[K * dim + k]; const wo = k * dim;
        for (let d = 0; d < dim; d++) s += th[wo + d] * flat[xo + d];
        p[k] = s; if (s > max) max = s;
      }
      let sum = 0;
      for (let k = 0; k < K; k++) { p[k] = Math.exp(p[k] - max); sum += p[k]; }
      loss -= w[i] * Math.log(Math.max(p[y[i]] / sum, 1e-300));
      if (!g) continue;
      for (let k = 0; k < K; k++) {
        const c = w[i] * (p[k] / sum - (y[i] === k ? 1 : 0));
        if (c === 0) continue;
        g[K * dim + k] += c; const wo = k * dim;
        for (let d = 0; d < dim; d++) g[wo + d] += c * flat[xo + d];
      }
    }
    let sq = 0;
    for (let j = 0; j < P; j++) { sq += th[j] * th[j]; if (g) g[j] = g[j] / wSum + l2 * th[j]; }
    return loss / wSum + 0.5 * l2 * sq;
  };
  const dot = (a, b) => { let s = 0; for (let j = 0; j < P; j++) s += a[j] * b[j]; return s; };

  let th = new Float64Array(P), g = new Float64Array(P), f = evaluate(th, g);
  const S = [], Y = [], MEMORY = 10;
  for (let it = 0; it < maxIter && Math.sqrt(dot(g, g)) >= tol; it++) {
    // Two-loop recursion: q becomes (approximate inverse Hessian) · gradient.
    const q = Float64Array.from(g), alpha = [];
    for (let j = S.length - 1; j >= 0; j--) {
      alpha[j] = dot(S[j], q) / dot(Y[j], S[j]);
      for (let t = 0; t < P; t++) q[t] -= alpha[j] * Y[j][t];
    }
    if (S.length) {
      const scale = dot(S.at(-1), Y.at(-1)) / dot(Y.at(-1), Y.at(-1));
      for (let t = 0; t < P; t++) q[t] *= scale;
    }
    for (let j = 0; j < S.length; j++) {
      const beta = dot(Y[j], q) / dot(Y[j], S[j]);
      for (let t = 0; t < P; t++) q[t] += (alpha[j] - beta) * S[j][t];
    }
    // Backtracking line search (Armijo). The first step has no curvature yet, so keep it short.
    const slope = -dot(g, q);
    let step = it ? 1 : 1 / Math.max(1, Math.sqrt(dot(g, g)));
    let next, nextG = new Float64Array(P), nextF;
    for (let ls = 0; ls < 30; ls++, step /= 2) {
      next = new Float64Array(P);
      for (let t = 0; t < P; t++) next[t] = th[t] - step * q[t];
      nextF = evaluate(next, nextG);
      if (nextF <= f + 1e-4 * step * slope) break;
    }
    const s = new Float64Array(P), yy = new Float64Array(P);
    for (let t = 0; t < P; t++) { s[t] = next[t] - th[t]; yy[t] = nextG[t] - g[t]; }
    if (dot(s, yy) > 1e-12) { S.push(s); Y.push(yy); if (S.length > MEMORY) { S.shift(); Y.shift(); } }
    const settled = Math.abs(f - nextF) < 1e-10 * Math.max(1, Math.abs(f));
    th = next; g = nextG; f = nextF;
    if (settled) break;
  }
  return { W: Float32Array.from(th.subarray(0, K * dim)), b: Float32Array.from(th.subarray(K * dim)), K, dim };
}

/**
 * The head trainer this SDK shipped until 2026-09-19: fixed-count full-batch gradient descent.
 * Kept under its own name so measurements made with it still reproduce.
 * Do not use it for anything new — see fitHead for what was wrong with its default.
 */
export function fitHeadGD(X, y, K, dim, { epochs = 250, lr = 3.0, l2 = 3e-3, weights = null } = {}) {
  const n = X.length;
  if (!n) throw new Error('fitHead: no examples');
  const W = new Float32Array(K * dim), b = new Float32Array(K);
  const gW = new Float32Array(K * dim), gb = new Float32Array(K), p = new Float32Array(K);
  const w = weights ?? new Array(n).fill(1);
  const wSum = w.reduce((a, v) => a + v, 0);

  for (let e = 0; e < epochs; e++) {
    gW.fill(0); gb.fill(0);
    for (let i = 0; i < n; i++) {
      const x = X[i]; let max = -Infinity;
      for (let k = 0; k < K; k++) {
        let s = b[k];
        for (let d = 0; d < dim; d++) s += W[k * dim + d] * x[d];
        p[k] = s; if (s > max) max = s;
      }
      let sum = 0;
      for (let k = 0; k < K; k++) { p[k] = Math.exp(p[k] - max); sum += p[k]; }
      for (let k = 0; k < K; k++) {
        const g = w[i] * (p[k] / sum - (y[i] === k ? 1 : 0));
        gb[k] += g;
        for (let d = 0; d < dim; d++) gW[k * dim + d] += g * x[d];
      }
    }
    for (let k = 0; k < K; k++) {
      b[k] -= lr * (gb[k] / wSum + l2 * b[k]);
      for (let d = 0; d < dim; d++) {
        const ix = k * dim + d;
        W[ix] -= lr * (gW[ix] / wSum + l2 * W[ix]);
      }
    }
  }
  return { W, b, K, dim };
}

/**
 * Nearest class mean. Average each answer's vectors, normalise, and score by cosine.
 *
 * No epochs, no learning rate, nothing to converge: the same rows give the same bytes on any
 * machine, by arithmetic rather than by a fixed epoch count. Measured against the two heads above
 * on human-written test rows (2026-09-19): at 8 examples per answer CLINC150 90.8
 * against 88.0, MASSIVE 74.6 against 67-71, Banking77 level; with generated prototypes and no
 * examples at all, +3 to +7 on all three. At 32 per answer Banking77 prefers kNN by 3.6 — an
 * answer with several modes does not have one mean — which is why the compiler measures all
 * three rather than assuming.
 *
 * Returned in the same shape as a linear head (W = unit centroids, b = 0), so it packs, unpacks
 * and costs exactly what a linear head does.
 */
export const CENTROID_SCALE = 20;
export function fitCentroids(X, y, K, dim, { weights = null } = {}) {
  if (!X.length) throw new Error('fitCentroids: no examples');
  const W = new Float32Array(K * dim), b = new Float32Array(K);
  for (let i = 0; i < X.length; i++) {
    const w = weights ? weights[i] : 1, off = y[i] * dim, x = X[i];
    for (let d = 0; d < dim; d++) W[off + d] += w * x[d];
  }
  for (let k = 0; k < K; k++) {
    let sq = 0;
    for (let d = 0; d < dim; d++) sq += W[k * dim + d] ** 2;
    const n = Math.sqrt(sq);
    // An answer with no rows keeps a zero vector: cosine 0 to everything, so it is never chosen
    // over an answer that has evidence, and never divides by zero.
    if (n) for (let d = 0; d < dim; d++) W[k * dim + d] /= n;
  }
  return { W, b, K, dim };
}

/** Softmax over scaled cosines to each centroid. Same return shape as `softmax`. */
export function centroidPredict(head, x) {
  const { W, K, dim } = head;
  let nx = 0;
  for (let d = 0; d < dim; d++) nx += x[d] * x[d];
  nx = Math.sqrt(nx) || 1;
  const s = new Float32Array(K);
  let max = -Infinity;
  for (let k = 0; k < K; k++) {
    let v = 0;
    for (let d = 0; d < dim; d++) v += W[k * dim + d] * x[d];
    s[k] = CENTROID_SCALE * v / nx; if (s[k] > max) max = s[k];
  }
  let sum = 0;
  for (let k = 0; k < K; k++) { s[k] = Math.exp(s[k] - max); sum += s[k]; }
  let top = 0;
  for (let k = 0; k < K; k++) { s[k] /= sum; if (s[k] > s[top]) top = k; }
  return { probs: Array.from(s), index: top, confidence: s[top] };
}

/** Per-class independent binary heads, for multi-label (`tags`). */
export function fitTagHeads(X, yMulti, K, dim, { rowWeight = null, ...opts } = {}) {
  const heads = [];
  for (let k = 0; k < K; k++) {
    const y = yMulti.map(set => set.has(k) ? 1 : 0);
    const pos = y.filter(v => v === 1).length;
    // Balance each tag against its own absence, or a rare tag never fires. Then scale
    // by the caller's per-row weight, which is how generated prototypes fade out as
    // real examples arrive.
    const balanced = pos && pos < y.length
      ? y.map(v => v === 1 ? (y.length - pos) / pos : 1)
      : y.map(() => 1);
    const weights = rowWeight ? balanced.map((w, i) => w * rowWeight[i]) : balanced;
    heads.push(fitHead(X, y, 2, dim, { ...opts, weights }));
  }
  return heads;
}

export function softmax(head, x) {
  const { W, b, K, dim } = head;
  const s = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    let v = b[k];
    for (let d = 0; d < dim; d++) v += W[k * dim + d] * x[d];
    s[k] = v;
  }
  let max = -Infinity;
  for (let k = 0; k < K; k++) if (s[k] > max) max = s[k];
  let sum = 0;
  for (let k = 0; k < K; k++) { s[k] = Math.exp(s[k] - max); sum += s[k]; }
  let top = 0;
  for (let k = 0; k < K; k++) { s[k] /= sum; if (s[k] > s[top]) top = k; }
  return { probs: Array.from(s), index: top, confidence: s[top] };
}

/** Class-balanced weights, so a rare answer is not simply ignored. */
export function balanceWeights(y, K) {
  const counts = new Array(K).fill(0);
  for (const v of y) counts[v]++;
  const nonEmpty = counts.filter(c => c > 0).length || 1;
  const target = y.length / nonEmpty;
  return y.map(v => counts[v] ? target / counts[v] : 1);
}

/** Deterministic PRNG, so holdout splits are reproducible across machines. */
export function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}
export function shuffled(arr, seed) {
  const a = [...arr], r = rng(seed);
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

/**
 * One number that makes a shim's confidence mean what it says.
 *
 * Measured on CLINC150 (150 answers, 32 examples each): a linear head is right 90% of the time
 * while reporting under 0.10 confidence on every single decision — expected calibration error
 * 89%. Spread over many answers, softmax probabilities are flat. The ordering is fine, so
 * thresholds still work, but "0.7" cannot be read as 70% and anything shown to a person, or
 * compared across shims, is nonsense.
 *
 * The fix is the standard one: raise the probabilities to a power and renormalise, with the power
 * chosen to minimise negative log-likelihood on held-out predictions. T < 1 sharpens an
 * under-confident head; T > 1 softens an over-confident one. It cannot change which answer wins —
 * only how sure the shim claims to be.
 */
export function applyTemperature(probs, T) {
  if (!T || T === 1) return probs;
  const p = probs.map(v => Math.pow(Math.max(v, 1e-12), 1 / T));
  const sum = p.reduce((a, b) => a + b, 0) || 1;
  return p.map(v => v / sum);
}

/** Fit T on held-out (probs, true answer) pairs. Grid then refine; 1 when there is nothing to fit. */
export function fitTemperature(samples) {
  if (!samples?.length) return 1;
  // With no held-out mistakes the likelihood has no maximum: sharper is always "better", the search
  // runs to the floor of the grid, and every confidence becomes 1.000. Found by the test suite on a
  // separable task (2026-09-19): T = 0.04, and an input sitting between three answers, raw confidence
  // 0.41, reported as 0.993. Nothing can be learned about where confidence should fall from rows
  // that are all right, so leave the head's own numbers alone.
  const top = p => { let i = 0; for (let j = 1; j < p.length; j++) if (p[j] > p[i]) i = j; return i; };
  if (samples.every(s => top(s.probs) === s.y)) return 1;
  const nll = T => {
    let s = 0;
    for (const { probs, y } of samples) s -= Math.log(Math.max(applyTemperature(probs, T)[y], 1e-12));
    return s / samples.length;
  };
  let best = 1, bestNll = nll(1);
  for (let e = -1.4; e <= 1.0; e += 0.05) {            // T from ~0.04 to 10
    const T = Math.pow(10, e), v = nll(T);
    if (v < bestNll) { bestNll = v; best = T; }
  }
  for (let step = best / 8; step > best / 200; step /= 2) {
    for (const T of [best - step, best + step]) {
      if (T <= 0.01) continue;
      const v = nll(T);
      if (v < bestNll) { bestNll = v; best = T; }
    }
  }
  return +best.toFixed(3);
}

/** Expected calibration error: mean gap between claimed confidence and how often it is right. */
export function calibrationError(samples, bins = 10) {
  if (!samples?.length) return null;
  const b = Array.from({ length: bins }, () => ({ n: 0, ok: 0, conf: 0 }));
  for (const s of samples) {
    const i = Math.min(bins - 1, Math.floor(s.confidence * bins));
    b[i].n++; b[i].ok += s.ok ? 1 : 0; b[i].conf += s.confidence;
  }
  return +b.reduce((a, x) => x.n ? a + (x.n / samples.length) * Math.abs(x.ok / x.n - x.conf / x.n) : a, 0).toFixed(4);
}

/**
 * How common each answer is in a stream of UNLABELLED input, from the shim's own calibrated
 * probabilities on it. EM re-estimation of the class prior (Saerens et al. 2002): start from the
 * prior the head was trained under, tilt every row's probabilities by (current / trained), average,
 * repeat. Deterministic, a few milliseconds, and measured to be close enough after 100 rows.
 */
export function estimatePrior(probRows, trainedPrior, { iters = 200, tol = 1e-7 } = {}) {
  const K = trainedPrior.length;
  let pi = Float64Array.from(trainedPrior);
  for (let it = 0; it < iters; it++) {
    const next = new Float64Array(K), r = new Float64Array(K);
    for (const p of probRows) {
      let z = 0;
      for (let k = 0; k < K; k++) { r[k] = p[k] * pi[k] / trainedPrior[k]; z += r[k]; }
      for (let k = 0; k < K; k++) next[k] += r[k] / (z || 1);
    }
    let moved = 0;
    for (let k = 0; k < K; k++) { next[k] /= probRows.length; moved += Math.abs(next[k] - pi[k]); }
    pi = next;
    if (moved < tol) break;
  }
  return Array.from(pi);
}

/** The lowest confidence at which the rows kept are `target` accurate, rows weighted by `w`. */
export function weightedGate(rows, target = 0.9) {
  const sorted = [...rows].sort((a, b) => b.c - a.c);
  let ok = 0, total = 0, kept = 0, gate = 1, coverage = 0;
  const all = sorted.reduce((a, r) => a + r.w, 0) || 1;
  for (const r of sorted) {
    ok += r.w * (r.ok ? 1 : 0); total += r.w; kept += r.w;
    if (ok / total >= target) { gate = r.c; coverage = kept / all; }
  }
  // Rounded down, never to nearest: a gate must not land above the row it was read from.
  return { threshold: gate >= 1 && !coverage ? 1 : Math.floor(gate * 1000) / 1000, coverage: +coverage.toFixed(3) };
}

