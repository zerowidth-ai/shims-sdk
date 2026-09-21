// The head trainers and the small numeric functions every gate is built from.
import test from 'node:test';
import assert from 'node:assert/strict';
import { fitHead, fitHeadGD, fitCentroids, centroidPredict, softmax, balanceWeights, defaultL2,
         applyTemperature, fitTemperature, calibrationError, estimatePrior, weightedGate } from '../src/math.mjs';
import { DIM } from '../src/format.mjs';
import { place, examples } from './world.mjs';

const LABELS = ['billing', 'technical', 'account'];
const rows = examples(LABELS, 8), X = rows.map(r => place(r.text)), y = rows.map(r => LABELS.indexOf(r.label));
const bytes = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
const accuracy = (predict, n = 10) => { const t = examples(LABELS, n, { from: 100 }); return t.filter(r => predict(place(r.text)) === LABELS.indexOf(r.label)).length / t.length; };

test('fitHead is deterministic to the byte — committing weights means nothing otherwise', () => {
  const a = fitHead(X, y, 3, DIM), b = fitHead(X, y, 3, DIM);
  assert.ok(bytes(a.W).equals(bytes(b.W))); assert.ok(bytes(a.b).equals(bytes(b.b)));
});

test('fitHead separates separable answers', () => {
  const h = fitHead(X, y, 3, DIM, { weights: balanceWeights(y, 3) });
  assert.equal(accuracy(v => softmax(h, v).index), 1);
});

test('fitHead throws on an option it does not know, rather than ignoring it', () => {
  // An experiment once passed `prior` to a fitHead that had no such option, and recorded "warm start does nothing".
  assert.throws(() => fitHead(X, y, 3, DIM, { epochs: 50 }), /unknown option epochs/);
  assert.throws(() => fitHead(X, y, 3, DIM, { prior: {} }), /unknown option prior/);
});

test('fitHead has actually converged: a looser tolerance and more iterations change nothing material', () => {
  const a = fitHead(X, y, 3, DIM), b = fitHead(X, y, 3, DIM, { maxIter: 5000, tol: 1e-8 });
  let worst = 0; for (let i = 0; i < a.W.length; i++) worst = Math.max(worst, Math.abs(a.W[i] - b.W[i]));
  assert.ok(worst < 1e-2, `weights moved by ${worst} after 10x the iterations`);
});

test('a lighter penalty lets the head commit: the over-regularisation bug, pinned', () => {
  // Until 2026-09-19 every head trained at l2 = 3e-3, and a many-answer head could rank but not commit
  // (mean confidence 0.03 on 77 answers). The mechanism is answer count, so test it on a many-answer head.
  const many = Array.from({ length: 40 }, (_, i) => `intent${i}`), r = examples(many, 6);
  const MX = r.map(e => place(e.text)), my = r.map(e => many.indexOf(e.label)), probe = MX.slice(0, 40);
  const meanConf = h => probe.reduce((a, v) => a + softmax(h, v).confidence, 0) / probe.length;
  const firm = meanConf(fitHead(MX, my, 40, DIM, { l2: 3e-3 })), light = meanConf(fitHead(MX, my, 40, DIM, { l2: 1e-4 }));
  assert.ok(light > firm + 0.3, `light ${light.toFixed(2)} should be far more committed than firm ${firm.toFixed(2)}`);
  assert.ok(defaultL2(40) < defaultL2(3), 'the default must loosen as answers are added');
  assert.equal(defaultL2(3), 3e-3, 'and stay where it was chosen for few-answer shims');
});

test('row weights are respected: a zero-weight row might as well not be there', () => {
  const poison = [...X, place('billing~0#999')], py = [...y, 2];          // a billing vector labelled `account`
  const clean = fitHead(X, y, 3, DIM), ignored = fitHead(poison, py, 3, DIM, { weights: [...X.map(() => 1), 0] });
  let worst = 0; for (let i = 0; i < clean.W.length; i++) worst = Math.max(worst, Math.abs(clean.W[i] - ignored.W[i]));
  assert.ok(worst < 1e-4);
});

test('fitHeadGD still exists, so experiments recorded against the old trainer reproduce', () => {
  const h = fitHeadGD(X, y, 3, DIM, { epochs: 50, lr: 3 });
  assert.equal(h.W.length, 3 * DIM);
});

test('centroids are unit vectors, classify separable answers, and survive an answer with no rows', () => {
  const c = fitCentroids(X, y, 4, DIM);                                   // K=4, nothing labelled 3
  for (let k = 0; k < 3; k++) { let sq = 0; for (let d = 0; d < DIM; d++) sq += c.W[k * DIM + d] ** 2; assert.ok(Math.abs(sq - 1) < 1e-5); }
  assert.equal(accuracy(v => centroidPredict(c, v).index), 1);
  const p = centroidPredict(c, X[0]); assert.ok(p.probs.every(Number.isFinite)); assert.notEqual(p.index, 3);
});

test('temperature can never change which answer wins', () => {
  const probs = [0.5, 0.3, 0.2];
  for (const T of [0.05, 0.5, 1, 2, 10]) { const q = applyTemperature(probs, T); assert.equal(q.indexOf(Math.max(...q)), 0); assert.ok(Math.abs(q.reduce((a, b) => a + b) - 1) < 1e-9); }
});

test('fitTemperature sharpens an under-confident head and softens an over-confident one', () => {
  const under = Array.from({ length: 60 }, (_, i) => ({ probs: [0.4, 0.3, 0.3], y: i < 55 ? 0 : 1 }));  // right 92% of the time, claims 40%
  const over = Array.from({ length: 60 }, (_, i) => ({ probs: [0.98, 0.01, 0.01], y: i % 2 ? 0 : 1 })); // sure, right half the time
  assert.ok(fitTemperature(under) < 1); assert.ok(fitTemperature(over) > 1); assert.equal(fitTemperature([]), 1);
});

test('2026-09-19 · fitTemperature refuses to fit rows that are all right', () => {
  // No mistakes means no maximum: sharper always scores better, and the search ran to T = 0.04.
  assert.equal(fitTemperature(Array.from({ length: 60 }, () => ({ probs: [0.4, 0.3, 0.3], y: 0 }))), 1);
});

test('calibrationError is zero for a perfectly calibrated set and large for a confidently wrong one', () => {
  const good = Array.from({ length: 100 }, (_, i) => ({ confidence: 0.8, ok: i < 80 })), bad = Array.from({ length: 100 }, () => ({ confidence: 0.95, ok: false }));
  assert.ok(calibrationError(good) < 0.01); assert.ok(calibrationError(bad) > 0.9);
});

test('weightedGate: the promise it returns is true of the rows it was read from', () => {
  const r = rows => rows.map(([c, ok, w = 1]) => ({ c, ok, w }));
  const set = r([[0.99, 1], [0.95, 1], [0.9, 1], [0.85, 1], [0.8, 1], [0.7, 1], [0.6, 1], [0.55, 1], [0.5, 1], [0.45, 0], [0.4, 0], [0.3, 0]]);
  const g = weightedGate(set, 0.9), kept = set.filter(s => s.c >= g.threshold);
  assert.ok(kept.filter(s => s.ok).length / kept.length >= 0.9);
  assert.equal(g.threshold, 0.45, 'and it is the WIDEST such gate: nine right and one wrong is exactly 90%');
  assert.deepEqual(weightedGate(r([[0.9, 0], [0.8, 0]]), 0.9), { threshold: 1, coverage: 0 }, 'no honest gate exists: stay shut');
});

test('weightedGate: weighting the hard answer up moves the gate up', () => {
  const easy = Array.from({ length: 20 }, (_, i) => ({ c: 0.6 + i * 0.01, ok: true, y: 0 })), hard = Array.from({ length: 20 }, (_, i) => ({ c: 0.6 + i * 0.01, ok: i >= 10, y: 1 }));
  const at = share => weightedGate([...easy.map(s => ({ ...s, w: 1 - share })), ...hard.map(s => ({ ...s, w: share }))], 0.9).threshold;
  assert.ok(at(0.9) > at(0.1), 'a stream that is mostly the hard answer needs a stricter gate');
});

test('estimatePrior recovers a skewed answer mix from calibrated probabilities, with no labels', () => {
  // 80/15/5 traffic, a head that is 85% sure of the right answer.
  const truth = [0.8, 0.15, 0.05], r = (() => { let s = 7; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; })();
  const probRows = Array.from({ length: 600 }, () => { const u = r(), k = u < 0.8 ? 0 : u < 0.95 ? 1 : 2, seen = r() < 0.85 ? k : (k + 1) % 3; return [0, 1, 2].map(j => j === seen ? 0.85 : 0.075); });
  const est = estimatePrior(probRows, [1 / 3, 1 / 3, 1 / 3]);
  assert.ok(Math.abs(est.reduce((a, b) => a + b) - 1) < 1e-6);
  assert.ok(est[0] > 0.7 && est[2] < 0.12, `estimated ${est.map(v => v.toFixed(2))}`);
});

test('estimatePrior leaves a balanced stream balanced', () => {
  const probRows = Array.from({ length: 300 }, (_, i) => [0, 1, 2].map(j => j === i % 3 ? 0.9 : 0.05));
  for (const v of estimatePrior(probRows, [1 / 3, 1 / 3, 1 / 3])) assert.ok(Math.abs(v - 1 / 3) < 0.02);
});
