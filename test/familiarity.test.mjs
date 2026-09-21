// Nearest-neighbour voting, the shipped reference set, and "have I seen anything like this".
import test from 'node:test';
import assert from 'node:assert/strict';
import { packReference, unpackReference, knnPredict, familiarityScore, cosine } from '../src/familiarity.mjs';
import { place, examples } from './world.mjs';

const LABELS = ['outerwear', 'accessories', 'sleepwear', 'shoes'];
const rows = examples(LABELS, 8), X = rows.map(r => place(r.text)), y = rows.map(r => LABELS.indexOf(r.label));

test('an exact match is not outvoted by strangers — the "anorak" bug', () => {
  // "anorak" sat at cosine 1.00 from its own training row (outerwear) and came back as accessories,
  // because votes were weighted by raw cosine and two unrelated rows at ~0.6 sum to more than 1.0.
  // The setting: one row for the word itself, the rest of the reference set unrelated to it.
  // Every stranger votes the same way, so under raw-cosine weighting four of them (~0.5 each) beat the one
  // exact match 2.0 to 1.0. A first draft spread them over two answers and passed by 1.00 to 0.98 — by luck.
  const vocab = [{ text: 'anorak', label: 0 }, ...['belt', 'scarf', 'beanie', 'gloves', 'watch', 'brooch'].map(text => ({ text, label: 1 }))];
  const ref = packReference(vocab.map(r => place(r.text)), vocab.map(r => r.label)), q = place('anorak');
  const strangers = vocab.slice(1).map(r => cosine(q, place(r.text)));
  assert.ok(Math.min(...strangers) > 0.35, 'the test only means something if strangers score well above zero, as they do in the real space');
  assert.equal(knnPredict(q, ref, 4, 5).index, 0);
});

test('kNN holds an answer whose examples sit in two unrelated places', () => {
  // "red" = blood, traffic lights. One direction cannot hold both without swallowing what lies between.
  const two = [...examples(['blood', 'lights'], 6).map(r => ({ ...r, y: 0 })), ...examples(['between'], 6).map(r => ({ ...r, y: 1 }))];
  const ref = packReference(two.map(r => place(r.text)), two.map(r => r.y));
  for (const t of ['blood#50', 'lights#51', 'blood#52', 'lights#53']) assert.equal(knnPredict(place(t), ref, 2, 5).index, 0, t);
  assert.equal(knnPredict(place('between#54'), ref, 2, 5).index, 1);
});

test('kNN probabilities are a distribution, and confidence is the winner\'s share', () => {
  const p = knnPredict(place('shoes#77'), packReference(X, y), 4, 5);
  assert.ok(Math.abs(p.probs.reduce((a, b) => a + b) - 1) < 1e-6); assert.equal(p.confidence, p.probs[p.index]); assert.equal(p.index, 3);
});

test('the reference set survives int8 packing and a JSON round trip', () => {
  const ref = packReference(X, y), b64 = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64');
  const shipped = JSON.parse(JSON.stringify({ q: b64(ref.q), scales: b64(ref.scales), spread: ref.spread, count: ref.count, labels: ref.labels }));
  const back = unpackReference(shipped);
  assert.equal(back.count, X.length); assert.deepEqual(Array.from(back.labels), y);
  for (const t of ['outerwear#90', 'shoes#91', 'a complete stranger']) {
    const v = place(t); assert.equal(familiarityScore(v, back), familiarityScore(v, ref)); assert.deepEqual(knnPredict(v, back, 4, 5), knnPredict(v, ref, 4, 5));
  }
});

test('familiarity: text like the examples scores high, a stranger scores zero', () => {
  const ref = packReference(X, y);
  assert.ok(familiarityScore(place('outerwear#200'), ref) > 0.2);
  assert.equal(familiarityScore(place('revent 80 cfm bathroom fan'), ref), 0);
  assert.equal(familiarityScore(X[0], ref), 1, 'an input identical to an example is as familiar as it gets');
  assert.equal(familiarityScore(X[0], null), null, 'no reference set, no opinion');
});

test('familiarity is a percentile of the examples\' own spacing, so it is comparable across shims', () => {
  // A tight shim and a loose one should both call their own typical input ~0.5, not 0.9 and 0.3.
  for (const scale of [0.15, 0.6]) {
    const r = examples(LABELS, 12, { scale }), ref = packReference(r.map(e => place(e.text)), r.map(e => LABELS.indexOf(e.label)));
    const typical = examples(LABELS, 12, { scale, from: 300 }).map(e => familiarityScore(place(e.text), ref)).sort((a, b) => a - b);
    const median = typical[typical.length >> 1];
    assert.ok(median > 0.2 && median < 0.8, `scale ${scale}: median familiarity of in-distribution input was ${median}`);
  }
});
