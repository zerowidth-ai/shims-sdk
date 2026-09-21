// Learning from use, as a memory: what a correction reaches, what it must never touch, and whether
// the result a caller gets back is one it can act on without knowing how memory works.
import test from 'node:test';
import assert from 'node:assert/strict';
import { compileShim } from '@zerowidth/shims-sdk/compile';
import { Shim, setEmbedder, LONG_INPUT_CHARS } from '@zerowidth/shims-sdk';
import { AdaptiveShim, AdaptiveBank, MEMORY_MATCH } from '@zerowidth/shims-sdk/adapt';
import { embed, place, cosine } from './world.mjs';

const LABELS = ['dresses', 'outerwear', 'shoes'];
const compiled = await compileShim({ name: 'cat', type: 'classify', question: 'which?', labels: LABELS,
  examples: LABELS.flatMap(l => Array.from({ length: 10 }, (_, i) => ({ label: l, text: `${l}#${i}` }))) }, { embed });
// A word the shim has never met, which it gets wrong, and two ways of typing it.
const typed = 'pinafore~0.1#1', rephrased = 'pinafore~0.1#2', unrelated = 'shoes#900';
const filler = k => `${`filler${k} `.repeat(20).trim()}.`;

test('the premise: a rephrasing sits inside the memory\'s reach and unrelated input sits well outside it', () => {
  assert.ok(cosine(place(typed), place(rephrased)) > MEMORY_MATCH); assert.ok(cosine(place(typed), place(unrelated)) < MEMORY_MATCH - 0.1);
});

test('a correction fixes that input and its rephrasings at once, and says where the answer came from', () => {
  const shim = new AdaptiveShim(compiled), before = shim.decideVector(place(typed));
  shim.record({ text: typed, vector: place(typed), label: 'dresses', verdict: 'yes' });
  for (const t of [typed, rephrased]) {
    const r = shim.decideVector(place(t));
    assert.equal(r.answer, 'dresses'); assert.equal(r.remembered.text, typed); assert.ok(r.remembered.similarity >= MEMORY_MATCH);
    assert.equal(r.shipped.answer, before.answer, 'and what the shipped shim would have said is still there to see');
  }
});

test('a memory cannot touch what it is not near', () => {
  const plain = new Shim(compiled), shim = new AdaptiveShim(compiled);
  shim.record({ text: typed, vector: place(typed), label: 'dresses', verdict: 'yes' });
  shim.record({ text: 'outerwear#5', vector: place('outerwear#5'), label: 'outerwear', verdict: 'no' });
  for (const t of [unrelated, 'dresses#901', 'shoes#902', 'a stranger entirely']) {
    const a = shim.decideVector(place(t)), b = plain.decideVector(place(t));
    assert.equal(a.answer, b.answer); assert.equal(a.confidence, b.confidence); assert.equal(a.action, b.action); assert.equal(a.remembered, undefined); assert.equal(a.rejected, null);
  }
});

test('2026-09-19 · the action on an adaptive result is one a caller can follow: remembered means act, rejected means say nothing', () => {
  // The result used to pair the REMEMBERED answer with the base shim's action for a DIFFERENT answer — often
  // "refuse", since a corrected word is by definition one the shim did not know.
  const shim = new AdaptiveShim(compiled);
  assert.notEqual(shim.decideVector(place(typed)).action, 'act', 'premise: on its own the shim would not act on this');
  shim.record({ text: typed, vector: place(typed), label: 'dresses', verdict: 'yes' });
  assert.equal(shim.decideVector(place(rephrased)).action, 'act');

  const no = new AdaptiveShim(compiled), sure = no.decideVector(place('shoes#3'));
  assert.equal(sure.action, 'act'); assert.equal(sure.answer, 'shoes');
  no.record({ text: 'shoes#3', vector: place('shoes#3'), label: 'shoes', verdict: 'no' });
  const after = no.decideVector(place('shoes#3'));
  assert.equal(after.action, 'refuse'); assert.equal(after.rejected.text, 'shoes#3'); assert.equal(after.shipped.action, 'act');
});

test('a "no" only silences the answer it was said about', () => {
  const shim = new AdaptiveShim(compiled);
  shim.record({ text: 'shoes#4', vector: place('shoes#4'), label: 'dresses', verdict: 'no' });   // "not dresses" — but the shim says shoes
  const r = shim.decideVector(place('shoes#4')); assert.equal(r.rejected, null); assert.equal(r.action, 'act');
});

test('one verdict per input and answer; choosing a different answer replaces the old choice', () => {
  const shim = new AdaptiveShim(compiled), v = place(typed);
  shim.record({ text: typed, vector: v, label: 'outerwear', verdict: 'yes' }); shim.record({ text: typed, vector: v, label: 'dresses', verdict: 'yes' });
  assert.equal(shim.feedback.length, 1); assert.equal(shim.decideVector(v).answer, 'dresses');
  shim.record({ text: typed, vector: v, label: 'dresses', verdict: 'no' }); assert.equal(shim.feedback.length, 1); assert.equal(shim.feedback[0].verdict, 'no');
  const id = shim.feedback[0].id; shim.forget(id); assert.equal(shim.feedback.length, 0);
});

test('memory survives being saved without its vectors and restored through the encoder', async () => {
  const a = new AdaptiveShim(compiled); a.record({ text: typed, vector: place(typed), label: 'dresses', verdict: 'yes' });
  const saved = JSON.parse(JSON.stringify(a.export())); assert.equal(saved[0].vector, undefined);
  const b = new AdaptiveShim(compiled); await b.restore(saved, embed);
  assert.equal(b.decideVector(place(rephrased)).answer, 'dresses');
  await b.restore([{ text: 'no label here' }], embed); assert.equal(b.feedback.length, 1, 'a malformed entry restores nothing and erases nothing');
});

test('tags shims are refused rather than half-supported, and thirty outcomes are needed before recalibrating', async () => {
  const tags = await compileShim({ name: 't', type: 'tags', question: 'q', labels: ['a', 'b'], examples: [...Array.from({ length: 8 }, (_, i) => ({ text: `a#${i}`, labels: ['a'] })), ...Array.from({ length: 8 }, (_, i) => ({ text: `b#${i}`, labels: ['b'] }))] }, { embed });
  assert.throws(() => new AdaptiveShim(tags), /classify shims only/);
  const shim = new AdaptiveShim(compiled); shim.record({ text: typed, vector: place(typed), label: 'dresses', verdict: 'yes' });
  assert.equal(shim.recalibrate().applied, false);
});

test('2026-09-21 · recalibrating a tags shim declines and says why, as refitGate does', async () => {
  // refitGate already guarded tags. recalibrate did not: it asked rawProbs for a distribution a tags shim
  // does not have, got [] back without complaint, and threw a TypeError from inside the temperature fit.
  const tags = await compileShim({ name: 't', type: 'tags', question: 'q', labels: ['a', 'b'], examples: [...Array.from({ length: 8 }, (_, i) => ({ text: `a#${i}`, labels: ['a'] })), ...Array.from({ length: 8 }, (_, i) => ({ text: `b#${i}`, labels: ['b'] }))] }, { embed });
  const shim = new Shim(tags), rows = Array.from({ length: 40 }, (_, i) => ({ vector: place(`a#${100 + i}`), label: 'a' }));
  const r = shim.recalibrate(rows);
  assert.equal(r.applied, false); assert.match(r.why, /tags shims apply no temperature/);
  assert.throws(() => shim.rawProbs(place('a#1')), /rawProbs is for classify shims/);
});

test('2026-09-19 · an adaptive bank decides long input, with memory applied to the piece it was about', async () => {
  // Routing every caller through Shim.decideChunks broke this for an hour: AdaptiveShim had no such method,
  // and nothing exercised an adaptive bank on a long message.
  setEmbedder(embed);
  try {
    const bank = new AdaptiveBank({ category: compiled });
    const short = await bank.decide('shoes#905'); assert.equal(short.category.answer, 'shoes'); assert.ok(short.category.vector);
    const message = [filler(1), 'outerwear#906.', filler(2), filler(3)].join(' '); assert.ok(message.length > LONG_INPUT_CHARS);
    const long = await bank.decide(message);
    assert.equal(long.category.answer, 'outerwear'); assert.equal(long.category.from, 'outerwear#906.'); assert.deepEqual(Array.from(long.category.vector), Array.from(place('outerwear#906.')), 'the vector handed back is the chosen piece\'s, so a correction lands on the right text');
    bank.shims.category.record({ text: 'outerwear#906.', vector: long.category.vector, label: 'dresses', verdict: 'yes' });
    const again = await bank.decide(message); assert.equal(again.category.answer, 'dresses'); assert.equal(again.category.action, 'act');
  } finally { setEmbedder(null); }
});
