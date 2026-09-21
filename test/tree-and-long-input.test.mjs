// Trees and long input, through the real compiler and the real runtime.
import test from 'node:test';
import assert from 'node:assert/strict';
import { compileShim } from '@zerowidth/shims-sdk/compile';
import { Shim, setEmbedder, LONG_INPUT_CHARS } from '@zerowidth/shims-sdk';
import { embed, place } from './world.mjs';

const compile = (src, opts = {}) => compileShim({ type: 'classify', question: 'which?', examples: [], ...src }, { embed, ...opts });
const FAMILIES = { money: ['refund', 'invoice', 'chargeback'], access: ['password', 'twofactor', 'lockout'], product: ['bug', 'outage', 'feature'] };
const LABELS = Object.values(FAMILIES).flat(), familyOf = l => Object.keys(FAMILIES).find(f => FAMILIES[f].includes(l));
const text = (l, i, noise = 0.3) => `${familyOf(l)}+${l}~${noise}#${i}`;
const rows = (n, from = 0, noise) => LABELS.flatMap(l => Array.from({ length: n }, (_, i) => ({ label: l, text: text(l, from + i, noise) })));

test('a tree pinned in the source is the tree that ships, and a decision says which group it came through', async () => {
  const c = await compile({ name: 'pinned', labels: LABELS, examples: rows(9), tree: FAMILIES });
  assert.equal(c.head, 'tree'); assert.deepEqual(c.report.structure.groups, Object.values(FAMILIES)); assert.equal(c.report.structure.pinned, true);
  const shim = new Shim(c), r = shim.decideVector(place(text('lockout', 800)));
  assert.equal(r.answer, 'lockout'); assert.deepEqual(r.path.siblings, FAMILIES.access); assert.ok(r.path.groupConfidence > 0.5);
  assert.deepEqual(shim.groups, Object.values(FAMILIES));
});

test('a pinned tree that loses an answer, invents one, or places one twice is refused in a sentence', async () => {
  const src = { name: 'badtree', labels: LABELS, examples: rows(4) };
  await assert.rejects(compile({ ...src, tree: { money: FAMILIES.money, access: FAMILIES.access } }), /leaves answers out.*bug/);
  await assert.rejects(compile({ ...src, tree: { ...FAMILIES, extra: ['refunds'] } }), /not in labels.*refunds/);
  await assert.rejects(compile({ ...src, tree: { ...FAMILIES, again: ['bug'] } }), /more than one group/);
});

test('"tree": "off" means flat, and says so; otherwise a tree is tried and its score recorded either way', async () => {
  const off = await compile({ name: 'flat', labels: LABELS, examples: rows(9, 0, 1.2), tree: 'off' });
  assert.notEqual(off.head, 'tree'); assert.equal(off.report.structure.chosen, 'flat'); assert.match(off.report.structure.reason, /off/);
  const auto = await compile({ name: 'auto', labels: LABELS, examples: rows(9, 0, 1.2) });
  assert.ok(auto.report.structure.tried.length > 0); for (const t of auto.report.structure.tried) assert.equal(typeof t.macroRecall, 'number');
  assert.equal(auto.head === 'tree', auto.report.structure.chosen === 'tree');
});

test('a tree shim survives JSON and decides the same afterwards', async () => {
  const c = await compile({ name: 'roundtrip', labels: LABELS, examples: rows(9), tree: FAMILIES }), a = new Shim(c), b = new Shim(JSON.parse(JSON.stringify(c)));
  for (const r of rows(2, 850)) assert.deepEqual(a.decideVector(place(r.text)), b.decideVector(place(r.text)));
});

// ---- long input ----------------------------------------------------------------------------------
const TOPICS = ['billing', 'technical', 'account'];
const topical = await compile({ name: 'topical', labels: TOPICS, examples: TOPICS.flatMap(l => Array.from({ length: 12 }, (_, i) => ({ label: l, text: `${l}#${i}` }))) });
// Filler sentences that each FILL a piece (160 characters), so nothing can be packed in beside them — a short
// needle would otherwise share a piece with its neighbour, which is correct packing and useless for this test.
const filler = k => `${`filler${k} `.repeat(20).trim()}.`;

test('a long message is decided on the piece the shim recognises, wherever it sits, and says which', async () => {
  setEmbedder(embed);
  try {
    const shim = new Shim(topical);
    for (const position of [0, 2, 4]) {
      const parts = [filler(1), filler(2), filler(3), filler(4)]; parts.splice(position, 0, 'technical#777.');
      const message = parts.join(' '); assert.ok(message.length > LONG_INPUT_CHARS);
      const r = await shim.decide(message);
      assert.equal(r.answer, 'technical', `needle at position ${position}`); assert.equal(r.from, 'technical#777.'); assert.equal(r.chunks, 5);
    }
  } finally { setEmbedder(null); }
});

test('a short message is never split', async () => {
  setEmbedder(embed);
  try { const r = await new Shim(topical).decide('billing#778'); assert.equal(r.chunks, undefined); assert.equal(r.answer, 'billing'); }
  finally { setEmbedder(null); }
});

test('whatever the reduction, the action reported is the action its own confidence and familiarity earn', async () => {
  setEmbedder(embed);
  try {
    const shim = new Shim(topical);
    // Two recognisable pieces that disagree, among filler: pooling must lower the confidence, and the action has to follow it.
    const message = [filler(1), 'billing#779.', filler(2), 'technical#780.', filler(3)].join(' ');
    for (const reduce of ['best', 'pooled']) {
      const r = await shim.decide(message, { reduce });
      assert.equal(r.action, shim.actionFor(r.familiarity, r.confidence), `${reduce}: confidence ${r.confidence.toFixed(2)}, gate ${shim.gates.confidence}, reported "${r.action}"`);
    }
  } finally { setEmbedder(null); }
});

test('2026-09-21 · pooling a long message leaves a tags decision a set of tags', async () => {
  // A tags result carries a score per tag and no `probs`. Pooling read `probs` anyway: the answer came back
  // as the first label — a string where every caller expects an array — at confidence 0, with a zeroed
  // distribution attached. Reachable from decide(), a bank, and a system spec's "reduce".
  const tagged = await compile({ name: 'flags', type: 'tags', labels: ['urgent', 'legal'], examples: [
    ...Array.from({ length: 10 }, (_, i) => ({ text: `urgent#${i}`, labels: ['urgent'] })),
    ...Array.from({ length: 10 }, (_, i) => ({ text: `legal#${i}`, labels: ['legal'] })),
    ...Array.from({ length: 10 }, (_, i) => ({ text: `neither#${i}`, labels: [] }))] });
  setEmbedder(embed);
  try {
    // "~0.1" sits the piece close to the examples, so it is familiar and there IS evidence to pool —
    // with nothing familiar, pooling falls back to the best piece before it ever reads `probs`.
    const shim = new Shim(tagged), message = [filler(1), 'legal~0.1#880.', filler(2), filler(3)].join(' ');
    assert.ok(message.length > LONG_INPUT_CHARS);
    const best = await shim.decide(message, { reduce: 'best' }), pooled = await shim.decide(message, { reduce: 'pooled' });
    assert.deepEqual(best.answer, ['legal']);
    assert.deepEqual(pooled.answer, best.answer); assert.equal(pooled.confidence, best.confidence); assert.equal(pooled.probs, undefined);
  } finally { setEmbedder(null); }
});
