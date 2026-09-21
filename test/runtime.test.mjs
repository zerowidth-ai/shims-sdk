// The runtime, fed by the real compiler: what a shim does with a vector, and whether the gates it
// reports are the gates it uses.
import test from 'node:test';
import assert from 'node:assert/strict';
import { compileShim } from '@zerowidth/shims-sdk/compile';
import { Shim, setEmbedder, ENCODER_ID } from '@zerowidth/shims-sdk';
import { fitCentroids, centroidPredict, softmax } from '@zerowidth/shims-sdk/math';
import { packHead, unpackHead, DIM } from '@zerowidth/shims-sdk/format';
import { embed, place, examples } from './world.mjs';

const compile = src => compileShim({ type: 'classify', question: 'which?', examples: [], ...src }, { embed });
const LABELS = ['billing', 'technical', 'account'];
const noisy = await compile({ name: 'noisy', labels: LABELS, examples: examples(LABELS, 12, { scale: 1.1 }) });

// Four answers, two of them easy and two entangled: a hard row is pure, leaning 2:1, or torn 1:1 between
// hardA and hardB — and a torn row is a coin toss by construction, so this shim makes real mistakes.
const MIX = ['easy1', 'easy2', 'hardA', 'hardB'];
const hardText = (own, other, i) => { const k = i % 14; return k < 5 ? `${own}~0.2#${i}` : k < 10 ? `${own}+${own}+${other}~0.2#${i}` : `hardA+hardB~0.2#${i * 2 + (own === 'hardA' ? 0 : 1)}`; };
const mixRow = (l, i) => ({ label: l, text: l === 'hardA' ? hardText('hardA', 'hardB', i) : l === 'hardB' ? hardText('hardB', 'hardA', i) : `${l}#${i}` });
const mixStream = (hardShare, from, n) => Array.from({ length: n }, (_, i) => mixRow((i % 20) < hardShare * 20 ? (i % 2 ? 'hardA' : 'hardB') : (i % 2 ? 'easy1' : 'easy2'), from + i));
const mixed = await compile({ name: 'mixed', labels: MIX, examples: MIX.flatMap(l => Array.from({ length: 14 }, (_, i) => mixRow(l, i))) });
const whenActing = (shim, rows) => { const a = rows.map(r => ({ r, x: shim.decideVector(place(r.text)) })).filter(o => o.x.action === 'act'); return { share: a.length / rows.length, right: a.filter(o => o.x.answer === o.r.label).length / (a.length || 1) }; };

test('a shim refuses weights it cannot safely run', () => {
  assert.throws(() => new Shim({ ...noisy, format: 1 }), /format 1/);
  assert.throws(() => new Shim({ ...noisy, encoder: 'some/other-encoder' }), /compiled for some\/other-encoder/);
  assert.equal(noisy.encoder, ENCODER_ID);
});

test('act, suggest, refuse: the three things a decision can be', () => {
  const shim = new Shim(mixed);
  assert.equal(shim.decideVector(place('easy1#4000')).action, 'act');
  assert.equal(shim.decideVector(place('an input from another product entirely')).action, 'refuse', 'far from every example: say nothing');
  const torn = shim.decideVector(place('hardA+hardB~0.2#4001'));
  assert.notEqual(torn.action, 'act', `torn between two answers at confidence ${torn.confidence}, gate ${shim.gates.confidence}`);
});

test('a decision reports what the gates need: calibrated and raw confidence, familiarity, every answer\'s share', () => {
  const r = new Shim(noisy).decideVector(place('billing#4100'));
  assert.equal(r.answer, 'billing'); assert.deepEqual(Object.keys(r.probs), LABELS);
  assert.ok(Math.abs(Object.values(r.probs).reduce((a, b) => a + b) - 1) < 1e-6); assert.equal(r.confidence, r.probs.billing);
  assert.ok(r.rawConfidence > 0 && r.rawConfidence <= 1); assert.ok(r.familiarity >= 0 && r.familiarity <= 1);
});

test('whatever head the build chose is the head that answers', () => {
  for (const c of [noisy, mixed]) {
    const shim = new Shim(c), v = place(`${c.labels[0]}#4200`);
    if (c.head === 'linear') assert.equal(shim.rawProbs(v).indexOf(Math.max(...shim.rawProbs(v))), softmax(unpackHead(c.heads[0]), v).index);
    if (c.head === 'centroid') assert.deepEqual(shim.rawProbs(v), centroidPredict(unpackHead(c.centroid), v).probs);
    assert.equal(shim.head, c.head);
  }
});

test('a centroid shim with its centroids stripped falls back to the linear head beside it', () => {
  const ex = examples(LABELS, 10), X = ex.map(r => place(r.text)), y = ex.map(r => LABELS.indexOf(r.label));
  const asCentroid = { ...noisy, head: 'centroid', centroid: packHead(fitCentroids(X, y, 3, DIM)) };
  assert.equal(new Shim(asCentroid).head, 'centroid');
  const { centroid, ...stripped } = asCentroid; const shim = new Shim(stripped);
  assert.equal(shim.head, 'linear'); assert.equal(shim.decideVector(place('technical#4300')).answer, 'technical');
});

test('2026-09-19 · isConfident uses the gates the shim is actually running, not the ones it was built with', () => {
  const shim = new Shim(mixed), r = shim.decideVector(place('easy1#4000'));
  assert.equal(shim.isConfident(r), true);
  shim._gate = 0.99999; assert.equal(shim.isConfident({ ...r, confidence: 0.9 }), false, 'a refitted gate must bind');
  shim._gate = undefined; shim._floor = 0.9999; assert.equal(shim.isConfident({ ...r, familiarity: 0.5 }), false, 'and so must a refitted floor');
});

test('refitFloor: needs enough input, then moves the floor to the traffic it was shown', () => {
  const shim = new Shim(noisy);
  assert.equal(shim.refitFloor([place('billing#1')]).applied, false);
  const out = shim.refitFloor(examples(LABELS, 20, { from: 4500, scale: 1.4 }).map(r => place(r.text)));
  assert.equal(out.applied, true); assert.equal(shim.gates.familiarity, out.after); assert.ok(out.after >= 0.005);
});

test('refitGate declines, with a reason, when it has nothing honest to work from', async () => {
  assert.match(new Shim(mixed).refitGate([place('easy1#1')]).why, /needs 100/);
  const young = new Shim(await compile({ name: 'young', labels: LABELS, examples: examples(LABELS, 5) }));
  const out = young.refitGate(examples(LABELS, 40, { from: 4600 }).map(r => place(r.text)));
  assert.equal(out.applied, false); assert.match(out.why, /no held-out rows/); assert.equal(young.gates.confidence, 1, 'a withheld gate stays withheld');
  assert.match(new Shim({ ...mixed, report: { ...mixed.report, gateRows: undefined } }).refitGate(mixStream(0.5, 4700, 120).map(r => place(r.text))).why, /no held-out rows/, 'weights from before gateRows shipped');
});

test('refitGate: traffic shaped like the build leaves the gate alone', () => {
  const shim = new Shim(mixed), out = shim.refitGate(mixStream(0.5, 9000, 300).map(r => place(r.text)));
  assert.equal(out.applied, true); assert.ok(Math.abs(out.after - out.before) < 0.03, `${out.before} → ${out.after}`);
  for (const share of Object.values(out.mix)) assert.ok(Math.abs(share - 0.25) < 0.05);
});

test('refitGate: when the hard answers are the busy ones, it sees the mix, raises the gate, and is right more often when it acts', () => {
  const asBuilt = new Shim(mixed), refit = new Shim(mixed), traffic = mixStream(0.9, 5000, 700);
  const out = refit.refitGate(mixStream(0.9, 9000, 300).map(r => place(r.text)));
  assert.ok(out.mix.hardA + out.mix.hardB > 0.8, `it should see that ~90% of traffic is the hard pair: ${JSON.stringify(out.mix)}`);
  assert.ok(out.after > out.before, `${out.before} → ${out.after}`);
  const before = whenActing(asBuilt, traffic), after = whenActing(refit, traffic);
  assert.ok(after.right >= before.right, `right when acting ${before.right} → ${after.right}`); assert.ok(after.share <= before.share, 'the price is acting on less, and it must be paid honestly');
  assert.equal(out.actsOnAfter <= out.actsOnBefore, true);
});

test('recalibrate needs thirty outcomes before it will touch the temperature', () => {
  const shim = new Shim(noisy), before = shim.temperature;
  const out = shim.recalibrate(examples(LABELS, 5, { from: 4800 }).map(r => ({ vector: place(r.text), label: r.label })));
  assert.equal(out.applied, false); assert.equal(shim.temperature, before);
});

test('decide() goes through whatever embedder is set, and agrees with decideVector', async () => {
  setEmbedder(embed);
  try { const shim = new Shim(noisy), a = await shim.decide('account#4900'), b = shim.decideVector(place('account#4900')); assert.equal(a.answer, b.answer); assert.equal(a.confidence, b.confidence); }
  finally { setEmbedder(null); }
});
