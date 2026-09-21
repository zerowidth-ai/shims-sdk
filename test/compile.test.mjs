// The compiler, end to end, on an encoder whose geometry is known. Each test names the promise it
// holds the build to; the ones marked with a date pin a defect that shipped and was found by hand.
import test from 'node:test';
import assert from 'node:assert/strict';
import { compileShim } from '@zerowidth/shims-sdk/compile';
import { Shim } from '@zerowidth/shims-sdk';
import { weightedGate } from '@zerowidth/shims-sdk/math';
import { embed, place, examples } from './world.mjs';

const compile = (src, opts = {}) => compileShim({ type: 'classify', question: 'which?', examples: [], ...src }, { embed, ...opts });
const LABELS = ['billing', 'technical', 'account'];
const strip = c => { const { compiledAt, embedded, ...rest } = c; return rest; };

test('the same source compiles to the same bytes', async () => {
  const src = { name: 'det', labels: LABELS, examples: examples(LABELS, 10, { scale: 0.8 }) };
  assert.deepEqual(strip(await compile(src)), strip(await compile(src)));
});

test('a separable task: scored, certified, and right on rows it never saw', async () => {
  const c = await compile({ name: 'easy', labels: LABELS, examples: examples(LABELS, 10) });
  assert.equal(c.report.scorable, true); assert.ok(c.report.accuracy > 0.95); assert.ok(c.report.suggestedThreshold < 1);
  const shim = new Shim(c), held = examples(LABELS, 10, { from: 500 });
  assert.equal(held.filter(r => shim.decideVector(place(r.text)).answer === r.label).length, held.length);
});

test('2026-09-19 · a shim that made no held-out mistakes keeps its own confidences', async () => {
  // Fitting a temperature to rows that are all right sharpens without limit. It shipped T = 0.04 and
  // reported an input sitting between three answers (raw confidence 0.41) as 0.993.
  const c = await compile({ name: 'perfect', labels: LABELS, examples: examples(LABELS, 10) });
  assert.equal(c.report.accuracy, 1); assert.equal(c.report.calibration.temperature, 1);
  const torn = new Shim(c).decideVector(place('billing+technical+account~0.1#2'));
  assert.ok(torn.confidence < 0.7, `an input torn three ways was reported at ${torn.confidence}`);
});

test('the gate is withheld until there are eight real examples per answer, and says what it would have been', async () => {
  const c = await compile({ name: 'young', labels: LABELS, examples: examples(LABELS, 5) });
  assert.equal(c.report.scorable, false); assert.equal(c.report.suggestedThreshold, 1); assert.equal(c.report.coverageAtThreshold, 0);
  assert.equal(typeof c.report.provisionalThreshold, 'number'); assert.equal(c.report.gateRows, null, 'no certified gate, nothing for refitGate to reweight');
});

test('an answer with too few examples and no prototypes makes a draft, and the runtime refuses a draft', async () => {
  const c = await compile({ name: 'draft', labels: LABELS, examples: [...examples(['billing', 'technical'], 5), { text: 'account#0', label: 'account' }] });
  assert.equal(c.draft, true); assert.throws(() => new Shim(c), /still a draft/);
});

test('gateRows are the rows the gate was read from: re-reading them gives the same gate', async () => {
  const c = await compile({ name: 'rows', labels: LABELS, examples: examples(LABELS, 12, { scale: 1.1 }) });
  const rows = c.report.gateRows; assert.equal(rows.length, c.report.evaluated);
  for (const [y, conf, ok] of rows) { assert.ok(y >= 0 && y < 3); assert.ok(conf >= 0 && conf <= 1); assert.ok(ok === 0 || ok === 1); }
  const again = weightedGate(rows.map(([, c_, ok]) => ({ c: c_, ok, w: 1 })), 0.9);
  assert.ok(Math.abs(again.threshold - c.report.suggestedThreshold) < 2e-3, `${again.threshold} vs ${c.report.suggestedThreshold}`);
  assert.ok(Math.abs(rows.filter(r => r[2]).length / rows.length - c.report.accuracy) < 1e-3, 'and their accuracy is the report\'s accuracy');
});

test('2026-09-19 · the gate never lands above the row it was read from', async () => {
  // Rounded to nearest, a boundary of 0.9998 became a gate of 1.000: coverage reported 100%, rows admitted 0.
  for (const [n, scale] of [[10, undefined], [14, 1.2], [12, 0.9]]) {
    const c = await compile({ name: 'round', labels: LABELS, examples: examples(LABELS, n, { scale }) });
    const admitted = c.report.gateRows.filter(r => r[1] >= c.report.suggestedThreshold).length / c.report.gateRows.length;
    assert.ok(admitted >= c.report.coverageAtThreshold - 1e-9, `reported ${c.report.coverageAtThreshold}, admitted ${admitted}`);
  }
});

test('the promise holds on the rows it was made about: above the gate, at least 90% right', async () => {
  const c = await compile({ name: 'promise', labels: LABELS, examples: examples(LABELS, 14, { scale: 1.2 }) });
  if (c.report.suggestedThreshold === 1) return;                          // nothing promised
  const kept = c.report.gateRows.filter(r => r[1] >= c.report.suggestedThreshold);
  assert.ok(kept.filter(r => r[2]).length / kept.length >= 0.9);
  assert.ok(Math.abs(kept.length / c.report.gateRows.length - c.report.coverageAtThreshold) < 0.02);
});

test('2026-09-19 · folds train on the prototypes too, because the shipped head does', async () => {
  // Every example is a word of its own, met once. Alone they cannot be cross-validated above chance: hold a
  // word out and nothing like it remains. The prototypes contain each word's twin. The shipped head is trained
  // on both, so it knows every word — and the report must describe THAT head. Before the fix it described a
  // head trained on the examples alone, and under-reported `category` by thirty points.
  const word = (l, i, seed) => `${l}word${i}~0.05#${seed}`;
  const src = { name: 'vocab', labels: LABELS, examples: LABELS.flatMap(l => Array.from({ length: 8 }, (_, i) => ({ text: word(l, i, 0), label: l }))),
    prototypes: Object.fromEntries(LABELS.map(l => [l, Array.from({ length: 8 }, (_, i) => word(l, i, 1))])) };
  const withTwins = await compile(src), alone = await compile({ ...src, prototypes: undefined });
  assert.ok(alone.report.accuracy < 0.6, `alone should be near chance, was ${alone.report.accuracy}`);
  assert.ok(withTwins.report.accuracy > 0.9, `with its prototypes in the folds it should be near perfect, was ${withTwins.report.accuracy}`);
  assert.ok(withTwins.headChoice.knn > 0.9, 'kNN in particular can only score this if the twins were in its folds');
});

test('prototypes are trained on and never scored: the report is about real examples', async () => {
  const c = await compile({ name: 'scored', labels: LABELS, examples: examples(LABELS, 9), prototypes: Object.fromEntries(LABELS.map(l => [l, examples([l], 6, { from: 900 }).map(r => r.text)])) });
  assert.equal(c.report.evaluated, 27); assert.equal(c.report.scoredOn, 'examples'); assert.equal(c.reference.count, 27 + 18, 'but they do ship in the reference set');
});

test('with no examples at all, prototypes compile to a working shim with its gate withheld', async () => {
  const c = await compile({ name: 'protos', labels: LABELS, prototypes: Object.fromEntries(LABELS.map(l => [l, examples([l], 8).map(r => r.text)])) });
  assert.ok(!c.draft); assert.equal(c.report.scoredOn, 'prototypes'); assert.equal(c.report.suggestedThreshold, 1);
  assert.equal(new Shim(c).decideVector(place('technical#700')).answer, 'technical');
});

test('an answer that lives in two places gets nearest-neighbours, not a single direction', async () => {
  const labels = ['red', 'other'], ex = [...examples(['blood', 'lights'], 8).map(r => ({ text: r.text, label: 'red' })), ...examples(['between'], 16).map(r => ({ text: r.text, label: 'other' }))];
  // `between` sits on the line from blood to lights, where a single "red" direction points.
  const c = await compile({ name: 'modes', labels, examples: ex.map(r => r.label === 'other' ? { ...r, text: r.text.replace('between', 'blood+lights') } : r) });
  assert.ok(c.headChoice.knn >= c.headChoice.linear, JSON.stringify(c.headChoice));
});

test('every candidate head is measured, the penalty is chosen from the grid, and both are recorded', async () => {
  const c = await compile({ name: 'choice', labels: LABELS, examples: examples(LABELS, 10, { scale: 0.9 }) });
  for (const h of ['linear', 'knn', 'centroid']) { assert.equal(typeof c.headChoice[h], 'number'); assert.equal(typeof c.headChoice.coverage[h], 'number'); }
  assert.ok([3e-3, 3e-4, 3e-5].includes(c.l2)); assert.equal(c.headChoice.l2Chosen, c.l2); assert.ok(['linear', 'knn', 'centroid', 'tree'].includes(c.head));
  assert.equal(c.report.macroRecall, c.headChoice[c.head] ?? c.report.macroRecall, 'the report describes the head that ships');
});

test('the coverage tie-break can be switched off, and says so when it fires', async () => {
  const src = { name: 'tb', labels: LABELS, examples: examples(LABELS, 10, { scale: 1.0 }) };
  const on = await compile(src), off = await compile(src, { coverageTiebreak: false });
  assert.equal(off.headChoice.coverageOverride, null);
  if (on.headChoice.coverageOverride) { const o = on.headChoice.coverageOverride; assert.equal(on.head, o.to); assert.ok(o.accuracyGap < o.standardError); assert.ok(o.coverage[1] >= o.coverage[0] + 0.10); }
});

test('a source that names an answer it does not have fails with a sentence, not a stack trace', async () => {
  await assert.rejects(compile({ name: 'bad1', labels: LABELS, examples: [...examples(LABELS, 4), { text: 'x#1', label: 'refunds' }] }), /refunds/);
  await assert.rejects(compile({ name: 'bad2', labels: LABELS, examples: examples(LABELS, 4), prototypes: { refunds: ['x#1', 'x#2', 'x#3'] } }), /refunds/);
});

test('the familiarity floor is fitted, bounded below, and refitted from a real sample when one is given', async () => {
  const src = { name: 'floor', labels: LABELS, examples: examples(LABELS, 10) };
  const c = await compile(src); assert.ok(c.report.familiarityFloor >= 0.02 && c.report.familiarityFloor < 0.5); assert.equal(c.report.floorFittedOn, 'examples');
  const r = await compile({ ...src, realSample: examples(LABELS, 10, { from: 800, scale: 0.7 }).map(e => e.text) }); assert.equal(r.report.floorFittedOn, 'realSample');
});

test('the floor does what it says: about one in twenty in-scope inputs is turned away, and a typical one reads ~0.5', async () => {
  const c = await compile({ name: 'fl', labels: LABELS, examples: examples(LABELS, 20) }), shim = new Shim(c);
  const fam = examples(LABELS, 200, { from: 5000 }).map(r => shim.decideVector(place(r.text)).familiarity).sort((a, b) => a - b);
  const refused = fam.filter(f => f < c.report.familiarityFloor).length / fam.length, median = fam[fam.length >> 1];
  assert.ok(refused < 0.12, `aimed at 5%, refused ${(100 * refused).toFixed(1)}%`); assert.ok(median > 0.3 && median < 0.7, `median familiarity ${median}`);
});

test('a tags shim compiles, ships a reference set, and its answer is a set', async () => {
  const tags = ['urgent', 'legal'], ex = [...examples(['urgent'], 8).map(r => ({ text: r.text, labels: ['urgent'] })), ...examples(['legal'], 8).map(r => ({ text: r.text, labels: ['legal'] })), ...examples(['neither'], 8).map(r => ({ text: r.text, labels: [] }))];
  const c = await compile({ name: 'tags', type: 'tags', labels: tags, examples: ex });
  assert.ok(c.reference.count > 0); const out = new Shim(c).decideVector(place('urgent#400'));
  assert.ok(Array.isArray(out.answer)); assert.ok(out.answer.includes('urgent')); assert.ok(['act', 'suggest', 'refuse'].includes(out.action));
});

test('2026-09-21 · a shim described by prototypes compiles with no examples key at all', async () => {
  // validateSource and readiness both read a missing "examples" as empty. compileShim read `src.examples.map`
  // and died with a TypeError on the one kind of source most likely to omit it: a described shim nobody has
  // written an example for yet. Called directly — the `compile` helper above fills `examples` in.
  const prototypes = Object.fromEntries(LABELS.map(l => [l, Array.from({ length: 8 }, (_, i) => `${l}#${600 + i}`)]));
  const c = await compileShim({ name: 'described', type: 'classify', question: 'which?', labels: LABELS, prototypes }, { embed });
  assert.equal(c.draft, undefined, 'prototypes are enough to compile'); assert.equal(c.examples, 0);
  assert.equal(new Shim(c).decideVector(place('technical#690')).answer, 'technical');
  const draft = await compileShim({ name: 'empty', labels: LABELS }, { embed });
  assert.equal(draft.draft, true, 'and with neither, it is a draft rather than a crash');
});

test('2026-09-21 · the reserved schema type is refused by name, not by a TypeError', async () => {
  // The validator accepts type "schema" so the format can be written down; nothing builds it. It used to
  // pass validation and then throw "Cannot read properties of undefined (reading 'length')".
  const src = { name: 'form', type: 'schema', fields: { area: { labels: ['a', 'b'] } }, examples: [{ text: 'a#1', values: { area: 'a' } }] };
  await assert.rejects(() => compileShim(src, { embed }), /form: "schema" is a reserved format/);
});
