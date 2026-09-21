// Capture. refitFloor, refitGate and recalibrate all need real traffic, and traffic is only captured
// if capturing is what happens by default on the path an app actually uses.
import test from 'node:test';
import assert from 'node:assert/strict';
import { compileShim } from '@zerowidth/shims-sdk/compile';
import { Shim, Bank, setEmbedder, LONG_INPUT_CHARS } from '@zerowidth/shims-sdk';
import { System } from '@zerowidth/shims-sdk/system';
import { AdaptiveBank } from '@zerowidth/shims-sdk/adapt';
import { observe, observing, outcome, ring } from '@zerowidth/shims-sdk/observe';
import { embed, place } from './world.mjs';

const build = async (name, labels, n = 12) => compileShim({ name, type: 'classify', question: 'q', labels, examples: labels.flatMap(l => Array.from({ length: n }, (_, i) => ({ label: l, text: `${l}#${i}` }))) }, { embed });
const area = await build('area', ['billing', 'technical', 'account']), tone = await build('tone', ['calm', 'angry']);
const filler = k => `${`filler${k} `.repeat(20).trim()}.`, long = [filler(1), 'technical#4.', filler(2), filler(3)].join(' ');
/** Run with the fake encoder and a fresh ring, and always switch both off again. */
async function watching(fn) { const log = ring(100); setEmbedder(embed); observe(log.push); try { return await fn(log); } finally { observe(null); setEmbedder(null); } }

test('nothing is recorded, and no id is handed out, unless something is observing', async () => {
  setEmbedder(embed);
  try { assert.equal(observing(), false); const r = await new Shim(area).decide('billing#3'); assert.equal(r.id, undefined); } finally { setEmbedder(null); }
});

test('a decision is recorded with what a refit needs: the vector, the answer, the action, the gates in force', () => watching(async log => {
  const r = await new Shim(area).decide('billing#3'); const row = log.rows[0];
  assert.equal(log.rows.length, 1); assert.equal(row.id, r.id); assert.equal(row.shim, 'area'); assert.equal(row.answer, 'billing'); assert.equal(row.action, r.action);
  assert.deepEqual(Array.from(row.vector), Array.from(place('billing#3'))); assert.equal(row.words, 1); assert.deepEqual(row.gates, new Shim(area).gates); assert.ok(row.ms >= 0);
}));

test('keepText: false records everything except what the person typed', () => watching(async log => {
  await new Shim(area).decide('billing#3', { keepText: false }); assert.equal(log.rows[0].text, null); assert.ok(log.rows[0].vector); assert.equal(log.rows[0].chars, 9);
}));

test('2026-09-19 · a long message is recorded with the vector of the piece it was decided on', () => watching(async log => {
  // It was recorded with `vector: null`, so no long decision could ever reach refitFloor, refitGate or recalibrate.
  assert.ok(long.length > LONG_INPUT_CHARS); const r = await new Shim(area).decide(long);
  assert.equal(r.answer, 'technical'); assert.deepEqual(Array.from(log.rows[0].vector), Array.from(place('technical#4.'))); assert.equal(log.vectorsFor('area', {}).length, 1);
}));

test('2026-09-19 · a bank records every shim it asks, each under its own id', () => watching(async log => {
  // Bank.decide() recorded nothing. It is how an app runs several shims over one input — the pattern the README
  // leads with — so a deployment wired that way captured no traffic at all.
  const out = await new Bank({ area, tone }).decide('billing#3');
  assert.equal(log.rows.length, 2); assert.deepEqual(log.rows.map(r => r.shim).sort(), ['area', 'tone']); assert.notEqual(out.area.id, out.tone.id);
  assert.equal(log.rows.find(r => r.shim === 'area').field, 'area'); assert.equal(log.rows[0].via, 'bank');
  const viaLong = await new Bank({ area, tone }).decide(long); assert.ok(viaLong.area.id); assert.equal(log.rows.length, 4); assert.ok(log.rows[2].vector);
}));

test('2026-09-19 · a system records every shim it asks, once, and says which decision an outcome belongs to', () => watching(async log => {
  const spec = { name: 'support', steps: [{ id: 'area', kind: 'shim', shim: 'area' }, { id: 'tone', kind: 'shim', shim: 'tone', always: true }] };
  const r = await new System(spec, { area: new Shim(area), tone: new Shim(tone) }).run('account#3');
  assert.equal(r.answer, 'account'); assert.equal(log.rows.length, 2); assert.ok(log.rows.every(x => x.via === 'system' && x.system === 'support'));
  assert.equal(r.id, r.results.area.id, 'the id to report an outcome against is the step whose answer became the system\'s');
  outcome(r.id, 'accepted'); assert.equal(log.for('area', { outcome: 'accepted' }).length, 1); assert.equal(log.for('tone', { outcome: 'accepted' }).length, 0);
}));

test('2026-09-19 · an adaptive bank records what was SAID — the remembered answer — and that memory spoke', () => watching(async log => {
  const bank = new AdaptiveBank({ category: area }), first = await bank.decide('pinafore~0.1#1');
  bank.shims.category.record({ text: 'pinafore~0.1#1', vector: first.category.vector, label: 'billing', verdict: 'yes' });
  const again = await bank.decide('pinafore~0.1#2'), row = log.rows.at(-1);
  assert.equal(again.category.answer, 'billing'); assert.equal(row.answer, 'billing'); assert.equal(row.action, 'act'); assert.equal(row.memory, 'yes'); assert.equal(log.rows[0].memory, null);
}));

test('outcomes attach to the decision they name, and what a person corrected TO becomes the label', () => watching(async log => {
  const shim = new Shim(area), a = await shim.decide('billing#3'), b = await shim.decide('technical#3'), c = await shim.decide('account#3');
  outcome(a.id, 'accepted'); outcome(b.id, 'corrected', { label: 'account' }); outcome(c.id, 'ignored'); outcome('no-such-id', 'accepted');
  assert.deepEqual(log.outcomesFor('area').map(r => r.label), ['billing', 'account'], 'accepted keeps the shim\'s answer; corrected takes the person\'s; ignored teaches nothing');
  assert.equal(log.vectorsFor('area').length, 2, 'in-scope means a person went along with it or fixed it — not everything that arrived');
  assert.equal(log.summary('area').withOutcome, 3);
}));

test('the ring is bounded, copies vectors, and forgets ids it has dropped', () => watching(async () => {
  const small = ring(3), v = new Float32Array([1, 2, 3]);
  for (let i = 0; i < 5; i++) small.push({ kind: 'decision', id: `d${i}`, shim: 's', action: 'act', vector: v });
  assert.deepEqual(small.rows.map(r => r.id), ['d2', 'd3', 'd4']); v[0] = 99; assert.equal(small.rows[0].vector[0], 1, 'the runtime reuses buffers; a ring holding references would fill with one vector');
  small.push({ kind: 'outcome', id: 'd0', outcome: 'accepted' }); assert.equal(small.rows.filter(r => r.outcome).length, 0);
  small.clear(); assert.equal(small.rows.length, 0);
}));

test('a sink that throws never breaks a decision', async () => {
  setEmbedder(embed); observe(() => { throw new Error('disk full'); });
  try { assert.equal((await new Shim(area).decide('billing#3')).answer, 'billing'); } finally { observe(null); setEmbedder(null); }
});

test('the summary answers the first week\'s questions without a label: how often it acted, how much people typed, how long it took', () => watching(async log => {
  const shim = new Shim(area); for (const t of ['billing#3', 'technical#3', 'a message about something else entirely']) await shim.decide(t);
  const s = log.summary('area'); assert.equal(s.decisions, 3); assert.ok(Math.abs(s.acted + s.suggested + s.refused - 1) < 0.01); assert.ok(s.refused > 0); assert.equal(s.medianWords, 1); assert.equal(s.withOutcome, 0);
}));

test('a bank skips drafts rather than failing, shares one encode, and adds up what it costs', async () => {
  const draft = await compileShim({ name: 'd', type: 'classify', question: 'q', labels: ['a', 'b'], examples: [{ text: 'a#1', label: 'a' }] }, { embed });
  assert.equal(draft.draft, true); const bank = new Bank({ area, tone, draft }); assert.deepEqual(Object.keys(bank.shims), ['area', 'tone']);
  let encodes = 0; setEmbedder(async t => { encodes++; return embed(t); });
  try { const out = await bank.decide('technical#3'); assert.equal(encodes, 1); assert.equal(out.area.answer, 'technical'); assert.ok(bank.bytes > 0); } finally { setEmbedder(null); }
});
