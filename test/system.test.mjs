// Several shims wired into one decision. The wiring is data, so it can be wrong in ways code cannot:
// these tests hold it to what it says it does — who is asked, who decides, and when nobody should.
import test from 'node:test';
import assert from 'node:assert/strict';
import { compileShim } from '@zerowidth/shims-sdk/compile';
import { Shim, setEmbedder, LONG_INPUT_CHARS } from '@zerowidth/shims-sdk';
import { System, validateSystem, shimsUsed } from '@zerowidth/shims-sdk/system';
import { embed, place } from './world.mjs';

// A router over three areas, each with a two-answer specialist. A message sits between its area's
// centre and its own reason's, so the router and the right specialist both recognise it.
const AREAS = { billing: ['refund', 'invoice'], technical: ['bug', 'outage'], account: ['password', 'lockout'] };
const msg = (area, reason, i) => `${area}+${reason}~0.25#${i}`;
const build = async (name, labels, rowsOf) => new Shim(await compileShim({ name, type: 'classify', question: 'q', labels, examples: labels.flatMap(l => rowsOf(l)) }, { embed }));
const shims = { area: await build('area', Object.keys(AREAS), a => AREAS[a].flatMap(r => Array.from({ length: 6 }, (_, i) => ({ label: a, text: msg(a, r, i) })))) };
for (const [a, reasons] of Object.entries(AREAS)) shims[a] = await build(a, reasons, r => Array.from({ length: 10 }, (_, i) => ({ label: r, text: msg(a, r, i) })));
shims.tone = await build('tone', ['calm', 'angry'], t => Array.from({ length: 10 }, (_, i) => ({ label: t, text: `${t}#${i}` })));

const base = { name: 'support', steps: [
  { id: 'area', kind: 'shim', shim: 'area' },
  { id: 'reason', kind: 'route', on: 'area', routes: { billing: 'billing', technical: 'technical', account: 'account' } }] };
const run = async (spec, text, withShims = shims) => { setEmbedder(embed); try { return await new System(spec, withShims).run(text); } finally { setEmbedder(null); } };

test('a router hands to the right specialist: one encode, two heads, and a trace of who said what', async () => {
  const r = await run(base, msg('technical', 'outage', 900));
  assert.equal(r.outcome, 'routed'); assert.equal(r.answer, 'outage'); assert.equal(r.encodes, 1); assert.equal(r.heads, 2);
  assert.deepEqual(r.trace.map(t => [t.id, t.shim, t.answer, t.state]), [['area', 'area', 'technical', 'confident'], ['reason', 'technical', 'outage', 'confident']]);
});

test('rules run before any model, and a hit stops everything that is not marked always', async () => {
  const spec = { ...base, steps: [{ id: 'urgent', kind: 'rules', outcome: 'page-oncall', rules: [{ name: 'outage-word', match: '\\bdown\\b' }] }, ...base.steps, { id: 'tone', kind: 'shim', shim: 'tone', always: true }] };
  const r = await run(spec, 'the site is DOWN');
  assert.equal(r.outcome, 'page-oncall'); assert.equal(r.answer, 'outage-word');
  assert.deepEqual(r.trace.map(t => t.id), ['urgent', 'tone'], 'the router was never asked; the always-step still was');
  assert.equal(r.heads, 1);
});

test('2026-09-19 · an always-step annotates the decision: it never becomes the answer, and failing to recognise the message never escalates it', async () => {
  const spec = { ...base, steps: [...base.steps, { id: 'tone', kind: 'shim', shim: 'tone', always: true }] };
  const r = await run(spec, msg('billing', 'refund', 901));
  assert.equal(r.answer, 'refund'); assert.equal(r.results.tone.shim, 'tone');
  assert.notEqual(r.results.tone.state, 'confident', 'premise: the tone shim does not recognise a billing message');
  assert.equal(r.outcome, 'routed', 'an unclear TONE must not send a confidently routed message to a person');
});

test('a route to null is a deliberate dead end: routed, with the router\'s own answer', async () => {
  const spec = { ...base, steps: [base.steps[0], { ...base.steps[1], routes: { ...base.steps[1].routes, account: null }, endsAt: 'the accounts team' }] };
  const r = await run(spec, msg('account', 'lockout', 902));
  assert.equal(r.outcome, 'routed'); assert.equal(r.answer, 'account'); assert.equal(r.heads, 1); assert.equal(r.trace.at(-1).note, 'the accounts team');
});

test('input nobody recognises escalates, and says it was unfamiliar rather than unsure', async () => {
  const r = await run(base, 'a message about something else entirely');
  assert.equal(r.outcome, 'escalate'); assert.equal(r.answer, null); assert.equal(r.trace[0].state, 'unfamiliar'); assert.equal(r.trace.length, 1, 'and nothing downstream was asked');
});

test('what happens when a step is unsure is the spec\'s to say', async () => {
  const strict = { confidence: 0.999999 }, text = msg('billing', 'invoice', 903);
  assert.equal((await run({ ...base, steps: [{ ...base.steps[0], gate: strict }, base.steps[1]] }, text)).outcome, 'escalate');
  assert.equal((await run({ ...base, steps: [{ ...base.steps[0], gate: strict, onUnsure: 'triage-queue' }, base.steps[1]] }, text)).outcome, 'triage-queue');
  const open = await run({ ...base, steps: [{ ...base.steps[0], gate: { confidence: false, familiarity: false } }, base.steps[1]] }, 'a message about something else entirely');
  assert.equal(open.trace[0].state, 'confident', '`false` lets everything through');
});

test('2026-09-19 · a system runs on the gates its shims are actually running, not the ones they were built with', async () => {
  // refitGate() and refitFloor() change shim.gates. The system read shim.report directly, so a gate refitted for
  // real traffic bound Shim.decide() and was ignored the moment the same shim was wired into a system.
  const area = new Shim(JSON.parse(JSON.stringify({ ...shims.area.report && {}, ...(await compileShim({ name: 'area', type: 'classify', question: 'q', labels: Object.keys(AREAS), examples: Object.keys(AREAS).flatMap(a => AREAS[a].flatMap(r => Array.from({ length: 6 }, (_, i) => ({ label: a, text: msg(a, r, i) })))) }, { embed })) })));
  const text = msg('billing', 'refund', 904);
  assert.equal((await run(base, text, { ...shims, area })).trace[0].state, 'confident');
  area._gate = 0.999999;
  assert.equal((await run(base, text, { ...shims, area })).trace[0].state, 'unsure', 'a refitted confidence gate must bind inside a system');
  area._gate = undefined; area._floor = 0.999999;
  assert.equal((await run(base, text, { ...shims, area })).trace[0].state, 'unfamiliar', 'and so must a refitted familiarity floor');
});

test('a confident child rescues an unsure parent — by familiarity, and each specialist is asked only once', async () => {
  const spec = { ...base, steps: [{ ...base.steps[0], gate: { confidence: 0.999999 } }, { ...base.steps[1], rescue: { margin: 0.2 } }] };
  const r = await run(spec, msg('technical', 'bug', 5));
  const vote = r.trace.find(t => t.kind === 'rescue');
  assert.equal(vote.took, 'technical'); assert.equal(vote.votes[0].key, 'technical'); assert.ok(vote.margin >= 0.2);
  assert.equal(r.results.area.state, 'rescued'); assert.equal(r.results.area.rescuedBy, 'technical'); assert.equal(r.answer, 'bug');
  assert.equal(r.heads, 4, 'router + three specialists, and the one that won was not asked a second time to route');
});

test('a rescue needs a clear winner: when no specialist recognises the input, it escalates', async () => {
  const spec = { ...base, steps: [{ ...base.steps[0], gate: { confidence: 0.999999, familiarity: false } }, { ...base.steps[1], rescue: true }] };
  const r = await run(spec, 'a message about something else entirely');
  assert.equal(r.trace.find(t => t.kind === 'rescue').took, null); assert.equal(r.outcome, 'escalate'); assert.equal(r.answer, null);
});

test('a bank fills independent fields from one encode, each with its own state', async () => {
  const r = await run({ name: 'fields', steps: [{ id: 'f', kind: 'bank', shims: { area: 'area', tone: 'tone' } }] }, msg('account', 'password', 905));
  assert.equal(r.encodes, 1); assert.equal(r.heads, 2); assert.equal(r.results.f.area.answer, 'account'); assert.equal(r.results.f.area.state, 'applied');
  assert.equal(r.results.f.tone.state, 'silent', 'a field that does not recognise the input says nothing rather than guessing');
});

test('confidence along a path multiplies, and a path weaker than the floor is not an answer', async () => {
  // Confidences saturate at 1.0 on the fake encoder, so the floor that cannot be met is one above 1.
  const r = await run({ ...base, minPathConfidence: 1.0000001 }, msg('billing', 'refund', 3));
  assert.equal(r.weakPath, true); assert.equal(r.answer, null); assert.equal(r.outcome, 'escalate');
  const fine = await run({ ...base, minPathConfidence: 0.01 }, msg('billing', 'refund', 3));
  assert.equal(fine.weakPath, false); assert.equal(fine.answer, 'refund');
  assert.ok(Math.abs(fine.pathConfidence - fine.trace[0].confidence * fine.trace[1].confidence) < 1e-3);
});

test('a long message is split once for the whole system, and each shim reads the piece it knows', async () => {
  const filler = k => `${`filler${k} `.repeat(20).trim()}.`, text = [filler(1), `${msg('account', 'lockout', 907)}.`, filler(2), filler(3)].join(' ');
  assert.ok(text.length > LONG_INPUT_CHARS);
  const r = await run(base, text); assert.equal(r.answer, 'lockout'); assert.equal(r.encodes, 4); assert.equal(r.chunks.length, 4);
});

test('wiring that cannot work is refused when the system is built, in words', () => {
  const bad = (steps, re) => assert.throws(() => new System({ name: 'x', steps }, shims), re);
  bad([{ id: 'r', kind: 'route', on: 'later', routes: { a: 'area' } }, { id: 'later', kind: 'shim', shim: 'area' }], /"on" must name an earlier step/);
  bad([{ id: 'a', kind: 'shim', shim: 'no-such-shim' }], /no shim called "no-such-shim"/);
  bad([{ id: 'u', kind: 'rules', rules: [{ name: 'broken', match: '(' }] }], /not a valid pattern/);
  bad([{ id: 'a', kind: 'shim', shim: 'area' }, { id: 'a', kind: 'shim', shim: 'tone' }], /duplicate id/);
  bad([{ id: 'a', kind: 'shim', shim: 'area' }, { id: 'r', kind: 'route', on: 'a', routes: { billing: 'billing' }, rescue: true }], /at least two routes/);
  assert.deepEqual(validateSystem(base, Object.keys(shims)), []); assert.deepEqual(shimsUsed(base).sort(), ['account', 'area', 'billing', 'technical']);
});
