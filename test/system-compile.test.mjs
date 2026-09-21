// The build-time check of a system's wiring: the boring mistakes — a renamed answer with nowhere to
// go, a shim that is still a draft — caught by the build rather than by a user.
import test from 'node:test';
import assert from 'node:assert/strict';
import { compileSystem } from '@zerowidth/shims-sdk/compile/system';

const shim = (name, labels, report = {}) => ({ name, labels, head: 'linear', report: { suggestedThreshold: 0.7, familiarityFloor: 0.05, bytes: 1000, referenceBytes: 4000, ...report } });
const area = shim('area', ['billing', 'technical', 'feature-request']), billing = shim('billing', ['refund', 'invoice']), technical = shim('technical', ['bug', 'outage']);
const refundWhy = shim('refund-why', ['duplicate', 'unhappy']), all = [area, billing, technical, refundWhy];
const route = (extra = {}) => ({ id: 'reason', kind: 'route', on: 'area', routes: { billing: 'billing', technical: 'technical', 'feature-request': null }, ...extra });
const spec = (...steps) => ({ name: 'support', steps: [{ id: 'area', kind: 'shim', shim: 'area' }, ...steps] });

test('sound wiring compiles clean, and says what it will cost to run', () => {
  const r = compileSystem(spec(route()), all);
  assert.deepEqual(r.errors, []); assert.deepEqual(r.notes, []); assert.equal(r.headsPerInput, 2);
  assert.deepEqual(r.shims.map(s => s.name), ['area', 'billing', 'technical']); assert.equal(r.bytes, 3 * 5000, 'heads AND the examples they ship with');
});

test('an answer with nowhere to go is an error that names it — unless the step says carrying on is fine', () => {
  const partial = { billing: 'billing', technical: 'technical' };
  assert.match(compileSystem(spec(route({ routes: partial })), all).errors.join(), /"area" can answer feature-request with nowhere to route it/);
  assert.deepEqual(compileSystem(spec(route({ routes: partial, onMissing: 'continue' })), all).errors, []);
});

test('a route for an answer the shim never gives is an error: that is what a renamed answer looks like', () => {
  assert.match(compileSystem(spec(route({ routes: { ...route().routes, refunds: 'billing' } })), all).errors.join(), /routes refunds, which "area" never answers/);
});

test('a chain of routes is checked against everything the step above could have answered', () => {
  const deeper = { id: 'why', kind: 'route', on: 'reason', onMissing: 'continue', routes: { refund: 'refund-why' } };
  assert.deepEqual(compileSystem(spec(route(), deeper), all).errors, []);
  assert.match(compileSystem(spec(route(), { ...deeper, routes: { refund: 'refund-why', chargeback: 'refund-why' } }), all).errors.join(), /routes chargeback, which "reason" never answers/);
});

test('a draft cannot be wired in; a provisional shim and one that never earns a gate can, with a warning each', () => {
  const draft = { name: 'billing', labels: ['refund', 'invoice'], draft: true };
  assert.match(compileSystem(spec(route()), [area, draft, technical]).errors.join(), /"billing" is still a draft/);
  const notes = compileSystem(spec(route()), [area, shim('billing', ['refund', 'invoice'], { provisional: true }), shim('technical', ['bug', 'outage'], { suggestedThreshold: 1 })]).notes;
  assert.deepEqual(notes.map(n => n.code).sort(), ['no-threshold', 'provisional-step']); assert.ok(notes.every(n => n.level === 'warn'));
});

test('a rescue vote that leaves an answer without a candidate is said out loud', () => {
  const r = compileSystem(spec(route({ rescue: true })), all);
  assert.deepEqual(r.errors, []); assert.equal(r.notes[0].code, 'partial-ballot'); assert.match(r.notes[0].text, /asking 2 of its 3 answers — feature-request has no specialist/);
});

test('a bank counts every field it fills, and rules cost no heads', () => {
  const r = compileSystem({ name: 'form', steps: [{ id: 'u', kind: 'rules', rules: [{ name: 'x', match: 'y' }] }, { id: 'f', kind: 'bank', shims: { a: 'area', b: 'billing' } }] }, all);
  assert.deepEqual(r.errors, []); assert.equal(r.headsPerInput, 2);
});

test('2026-09-21 · a bank step with no shims is an error in the report, not a crash', () => {
  // validateSystem already says `needs "shims"`. The head count then read Object.keys(undefined) and threw,
  // so the CLI printed a stack trace where the error it had just found should have been.
  const r = compileSystem({ name: 'form', steps: [{ id: 'f', kind: 'bank' }] }, all);
  assert.match(r.errors.join('\n'), /step 0 \(f\): needs "shims"/); assert.equal(r.headsPerInput, 0);
});
