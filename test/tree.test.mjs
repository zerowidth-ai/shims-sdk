// Grouped answers: a root that picks the group, a small head per group that picks the answer.
import test from 'node:test';
import assert from 'node:assert/strict';
import { groupCandidates, learnGroups, fitTree, predictTree, packTree, unpackTree } from '../src/tree.mjs';
import { packReference } from '../src/familiarity.mjs';
import { packHead, unpackHead, DIM } from '../src/format.mjs';
import { place } from './world.mjs';

// Nine answers in three families. An answer's rows sit between its family's centre and its own, so
// siblings are close to each other and far from everyone else — the shape a tree is for.
const FAMILIES = { money: ['refund', 'invoice', 'chargeback'], access: ['password', 'twofactor', 'lockout'], product: ['bug', 'outage', 'feature'] };
const LABELS = Object.values(FAMILIES).flat(), K = LABELS.length, familyOf = l => Object.keys(FAMILIES).find(f => FAMILIES[f].includes(l));
const text = (l, i) => `${familyOf(l)}+${l}~0.3#${i}`;
const rows = (n, from = 0) => LABELS.flatMap(l => Array.from({ length: n }, (_, i) => ({ label: l, text: text(l, from + i) })));
const train = rows(8), X = train.map(r => place(r.text)), y = train.map(r => LABELS.indexOf(r.label));

test('no tree is even tried below eight answers, and the group counts it does try are sane', () => {
  for (let k = 2; k < 8; k++) assert.deepEqual(groupCandidates(k), []);
  for (const k of [8, 16, 53, 150]) for (const g of groupCandidates(k)) assert.ok(g >= 3 && g <= k / 2, `K=${k} G=${g}`);
});

test('learned groups are a partition: every answer in exactly one group, none oversized, same every time', () => {
  for (const G of [3, 4]) {
    const groups = learnGroups(X, y, K, G), flat = groups.flat().sort((a, b) => a - b);
    assert.deepEqual(flat, [...Array(K).keys()]); assert.ok(groups.every(g => g.length <= Math.ceil(2 * K / G)));
    assert.deepEqual(learnGroups(X, y, K, G), groups);
  }
});

test('groups learned from the shim\'s own confusions recover the families nobody told it about', () => {
  const groups = learnGroups(X, y, K, 3).map(g => g.map(i => LABELS[i]));
  for (const g of groups) assert.equal(new Set(g.map(familyOf)).size, 1, `a learned group mixes families: ${g.join(', ')}`);
});

test('with too few rows per answer to measure confusion, it falls back to centroids and still finds the families', () => {
  const few = rows(3), groups = learnGroups(few.map(r => place(r.text)), few.map(r => LABELS.indexOf(r.label)), K, 3).map(g => g.map(i => LABELS[i]));
  for (const g of groups) assert.equal(new Set(g.map(familyOf)).size, 1, g.join(', '));
});

test('a tree\'s answer is a distribution over ALL the answers, and it gets siblings right', () => {
  const groups = Object.values(FAMILIES).map(f => f.map(l => LABELS.indexOf(l))), tree = fitTree(X, y, K, DIM, groups);
  const held = rows(6, 500); let ok = 0;
  for (const r of held) { const p = predictTree(tree, place(r.text)); assert.ok(Math.abs(p.probs.reduce((a, b) => a + b) - 1) < 1e-6); assert.equal(p.confidence, p.probs[p.index]); assert.equal(p.group, tree.groupOf[p.index]); ok += LABELS[p.index] === r.label; }
  assert.ok(ok / held.length > 0.9, `${ok}/${held.length}`);
});

test('a group of one needs no leaf: the root\'s share goes straight to its only answer', () => {
  const groups = [[0, 1, 2], [3, 4, 5], [6, 7], [8]], tree = fitTree(X, y, K, DIM, groups);
  assert.equal(tree.leaves[3], null);
  const p = predictTree(tree, place(text('feature', 900))); assert.equal(LABELS[p.index], 'feature'); assert.ok(p.probs.every(Number.isFinite));
});

test('a packed tree decides exactly as the tree it was packed from', () => {
  const groups = Object.values(FAMILIES).map(f => f.map(l => LABELS.indexOf(l))), tree = fitTree(X, y, K, DIM, groups);
  const back = unpackTree(JSON.parse(JSON.stringify(packTree(tree, packHead))), packReference(X, y), K, unpackHead);
  for (const r of rows(3, 700)) { const a = predictTree(tree, place(r.text)), b = predictTree(back, place(r.text)); assert.equal(a.index, b.index); assert.ok(Math.abs(a.confidence - b.confidence) < 1e-5); }
});
