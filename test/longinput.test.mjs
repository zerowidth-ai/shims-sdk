// Input longer than a sentence: how it is cut up, and how the pieces become one decision.
import test from 'node:test';
import assert from 'node:assert/strict';
import { splitForDecision, LONG_INPUT_CHARS, CHUNK_CHARS, ENCODER_MAX_CHARS } from '../src/index.mjs';
import { reduceChunks } from '../src/reduce.mjs';

const sentence = n => `${'word '.repeat(Math.ceil(n / 5)).trim().slice(0, n - 1)}.`;

test('splitting loses nothing: every word of the input is in exactly one piece, in order', () => {
  const text = [sentence(90), sentence(140), 'A short one.', sentence(200), 'and a trailing fragment with no full stop'].join(' ');
  assert.equal(splitForDecision(text).join(' ').replace(/\s+/g, ' '), text.replace(/\s+/g, ' '));
});

test('no piece is ever longer than the encoder can read — the silent-truncation bug', () => {
  // A 1,200-character paragraph with no full stop used to come back as ONE piece and be truncated to a third of itself.
  const runOn = 'word '.repeat(600).trim();
  assert.ok(runOn.length > ENCODER_MAX_CHARS);
  const pieces = splitForDecision(runOn); assert.ok(pieces.length > 1); assert.ok(pieces.every(p => p.length <= ENCODER_MAX_CHARS));
  assert.equal(pieces.join(' '), runOn);
});

test('short input is left alone, and sentences are packed into pieces of about the target size', () => {
  assert.deepEqual(splitForDecision('black tie wedding guest dress'), ['black tie wedding guest dress']);
  const pieces = splitForDecision(Array.from({ length: 12 }, () => sentence(60)).join(' '));
  assert.ok(pieces.length >= 4 && pieces.every(p => p.length <= CHUNK_CHARS + 60), pieces.map(p => p.length).join(','));
  assert.ok(CHUNK_CHARS < LONG_INPUT_CHARS);
});

const result = (answer, confidence, familiarity, action = 'act') => ({ answer, confidence, familiarity, action, probs: { a: answer === 'a' ? confidence : 1 - confidence, b: answer === 'b' ? confidence : 1 - confidence } });

test('best: the decision is the chunk the shim RECOGNISES best, not the one it is surest about', () => {
  // A forced-choice head is confidently wrong on filler it has never seen. Familiarity asks the right question.
  const r = reduceChunks([result('a', 0.99, 0.02), result('b', 0.7, 0.9), result('a', 0.95, 0.1)], ['filler', 'the point', 'more filler'], ['a', 'b']);
  assert.equal(r.answer, 'b'); assert.equal(r.from, 'the point'); assert.equal(r.chunks, 3); assert.equal(r.confidence, 0.7);
});

test('one chunk is passed through untouched', () => {
  const only = result('a', 0.8, 0.5); assert.deepEqual(reduceChunks([only], ['x'], ['a', 'b']), { ...only, from: 'x', chunks: 1 });
});

test('pooled: familiar chunks outvote unfamiliar ones, and familiarity stays the best chunk\'s', () => {
  const r = reduceChunks([result('a', 0.9, 0.05), result('b', 0.8, 0.9), result('b', 0.7, 0.8)], ['filler', 'x', 'y'], ['a', 'b'], { mode: 'pooled' });
  assert.equal(r.answer, 'b'); assert.equal(r.familiarity, 0.9); assert.ok(Math.abs(r.probs.a + r.probs.b - 1) < 1e-9); assert.equal(r.confidence, r.probs.b);
});

test('pooled: when nothing is familiar there is nothing to pool, so it falls back to the single best chunk', () => {
  const r = reduceChunks([result('a', 0.9, 0), result('b', 0.6, 0)], ['x', 'y'], ['a', 'b'], { mode: 'pooled' });
  assert.equal(r.answer, 'a'); assert.equal(r.from, 'x');
});
