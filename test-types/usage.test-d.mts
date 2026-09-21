// Type tests: a consumer-style file that exercises every README sample against the hand-written
// declarations. Never run — `tsc -p test-types/tsconfig.json` must exit 0.
// A `@ts-expect-error` line is a promise that the types REJECT what follows it.

import {
  Shim, Bank, observe, observing, outcome, ring, preload, setEmbedder, currentEmbedder, encode, embed,
  splitForDecision, reduceChunks, withSurface, surfaceFeatures, headBytes,
  DIM, EMBED_DIM, ENCODER_ID, SURFACE_DIM, LONG_INPUT_CHARS, CHUNK_CHARS, ENCODER_MAX_CHARS, ENCODER_MAX_TOKENS,
} from '../src/index.mjs';
import type {
  Action, ClassifyDecision, Decision, Embedder, ShimSource, ShimWeights, ShimWeightsInput,
  TagsDecision, DecisionRow, Outcome, RefitGateResult, Vector,
} from '../src/index.mjs';
import { System, validateSystem, shimsUsed } from '../src/system.mjs';
import type { SystemSpec, SystemSpecInput, StepResult } from '../src/system.mjs';
import { AdaptiveShim, AdaptiveBank, MEMORY_MATCH } from '../src/adapt.mjs';
import { record, nextId } from '../src/observe.mjs';
import {
  fitHead, fitHeadGD, fitCentroids, centroidPredict, fitTagHeads, softmax, balanceWeights, rng, shuffled,
  applyTemperature, fitTemperature, calibrationError, estimatePrior, weightedGate, defaultL2, CENTROID_SCALE,
} from '../src/math.mjs';
import {
  packHead, unpackHead, validateSource, readiness, prototypeWeight, vectorCacheFile,
  FORMAT_VERSION, MIN_PER_ANSWER, PROTOTYPE_PARITY, PROTOTYPE_FLOOR,
} from '../src/format.mjs';
import { cosine, packReference, unpackReference, familiarityScore, knnPredict, KNN_SHARPNESS } from '../src/familiarity.mjs';
import { groupCandidates, learnGroups, fitTree, predictTree, packTree, unpackTree } from '../src/tree.mjs';
import { SCALE } from '../src/surface.mjs';
import { compileShim } from '../src/compile/compile.mjs';
import type { CompiledShim } from '../src/compile/compile.mjs';
import { compileSystem } from '../src/compile/system.mjs';

declare function route(answer: string): void;
declare function offer(answer: string): void;
declare function askAPerson(): void;
declare const userWentWithIt: boolean;
declare const userPickedOther: boolean;
declare const theirChoice: string;
declare const message: string;

/* ---------- a weights file as `import urgency from './urgency.weights.json'` types it ---------- */
// Not `as const`: every literal widens (format: number, type: string, head: string), which is
// exactly what resolveJsonModule hands a consumer.
const urgency = {
  format: 2, name: 'urgency', type: 'classify', question: 'How quickly does this need attention?',
  labels: ['now', 'soon', 'whenever'], encoder: 'Xenova/bge-small-en-v1.5', dim: 392,
  compiledAt: '2026-09-19T00:00:00.000Z', examples: 24, embedded: 24, head: 'linear',
  headChoice: {
    linear: 0.88, knn: 0.8, centroid: 0.85, coverage: { centroid: 0.4, linear: 0.5, knn: 0.3 }, coverageOverride: null,
    l2: { '0.003': 0.88 }, l2Chosen: 0.003, perAnswer: [{ label: 'now', linear: 0.9, knn: 0.8 }],
  },
  heads: [{ K: 3, W: '', b: '' }], l2: 0.003,
  report: {
    accuracy: 0.88, accuracyInterval: [0.69, 0.96], macroRecall: 0.88, confusion: [[8, 0, 0], [1, 7, 0], [0, 2, 6]],
    suggestedThreshold: 0.71, coverageAtThreshold: 0.6, evaluated: 24, perExample: [{ pred: 0, conf: 0.9, ok: true }],
    scorable: true, provisional: false, provisionalThreshold: null,
    calibration: { temperature: 1, eceBefore: 0.1, eceAfter: 0.1, fittedTemperature: 0.9, eceIfScaled: 0.12, applied: false, samples: 24 },
    gateRows: [[0, 0.91, 1], [1, 0.62, 0]], familiarityFloor: 0.05, familiarityFittedOn: 24, floorFittedOn: 'examples',
    chance: 0.3333, labels: ['now', 'soon', 'whenever'], recall: { now: 1, soon: 0.875, whenever: 0.75 },
    bytes: 4716, head: 'linear', structure: { chosen: 'flat', tried: [], reason: '3 answers is too few for grouping to matter' },
    prototypes: 0, prototypeWeight: 0.25, scoredOn: 'examples', notes: [{ level: 'info', code: 'few-examples', text: '…' }],
    referenceBytes: 9504,
  },
  reference: { q: '', scales: '', spread: [0.5, 0.6], count: 24, labels: [0, 1, 2] },
};
const tone = { ...urgency, name: 'tone', labels: ['warm', 'neutral', 'cold'] };
const halfWritten = { format: 2, name: 'wip', type: 'classify', labels: ['a', 'b'], draft: true, examples: 1 };

/* ---------- sdk/README.md: decide, then branch on the action ---------- */
const shim = new Shim<'classify'>(urgency);
const r = await shim.decide('the whole site is down for us');
if (r.action === 'act') route(r.answer);
else if (r.action === 'suggest') offer(r.answer);
else askAPerson();

const answer: string = r.answer;
const confidence: number = r.confidence;
const familiarity: number | null = r.familiarity;
const pNow: number = r.probs['now']!;
const rawConfidence: number = r.rawConfidence;
const maybeId: string | undefined = r.id;
const act: Action = r.action;
if (r.path) { const siblings: string[] = r.path.siblings; void siblings; }

// @ts-expect-error — 'maybe' is not an action
if (r.action === 'maybe') askAPerson();
// @ts-expect-error — a classify answer is one label, not a list
const wrongAnswer: string[] = r.answer;
// @ts-expect-error — weights need their answers
new Shim({ format: 2, name: 'x', type: 'classify', heads: [] });

// Without the assertion the decision is the union, and narrows on the answer.
const untyped = new Shim(urgency);
const u = await untyped.decide(message, { long: false, reduce: 'pooled', reducePower: 2, keepText: false });
if (typeof u.answer === 'string') route(u.answer); else u.answer.forEach(route);
const asUnion: Decision = u;
const tagShim = new Shim<'tags'>(urgency);
const t: TagsDecision = tagShim.decideVector(new Float32Array(DIM));
const tagScores: Record<string, number> = t.scores;
// @ts-expect-error — a tags decision has scores, not probs
t.probs;

/* ---------- no encoder wiring; preload() and setEmbedder() for control ---------- */
await preload();
await preload({ device: 'webgpu' });
await preload({                                            // "Where the encoder comes from": self-hosted model and runtime
  remoteHost: 'https://static.example.com/',
  remotePathTemplate: 'models/{model}/',
  wasmPaths: 'https://static.example.com/ort/',
});
// @ts-expect-error — a host is a URL string
await preload({ remoteHost: 443 });
const fake = (text: string): Float32Array => withSurface(new Float32Array(EMBED_DIM).fill(text.length / 100), text);
const fakeEmbedder: Embedder = async texts => Array.isArray(texts) ? texts.map(fake) : fake(texts);
setEmbedder(fakeEmbedder);
setEmbedder((texts: string | string[]) => Array.isArray(texts) ? texts.map(x => Array.from(fake(x))) : Array.from(fake(texts)));   // sync, number[]
setEmbedder(null);
// @ts-expect-error — an embedder takes text, not numbers
setEmbedder((n: number) => new Float32Array(n));
const inPlay: Embedder = currentEmbedder();
await inPlay.warm?.();
const one: Vector = await encode('hello');
const many: Vector[] = await encode(['hello', 'there']);
const real: Float32Array = await embed('hello', { dtype: 'q8' });
const reals: Float32Array[] = await embed(['hello']);
const pieces: string[] = splitForDecision(message, CHUNK_CHARS);
const sizes: number[] = [DIM, EMBED_DIM, SURFACE_DIM, LONG_INPUT_CHARS, ENCODER_MAX_CHARS, ENCODER_MAX_TOKENS, headBytes(3), headBytes(3, DIM), SCALE];
const encoderId: string = ENCODER_ID;
const features: number[] = surfaceFeatures('WHY?!');

/* ---------- docs/runtime.md: a Bank runs the encoder once ---------- */
const bank = new Bank({ urgency, tone, halfWritten, loaded: shim });
const out = await bank.decide(message);
const urgencyAnswer: string | string[] = out.urgency.answer;
const loadedAnswer: string = out.loaded.answer;            // a Shim<'classify'> keeps its decision type through the bank
const toneConfidence: number = out.tone.confidence;
const bankBytes: number = bank.bytes;
const bankShim: Shim = bank.shims.urgency;
// @ts-expect-error — no such shim in this bank
out.sentiment;

/* ---------- watching real traffic ---------- */
const log = ring(2000);
observe(log.push);
const watched = await shim.decide(message);                // watched.id exists only while something is observing
if (userWentWithIt) outcome(watched.id, 'accepted');
if (userPickedOther) outcome(watched.id, 'corrected', { label: theirChoice });
const summary = log.summary();
const counts: number[] = [summary.decisions, summary.acted, summary.suggested, summary.refused, summary.withOutcome];
const medians: (number | null)[] = [summary.medianWords, summary.medianMs];
const forUrgency = log.summary('urgency');
const refusedRows = log.for('urgency', { action: 'refuse', acted: false, outcome: 'ignored' });
const firstVector: Float32Array | null | undefined = refusedRows[0]?.vector;
const allOutcomes: Outcome[] = ['accepted', 'corrected', 'rejected', 'ignored'];
const isWatching: boolean = observing();
observe(entry => { if (entry.kind === 'decision') { const row: DecisionRow = entry; void row.gates.confidence; } else { void entry.outcome; } });
observe(null);
record({ kind: 'outcome', id: nextId(), outcome: 'ignored', label: null, at: Date.now() });
log.clear();

// @ts-expect-error — not one of the four outcomes
outcome(watched.id, 'liked');
// @ts-expect-error — the summary has no such field
summary.accuracy;
// @ts-expect-error — a filter's action is an Action
log.for('urgency', { action: 'escalate' });

/* ---------- refitFloor / refitGate / recalibrate ---------- */
const floor = shim.refitFloor(log.vectorsFor('urgency'), { target: 0.05, minSamples: 20, floor: 0.005 });
if (floor.applied) { const shares: number[] = [floor.refusedBefore, floor.refusedAfter, floor.before, floor.after]; void shares; }
else { const why: string = floor.why; void why; }
// @ts-expect-error — only an applied refit reports what it refused
floor.refusedBefore;

const gate: RefitGateResult = shim.refitGate(log.vectorsFor('urgency', { acted: true }), { target: 0.9 });
if (gate.applied) { const mix: Record<string, number> = gate.mix; void [mix, gate.actsOnBefore, gate.actsOnAfter]; } else void gate.why;

const recal = shim.recalibrate(log.outcomesFor('urgency'), { minSamples: 30 });
const recalNumbers: (number | null | undefined)[] = [recal.temperature, recal.before, recal.after, recal.samples, recal.fitted];
const recalFlags: [boolean, string | undefined] = [recal.applied, recal.why];
shim.recalibrate([{ vector: [0.1, 0.2], label: 'now' }, null]);
const gates: { confidence: number; familiarity: number } = shim.gates;
const safe: boolean = shim.isConfident(r) && shim.isConfident(r, 0.95) && shim.isConfident({ confidence: 0.9 });
const temperature: number = shim.temperature;
const groups: string[][] | null = shim.groups;
const threshold: number = shim.report.suggestedThreshold;
const again: Action = shim.actionFor(null, 0.9);
const probsByIndex: number[] = shim.rawProbs(one);
const long = shim.decideChunks(many, ['a', 'b'], { mode: 'pooled', power: 2 });
const chosenPiece: string = long.from;
const loadedShim: Shim = Shim.load(urgency);
// @ts-expect-error — the floor is read-only through `gates`; refit it instead
shim.gates = { confidence: 0.5, familiarity: 0.1 };
// @ts-expect-error — internals are not part of the API
shim._gate = 0.5;

/* ---------- strict weights are valid input too ---------- */
declare const compiledWeights: ShimWeights;
const asInput: ShimWeightsInput = compiledWeights;
new Shim(compiledWeights);
new Bank({ a: compiledWeights });
if (compiledWeights.type === 'classify') {
  const interval: [number, number] | null = compiledWeights.report.accuracyInterval;
  const centroid = compiledWeights.centroid ? unpackHead(compiledWeights.centroid) : null;
  void [interval, centroid, compiledWeights.headChoice.l2Chosen];
} else {
  const perTag: (number | null)[] = compiledWeights.report.perTag.map(p => p.macroRecall);
  void perTag;
  // @ts-expect-error — a tags report has no single accuracy
  compiledWeights.report.accuracy;
}

/* ---------- systems ---------- */
const spec: SystemSpec = {
  name: 'support', type: 'system', minPathConfidence: 0.5, onWeakPath: 'triage', reduce: 'best',
  input: 'a message a customer sends to support', outcomes: { routed: 'goes to that queue', triage: 'a person reads it' },
  steps: [
    { id: 'urgent', kind: 'rules', outcome: 'urgent', rules: [{ name: 'outage', match: '\\b(down|outage)\\b' }] },
    { id: 'area', kind: 'shim', shim: 'support-area', onUnsure: 'triage', gate: { confidence: 0.8, familiarity: false } },
    { id: 'specialist', kind: 'route', on: 'area', rescue: { margin: 0.25 }, endsAt: 'the product board', onMissing: 'continue',
      routes: { billing: 'support-billing', technical: 'support-technical', 'feature-request': null } },
    { id: 'notes', kind: 'bank', always: true, shims: { urgency: 'urgency', tone: 'tone' } },
  ],
};
const specFromJson = {                                     // kind: string, as a JSON import types it
  name: 'support', type: 'system',
  steps: [{ id: 'area', kind: 'shim', shim: 'urgency' }, { id: 'notes', kind: 'bank', shims: { tone: 'tone' }, always: true }],
};
const asSpecInput: SystemSpecInput = spec;
const problems: string[] = [...validateSystem(spec, ['urgency']), ...validateSystem(JSON.parse('{}'))];
const toLoad: string[] = [...shimsUsed(spec), ...shimsUsed(specFromJson)];
const system = new System(specFromJson, { urgency: shim, tone: new Shim(tone) });
new System(spec, { urgency: shim });
const ran = await system.run(message, { keepText: false });
const replay = await system.run(message, { chunks: pieces, vectors: many, reduce: 'pooled' });
if (ran.outcome === 'routed' && typeof ran.answer === 'string') route(ran.answer);
outcome(ran.id ?? undefined, 'accepted');
const pathNumbers: number[] = [ran.pathConfidence, ran.heads, ran.encodes];
const weak: boolean = ran.weakPath;
const areaResult = ran.results['area'];
if (areaResult && 'state' in areaResult && typeof areaResult.state === 'string') { const step = areaResult as StepResult; void [step.shim, step.rescuedBy]; }
for (const line of ran.trace) {
  if (line.kind === 'rules') void line.matched;
  else if (line.kind === 'bank') void line.fields;
  else if (line.kind === 'rescue') void [line.votes, line.took, line.margin];
  else if ('shim' in line) void [line.shim, line.confidence, line.state];
}
const gatesForStep: { conf: number; fam: number } = system.gatesFor({ gate: { confidence: false } }, shim);
// @ts-expect-error — a step kind the runtime does not have
const badSpec: SystemSpec = { name: 'x', steps: [{ id: 'a', kind: 'llm', shim: 'urgency' }] };
// @ts-expect-error — a system takes loaded shims, not raw weights
new System(spec, { urgency });

/* ---------- learning from use ---------- */
const adaptive = new AdaptiveShim(urgency, { match: MEMORY_MATCH });
const fromLoaded = new AdaptiveShim(shim);
const said = adaptive.decideVector(one);
const shipped: { answer: string; confidence: number; action: Action } = said.shipped;
if (said.remembered) void [said.remembered.text, said.remembered.similarity, said.remembered.id];
const entry = adaptive.record({ text: message, vector: one, label: 'now', verdict: 'yes' });
adaptive.forget(entry.id);
const stored = adaptive.export();
await adaptive.restore(stored);
await adaptive.restore(stored, fakeEmbedder);
const adaptiveRecal = adaptive.recalibrate({ minSamples: 10 });
const near = adaptive.nearest(one, 'no', 'now');
adaptive.reset();
const adaptiveBank = new AdaptiveBank({ urgency, tone: fromLoaded }, { match: 0.85 });
const adaptiveOut = await adaptiveBank.decide(message);
adaptive.record({ text: message, vector: adaptiveOut.urgency.vector, label: adaptiveOut.urgency.answer, verdict: 'no' });
// @ts-expect-error — a verdict is yes or no
adaptive.record({ text: message, label: 'now', verdict: 'maybe' });
// @ts-expect-error — stored feedback carries no vector
stored[0]!.vector;

/* ---------- authoring and compiling ---------- */
const source: ShimSource = {
  name: 'urgency', type: 'classify', question: 'How quickly does this need attention?',
  labels: ['now', 'soon', 'whenever'],
  examples: [
    { text: 'Production is down, every request returns 503.', label: 'now' },
    { text: 'Typo on the pricing page.', label: 'whenever' },
  ],
  prototypes: { soon: ['Can someone look at this before Friday?'] },
  tree: 'auto', context: 'Messages a customer sends to a B2B software support desk.', realSample: ['my invoice is wrong'],
};
const built = await compileShim(source, { embed: fakeEmbedder, coverageTiebreak: false });
if (built.draft) {
  const needed: number = built.report.needed;
  void [needed, built.report.counts, built.report.notes[0]?.text];
} else {
  const interval: [number, number] | null = built.report.accuracyInterval;
  const low: number | undefined = built.report.accuracyInterval?.[0];
  const errors = built.report.notes.filter(n => n.level === 'error').map(n => `${n.code}: ${n.text}`);
  const report: number[] = [built.report.accuracy, built.report.macroRecall, built.report.suggestedThreshold,
    built.report.coverageAtThreshold, built.report.evaluated, built.report.chance, built.report.bytes, built.report.referenceBytes];
  const recallNow: number | null | undefined = built.report.recall['now'];
  const structure: 'tree' | 'flat' = built.report.structure.chosen;
  const fresh = new Shim<'classify'>(built);
  void [interval, low, errors, report, recallNow, structure, fresh, built.report.confusion[0]?.[0], built.report.provisional, built.headChoice.coverage.knn];
}
// docs/compiler.md, as written.
const weights = await compileShim(source, { cacheDir: '.shim-cache' });
if (weights.draft) console.log(`needs ${weights.report.needed} more examples`);
else console.log(weights.report.accuracy, weights.report.accuracyInterval);

const tagsBuilt = await compileShim({ name: 'risk', type: 'tags', labels: ['a', 'b'], examples: [{ text: 'x', labels: ['a'] }, { text: 'y', labels: [] }] }, { cacheDir: '.shim-cache' });
if (!tagsBuilt.draft) void tagsBuilt.report.perTag[0]?.n;
const sourceFromJson = { name: 'urgency', type: 'classify', labels: ['now', 'soon'], examples: [{ text: 'x', label: 'now' }] };
const loose: CompiledShim = await compileShim(sourceFromJson);
const sourceProblems: string[] = validateSource(sourceFromJson);
const ready = readiness(source);
const readyNumbers: [boolean, boolean, number, string[]] = [ready.ready, ready.described, ready.needed, ready.short];

// @ts-expect-error — a source needs its answers
const noLabels: ShimSource = { name: 'x', type: 'classify', examples: [] };
// @ts-expect-error — 'regress' is not a shim type
const badType: ShimSource = { name: 'x', type: 'regress', labels: ['a', 'b'], examples: [] };
// @ts-expect-error — a classify example carries one label
const badExample: ShimSource = { name: 'x', type: 'classify', labels: ['a', 'b'], examples: [{ text: 't', labels: ['a'] }] };
// @ts-expect-error — compileShim needs a name
await compileShim({ labels: ['a', 'b'], examples: [] });
// @ts-expect-error — no such option
await compileShim(source, { epochs: 10 });

const checked = compileSystem(spec, [built, loose, compiledWeights]);
const verdict: [string[], number, number] = [checked.errors, checked.headsPerInput, checked.bytes];
for (const note of checked.notes) if (note.level === 'warn') void note.code;

/* ---------- the subpath modules ---------- */
const X = [new Float32Array(DIM), new Float32Array(DIM), [0.1, 0.2]], y = [0, 1, 0];
const head = fitHead(X, y, 2, DIM, { l2: defaultL2(2), weights: balanceWeights(y, 2), maxIter: 100, tol: 1e-5 });
const oldHead = fitHeadGD(X, y, 2, DIM, { epochs: 10, lr: 3 });
const means = fitCentroids(X, y, 2, DIM, { weights: new Float32Array(3) });
const tagHeads = fitTagHeads(X, [new Set([0]), new Set<number>(), new Set([0, 1])], 2, DIM, { rowWeight: [1, 1, 0.25] });
const prediction = softmax(head, X[0]!);
const centroidGuess: number = centroidPredict(means, X[1]!).confidence + CENTROID_SCALE;
const cooled: number[] = applyTemperature(prediction.probs, fitTemperature([{ probs: prediction.probs, y: 0 }]));
const ece: number | null = calibrationError([{ confidence: prediction.confidence, ok: true }], 10);
const prior: number[] = estimatePrior([prediction.probs], [0.5, 0.5], { iters: 50 });
const readGate: { threshold: number; coverage: number } = weightedGate([{ c: 0.9, ok: 1, w: 1 }, { c: 0.4, ok: false, w: 2 }], 0.9);
const order: number[] = shuffled([1, 2, 3], 7);
const draw: number = rng(7)();
const packed = packHead(head);
const roundTrip: Float32Array = unpackHead(packed, DIM).W;
const formatConsts: number[] = [FORMAT_VERSION, MIN_PER_ANSWER, PROTOTYPE_PARITY, PROTOTYPE_FLOOR, prototypeWeight(4), KNN_SHARPNESS];
const cacheName: string = vectorCacheFile();
const reference = packReference(X, y);
const quantised: Int8Array = reference.q;
const fam: number | null = familiarityScore(X[0]!, reference);
const knn = knnPredict(X[0]!, reference, 2);
const knnIndex: number | undefined = knn?.index;
const similarity: number = cosine(X[0]!, X[1]!, DIM);
const unpacked = unpackReference(compiledWeights.reference);
const candidateCounts: number[] = groupCandidates(16);
const learned: number[][] = learnGroups(X, y, 2, 2);
const tree = fitTree(X, y, 2, DIM, learned, { weights: [1, 1, 1] });
const treeGuess = predictTree(tree, X[0]!);
const treeNumbers: number[] = [treeGuess.index, treeGuess.confidence, treeGuess.group, treeGuess.groupConfidence, treeGuess.rootTop];
const packedTree = packTree(tree, packHead);
if (unpacked?.labels) { const rebuilt = unpackTree(packedTree, { ...unpacked, labels: unpacked.labels }, 2, h => unpackHead(h)); void rebuilt.groupOf; }
const reduced = reduceChunks([said, said], ['a', 'b'], ['now', 'soon', 'whenever'], { mode: 'best' });
const reducedFrom: string = reduced.from;

// @ts-expect-error — the old optimiser's knobs belong to fitHeadGD
fitHead(X, y, 2, DIM, { epochs: 250 });
// @ts-expect-error — FORMAT_VERSION is 2
const three: 3 = FORMAT_VERSION;
const plainHead: number = softmax({ W: [0], b: [0], K: 1, dim: 1 }, X[0]!).confidence;   // predictors only index
// @ts-expect-error — packing reads the float32 bytes, so a plain array will not do
packHead({ W: [0], b: [0], K: 1 });

void [answer, confidence, familiarity, pNow, rawConfidence, maybeId, act, wrongAnswer, asUnion, tagScores, inPlay, real, reals,
  sizes, encoderId, features, urgencyAnswer, loadedAnswer, toneConfidence, bankBytes, bankShim, counts, medians, forUrgency,
  firstVector, allOutcomes, isWatching, recalNumbers, recalFlags, gates, safe, temperature, groups, threshold, again, probsByIndex,
  chosenPiece, loadedShim, asInput, asSpecInput, problems, toLoad, replay, pathNumbers, weak, gatesForStep, badSpec, shipped,
  adaptiveRecal, near, sourceProblems, readyNumbers, noLabels, badType, badExample, verdict, oldHead, tagHeads, centroidGuess,
  cooled, ece, prior, readGate, order, draw, roundTrip, formatConsts, cacheName, quantised, fam, knnIndex, similarity,
  candidateCounts, treeNumbers, reducedFrom, three, plainHead];
export type { ClassifyDecision };
