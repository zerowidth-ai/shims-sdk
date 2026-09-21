// Does the suite catch the bugs it was written for?
//
// A test that has never failed has proved nothing. Each entry below puts one defect that actually
// shipped back into the source, runs the suite, and expects a failure. The "anorak" test passed
// this check by luck on its first draft — it would have passed with or without the bug — and only
// this script showed it. When you fix a bug: add the test, then add the bug here.
//
//   node test/mutations.mjs        (restores every file it touches)
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawnSync } from 'node:child_process';

const PKG = path.resolve(import.meta.dirname, '..');
// [what the bug was, the file, the line as it stands, the line as the bug had it, the test that must catch it]
const MUTATIONS = [
  ['kNN votes weighted by raw cosine', 'src/familiarity.mjs', 'const w = Math.exp((b.c - 1) / KNN_SHARPNESS);', 'const w = Math.max(0, b.c);', /anorak/],
  ['folds train on examples only, not prototypes', 'src/compile/compile.mjs', '    const head = always\n      ? fit(', '    const head = false\n      ? fit(', /folds train on the prototypes/],
  ["isConfident reads the build's gate, not the live one", 'src/index.mjs', 'isConfident(result, threshold = this.gates.confidence) {', 'isConfident(result, threshold = this.report?.suggestedThreshold ?? 0.7) {', /isConfident/],
  ["isConfident reads the build's floor, not the live one", 'src/index.mjs', 'result.familiarity > this.gates.familiarity;', 'result.familiarity > (this.report?.familiarityFloor ?? 0.25);', /isConfident/],
  ['fitHead silently ignores unknown options', 'src/math.mjs', '  if (unknown.length) throw new Error(', '  if (false) throw new Error(', /option it does not know/],
  ['temperature fitted to rows that are all right', 'src/math.mjs', '  if (samples.every(s => top(s.probs) === s.y)) return 1;', '', /no held-out mistakes|all right/],
  ['gate rounded to nearest instead of down', 'src/compile/compile.mjs', 'threshold: Math.floor(kept[kept.length - 1].c * 1000) / 1000,', 'threshold: +kept[kept.length - 1].c.toFixed(3),', /never lands above/],
  ['prototype keys not validated', 'src/format.mjs', '        if (!src.labels?.includes(l)) errs.push(`prototypes: "${l}" is not in labels`);', '', /names an answer it does not have/],
  ['default penalty fixed at 3e-3 whatever the answer count', 'src/math.mjs', 'export const defaultL2 = K => Math.min(3e-3, 0.012 / K);', 'export const defaultL2 = K => 3e-3;', /lighter penalty/],
  ['refitGate invents a gate where the build withheld one', 'src/index.mjs', '    if (!rows?.length) return no(', '    if (false) return no(', /refitGate declines/],
  ['gateRows not shipped', 'src/compile/compile.mjs', '      gateRows: scorable ? calibrated.map(', '      gateRows: false ? calibrated.map(', /gateRows are the rows/],
  ['centroid head chosen at build and never used at runtime', 'src/index.mjs', '    if (this.centroid) return centroidPredict(this.centroid, vec);', '', /head the build chose/],
  ['tree grouping takes the first of several zero-confusion merges', 'src/tree.mjs', ' + 1e-3 * alike[i][j]));', '));', /recover the families/],
  ['pooled long input keeps the best chunk\'s action', 'src/index.mjs', "    return r.chunks > 1 ? { ...r, action: this.actionFor(r.familiarity, r.confidence) } : r;", '    return r;', /action reported is the action/],
  ['a run-on paragraph comes back as one piece longer than the encoder reads', 'src/index.mjs', '    if (c.length <= ENCODER_MAX_CHARS) { safe.push(c); continue; }', '    if (true) { safe.push(c); continue; }', /silent-truncation/],
  ['the long-input decision is the most CONFIDENT chunk, not the most familiar', 'src/reduce.mjs', 'if ((r.familiarity ?? 0) > (results[at].familiarity ?? 0)) at = i;', 'if ((r.confidence ?? 0) > (results[at].confidence ?? 0)) at = i;', /RECOGNISES best/],
  ["a system reads a shim's built gate, not its live one", 'src/system.mjs', ": shim.gates.confidence;", ": (shim.report?.suggestedThreshold ?? 0.7);", /gates its shims are actually running/],
  ["a system reads a shim's built floor, not its live one", 'src/system.mjs', ": shim.gates.familiarity;", ": (shim.report?.familiarityFloor ?? 0.25);", /gates its shims are actually running/],
  ['an always-step that cannot answer escalates a routed message', 'src/system.mjs', "if (next !== 'continue' && !outcome && !step.always) outcome = next;", "if (next !== 'continue' && !outcome) outcome = next;", /always-step annotates/],
  ['a remembered answer is returned with the base shim\'s action', 'src/adapt.mjs', "answer: yes.entry.label, action: 'act',", 'answer: yes.entry.label,', /remembered means act/],
  ['a remembered "no" still says act', 'src/adapt.mjs', "action: no ? 'refuse' : r.action,", 'action: r.action,', /remembered means act/],
  ['an adaptive bank cannot decide long input', 'src/adapt.mjs', '  decideChunks(vectors, chunks, opts = {}) {', '  decideChunksGone(vectors, chunks, opts = {}) {', /adaptive bank decides long input/],
  ['a sibling vote is won without a margin', 'src/system.mjs', 'const took = (top.familiarity >= top.floor && gap >= margin) ? top : null;', 'const took = top;', /needs a clear winner/],
  ['a specialist is evaluated twice when it is voted on and then routed to', 'src/system.mjs', '      if (!seen.has(name)) { seen.set(name, decide(this.shims[name])); heads++; }', '      { seen.set(name, decide(this.shims[name])); heads++; }', /asked only once/],
  ['a bank records nothing', 'src/index.mjs', "out[key] = shim._observed(shim.decideVector(vec), text_, vec, t0, opts, { via: 'bank', field: key });", 'out[key] = shim.decideVector(vec);', /a bank records every shim/],
  ['a system records nothing', 'src/system.mjs', "      if (chunks.length < 2) return shim._observed(shim.decideVector(vectors[0]), text_, vectors[0], t0, opts, { via: 'system', system: this.name });", '      if (chunks.length < 2) return shim.decideVector(vectors[0]);', /a system records every shim/],
  ['a long decision is recorded without the vector it was made on', 'src/index.mjs', '      return this._observed(result, text_, vector, t0, opts);', '      return this._observed(result, text_, null, t0, opts);', /long message is recorded with the vector/],
  ['an adaptive bank records nothing', 'src/adapt.mjs', 'out[key] = { ...shim.observed(shim.decideVector(vec), text_, vec, t0, key), vector: vec };', 'out[key] = { ...shim.decideVector(vec), vector: vec };', /adaptive bank records what was SAID/],
  ['the ring keeps references to vectors the runtime reuses', 'src/observe.mjs', 'vector: e.vector ? new Float32Array(e.vector) : null', 'vector: e.vector ?? null', /ring is bounded, copies vectors/],
  ['a sink that throws takes the decision down with it', 'src/observe.mjs', '  try { sink(entry); } catch { /* a broken sink must never break a decision */ }', '  sink(entry);', /sink that throws never breaks/],
  ['an ignored suggestion is treated as in-scope evidence', 'src/observe.mjs', ": this.for(name).filter(r => r.outcome === 'accepted' || r.outcome === 'corrected');", ': this.for(name).filter(r => r.outcome);', /outcomes attach to the decision/],
  ["a system's answer has no id to report an outcome against", 'src/system.mjs', '             answer: weak ? null : answer, id: decidedBy, results, trace,', '             answer: weak ? null : answer, results, trace,', /a system records every shim/],
  ['an answer with nowhere to route goes unreported at build', 'src/compile/system.mjs', "        if (missing.length && step.onMissing !== 'continue')", '        if (false)', /nowhere to go is an error/],
  ['a draft can be wired into a system', 'src/compile/system.mjs', '      if (c?.draft) errors.push(', '      if (false) errors.push(', /draft cannot be wired in/],
  ['a source with no examples key crashes the compiler', 'src/compile/compile.mjs', '  if (!Array.isArray(src.examples)) src = { ...src, examples: [] };', '', /no examples key at all/],
  ['the reserved schema type dies on a TypeError', 'src/compile/compile.mjs', "  if (type === 'schema') throw new Error(", "  if (false) throw new Error(", /schema type is refused by name/],
  ['pooling a tags decision turns its answer into a string', 'src/reduce.mjs', '  if (!results[at].probs) return pick;', '', /leaves a tags decision a set of tags/],
  ['recalibrate runs on a tags shim', 'src/index.mjs', "    if (this.type === 'tags') return { applied: false, temperature: this._temperature,", "    if (false) return { applied: false, temperature: this._temperature,", /recalibrating a tags shim declines/],
  ['a bank step with no shims takes the system check down', 'src/compile/system.mjs', "Object.keys(s.shims ?? {}).length", "Object.keys(s.shims).length", /bank step with no shims is an error in the report/],
];

// A package built on this one can check its own bugs in the same run: pass modules that default-export
// `{ root, mutations }`. Its entries name files relative to `root`, and `root`'s suite runs beside this one.
const extras = await Promise.all(process.argv.slice(2).map(async f => (await import(pathToFileURL(path.resolve(f)).href)).default));
const ENTRIES = [...MUTATIONS.map(m => [PKG, ...m]), ...extras.flatMap(e => e.mutations.map(m => [e.root, ...m]))];
const SUITES = [PKG, ...extras.map(e => e.root)];
// The reporter is named because its output is what gets parsed. Left to its default, Node 22 prints
// TAP when stdout is a pipe and Node 24 prints spec, and under TAP every mutation reads as MISSED:
// the suite "failed" nothing this script could see. `seen` is the guard for the next time the format
// moves: no test lines at all is an error, not a verdict.
let seen = 0;
const failures = () => SUITES.flatMap(dir => {
  const out = spawnSync('node', ['--test', '--test-reporter=spec', 'test/*.test.mjs'], { cwd: dir, encoding: 'utf8' }).stdout;
  seen += (out.match(/^[✔✖] /gm) ?? []).length;
  return [...new Set([...out.matchAll(/^✖ (.+?) \(\d/gm)].map(m => m[1]))];
});

const clean = failures();
if (!seen) { console.error('no test results could be read from the runner\'s output — has the reporter format changed?'); process.exit(1); }
if (clean.length) { console.error(`the suite is not green to begin with:\n  ${clean.join('\n  ')}`); process.exit(1); }
let bad = 0;
for (const [root, name, file, from, to, expected] of ENTRIES) {
  const full = path.join(root, file);
  const src = fs.existsSync(full) ? fs.readFileSync(full, 'utf8') : '';
  if (!src.includes(from)) { console.log(`  STALE   ${name} — the line it replaces is gone; update this entry`); bad++; continue; }
  fs.writeFileSync(full, src.replace(from, to));
  let failed; try { failed = failures(); } finally { fs.writeFileSync(full, src); }
  const byItsOwnTest = failed.some(f => expected.test(f));
  if (!byItsOwnTest) bad++;
  console.log(`  ${byItsOwnTest ? 'caught ' : failed.length ? 'INDIRECT' : 'MISSED '} ${name}${byItsOwnTest ? '' : failed.length ? ` — only by: ${failed[0]}` : ''}`);
}
console.log(`\n${ENTRIES.length - bad}/${ENTRIES.length} caught by the test written for them`);
process.exit(bad ? 1 : 0);
