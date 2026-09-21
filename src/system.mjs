// Several shims wired into one decision: a router handing off to specialists, rules that run
// before any model, a bank of independent fields, a person at the end.
//
// Wired by hand in application code, the interesting part — the wiring — cannot be reviewed,
// compiled, or reused. A system is that wiring as data: which
// shim asks first, what each of its answers hands to, where the gates sit, and what happens when
// nothing is confident. It ships next to the shims it names.
//
//   { "name": "support", "type": "system",
//     "steps": [
//       { "id": "urgent",     "kind": "rules", "outcome": "urgent",
//         "rules": [{ "name": "outage", "match": "\\b(down|outage)\\b" }] },
//       { "id": "area",       "kind": "shim",  "shim": "support-area" },
//       { "id": "specialist", "kind": "route", "on": "area",
//         "routes": { "billing": "support-billing", "feature-request": null } } ] }
//
// Gates default to what each shim earned at build: its fitted confidence threshold and its fitted
// familiarity floor. A step can override either, or say `false` to let everything through.
import { splitForDecision, LONG_INPUT_CHARS, currentEmbedder } from './index.mjs';
import { observing } from './observe.mjs';

const KINDS = ['rules', 'shim', 'route', 'bank'];

export function validateSystem(spec, have = []) {
  const errs = [];
  if (!spec?.name) errs.push('missing "name"');
  if (!Array.isArray(spec?.steps) || !spec.steps.length) errs.push('needs at least one step');
  const ids = new Set();
  for (const [i, s] of (spec?.steps ?? []).entries()) {
    const at = `step ${i}${s?.id ? ` (${s.id})` : ''}`;
    if (!s?.id) errs.push(`${at}: missing "id"`);
    if (ids.has(s?.id)) errs.push(`${at}: duplicate id`);
    ids.add(s?.id);
    if (!KINDS.includes(s?.kind)) { errs.push(`${at}: kind must be one of ${KINDS.join(', ')}`); continue; }
    if (s.kind === 'rules') {
      if (!Array.isArray(s.rules) || !s.rules.length) errs.push(`${at}: needs "rules"`);
      for (const r of s.rules ?? []) {
        if (!r.name || !r.match) errs.push(`${at}: every rule needs "name" and "match"`);
        else { try { new RegExp(r.match, r.flags ?? 'i'); } catch (e) { errs.push(`${at}: rule "${r.name}" is not a valid pattern: ${e.message}`); } }
      }
    }
    if (s.kind === 'shim' && !s.shim) errs.push(`${at}: missing "shim"`);
    if (s.kind === 'bank' && !Object.keys(s.shims ?? {}).length) errs.push(`${at}: needs "shims"`);
    if (s.kind === 'route') {
      if (!s.on || !ids.has(s.on)) errs.push(`${at}: "on" must name an earlier step`);
      if (!Object.keys(s.routes ?? {}).length) errs.push(`${at}: needs "routes"`);
      if (s.onMissing && !['continue', 'escalate'].includes(s.onMissing) && typeof s.onMissing !== 'string')
        errs.push(`${at}: "onMissing" should be "continue", or the outcome to stop with`);
      if (s.rescue != null) {
        const ok = s.rescue === true || s.rescue === 'siblings' || (typeof s.rescue === 'object' && !Array.isArray(s.rescue));
        if (!ok) errs.push(`${at}: "rescue" should be true, "siblings", or { margin }`);
        else if (Object.values(s.routes ?? {}).filter(Boolean).length < 2)
          errs.push(`${at}: "rescue" needs at least two routes with a shim to vote between`);
      }
    }
    for (const name of namesIn(s)) if (have.length && !have.includes(name)) errs.push(`${at}: no shim called "${name}"`);
  }
  return errs;
}
const namesIn = s => s.kind === 'shim' ? [s.shim]
  : s.kind === 'bank' ? Object.values(s.shims ?? {})
  : s.kind === 'route' ? Object.values(s.routes ?? {}).filter(Boolean) : [];

/** Every shim a system can reach, so a caller knows what to load. */
export const shimsUsed = spec => [...new Set((spec.steps ?? []).flatMap(namesIn))];

export class System {
  /** @param {object} spec the wiring · @param {Record<string, Shim>} shims by name */
  constructor(spec, shims) {
    const errs = validateSystem(spec, Object.keys(shims));
    if (errs.length) throw new Error(`system "${spec?.name ?? '?'}": ${errs.join('; ')}`);
    this.spec = spec;
    this.shims = shims;
    this.name = spec.name;
  }

  gatesFor(step, shim) {
    const g = step.gate ?? {};
    // The shim's LIVE gates, not its report: refitGate() and refitFloor() move them for real traffic,
    // and until 2026-09-19 a gate refitted that way bound Shim.decide() and was ignored here.
    const conf = g.confidence === false ? 0 : typeof g.confidence === 'number' ? g.confidence : shim.gates.confidence;
    const fam = g.familiarity === false ? 0 : typeof g.familiarity === 'number' ? g.familiarity : shim.gates.familiarity;
    return { conf, fam };
  }

  /**
   * A confident child can rescue an unsure parent.
   *
   * When a router cannot tell which specialist to hand to, ask them all and see which one
   * RECOGNISES the input. Found on "i cant seem to connect my stripe account", where the word
   * "account" is also the name of a sibling branch: the router slid from 78% to 45% as the word
   * finished, fell under its 0.68 gate, and sent the message to a person — while the technical
   * specialist sat at 87% and had never once wavered. A correct answer was being thrown away
   * because the question before it lost its nerve.
   *
   * Familiarity is the ballot, NOT confidence. A specialist can only answer in its own categories,
   * so it is structurally forced to pick one of them and will say 99% about anything: the billing
   * shim calls "please close my account" a failed-payment. Familiarity is the only number here
   * that can say "not mine" — on that stripe probe the three specialists read 90 / 25 / 24, and
   * on a real login problem they read 4 / 30 / 82.
   *
   * A vote only counts if the winner is somewhere it belongs (its own fitted familiarity floor)
   * AND beats the runner-up by `margin`. A three-way tie means nobody recognises it, which is
   * what escalation is for.
   */
  askSiblings(stepId, ask) {
    const route = this.spec.steps.find(s => s.kind === 'route' && s.on === stepId && s.rescue);
    if (!route) return null;
    const opt = (route.rescue === true || route.rescue === 'siblings') ? {} : route.rescue;
    const margin = opt.margin ?? 0.25;
    const votes = Object.entries(route.routes).filter(([, n]) => n).map(([key, n]) => {
      const d = ask(n);
      return { key, shim: n, familiarity: d.familiarity, floor: this.shims[n].gates.familiarity };
    }).sort((a, b) => b.familiarity - a.familiarity);
    if (votes.length < 2) return null;
    const [top, next] = votes;
    const gap = top.familiarity - next.familiarity;
    const took = (top.familiarity >= top.floor && gap >= margin) ? top : null;
    return { votes: votes.map(v => ({ key: v.key, familiarity: +v.familiarity.toFixed(3) })),
             took: took?.key ?? null, by: took?.shim ?? null, margin: +gap.toFixed(3) };
  }

  /**
   * One encode for the whole system: every shim reads the same vector, or — for input past the
   * encoder's ceiling — every chunk, each weighted by how much of it that shim recognises. That
   * is how a shim finds the sentences in a long message that are its business, and why two shims
   * can draw on different parts of the same message.
   */
  async run(text, opts = {}) {
    const text_ = String(text ?? '');
    const embedder = currentEmbedder();
    // `opts.vectors` lets a caller that has already encoded this input — a stream replaying with
    // different gates, a page re-running a whole inbox — skip the expensive part.
    const chunks = opts.chunks ?? (text_.length > LONG_INPUT_CHARS ? splitForDecision(text_) : [text_]);
    const vectors = opts.vectors ?? await embedder(chunks);
    // A system can say how its own input is shaped. `"reduce": "pooled"` on the spec is for
    // documents whose evidence is spread across sentences; the default reads the most familiar
    // sentence. See reduce.mjs — neither wins in general, which is why this is a declaration
    // about the input rather than a default.
    const red = { mode: opts.reduce ?? this.spec.reduce, power: opts.reducePower ?? this.spec.reducePower };
    const t0 = observing() ? performance.now() : 0;
    // Every shim a system asks is recorded, once, with the vector it decided on.
    const decide = shim => {
      if (chunks.length < 2) return shim._observed(shim.decideVector(vectors[0]), text_, vectors[0], t0, opts, { via: 'system', system: this.name });
      const r = shim.decideChunks(vectors, chunks, red);
      return shim._observed(r, text_, vectors[chunks.indexOf(r.from)] ?? null, t0, opts, { via: 'system', system: this.name });
    };
    // One evaluation per shim per input. A sibling vote asks specialists the router may then hand
    // to anyway, and a head asked twice is still one head.
    const seen = new Map();
    const ask = name => {
      if (!seen.has(name)) { seen.set(name, decide(this.shims[name])); heads++; }
      return seen.get(name);
    };

    const trace = [], results = {};
    let heads = 0, outcome = null, answer = null;
    // Confidence along a path multiplies: five nodes at 90% is 59%, and every node can clear its
    // own gate while the path as a whole is a guess. "where is my parcel" routed to billing ->
    // invoice-question with both nodes past their gates and 32% joint. `minPathConfidence` gates
    // the path itself.

    for (const step of this.spec.steps) {
      // `always` steps run on every branch, including after something has already decided the
      // outcome: urgency and tone are asked of every message, wherever it was routed.
      if (outcome && !step.always) continue;
      if (step.kind === 'rules') {
        const hit = step.rules.map(r => ({ ...r, re: new RegExp(r.match, r.flags ?? 'i') })).find(r => r.re.test(text_));
        trace.push({ id: step.id, kind: 'rules', matched: hit?.name ?? null });
        if (hit) { outcome = step.outcome ?? 'stopped'; answer = hit.name; }
        continue;
      }
      if (step.kind === 'bank') {
        const fields = {};
        for (const [field, name] of Object.entries(step.shims)) {
          const shim = this.shims[name], r = ask(name), { conf, fam } = this.gatesFor(step, shim);
          fields[field] ={ ...r, state: r.familiarity < fam ? 'silent' : r.confidence >= conf ? 'applied' : 'offered' };
        }
        results[step.id] = fields;
        trace.push({ id: step.id, kind: 'bank', fields: Object.fromEntries(Object.entries(fields).map(([f, r]) => [f, { answer: r.answer, state: r.state }])) });
        continue;
      }
      // shim / route: one decision, gated
      const name = step.kind === 'shim' ? step.shim : this.spec.steps.find(s => s.id === step.on) && step.routes[results[step.on]?.answer];
      if (step.kind === 'route' && !results[step.on]) { trace.push({ id: step.id, kind: 'route', skipped: 'the step it routes on did not decide' }); continue; }
      if (step.kind === 'route' && name === undefined) {
        // A deeper node that only some branches have. "continue" means this path simply stops
        // here with what it already decided, rather than escalating for lack of a sub-question.
        const missing = step.onMissing ?? 'escalate';
        trace.push({ id: step.id, kind: 'route', on: results[step.on].answer, to: null,
                     note: missing === 'continue' ? 'nothing deeper for that answer' : 'no route for that answer' });
        if (missing !== 'continue') outcome = outcome ?? missing;
        continue;
      }
      if (name === null) {                                    // a deliberate dead end, e.g. a board
        trace.push({ id: step.id, kind: 'route', on: results[step.on].answer, to: null, note: step.endsAt ?? 'no specialist' });
        outcome = 'routed'; answer = results[step.on].answer;
        continue;
      }
      const shim = this.shims[name], r = ask(name), { conf, fam } = this.gatesFor(step, shim);
      let state = r.familiarity < fam ? 'unfamiliar' : r.confidence >= conf ? 'confident' : 'unsure';
      let answerOf = r.answer, rescue = null;

      if (state !== 'confident' && !step.always) {
        rescue = this.askSiblings(step.id, ask);
        if (rescue?.took) { state = 'rescued'; answerOf = rescue.took; }
      }

      results[step.id] = { ...r, answer: answerOf, shim: name, state, rescuedBy: rescue?.by ?? null };
      trace.push({ id: step.id, kind: step.kind, shim: name, answer: answerOf, confidence: r.confidence, familiarity: r.familiarity, state });
      if (rescue) trace.push({ id: `${step.id}+siblings`, kind: 'rescue', on: step.id, was: r.answer,
                               votes: rescue.votes, took: rescue.took, margin: rescue.margin });
      // An `always` step is a note attached to whatever the routing decided — urgency, tone — so it
      // never becomes the answer, and it never overrides an outcome an earlier step reached.
      if (state === 'confident' || state === 'rescued') { if (!step.always) answer = answerOf; }
      else {
        const next = state === 'unfamiliar' ? (step.onUnfamiliar ?? 'escalate') : (step.onUnsure ?? 'escalate');
        // An always-step that cannot answer leaves the decision alone. A route that succeeds only gets
        // its outcome at the end of the run, so `outcome` is still empty here — and an annotation that
        // did not recognise the message used to fill it with "escalate", sending a confidently routed
        // message to a person because its TONE was unclear (found by the test suite, 2026-09-19).
        if (next !== 'continue' && !outcome && !step.always) outcome = next;
      }
    }

    const routing = trace.filter(t => t.confidence != null && !this.spec.steps.find(s => s.id === t.id)?.always);
    const pathConfidence = routing.reduce((a, t) => a * t.confidence, 1);
    const floor = this.spec.minPathConfidence ?? 0;
    const weak = routing.length > 1 && pathConfidence < floor;
    if (weak && !outcome) outcome = this.spec.onWeakPath ?? 'escalate';
    // The decision an app should report an outcome against: the step whose answer became the system's.
    const decidedBy = weak || answer == null ? null : Object.values(results).find(r => r?.answer === answer && r.id)?.id ?? null;
    return { outcome: outcome ?? (answer != null ? 'routed' : 'escalate'),
             answer: weak ? null : answer, id: decidedBy, results, trace,
             pathConfidence: +pathConfidence.toFixed(4), weakPath: weak,
             heads, encodes: chunks.length, chunks: chunks.length > 1 ? chunks : null };
  }
}
