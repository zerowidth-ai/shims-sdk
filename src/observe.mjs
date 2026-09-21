// What a shim saw, what it said, and what happened next.
//
// Two things in this SDK need real traffic and have never had any.
//
// `refitFloor` needs a few hundred in-scope vectors, because a familiarity floor fitted on a
// shim's own examples is calibrated to the distribution it was built from: measured against real
// search queries, floors aiming to wrongly refuse one request in twenty refused between one in
// three and five in six. `recalibrate` needs thirty outcomes per shim and the committed feedback
// files hold five to ten, so it has never once been runnable.
//
// Both are blocked on capture rather than on cleverness, and capture only happens if it is the
// default. An integration that has to remember to log will not.
//
// What is recorded is deliberately the passive signal. `adapt.mjs` already keeps explicit verdicts
// — someone clicking a chip to teach it — and LOOP.md's own premise is that explicit feedback
// "collects almost nothing, and what it does collect skews toward the annoyed". So the mechanism
// that exists captures the signal the design says will not arrive. This captures the other kind:
// every decision, whether the gate fired, and what the person did next if the app can see it.
//
// Nothing is sent anywhere. There is no default sink — an app that does not call `observe()` pays
// nothing and records nothing, and an app that does chooses where it goes.
//
//   import { observe, outcome, ring } from '@zerowidth/shims-sdk/observe';
//   const log = ring(2000);
//   observe(log.push);                        // or a sink writing to IndexedDB, a file, a POST
//
//   const r = await shim.decide(text);        // r.id exists only while something is observing
//   if (userWentWithIt) outcome(r.id, 'accepted');
//   if (userPickedOther) outcome(r.id, 'corrected', { label: theirChoice });
//
//   log.summary()                             // abstention rate, median words, median ms
//   shim.refitFloor(log.vectorsFor('category'))    // needs a few hundred, not a few

/** @typedef {'act'|'suggest'|'refuse'} Action */

let sink = null;

/**
 * Where decisions go. Called synchronously on every decide, so it must be cheap — push to an
 * array, not a network call. Pass null to stop.
 */
export function observe(fn) { sink = typeof fn === 'function' ? fn : null; }
export const observing = () => sink != null;

/**
 * Record one decision. Called by the runtime; an app should not need to.
 * Vectors are kept by reference rather than copied — a sink that retains them past the next
 * decide should copy, and `ring` below does.
 */
export function record(entry) {
  if (!sink) return;
  try { sink(entry); } catch { /* a broken sink must never break a decision */ }
}

/**
 * The outcome of a decision, reported by the app when it can see one.
 *
 * This is the part LOOP.md asks for and nothing implemented: the user's next action is the label,
 * and nobody knows they are annotating. `accepted` and `ignored` are deliberately different —
 * a suggestion scrolled past is not a suggestion refused, and collapsing them teaches a shim to
 * be timid.
 *
 * @param {string} id the `id` from the decision this refers to
 * @param {'accepted'|'corrected'|'rejected'|'ignored'} outcome
 * @param {{label?: string}} [opts] the right answer, when the outcome is `corrected`
 */
export function outcome(id, outcome, { label = null } = {}) {
  record({ kind: 'outcome', id, outcome, label, at: Date.now() });
}

let seq = 0;
export const nextId = () => `d${Date.now().toString(36)}${(seq = (seq + 1) % 1e6).toString(36)}`;

/**
 * A bounded in-memory sink, which is what most apps want to start with.
 *
 * Bounded because an unbounded one in a long-lived tab is a leak, and because the two consumers
 * both want recent traffic rather than all of it. Vectors are copied on the way in: the runtime
 * reuses buffers, and a ring that held references would quietly fill with the same vector.
 */
export function ring(limit = 1000) {
  const rows = [];
  const byId = new Map();
  return {
    push(e) {
      if (e.kind === 'outcome') {
        const d = byId.get(e.id);
        if (d) { d.outcome = e.outcome; d.correctedTo = e.label; }
        return;
      }
      const copy = { ...e, vector: e.vector ? new Float32Array(e.vector) : null };
      rows.push(copy);
      byId.set(copy.id, copy);
      while (rows.length > limit) byId.delete(rows.shift().id);
    },
    get rows() { return rows; },
    clear() { rows.length = 0; byId.clear(); },

    /** Decisions for one shim, filtered. */
    for(name, { action = null, acted = null, outcome = null } = {}) {
      return rows.filter(r => r.shim === name
        && (action == null || r.action === action)
        && (acted == null || (r.action === 'act') === acted)
        && (outcome == null || r.outcome === outcome));
    },

    /**
     * Vectors to hand `refitFloor`, and the awkward part of this whole module.
     *
     * That fit needs text the app believes is IN SCOPE, and there is no automatic signal for it.
     * "Decisions it acted on" was the first default here and it is wrong twice: a shim whose gate
     * is 1 never acts, so the shim that most needs refitting produces nothing, and gating on
     * familiarity instead is circular — you would only ever see what already clears the floor.
     *
     * So in-scope has to be something the app knows and reports. Default is decisions a person
     * went along with, which is the strongest evidence available that the shim was asked a
     * question it was for. Pass `{ acted: true }` or your own filter if you have better.
     *
     * Feeding this raw traffic is the one thing not to do: if half of it is out of scope the floor
     * goes to zero and refusal stops working, which is the failure the gate exists to prevent.
     */
    vectorsFor(name, opts = null) {
      const rows = opts ? this.for(name, opts)
        : this.for(name).filter(r => r.outcome === 'accepted' || r.outcome === 'corrected');
      return rows.map(r => r.vector).filter(Boolean);
    },

    /** Rows to hand `recalibrate`: the ones where the app saw the person go along with it. */
    outcomesFor(name) {
      return this.for(name).filter(r => r.vector && (r.outcome === 'accepted' || r.outcome === 'corrected'))
        .map(r => ({ vector: r.vector, label: r.correctedTo ?? r.answer }));
    },

    /** What a first week actually wants to know, none of which needs a label. */
    summary(name = null) {
      const rs = name ? rows.filter(r => r.shim === name) : rows;
      const n = rs.length || 1;
      const count = a => rs.filter(r => r.action === a).length;
      const words = rs.map(r => r.words).filter(Number.isFinite).sort((a, b) => a - b);
      return {
        decisions: rs.length,
        acted: +(count('act') / n).toFixed(3),
        suggested: +(count('suggest') / n).toFixed(3),
        refused: +(count('refuse') / n).toFixed(3),
        // How much people type. Not because short input is hard as such — a frontier model scores
        // 100% on two-word clothing queries — but because very short input is usually a NAME, and a
        // shim reads meaning better than it knows vocabulary. Worth knowing before wondering why.
        medianWords: words.length ? words[words.length >> 1] : null,
        medianMs: (() => { const m = rs.map(r => r.ms).filter(Number.isFinite).sort((a, b) => a - b);
          return m.length ? +m[m.length >> 1].toFixed(1) : null; })(),
        // How much of this traffic can drive a refit. Zero here means the app is not reporting
        // outcomes, and both refitFloor and recalibrate will stay unusable however long it runs.
        withOutcome: rs.filter(r => r.outcome).length,
      };
    },
  };
}
