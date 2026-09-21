// Check the wiring. A system names shims and routes answers to them; the things that go wrong are
// the boring ones — a renamed answer with no route, a shim that no longer exists, a route to a
// shim that is still a draft — and they are exactly what a build should catch rather than a user.
import { validateSystem, shimsUsed } from '../system.mjs';

export function compileSystem(spec, compiled) {
  const byName = Object.fromEntries(compiled.map(c => [c.name, c]));
  const errors = validateSystem(spec, Object.keys(byName));
  const notes = [];

  // What a step can answer: a shim step, its shim's labels; a route step, everything its targets
  // can answer — which is how a chain of routes is checked past the first level.
  const answersOf = step => {
    if (!step) return null;
    if (step.kind === 'shim') return byName[step.shim]?.labels ?? null;
    if (step.kind !== 'route') return null;
    const names = Object.values(step.routes ?? {}).filter(Boolean);
    const labels = names.flatMap(n => byName[n]?.labels ?? []);
    return labels.length ? [...new Set(labels)] : null;
  };

  for (const step of spec.steps ?? []) {
    if (step.kind === 'route') {
      const on = (spec.steps ?? []).find(s => s.id === step.on);
      const canAnswer = answersOf(on);
      if (canAnswer) {
        const routed = Object.keys(step.routes ?? {});
        const missing = canAnswer.filter(l => !routed.includes(l));
        const unknown = routed.filter(l => !canAnswer.includes(l));
        // Only a step that stops on a missing route has to cover every answer; one that carries
        // on is a deeper question some branches simply do not have.
        if (missing.length && step.onMissing !== 'continue')
          errors.push(`step ${step.id}: "${step.on}" can answer ${missing.join(', ')} with nowhere to route it — route them, or set onMissing: "continue"`);
        if (unknown.length) errors.push(`step ${step.id}: routes ${unknown.join(', ')}, which "${step.on}" never answers`);
      }
      // A sibling vote is only a fair ballot if every answer the parent could give has a
      // specialist standing for it. Where one does not — a branch that ends at a board, or a
      // deeper question only some branches have — that answer can never win a rescue, so an
      // unsure parent that was right about it gets overruled or escalated either way. Worth
      // saying out loud rather than discovering as a silent bias toward the branches that vote.
      if (step.rescue && canAnswer) {
        const standing = Object.entries(step.routes ?? {}).filter(([, n]) => n).map(([k]) => k);
        const silent = canAnswer.filter(l => !standing.includes(l));
        if (standing.length < 2) errors.push(`step ${step.id}: "rescue" needs at least two routes with a shim to vote between`);
        else if (silent.length) notes.push({ level: 'warn', code: 'partial-ballot',
          text: `"${step.id}" rescues an unsure "${step.on}" by asking ${standing.length} of its ${canAnswer.length} answers` +
                ` — ${silent.join(', ')} ${silent.length === 1 ? 'has' : 'have'} no specialist to stand for` +
                ` ${silent.length === 1 ? 'it' : 'them'} and can never win the vote.` });
      }
    }
    for (const name of [step.shim, ...Object.values(step.routes ?? {}), ...Object.values(step.shims ?? {})].filter(Boolean)) {
      const c = byName[name];
      if (c?.draft) errors.push(`step ${step.id}: "${name}" is still a draft`);
      else if (c && c.report?.provisional) notes.push({ level: 'warn', code: 'provisional-step',
        text: `"${name}" has too few examples for its score to settle — this system routes on a number that will move.` });
      else if (c && c.report?.suggestedThreshold >= 1) notes.push({ level: 'warn', code: 'no-threshold',
        text: `"${name}" never reaches 90% accuracy at any confidence, so every decision through it escalates.` });
    }
  }

  const used = shimsUsed(spec).map(n => byName[n]).filter(Boolean);
  // Worst case is one root decision plus the deepest chain of routes it can set off.
  // A bank step with no "shims" is already in `errors`; it must not take the whole check down with it.
  const perStep = (spec.steps ?? []).map(s => s.kind === 'bank' ? Object.keys(s.shims ?? {}).length : s.kind === 'rules' ? 0 : 1);
  return {
    name: spec.name, type: 'system', input: spec.input ?? null,
    shims: used.map(c => ({ name: c.name, answers: c.labels?.length ?? 0, head: c.head,
      threshold: c.report?.suggestedThreshold, floor: c.report?.familiarityFloor, bytes: c.report?.bytes ?? 0 })),
    steps: (spec.steps ?? []).map(s => ({ id: s.id, kind: s.kind,
      shim: s.shim ?? null, routes: s.routes ?? null, rules: (s.rules ?? []).map(r => r.name) })),
    outcomes: spec.outcomes ?? null,
    headsPerInput: perStep.reduce((a, b) => a + b, 0),
    bytes: used.reduce((a, c) => a + (c.report?.bytes ?? 0) + (c.report?.referenceBytes ?? 0), 0),
    errors, notes,
  };
}
