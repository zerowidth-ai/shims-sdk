#!/usr/bin/env node
// shim-compile <dir|file...> [--check]
//
// Also checks any *.system.json it finds beside them: the wiring that turns several shims into
// one decision. Systems compile to nothing — they are checked, and the apps read the spec.
//
// Compiles every *.shim.json it is given and writes the weights next to the source.
// With --check it compiles but writes nothing, and exits non-zero on any error-level
// note — which is what you want in CI.

import fs from 'node:fs/promises';
import path from 'node:path';
import { compileShim } from './compile.mjs';
import { compileSystem } from './system.mjs';

const args = process.argv.slice(2);
const check = args.includes('--check');
const targets = args.filter(a => !a.startsWith('--'));
if (!targets.length) {
  console.error('usage: shim-compile <dir|file.shim.json ...> [--check]');
  process.exit(2);
}

const C = { dim: '\x1b[2m', red: '\x1b[31m', yellow: '\x1b[33m', green: '\x1b[32m', bold: '\x1b[1m', off: '\x1b[0m' };
const pct = v => (100 * v).toFixed(0) + '%';
const kb = b => (b / 1024).toFixed(1) + 'KB';

async function sources(t, ext = '.shim.json') {
  const st = await fs.stat(t);
  if (st.isFile()) return t.endsWith(ext) ? [t] : [];
  const names = await fs.readdir(t);
  return names.filter(n => n.endsWith(ext)).map(n => path.join(t, n));
}

const files = (await Promise.all(targets.map(t => sources(t)))).flat();   // not .map(sources): map passes the index as the second argument
if (!files.length) { console.error('no *.shim.json found'); process.exit(2); }


let failed = 0, warned = 0;
const built = [];
console.log();

for (const file of files) {
  const src = JSON.parse(await fs.readFile(file, 'utf8'));
  const cacheDir = path.join(path.dirname(file), '.shim-cache');
  let out;
  try {
    out = await compileShim(src, { cacheDir });
  } catch (e) {
    console.log(`${C.red}✗${C.off} ${C.bold}${path.basename(file)}${C.off}\n   ${e.message}\n`);
    failed++; continue;
  }

  const r = out.report;
  if (out.draft) {
    const counts = Object.entries(r.counts ?? {})
      .map(([l, n]) => `${l} ${n}/${r.minPerAnswer ?? 3}`).join('  ');
    console.log(`${C.dim}·${C.off} ${C.bold}${out.name}${C.off} ${C.dim}draft · ${out.examples} examples${C.off}`);
    if (out.question) console.log(`   ${C.dim}"${out.question}"${C.off}`);
    console.log(`   ${C.dim}${counts}${C.off}`);
    for (const n of r.notes ?? []) console.log(`   ${C.dim}${n.text}${C.off}`);
    if (!check) {
      const dest = file.replace(/\.shim\.json$/, '.weights.json');
      await fs.writeFile(dest, JSON.stringify(out, null, 1) + '\n');
      console.log(`   ${C.dim}→ ${path.basename(dest)}${C.off}`);
    }
    console.log();
    continue;
  }
  const errs = (r.notes ?? []).filter(n => n.level === 'error');
  const warns = (r.notes ?? []).filter(n => n.level === 'warn');
  const mark = errs.length ? `${C.red}✗${C.off}` : warns.length ? `${C.yellow}!${C.off}` : `${C.green}✓${C.off}`;

  console.log(`${mark} ${C.bold}${out.name}${C.off} ${C.dim}${out.type} · ${out.examples} examples · ${kb(r.bytes)}${C.off}`);
  if (out.question) console.log(`   ${C.dim}"${out.question}"${C.off}`);

  if (out.type === 'tags') {
    console.log(`   balanced accuracy ${pct(r.macroRecall)} ${C.dim}(chance 50%)${C.off}`);
    for (const p of r.perTag) console.log(`     ${C.dim}${p.label.padEnd(16)}${C.off}${pct(p.macroRecall)}`);
  } else {
    if (r.provisional) {
      console.log(`   ${C.dim}provisional — too few examples for the score to settle${C.off}`);
    }
    {
    // The interval, not the point estimate, is what a reviewer should read: on a 24-example shim
    // the same number is worth about ±9 points (measured across 120 small shims).
    const ci = r.accuracyInterval ? ` ${C.dim}(95%: ${pct(r.accuracyInterval[0])}–${pct(r.accuracyInterval[1])})${C.off}` : '';
    console.log(`   accuracy ${C.bold}${pct(r.accuracy)}${C.off}${ci}   balanced ${pct(r.macroRecall)} ${C.dim}(chance ${pct(r.chance)})${C.off}   ${C.dim}${r.evaluated} held out · ${r.head} head${C.off}`);
    const recalls = Object.entries(r.recall).map(([l, v]) => `${l} ${v === null ? '—' : pct(v)}`).join('  ');
    console.log(`   ${C.dim}recall:${C.off} ${recalls}`);
    }
    if (!r.provisional && r.coverageAtThreshold) {
      console.log(`   ${C.dim}90% accurate above confidence ${r.suggestedThreshold} — covers ${pct(r.coverageAtThreshold)} of inputs${C.off}`);
    } else if (!r.provisional) {
      console.log(`   ${C.dim}never reaches 90% accuracy at any confidence — escalate everything for now${C.off}`);
    }
  }

  for (const n of r.notes ?? []) {
    const col = n.level === 'error' ? C.red : n.level === 'warn' ? C.yellow : C.dim;
    console.log(`   ${col}${n.level}${C.off} ${n.text}`);
  }

  built.push(out);
  if (!check) {
    const dest = file.replace(/\.shim\.json$/, '.weights.json');
    await fs.writeFile(dest, JSON.stringify(out, null, 1) + '\n');
    console.log(`   ${C.dim}→ ${path.basename(dest)}${C.off}`);
  }
  console.log();
  if (errs.length) failed++;
  if (warns.length) warned++;
}

/* ---------- systems: the wiring between shims ---------- */
const systemFiles = (await Promise.all(targets.map(t => sources(t, '.system.json')))).flat();
for (const file of systemFiles) {
  const spec = JSON.parse(await fs.readFile(file, 'utf8'));
  const r = compileSystem(spec, built);
  const mark = r.errors.length ? `${C.red}✗${C.off}` : r.notes.length ? `${C.yellow}!${C.off}` : `${C.green}✓${C.off}`;
  console.log(`${mark} ${C.bold}${r.name}${C.off} ${C.dim}system · ${r.shims.length} shims · ${kb(r.bytes)} · ${r.headsPerInput} head${r.headsPerInput === 1 ? '' : 's'} per input${C.off}`);
  if (r.input) console.log(`   ${C.dim}in: ${r.input}${C.off}`);
  for (const s of r.steps) {
    if (s.kind === 'rules') console.log(`   ${C.dim}${s.id.padEnd(11)} rules · ${s.rules.join(', ')}${C.off}`);
    else if (s.kind === 'route') console.log(`   ${C.dim}${s.id.padEnd(11)} route · ${Object.entries(s.routes).map(([k, v]) => `${k}→${v ?? '·'}`).join('  ')}${C.off}`);
    else console.log(`   ${C.dim}${s.id.padEnd(11)} ${s.kind} · ${s.shim ?? Object.values(spec.steps.find(x => x.id === s.id).shims ?? {}).join(', ')}${C.off}`);
  }
  for (const e of r.errors) console.log(`   ${C.red}error${C.off} ${e}`);
  for (const n of r.notes) console.log(`   ${C.yellow}${n.level}${C.off} ${n.text}`);
  console.log();
  if (r.errors.length) failed++; else if (r.notes.length) warned++;
}

const summary = `${files.length - failed}/${files.length} compiled` + (warned ? `, ${warned} with warnings` : '');
console.log(check ? `${summary} (check only, nothing written)\n` : `${summary}\n`);
process.exit(check && failed ? 1 : 0);
