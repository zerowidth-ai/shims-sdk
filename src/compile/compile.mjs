// Compile a *.shim.json (task + examples) into committed weights plus a report.
//
// The report is the part a person reads in a pull request. It answers three
// questions: how good is this shim, which answers does it get wrong, and is this
// task even the right shape for a shim at all.

import { embed } from '../index.mjs';
import { fitHead, fitTagHeads, softmax, balanceWeights, shuffled, fitTemperature, applyTemperature, calibrationError, fitCentroids, centroidPredict } from '../math.mjs';
import { packHead, validateSource, readiness, MIN_PER_ANSWER, PROTOTYPE_PARITY, prototypeWeight, headBytes, DIM, EMBED_DIM, ENCODER_ID, FORMAT_VERSION, vectorCacheFile, bytesToBase64 } from '../format.mjs';
import { packReference, unpackReference, knnPredict, familiarityScore } from '../familiarity.mjs';
import { groupCandidates, learnGroups, fitTree, predictTree, packTree } from '../tree.mjs';

/** A grouping written into the source: { "billing": ["refund", …], … } → arrays of answer indices. */
function pinnedGroups(tree, labels) {
  const groups = Object.values(tree).map(ls => ls.map(l => labels.indexOf(l)));
  const flat = groups.flat();
  if (flat.some(i => i < 0)) throw new Error(`tree names an answer that is not in labels: ${Object.values(tree).flat().filter(l => !labels.includes(l)).join(', ')}`);
  if (new Set(flat).size !== flat.length || flat.length !== labels.length)
    throw new Error('tree must place every answer in exactly one group');
  return groups;
}

const FOLDS = 5;
const SEED = 20260916;

/* ---------- embedding cache: recompiles should be instant ---------- */
// The cache is the only part of the compiler that touches a filesystem, so the Node modules it
// needs are loaded here, on first use, and only when a caller asks for a cache. Everything else in
// this file is plain arithmetic, which is what lets the same compiler run in a browser: pass no
// `cacheDir`, and an `embed` that reaches whatever encoder the page already has.
async function embedCached(texts, cacheDir, embedFn = embed) {
  // No cache directory means no cache: the test suite compiles with an injected encoder and must
  // not leave, or read, vectors on disk that a real build could pick up.
  if (!cacheDir) return { vectors: (await embedFn(texts)).map(v => Float32Array.from(v)), embedded: texts.length };
  const [fs, path, crypto] = await Promise.all([import('node:fs/promises'), import('node:path'), import('node:crypto')]);
  const hash = s => crypto.createHash('sha1').update(s).digest('hex').slice(0, 16);
  await fs.mkdir(cacheDir, { recursive: true });
  const file = path.join(cacheDir, vectorCacheFile());
  let cache = {};
  try { cache = JSON.parse(await fs.readFile(file, 'utf8')); } catch {}

  // Cached vectors carry the surface features too, so anything stored at a different
  // width is from an older format and must be recomputed rather than quietly mixed in.
  for (const k of Object.keys(cache)) if (cache[k]?.length !== DIM) delete cache[k];

  const missing = texts.filter(t => !cache[hash(t)]);
  if (missing.length) {
    const fresh = await embedFn(missing);
    missing.forEach((t, i) => { cache[hash(t)] = Array.from(fresh[i]); });
    await fs.writeFile(file, JSON.stringify(cache));
  }
  return { vectors: texts.map(t => Float32Array.from(cache[hash(t)])), embedded: missing.length };
}

/* ---------- evaluation ---------- */
// Five folds over nine examples leaves two per fold and most folds degenerate. Below
// LOO_BELOW, hold out one example at a time instead — slower, but the only way to get
// a real number out of a small set.
const LOO_BELOW = 25;

function foldIndices(n, folds, seed) {
  const order = shuffled([...Array(n).keys()], seed);
  const k = n < LOO_BELOW ? n : folds;
  return Array.from({ length: k }, (_, f) => ({
    test: order.filter((_, i) => i % k === f),
    train: order.filter((_, i) => i % k !== f),
  }));
}

/**
 * 95% Wilson interval for a proportion — what an accuracy measured on `n` decisions is actually
 * worth.
 *
 * A shim built from 24 examples reports accuracy to four decimal places. What that number is really
 * worth was measured: across 120 small shims, held-out accuracy sat a
 * standard deviation of 7.5-8.8 points from the reported figure whenever the answers were
 * confusable, and about one shim in six overstated by more than five points. The reports are not
 * biased — cross-validation on few examples trains each fold on fewer still, so it tends to
 * understate the head that ships — they are simply imprecise, and printing 0.9583 hides that.
 *
 * Wilson rather than the textbook normal interval because it stays inside [0,1] and does not
 * collapse to zero width when a small shim scores 100%, which is exactly the case that misleads.
 */
function wilson(p, n, z = 1.96) {
  if (!n) return null;
  const d = 1 + z * z / n;
  const centre = (p + z * z / (2 * n)) / d;
  const half = (z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d;
  return [+Math.max(0, centre - half).toFixed(4), +Math.min(1, centre + half).toFixed(4)];
}

/** Confidence above which kept decisions are `target` accurate, and the coverage that leaves. */
function thresholdAt(scored, target = 0.9) {
  const sorted = [...scored].sort((a, b) => b.c - a.c);
  for (let i = sorted.length - 1; i >= 0; i--) {
    const kept = sorted.slice(0, i + 1);
    if (kept.filter(s => s.ok).length / kept.length >= target)
      // Rounded DOWN. Rounding to nearest can land above the row the gate was read from, which then
      // fails its own gate: a boundary at 0.9998 became a gate of 1.000 that admitted none of the rows
      // it claimed to cover (found by the test suite, 2026-09-19).
      return { threshold: Math.floor(kept[kept.length - 1].c * 1000) / 1000, coverage: +((i + 1) / sorted.length).toFixed(3) };
  }
  return { threshold: 1, coverage: 0 };
}

/**
 * One temperature and one gate from a head's held-out predictions.
 *
 * Fitting maximises likelihood, which is not the same as calibration, and on a few dozen held-out
 * rows it can make matters worse (measured: three shims here went 29→48, 14→25, 27→44). So the
 * temperature is kept only when it improves calibration on those same rows; otherwise the
 * confidences stay as the head produced them. The gate is read off the CALIBRATED confidences,
 * since that is what the runtime reports.
 */
function calibrateAndGate(ev) {
  const scoreAt = T => ev.cal.map(c => {
    const p = applyTemperature(c.probs, T);
    let i = 0; for (let j = 1; j < p.length; j++) if (p[j] > p[i]) i = j;
    return { confidence: p[i], ok: i === c.y, c: p[i] };
  });
  const fitted = fitTemperature(ev.cal);
  const plain = scoreAt(1), scaled = scoreAt(fitted);
  const eceBefore = calibrationError(plain), eceScaled = calibrationError(scaled);
  const keep = eceScaled != null && eceBefore != null && eceScaled < eceBefore;
  const calibrated = keep ? scaled : plain;
  return {
    calibrated,
    calibration: { temperature: keep ? fitted : 1, eceBefore, eceAfter: keep ? eceScaled : eceBefore,
                   fittedTemperature: fitted, eceIfScaled: eceScaled, applied: keep, samples: ev.cal.length },
    gate: calibrated.length ? thresholdAt(calibrated)
                            : { threshold: ev.suggestedThreshold, coverage: ev.coverageAtThreshold },
  };
}

/**
 * `always` are rows every fold trains on and none is scored on — the prototypes, when real examples
 * exist. Until 2026-09-19 they were left out of the folds altogether, so a shim with both was SCORED
 * as a head trained on its examples alone and SHIPPED as a head trained on examples plus prototypes:
 * head, penalty, temperature and gate were all being chosen for a model that was not the one built.
 * `fit` receives a per-row scale (1 for an example, the prototype weight for a prototype).
 */
function crossValidate(X, y, K, fit, predictTop, always = null) {
  const conf = Array.from({ length: K }, () => new Array(K).fill(0));
  const scored = [];
  // Held-out probability vectors, kept so a temperature can be fitted on predictions this
  // shim never saw — calibrating on the training rows would just learn the overfit.
  const cal = [];
  // Per-example verdicts, aligned to input order. Every example is predicted exactly
  // once, by a head that never saw it — so a tool listing the examples can show honest
  // held-out answers rather than the model grading its own homework.
  const perExample = new Array(X.length).fill(null);
  for (const { train, test } of foldIndices(X.length, FOLDS, SEED)) {
    if (!train.length || !test.length) continue;
    const seen = new Set(train.map(i => y[i]));
    if (seen.size < 2) continue;                      // fold has only one answer; skip
    const head = always
      ? fit([...train.map(i => X[i]), ...always.X], [...train.map(i => y[i]), ...always.y], [...train.map(() => 1), ...always.scale])
      : fit(train.map(i => X[i]), train.map(i => y[i]), null);
    for (const i of test) {
      const r = predictTop(head, X[i]);
      conf[y[i]][r.index]++;
      scored.push({ ok: r.index === y[i], c: r.confidence });
      if (r.probs) cal.push({ probs: Array.from(r.probs), y: y[i] });
      perExample[i] = { pred: r.index, conf: +r.confidence.toFixed(3), ok: r.index === y[i] };
    }
  }
  const total = scored.length || 1;
  const acc = scored.filter(s => s.ok).length / total;

  // Macro-recall: mean per-answer recall. This is the honest number — a shim that
  // always says the most common answer scores exactly 1/K here, no matter how
  // flattering its accuracy looks on an imbalanced task.
  const recalls = conf.map((row, k) => {
    const n = row.reduce((a, b) => a + b, 0);
    return n ? row[k] / n : null;
  });
  const present = recalls.filter(r => r !== null);
  const macroRecall = present.length ? present.reduce((a, b) => a + b, 0) / present.length : 0;

  const { threshold, coverage } = thresholdAt(scored);
  return { accuracy: +acc.toFixed(4), accuracyInterval: wilson(acc, scored.length),
           macroRecall: +macroRecall.toFixed(4), confusion: conf,
           suggestedThreshold: threshold, coverageAtThreshold: coverage,
           evaluated: scored.length, perExample, cal, scored };
}

/** Build an in-memory reference from raw vectors, for cross-validating the kNN head. */
function packAsRef(X, y) {
  const r = packReference(X, y);
  return { q: r.q, scales: r.scales, spread: r.spread, count: r.count, labels: r.labels };
}

/**
 * When kNN clearly beats a linear head on one answer, that answer is multi-modal —
 * it covers several unrelated situations that happen to share a name. The fix is more
 * examples PER SITUATION, or splitting the answer, not more examples overall.
 */
function diagnoseModality(choice, labels) {
  if (!choice) return [];
  const notes = [];
  for (const p of choice.perAnswer) {
    if (p.linear === null || p.knn === null) continue;
    if (p.knn - p.linear >= 0.15) {
      notes.push({ level: 'info', code: 'multi-modal',
        text: `"${p.label}" is found ${(100 * p.knn).toFixed(0)}% of the time by nearest-neighbour but only ${(100 * p.linear).toFixed(0)}% by a single direction. ` +
              `That gap means its examples sit in several unrelated places rather than one — like "red" covering blood, traffic lights and fruit. ` +
              `Add examples for each situation separately, or split the answer; adding more of the same will not close it.` });
    }
  }
  return notes;
}

/** Say what the compiler decided about structure, and why, in words a reviewer can check. */
function diagnoseTree(choice, useTree, groups, labels) {
  if (!choice) return [];
  const pts = v => (100 * v).toFixed(1);
  const best = choice.best.ev.macroRecall, flat = choice.summary.flat;
  if (useTree) {
    const shown = groups.filter(g => g.length > 1).slice(0, 3)
      .map(g => g.slice(0, 3).map(i => labels[i]).join(' / ') + (g.length > 3 ? ` +${g.length - 3}` : '')).join('; ');
    return [{ level: 'info', code: 'tree',
      text: `${choice.summary.pinned ? 'Using the grouping in the source' : `Grouped into ${groups.length}`}: a nearest-neighbour root picks the group, a small head picks the answer. ` +
            `Held-out ${pts(best)}% vs ${pts(flat)}% flat${choice.summary.pinned && best <= flat ? ' — the pinned tree is not beating flat; consider removing it' : ''}. ` +
            `Groups include ${shown}.` }];
  }
  return [{ level: 'info', code: 'tree-not-used',
    text: `Tried grouping the answers (best ${pts(best)}% held-out) — flat is as good or better (${pts(flat)}%), so this shim stays flat.` }];
}

/* ---------- the linter ---------- */
/**
 * A shim answering from generated prototypes alone is answering from whatever world the
 * generator imagined. Asked for "legal-exposure concerns in this clause" with nothing
 * else to go on, it wrote about software warranties and vendor subcontractors — good
 * prototypes for a SaaS procurement team, useless for a ship charter. The shim then
 * cross-validated at 88% against its own generated data, and on real charterparty text
 * sat at a median familiarity of 0.12, below its own refusal line on 21 of 32 inputs.
 *
 * The compiler cannot detect the mismatch — it has no sample of the real input. What it
 * can do is notice when nothing has pinned the register down, and say so before the
 * shim ships.
 */
function diagnoseContext(src, protoCount) {
  if (!protoCount || src.examples.length) return [];
  if ((src.context ?? '').trim().length >= 40) return [];
  return [{ level: 'warn', code: 'no-context',
    text: 'Every answer here comes from generated prototypes, and nothing tells the ' +
          'generator what world the inputs come from — so it wrote for the one it ' +
          'assumed. Add a `context` describing the real setting (its vocabulary, its ' +
          'document conventions, its typical length) and re-describe. Until then, ' +
          'check familiarity on real input: if it reads low, the prototypes are for a ' +
          'different world than the one this shim is deployed in.' }];
}

function diagnose(evalOut, K, labels, exampleCount, texts) {
  const notes = [];
  const chance = 1 / K;
  const lift = evalOut.macroRecall - chance;

  // Below this there is not enough evidence to accuse the task of anything.
  const perAnswer = exampleCount / K;
  if (perAnswer < 8) {
    notes.push({ level: 'info', code: 'too-early',
      text: `${exampleCount} examples across ${K} answers is too few to judge quality — roughly ${Math.ceil(8 * K)} is where the score starts meaning something. Keep going.` });
  } else if (lift < 0.06) {
    notes.push({ level: 'error', code: 'not-learnable',
      text: `This task does not appear learnable from embeddings — balanced accuracy is ${(100 * evalOut.macroRecall).toFixed(0)}%, and chance is ${(100 * chance).toFixed(0)}%. ` +
            `Sentence embeddings encode what text is ABOUT, not how it is written. If the distinction is syntactic (punctuation, word order, a keyword, a format), a rule will beat this shim outright. If it is semantic, the examples may be inconsistent.` });
  } else if (perAnswer >= 8 && lift < 0.15) {
    notes.push({ level: 'warn', code: 'weak',
      text: `Only ${(100 * lift).toFixed(0)} points above chance. Usually means too few examples, or answers that overlap in meaning.` });
  }

  // Per-answer recall, then the comparison between answers — a single answer doing
  // far worse than its siblings is usually a definition problem, not a data problem.
  const recalls = evalOut.confusion.map((row, k) => {
    const n = row.reduce((a, b) => a + b, 0);
    return { k, n, r: n ? row[k] / n : null };
  });
  recalls.forEach(({ k, n, r }) => {
    if (n && n < 5) notes.push({ level: 'warn', code: 'thin-answer',
      text: `"${labels[k]}" has only ${n} examples — too few to learn or to measure.` });
    else if (r !== null && r < 0.4) notes.push({ level: 'warn', code: 'low-recall',
      text: `"${labels[k]}" is found only ${(100 * r).toFixed(0)}% of the time. Add examples of it, or it will rarely be predicted.` });
  });

  const present = recalls.filter(x => x.r !== null && x.n >= 5);
  if (present.length >= 2) {
    const best = present.reduce((a, b) => (b.r > a.r ? b : a));
    const worst = present.reduce((a, b) => (b.r < a.r ? b : a));
    if (best.r > 0 && worst.r < 0.65 * best.r && worst.n >= 10) {
      notes.push({ level: 'warn', code: 'incoherent-answer',
        text: `"${labels[worst.k]}" (${(100 * worst.r).toFixed(0)}%) is far behind "${labels[best.k]}" (${(100 * best.r).toFixed(0)}%). ` +
              `That gap usually means the answer is not a real category — a middle or "other" bucket defined by what it ISN'T rather than by any property of its own. More examples will not fix it; a sharper definition, or dropping the answer, will.` });
    }
  }

  // What the examples actually cover. Anything outside this range is unmapped.
  if (texts?.length) {
    const lens = texts.map(t => t.split(/\s+/).filter(Boolean).length).sort((a, b) => a - b);
    const at = q => lens[Math.min(lens.length - 1, Math.floor(q * lens.length))];
    const p10 = at(0.10), p90 = at(0.90);
    if (p10 >= 8) notes.push({ level: 'info', code: 'length-gap',
      text: `Nine in ten examples run ${p10}-${p90} words. Short inputs are unmapped territory — this shim will answer them, but with nothing to go on.` });
  }

  if (exampleCount >= 8 * K && exampleCount < 120) notes.push({ level: 'info', code: 'few-examples',
    text: `${exampleCount} examples. Accuracy usually keeps climbing well past this.` });

  return notes;
}

/* ---------- compile ---------- */
export async function compileShim(src, { cacheDir, ...opts } = {}) {
  const errors = validateSource(src);
  if (errors.length) throw new Error(`${src?.name ?? 'shim'}: ${errors.join('; ')}`);

  const type = src.type ?? 'classify';
  // `schema` is a format the validator reserves and nothing builds. It used to get past here and
  // die on `src.labels.length`, which tells the person editing the JSON nothing.
  if (type === 'schema') throw new Error(`${src.name}: "schema" is a reserved format — the compiler builds classify and tags shims`);
  // A shim described by prototypes alone may have no "examples" key at all. validateSource and
  // readiness both read that as empty; everything below reads `src.examples` directly.
  if (!Array.isArray(src.examples)) src = { ...src, examples: [] };

  // Unfinished is not broken. Hand back a stub an authoring tool can render as a draft.
  const ready = readiness(src);
  if (!ready.ready) {
    return {
      format: FORMAT_VERSION, name: src.name, type,
      question: src.question ?? null, labels: src.labels,
      encoder: ENCODER_ID, dim: DIM, draft: true,
      compiledAt: new Date().toISOString(),
      examples: (src.examples ?? []).length,
      report: {
        draft: true, counts: ready.counts, needed: ready.needed, minPerAnswer: MIN_PER_ANSWER,
        bytes: headBytes(src.labels.length, DIM),
        notes: [{ level: 'info', code: 'draft',
          text: ready.short.length
            ? `Needs ${ready.needed} more example${ready.needed === 1 ? '' : 's'} before it can compile — at least ${MIN_PER_ANSWER} for each of: ${ready.short.join(', ')}.`
            : 'Needs at least two answers.' }],
      },
    };
  }
  // Generated prototypes train alongside real examples, weighted down as real data
  // arrives. They are what lets a shim answer before anyone has written an example.
  const protoRows = Object.entries(src.prototypes ?? {})
    .flatMap(([label, list]) => (list ?? []).map(text => ({ text, label, proto: true })));
  const rows = [...src.examples.map(e => ({ ...e, proto: false })), ...protoRows];
  const texts = rows.map(e => e.text);
  // `opts.embed` replaces the encoder. It exists for the test suite, which needs vectors it can
  // reason about; a real build never passes it.
  const embedFn = opts.embed ?? embed;
  const { vectors, embedded } = await embedCached(texts, cacheDir, embedFn);
  const realPerAnswer = src.labels.length ? src.examples.length / src.labels.length : 0;
  const pWeight = prototypeWeight(realPerAnswer);

  const out = {
    format: FORMAT_VERSION, name: src.name, type,
    question: src.question ?? null, labels: src.labels,
    encoder: ENCODER_ID, dim: DIM,
    compiledAt: new Date().toISOString(),
    examples: src.examples.length, embedded,
  };

  if (type === 'tags') {
    const K = src.labels.length;

    // Each row carries a SET of tags. A hand-written example may have several; a
    // generated prototype is written for exactly one, and counts as a negative for
    // every other tag — which is what makes one-vs-rest work with no examples at all.
    const yMulti = rows.map(e => new Set(
      (e.proto ? [e.label] : (e.labels ?? [])).map(l => src.labels.indexOf(l)).filter(i => i >= 0)));
    const rowWeight = rows.map(e => e.proto ? pWeight : 1);

    out.heads = fitTagHeads(vectors, yMulti, K, DIM, { rowWeight }).map(packHead);

    // Per-tag cross-validation: each tag is its own yes/no question, so a tag that is
    // rare across the set cannot hide behind the ones that are common.
    const per = src.labels.map((label, k) => {
      const y = yMulti.map(set => set.has(k) ? 1 : 0);
      if (!y.includes(1) || !y.includes(0)) return { label, accuracy: null, macroRecall: null, n: 0 };
      const ev = crossValidate(vectors, y, 2,
        (X, yy) => fitHead(X, yy, 2, DIM, { weights: balanceWeights(yy, 2) }),
        (h, x) => softmax(h, x));
      return { label, accuracy: ev.accuracy, macroRecall: ev.macroRecall,
               n: y.filter(Boolean).length };
    });
    const scored = per.filter(p => p.macroRecall !== null);
    const macro = scored.length
      ? +(scored.reduce((a, p) => a + p.macroRecall, 0) / scored.length).toFixed(4) : 0;

    // Gates, which tags did without until 2026-09-18 — the branch cross-validated every tag and
    // then fitted neither a threshold nor a floor, so the runtime fell back to the hardcoded
    // 0.25/0.7 that fitted gates exist to replace. Measured on a 12-tag CLINC task, that floor
    // refused 32% of in-scope input while `suggest` was unreachable.
    //
    // The threshold has to be fitted on the WHOLE TAG SET, not per tag. Fitting it on one-vs-rest
    // scores puts it at 0.501 on every task however hard, because a tag that fires is nearly
    // always right when it is competing against K-1 obvious negatives. What separates is the
    // item-level confidence — the weakest flag raised — which on a 60-tag task runs 0.86 when the
    // set is right and 0.58 when it is not. So: fold over items, fit all K heads on the training
    // side, and score the tag set each fold produces.
    const setScores = [];
    for (const { train: tr, test: te } of foldIndices(vectors.length, FOLDS, SEED)) {
      if (tr.length < 2 || !te.length) continue;
      const heads = fitTagHeads(tr.map(i => vectors[i]), tr.map(i => yMulti[i]), K, DIM,
        { rowWeight: tr.map(i => rowWeight[i]) });
      for (const i of te) {
        const scores = heads.map(h => softmax(h, vectors[i]).probs[1]);
        const fired = scores.map((p, k) => [p, k]).filter(([p]) => p >= 0.5);
        const want = yMulti[i];
        const ok = fired.length === want.size && fired.every(([, k]) => want.has(k));
        // Same definition the runtime uses, so the threshold is fitted on the number it gates.
        const c = fired.length ? Math.min(...fired.map(([p]) => p)) : 1 - Math.max(...scores);
        setScores.push({ ok, c });
      }
    }
    const tagGate = setScores.length >= 20 ? thresholdAt(setScores) : { threshold: 1, coverage: 0 };

    const floorTarget = 0.05;
    let familiarityFloor = 0.25, floorSamples = 0;
    const realIdx = rows.map((e, i) => (e.proto ? -1 : i)).filter(i => i >= 0);
    if (realIdx.length >= 4 * K) {
      const fX = realIdx.map(i => vectors[i]);
      const fY = realIdx.map(i => [...yMulti[i]][0] ?? 0);
      const fams = [];
      for (const { train: tr, test: te } of foldIndices(fX.length, FOLDS, SEED)) {
        if (tr.length < 2 || !te.length) continue;
        const fref = packAsRef(tr.map(i => fX[i]), tr.map(i => fY[i]));
        for (const i of te) fams.push(familiarityScore(fX[i], fref));
      }
      fams.sort((a, b) => a - b);
      floorSamples = fams.length;
      if (fams.length >= 20) familiarityFloor = +Math.max(0.02, fams[Math.floor(floorTarget * fams.length)]).toFixed(3);
    }

    // Tag shims get a reference set too — without it a tag shim cannot tell familiar
    // input from foreign, which is the whole point of not answering.
    const ref = packReference(vectors, yMulti.map(set => [...set][0] ?? 0));
    out.reference = { q: bytesToBase64(ref.q), scales: bytesToBase64(ref.scales), spread: ref.spread,
                      count: ref.count, labels: ref.labels };
    out.head = 'linear';

    const scorable = src.examples.length >= 4 * K;
    out.report = {
      perTag: per, macroRecall: macro, chance: 0.5,
      provisional: !scorable, scorable,
      // Withheld the same way a classify threshold is: a number nobody can justify is worse than
      // no number, because the runtime will act on it.
      suggestedThreshold: scorable ? tagGate.threshold : 1,
      coverageAtThreshold: scorable ? tagGate.coverage : 0,
      provisionalThreshold: scorable ? null : tagGate.threshold,
      familiarityFloor, familiarityFittedOn: floorSamples, floorFittedOn: 'examples',
      prototypes: protoRows.length, prototypeWeight: +pWeight.toFixed(3),
      scoredOn: src.examples.length ? 'examples' : 'prototypes',
      recall: Object.fromEntries(per.map(p => [p.label, p.macroRecall])),
      bytes: headBytes(2, DIM) * K,
      referenceBytes: ref.q.byteLength + ref.scales.byteLength,
      notes: [
        ...(macro - 0.5 < 0.06 && scorable
          ? [{ level: 'error', code: 'not-learnable',
               text: 'These tags are not separable from embeddings — each one scores at chance. ' +
                     'Sentence embeddings encode what text is ABOUT; if a tag turns on wording or ' +
                     'a keyword, a rule will beat this shim outright.' }]
          : []),
        ...per.filter(p => p.macroRecall !== null && p.macroRecall - 0.5 < 0.08)
              .map(p => ({ level: 'warn', code: 'weak-tag',
                text: `"${p.label}" scores ${(100 * p.macroRecall).toFixed(0)}% against a 50% coin flip — ` +
                      `it is not being learned. Either it overlaps another tag, or what triggers it is not semantic.` })),
        ...diagnoseContext(src, protoRows.length),
        ...(!scorable ? [{ level: 'info', code: 'too-early',
              text: `${src.examples.length} hand-written examples across ${K} tags is too few to judge — ` +
                    `the score comes from generated prototypes. Roughly ${4 * K} real examples is where it starts meaning something.` }] : []),
      ],
    };
    } else {
    const K = src.labels.length;
    const y = rows.map(e => src.labels.indexOf(e.label));
    const base = balanceWeights(y, K);
    const weights = base.map((w, i) => rows[i].proto ? w * pWeight : w);

    // A linear head gives each answer one direction, so its decision region is convex.
    // An answer whose examples sit in several unrelated places cannot fit in one.
    // kNN has no such constraint and costs nothing extra — the reference set is
    // already shipped for familiarity. So measure both and keep the better one.
    // Scored on REAL examples only where any exist — a shim graded on the prototypes
    // it was trained on is marking its own homework. With no real examples the
    // prototypes are all there is, and the report says so.
    const realIdx = rows.map((r, i) => r.proto ? -1 : i).filter(i => i >= 0);
    const evalIdx = realIdx.length >= 2 * K ? realIdx : rows.map((_, i) => i);
    const eX = evalIdx.map(i => vectors[i]), eY = evalIdx.map(i => y[i]);
    const protoIdx = evalIdx === realIdx ? rows.map((r, i) => r.proto ? i : -1).filter(i => i >= 0) : [];
    const always = protoIdx.length ? { X: protoIdx.map(i => vectors[i]), y: protoIdx.map(i => y[i]), scale: protoIdx.map(() => pWeight) } : null;
    const foldWeights = (yy, scale) => { const bal = balanceWeights(yy, K); return scale ? bal.map((w, i) => w * scale[i]) : bal; };
    // How hard to regularise is a property of the task, not a constant. A clean, separable task
    // wants a light penalty — Banking77 goes 78 → 85 between 3e-3 and 1e-4 — and a noisy one wants
    // a firm one: `food`, `lease-clause` and `category` all cross-validate a point or two better at
    // the old 3e-3. So measure it, the same way the head itself is chosen. The strongest penalty
    // wins a tie, because it is the smaller model.
    const L2_GRID = [3e-3, 3e-4, 3e-5];
    const linearTried = L2_GRID.map(l2 => ({ l2, ev: crossValidate(eX, eY, K,
      (X, yy, scale) => fitHead(X, yy, K, DIM, { l2, weights: foldWeights(yy, scale) }),
      (h, x) => softmax(h, x), always) }));
    const linearBest = linearTried.reduce((a, b) => b.ev.macroRecall > a.ev.macroRecall ? b : a);
    const evLinear = linearBest.ev, l2 = linearBest.l2;
    const evKnn = crossValidate(eX, eY, K,
      (X, yy) => ({ ref: packAsRef(X, yy), K }),
      (m, x) => knnPredict(x, m.ref, m.K, 5) ?? { index: 0, probs: [], confidence: 0 }, always);

    // The third flat head: nearest class mean. Closed-form, so it is also the cheapest thing here
    // to cross-validate. Measured 2026-09-19 it beats both of the above at 8 examples per answer and
    // with prototypes alone, and loses to kNN on fine-grained answers once there is plenty of data.
    const evCentroid = crossValidate(eX, eY, K,
      (X, yy, scale) => fitCentroids(X, yy, K, DIM, { weights: foldWeights(yy, scale) }),
      (h, x) => centroidPredict(h, x), always);

    // Between the two single-direction heads, whichever measures better; the centroid on a tie,
    // because it is the one with nothing to tune. The first version of this preferred the centroid
    // unless linear won by two points, on the strength of the benchmarks — and on seventeen small
    // hand-built shims that was a regression on four of them. There the centroid wins five, ties two
    // and loses ten, mostly the shims with many examples per answer, which is the same shape
    // Banking77 showed at 32. Nearest-neighbours still has to beat the
    // better of the two by two points, as before.
    let useLinear = evLinear.macroRecall > evCentroid.macroRecall;
    let evSimple = useLinear ? evLinear : evCentroid;
    let useKnn = evKnn.macroRecall > evSimple.macroRecall + 0.02;

    // Accuracy picks the head; what a shim is FOR is the share of input it can act on. Usually the
    // two agree. When they do not, and the accuracy gap is inside the noise of the held-out rows —
    // less than one standard error — the pick is a coin toss on accuracy and should not be one on
    // usefulness. `fit` is the case that prompted this: centroid 64.4 against linear 62.8 on 74 rows,
    // one row's difference, while linear could act on 17.6% of input and the centroid on 2.7%. So a
    // rival within one standard error that can act on at least ten points more takes the pick. Ten
    // points because coverage on a few dozen rows is itself noisy, and a three-point edge is a row.
    const flat = [
      { name: 'centroid', ev: evCentroid }, { name: 'linear', ev: evLinear }, { name: 'knn', ev: evKnn },
    ].map(c => ({ ...c, coverage: calibrateAndGate(c.ev).gate.coverage }));
    const scorableForGate = src.examples.length >= 8 * K;   // same test as `scorable` below: no gate, nothing to break a tie on
    let picked = flat.find(c => c.name === (useKnn ? 'knn' : useLinear ? 'linear' : 'centroid'));
    let coverageOverride = null;
    if (scorableForGate && opts.coverageTiebreak !== false) {
      const n = picked.ev.evaluated || 1, p = picked.ev.macroRecall;
      const se = Math.sqrt(Math.max(p * (1 - p), 1e-6) / n);
      const rival = flat.filter(c => c !== picked && picked.ev.macroRecall - c.ev.macroRecall < se
                                     && c.coverage >= picked.coverage + 0.10)
                        .sort((a, b) => b.coverage - a.coverage)[0];
      if (rival) {
        coverageOverride = { from: picked.name, to: rival.name, accuracyGap: +(picked.ev.macroRecall - rival.ev.macroRecall).toFixed(4),
                             standardError: +se.toFixed(4), coverage: [picked.coverage, rival.coverage] };
        picked = rival; useKnn = rival.name === 'knn'; useLinear = rival.name === 'linear'; evSimple = rival.ev;
      }
    }

    // A tree is the third structure, measured the same way: each fold learns its own grouping
    // from only its training rows, so the score is not flattered by groups built on the test
    // rows. Chosen only when it beats the better flat head by a point — on a narrow set with
    // plenty of data it does not (Banking77 at 32/answer), and a tree that merely ties is
    // extra structure for nothing. A grouping pinned in the source is always used.
    const pinned = src.tree && typeof src.tree === 'object' ? pinnedGroups(src.tree, src.labels) : null;
    let treeChoice = null;
    if (src.tree !== 'off' && (pinned || groupCandidates(K).length)) {
      const options = pinned ? [{ G: pinned.length, pinned }] : groupCandidates(K).map(G => ({ G }));
      const tried = options.map(o => ({ ...o, ev: crossValidate(eX, eY, K,
        (X, yy, scale) => fitTree(X, yy, K, DIM, o.pinned ?? learnGroups(X, yy, K, o.G), { weights: foldWeights(yy, scale) }),
        (t, x) => predictTree(t, x), always) }));
      const best = tried.reduce((a, b) => b.ev.macroRecall > a.ev.macroRecall ? b : a);
      const flatBest = Math.max(evLinear.macroRecall, evKnn.macroRecall, evCentroid.macroRecall);
      treeChoice = {
        best, use: !!pinned || best.ev.macroRecall > flatBest + 0.01,
        summary: {
          pinned: !!pinned, flat: +flatBest.toFixed(4),
          tried: tried.map(t => ({ groups: t.G, macroRecall: +t.ev.macroRecall.toFixed(4) })),
        },
      };
    }
    const useTree = !!treeChoice?.use;
    const ev = useTree ? treeChoice.best.ev : useKnn ? evKnn : evSimple;
    out.head = useTree ? 'tree' : useKnn ? 'knn' : useLinear ? 'linear' : 'centroid';

    // Calibration. Measured on CLINC150: a 150-answer linear head is right 90% of the time while
    // reporting under 0.10 confidence on every decision (ECE 89%) — spread over many answers,
    // softmax probabilities are flat. One temperature, fitted on held-out predictions, makes the
    // number mean what it says without changing which answer wins.
    const { calibrated, calibration, gate } = calibrateAndGate(ev);
    const temperature = calibration.temperature;

    // The familiarity floor, fitted rather than assumed. The old constant 0.25 turned away 32% of
    // real CLINC150 requests; a floor allowing 5% false refusals sits near 0.05 there and still
    // catches ~79% of genuinely out-of-scope input. Each fold scores its held-out rows against a
    // reference built only from that fold's training rows.
    const floorTarget = 0.05;
    let familiarityFloor = 0.25, floorSamples = 0;
    if (evalIdx.length >= 4 * K) {
      const fams = [];
      for (const { train: tr, test: te } of foldIndices(eX.length, FOLDS, SEED)) {
        if (tr.length < 2 || !te.length) continue;
        const fref = packAsRef(tr.map(i => eX[i]), tr.map(i => eY[i]));
        for (const i of te) fams.push(familiarityScore(eX[i], fref));
      }
      fams.sort((a, b) => a - b);
      floorSamples = fams.length;
      // A floor of 0 would disable refusal entirely, which a handful of rows can easily produce.
      // 0.02 still turns away input that resembles nothing in the examples at all (gibberish and
      // other-domain text score 0.00), without the 0.25 constant's 32% false-refusal rate.
      if (fams.length >= 20) familiarityFloor = +Math.max(0.02, fams[Math.floor(floorTarget * fams.length)]).toFixed(3);
    }
    let floorFittedOn = 'examples';

    // Fitting the floor on the shim's own examples calibrates it to the distribution the shim was
    // BUILT from, and real input is a different distribution sitting lower. Measured on the shop
    // shims against real Amazon search queries: a floor aiming to refuse one real request in twenty
    // refused between one in three and five in six. The mechanism was fine — genuinely irrelevant
    // queries were still turned away at 96-98% — the threshold was calibrated on the wrong
    // population.
    //
    // `realSample` fixes that, and the thing it asks for is deliberately weak: text the app
    // believes is IN SCOPE, unlabelled. Not "this query means dresses", just "this is the kind of
    // thing we are for". Any deployed app has that on day one, before a single label exists.
    //
    // It must be in-scope. Fitting on raw traffic that is half out-of-scope would drag the floor to
    // zero and disable refusal entirely, which is the failure this whole gate exists to prevent.
    if (src.realSample?.length >= 20) {
      const vecs = await embedFn(src.realSample.map(String), { cacheDir });
      const fams = vecs.map(v => familiarityScore(v, packReference(vectors, y))).filter(v => v != null);
      if (fams.length >= 20) {
        fams.sort((a, b) => a - b);
        familiarityFloor = +Math.max(0.005, fams[Math.floor(floorTarget * fams.length)]).toFixed(3);
        floorSamples = fams.length;
        floorFittedOn = 'realSample';
      }
    }
    out.headChoice = {
      linear: +evLinear.macroRecall.toFixed(4), knn: +evKnn.macroRecall.toFixed(4),
      centroid: +evCentroid.macroRecall.toFixed(4),
      // What each could ACT on: coverage at its own 90% gate, after its own calibration.
      coverage: Object.fromEntries(flat.map(c => [c.name, c.coverage])),
      coverageOverride,
      l2: Object.fromEntries(linearTried.map(t => [t.l2, +t.ev.macroRecall.toFixed(4)])), l2Chosen: l2,
      perAnswer: src.labels.map((l, k) => {
        const rl = evLinear.confusion[k], rk = evKnn.confusion[k];
        const nl = rl.reduce((a, b) => a + b, 0), nk = rk.reduce((a, b) => a + b, 0);
        return { label: l, linear: nl ? +(rl[k] / nl).toFixed(3) : null,
                 knn: nk ? +(rk[k] / nk).toFixed(3) : null };
      }),
    };
    // The flat head always ships, so a runtime that predates trees still has something to run.
    out.heads = [packHead(fitHead(vectors, y, K, DIM, { weights, l2 }))];
    out.l2 = l2;
    // Centroids are fitted on every row the linear head sees, prototypes included and faded by the
    // same weight. Class balancing is a no-op for a per-answer mean, so only the fade matters.
    if (out.head === 'centroid') out.centroid = packHead(fitCentroids(vectors, y, K, DIM, { weights }));
    let treeGroups = null, treeBytes = 0;
    if (useTree) {
      treeGroups = treeChoice.best.pinned ?? learnGroups(eX, eY, K, treeChoice.best.G);
      const tree = fitTree(vectors, y, K, DIM, treeGroups, { weights });
      out.tree = packTree(tree, packHead);
      treeBytes = tree.leaves.reduce((a, h) => a + (h ? headBytes(h.K, DIM) : 0), 0);
    }
    // Below a few examples per answer, a held-out score swings wildly with the split —
    // it is a measurement of the sample size, not of the task. Mark it PROVISIONAL
    // rather than hiding it: the person building the shim needs to watch the number
    // move as they add examples, they just must not quote it anywhere.
    // Per-row verdicts stay visible at every size; each one is a real prediction from
    // a head that never saw that row, and knowing WHICH rows miss is the whole point.
    const scorable = src.examples.length >= 8 * K;
    // `cal` and `scored` are the raw held-out predictions the temperature and the thresholds were
    // fitted on. They are build-time scaffolding: useful here, and 465KB of a 1MB weights file if
    // they ship. Everything downstream reads the fitted numbers, not the samples.
    const { cal: _cal, scored: _scored, ...evReport } = ev;
    out.report = {
      ...evReport,
      scorable,
      provisional: !scorable,
      // A threshold learned from too few points would gate everything; stay closed.
      // But closed is not the same as unusable. The number IS computable from
      // prototypes — it is just optimistic, because generated text clusters more
      // tightly than the real thing (measured: 46% synthetic vs 23% hand-written on
      // the same shim). So ship both. `suggestedThreshold` is the gate you can trust
      // and stays shut until real examples earn it; `provisionalThreshold` is the
      // computed one, and a caller who passes it to isConfident has chosen to.
      suggestedThreshold: scorable ? gate.threshold : 1,
      coverageAtThreshold: scorable ? gate.coverage : 0,
      provisionalThreshold: scorable ? null : gate.threshold,
      calibration,
      // The held-out rows the gate was read off: [true answer, calibrated confidence, right?].
      // The gate is a promise about a stream shaped like these rows — every answer about equally
      // common — and no real stream is. `shim.refitGate()` reweights them to the answer mix it
      // estimates from unlabelled traffic, which it can only do if they ship.
      gateRows: scorable ? calibrated.map((s, i) => [ev.cal[i].y, +s.c.toFixed(4), s.ok ? 1 : 0]) : null,
      // Below this familiarity the shim is looking at something unlike anything it has seen.
      // Fitted per shim where there is enough data; otherwise the old constant.
      // `floorFittedOn` matters more than the number: "examples" means the floor is calibrated to
      // the distribution the shim was built from, which measured 34-83% false refusal on real
      // queries. "realSample" means it was fitted on text the app says is in scope.
      familiarityFloor, familiarityFittedOn: floorSamples, floorFittedOn,
      chance: +(1 / K).toFixed(4),
      labels: src.labels,
      recall: Object.fromEntries(src.labels.map((l, k) => {
        const n = ev.confusion[k].reduce((a, b) => a + b, 0);
        return [l, n ? +(ev.confusion[k][k] / n).toFixed(3) : null];
      })),
      // A centroid shim ships two flat heads: the centroids it uses and the linear one it falls back to.
      bytes: headBytes(K, DIM) * (out.centroid ? 2 : 1) + treeBytes,
      head: out.head,
      structure: treeChoice ? {
        chosen: useTree ? 'tree' : 'flat', ...treeChoice.summary,
        groups: treeGroups ? treeGroups.map(g => g.map(i => src.labels[i])) : null,
      } : { chosen: 'flat', tried: [], reason: src.tree === 'off' ? 'tree: "off" in source' : `${K} answers is too few for grouping to matter` },
      prototypes: protoRows.length,
      prototypeWeight: +pWeight.toFixed(3),
      scoredOn: realIdx.length >= 2 * K ? 'examples' : 'prototypes',
      notes: [...diagnose(ev, K, src.labels, src.examples.length, texts),
              ...diagnoseModality(out.headChoice, src.labels),
              ...(out.headChoice.coverageOverride && out.head === out.headChoice.coverageOverride.to
                ? [(o => ({ level: 'info', code: 'head-by-coverage',
                    text: `Shipping the ${o.to} head rather than ${o.from}. Their accuracy is inside the noise of ` +
                          `${ev.evaluated} held-out rows (${(100 * o.accuracyGap).toFixed(1)} points apart, ±${(100 * o.standardError).toFixed(1)}), ` +
                          `but ${o.to} can act on ${(100 * o.coverage[1]).toFixed(0)}% of input at the 90% gate and ${o.from} on ${(100 * o.coverage[0]).toFixed(0)}%.` }))(out.headChoice.coverageOverride)]
                : []),
              ...diagnoseContext(src, protoRows.length),
              ...diagnoseTree(treeChoice, useTree, treeGroups, src.labels)],
    };
    // Ship the training set itself, quantised. Familiarity reads the semantic half;
    // a kNN head reads all of it. Same bytes either way.
    const ref = packReference(vectors, y);
    out.reference = { q: bytesToBase64(ref.q), scales: bytesToBase64(ref.scales), spread: ref.spread,
                      count: ref.count, labels: ref.labels };
    out.report.referenceBytes = ref.q.byteLength + ref.scales.byteLength;
  }
  return out;
}
