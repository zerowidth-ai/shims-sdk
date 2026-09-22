# The compiler

The other half of [`@zerowidth/shims-sdk`](../README.md). It reads a `*.shim.json` — a question, its
answers, and examples — and writes `*.weights.json`, the file an app imports and hands to the
[runtime](runtime.md). Installing the package puts `shim-compile` on your path:

```bash
npx shim-compile machines            # compiles every machines/*.shim.json
npx shim-compile machines --check    # compiles, writes nothing, exits 1 on any error — for CI
```

The first run downloads the encoder (32.4MB) and embeds every example. Vectors are cached in
`.shim-cache/` beside the sources, so a rebuild after adding one example embeds one example.
Add `.shim-cache/` to `.gitignore`.

**Commit both files.** The source is what a teammate reviews — they can argue with a label in a
pull request. The weights are a build artifact nobody reads, like a lockfile.

**How reproducible a build is.** The same source file produces the same heads, gates and report,
with or without a vector cache; only `compiledAt` changes, so a rebuild with nothing else changed
shows a one-line diff. Every text is embedded in a forward pass of its own, because a quantised
encoder's vector for a text shifts with whatever shares its batch (cosine 0.9995 against the same
text alone). A vector therefore depends on its text and nothing else, and the compiler sees exactly
the vector the runtime will. What still moves the numbers without any example changing is their
order: it decides which examples are held out together, and on a small shim that is worth a couple
of points on the gate.

## What the build decides for you

Everything below is measured on held-out examples during the build and recorded in the weights
file. Rebuild with more examples and any of it can change.

- **The head.** A class mean per answer, a fitted linear head, or nearest neighbours over the
  shipped examples. The build cross-validates all three and ships the one that measures best. For
  the linear head it cross-validates the penalty too.
- **Flat or tree.** With enough answers the build groups the ones the shim confuses, tests that
  tree against the flat shim, and keeps the tree only when it wins by a point.
- **The confidence gate.** The threshold above which the shim is 90% accurate on held-out
  examples, and the share of inputs that clears it. The build withholds the gate when there are
  too few examples to justify one; until then every answer should escalate.
- **The familiarity floor.** How far an input can sit from every shipped example before the shim
  declines to answer.
- **The temperature.** One number that makes reported confidence match observed accuracy. The
  build keeps it only when it improves held-out calibration.

## The report

```
✓ urgency classify · 35 examples · 4.6KB
   "How quickly does this need attention?"
   accuracy 97% (95%: 85%–99%)   balanced 97% (chance 33%)   35 held out · linear head
   recall: now 100%  soon 91%  whenever 100%
   90% accurate above confidence 0.48 — covers 100% of inputs
```

The size on the first line is the head alone. The weights file also carries the examples as int8
vectors, which is usually the larger half. Read the interval before the number: on a 24-example
shim the same accuracy figure is worth about nine points either way.

Notes follow the scores. An `error` note fails a `--check` run. The ones worth knowing:

| code | what it means |
|---|---|
| `draft` | An answer has fewer than 3 examples. The build returns a stub and no weights. |
| `not-learnable` | Balanced accuracy sits near chance. The encoder cannot see the distinction — it is probably about form (a question mark, a keyword) and a rule will do better. |
| `thin-answer`, `low-recall` | One answer has too few examples, or is found too rarely, for the overall score to describe it. |
| `multi-modal` | One answer's examples sit in several places. It may be several situations under one name. |
| `incoherent-answer` | One answer is found far less often than the best one despite having enough examples. It is usually a middle or "other" bucket, and a sharper definition helps where more examples will not. |
| `no-context` | Every answer comes from generated prototypes and no `context` says where real inputs come from, so the prototypes were written for an assumed setting. |
| `too-early` | Too few examples for the score to mean anything yet, so no gate ships. |
| `length-gap` | Examples are all long. Short inputs are unmapped. |

## Source format

```json
{
  "name": "urgency",
  "type": "classify",
  "question": "How quickly does this need attention?",
  "context": "Support tickets from a B2B invoicing product. One or two sentences, written in a hurry.",
  "labels": ["now", "soon", "whenever"],
  "examples": [
    { "text": "Production is down, every request returns 503.", "label": "now" },
    { "text": "Typo on the pricing page.", "label": "whenever" }
  ]
}
```

`name` becomes the file name, so keep it to lowercase letters, digits and hyphens. `type` is
`classify` (one answer of K) or `tags` (any of K; see the runtime README for what `tags` still
lacks). `tree` is `"auto"` (the default), `"off"`, or an object of named groups to pin. `prototypes` and `definitions` are optional maps from an answer to generated sentences
and a one-line meaning; prototypes train alongside examples and count for less as real examples
arrive. Avoid a middle or "other" answer: an answer defined by what it is not does not learn.

## Systems

A `*.system.json` beside the shims wires several into one decision: rules first, then a shim, a
route on its answer, a bank of independent fields. Systems compile to nothing. The CLI checks them —
a route to a shim that does not exist or is still a draft, an answer with nowhere to go, a route on
a later step, an invalid rule pattern, a step routing on a score that is still provisional — and the
app hands the spec to `System` from `@zerowidth/shims-sdk/system`.

## From code

```js
import { compileShim } from '@zerowidth/shims-sdk/compile';

const weights = await compileShim(source, { cacheDir: '.shim-cache' });
if (weights.draft) console.log(`needs ${weights.report.needed} more examples`);
else console.log(weights.report.accuracy, weights.report.accuracyInterval);
```

`cacheDir` is optional. Without it nothing touches the disk and every example is embedded on each
call. The CLI runs in Node 22 or later.

### In a browser

The compiler is arithmetic over vectors, and the vector cache is the only part of it that reaches a
filesystem, so it runs in a browser too. Leave `cacheDir` off and pass the encoder the page already
has:

```js
import { encode } from '@zerowidth/shims-sdk';
import { compileShim } from '@zerowidth/shims-sdk/compile';

const weights = await compileShim(source, { embed: encode });   // the SDK's worker does the embedding
```

A build with a few dozen examples takes about as long as embedding them, plus a few hundred
milliseconds of fitting. The result is the same weights file the CLI writes, so it can be saved from
the page and loaded by `new Shim(weights)` anywhere.
