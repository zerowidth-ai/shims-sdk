# shims

Tiny task-specific decision models that run **in the client**. One shared sentence encoder
(32.4MB, cached by the browser) serves every shim in an app; each shim adds a head plus its own
examples, quantised — about 18KB for a three-answer shim, a few hundred for one that carries a
vocabulary. No server call at decision time.

```bash
npm install @zerowidth/shims-sdk
```

One package: the runtime an app imports, and the compiler and its CLI behind `@zerowidth/shims-sdk/compile`.
A bundler ships only what the app imports, so an app that only decides never carries the compiler.
An app that builds shims in the page can: the compiler runs in a browser as well as in Node.

```js
import { Shim } from '@zerowidth/shims-sdk';
import urgency from './machines/urgency.weights.json';   // committed like any other file

const shim = new Shim(urgency);
const r = await shim.decide('the whole site is down for us');
// { answer: 'now', confidence: 0.86, familiarity: 0.91, action: 'act', probs: {…} }

if (r.action === 'act') route(r.answer);      // clears the gate it earned at build
else if (r.action === 'suggest') offer(r.answer);
else askAPerson();                            // 'refuse' — nothing like this in its examples
```

No encoder wiring: the SDK starts its own worker the first time anything needs a vector, so inference
never lands on the UI thread. `preload()` warms it early; `setEmbedder()` takes over for a server, a
test, or an app that already runs a worker of its own.

A shim is authored as JSON — a question, its answers, and examples — and compiled to weights:

```bash
npx shim-compile machines     # compiles machines/*.shim.json, writes *.weights.json beside them
```

[`examples/urgency`](examples/urgency) is one shim start to finish: the source, its compiled weights,
and a script that decides — including the message it refuses.

## What it does

- **Decides between answers your product already owns** — filters, queues, intents, clause types.
- **Declines.** Every shim ships its training set as int8 vectors and scores how close an input is
  to anything it has seen, so it can stay silent instead of guessing.
- **Earns the right to act.** The build cross-validates and hands back a confidence threshold only
  when there are enough real examples to justify one, and reports it as an interval rather than a
  number.
- **Picks its own head.** The build cross-validates three ways of deciding — a class mean per
  answer, a fitted linear head, nearest neighbours over the shipped examples — and ships whichever
  measures best on that shim's examples, and how firmly to regularise the linear one the same way.
  The choices are made at build time and recorded in the weights file; rebuild with more examples
  and they can change.
- **Groups itself when that helps.** With enough answers the compiler tries a tree — a
  nearest-neighbour root over learned groups, a small linear head per group — and ships it only if
  it beats a flat shim.
- **Learns from use.** `@zerowidth/shims-sdk/adapt` remembers accepted and rejected suggestions and
  applies them to closely-worded inputs immediately; examples fold in at the next build.
- **Reports what it saw.** `observe()` captures every decision — vector, answer, gates, latency,
  word count — so a deployment can refit its floor and its calibration on real traffic. Off by
  default and free when off.

## What it does not do

Generate text. Extract values (sizes, prices, dates — rules do that better). Judge things nobody
can check. Work in languages the encoder does not cover.

## Tests

```bash
pnpm test             # ~2 seconds, no model and no network
pnpm test:mutations   # puts real past bugs back, one at a time, and expects the test written for each to fail
```

The suite compiles and runs shims on a fake encoder (`test/world.mjs`) whose geometry is
known, so a test can state a promise — *the gate never lands above the row it was read from* — rather
than a number. When you fix a bug, add its test, then add the bug to `test/mutations.mjs`: a test that
has never been seen to fail has not been shown to test anything.

## Where a shim works

What decides this is how much the input leans on **knowing things** rather than on how long it is.
A shim reads meaning well and knows little. On a sentence cut to its first two words a shim scores
21% — and so does a frontier model (20%): that is the input having nothing left in it, not the
shim failing. But on real two-word queries, which are short *and* complete — "anorak", "pinafore
denim" — a frontier model scores 94–100% where a shim scores 26–62%, because telling outerwear
from dresses there is vocabulary, and a 32MB encoder with a few examples per answer does not have it.

| shape | typical input | how it goes |
|---|---|---|
| a support message, a form, a clause | 10–90 words | the good case |
| a benchmark utterance | ~8 words | 88–93% |
| a site search box | 2–5 words, mostly product and brand vocabulary | the hard case — 40–70 points under a frontier model |

Before putting a shim somewhere, measure what people type there. `observe()`'s `summary()` reports
`medianWords` for exactly this reason.

## Before shipping one

Three things the build cannot know, and all of them need real traffic rather than more examples:

- **The familiarity floor does not transfer.** It is fitted on the shim's own examples, and against
  real queries that refused 34–83% of in-scope traffic. Collect a few hundred in-scope inputs and
  call `refitFloor()` before trusting the gate.
- **The confidence gate assumes every answer is equally common.** Uneven traffic alone leaves it
  intact; hard answers that are also the busy ones take a gate promising 90% down to 80–84%. About
  a hundred unlabelled inputs and `refitGate()` bring it back (89–92%), at the cost of acting less.
- **A score is only as real as the rows it was scored on.** A *test set* written by a model measures
  about twice as easy as text a person types. A shim *built* from generated text and cross-validated
  on it errs the other way — measured, its report came in under its accuracy on human text. Neither
  is a measurement of your traffic.

## What is in the package

| import | |
|---|---|
| `@zerowidth/shims-sdk` | the runtime: `Shim`, `Bank`, the encoder, `observe()` and the refits. [docs/runtime.md](docs/runtime.md) |
| `@zerowidth/shims-sdk/system` | `System`: several shims wired into one decision |
| `@zerowidth/shims-sdk/adapt` | `AdaptiveShim`, `AdaptiveBank`: remember corrections between builds |
| `@zerowidth/shims-sdk/compile` | `compileShim`: `*.shim.json` → weights and a report a reviewer reads. [docs/compiler.md](docs/compiler.md) |
| `@zerowidth/shims-sdk/compile/system` | `compileSystem`: checks a system's wiring against the shims it names |
| `shim-compile` | the CLI over both |
| `/math` `/format` `/familiarity` `/tree` `/surface` `/observe` | the pieces, for tooling and experiments |

Plain ES modules with hand-written type declarations, and no build step. The compiler and the runtime
ship as one package because a weights file records the format it was compiled to and the runtime
reads that format: one version means they cannot disagree.

## Status

Pre-1.0. Interfaces change. Every number in these docs comes from public benchmarks, generated
text, or examples we wrote; none of it has met a real deployment's traffic yet.
[docs/runtime.md](docs/runtime.md) says what each one was measured on, and what shims are bad at.

## License

[Apache-2.0](LICENSE)
