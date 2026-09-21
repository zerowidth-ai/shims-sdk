# The runtime

The half of [`@zerowidth/shims-sdk`](../README.md) an app imports. One encoder is shared by every
shim in an app; each shim is a few thousand numbers on top of it.

```js
import { Bank } from '@zerowidth/shims-sdk';
import urgency from '../machines/urgency.weights.json';
import tone    from '../machines/tone.weights.json';

const bank = new Bank({ urgency, tone });
const out = await bank.decide(message);
// { urgency: { answer: 'now', confidence: 0.82, probs: {…} },
//   tone:    { answer: 'neutral', confidence: 0.64, probs: {…} } }
```

`Bank.decide` runs the encoder **once** no matter how many shims it holds. Encoding is
the expensive part; each additional shim costs a dot product.

## What a shim costs

| | |
|---|---|
| encoder, shared by all shims | **32.4MB** (`bge-small-en-v1.5`, q8), plus the ONNX runtime's `.wasm` (22MB, about 5MB compressed). Both cached after first load |
| a 3-answer shim, 24 examples | **18KB** — 9KB of heads, 9KB of examples |
| a 10-answer shim, 180 examples | 85KB — 15KB of head, 70KB of examples |
| a 16-answer shim carrying a vocabulary, 1,030 rows | **423KB** — 25KB of head, 398KB of examples |
| an 85-answer shim carrying a vocabulary, 2,016 rows | 910KB — 130KB of head, 780KB of examples |
| a short query | **~15ms** in a browser |
| a support-message-length input | ~30ms |
| each extra shim on the same text | microseconds |

A head is `K × (392 + 1)` float32 — 384 embedding dimensions plus 8 surface features, one
weight each per answer, plus a bias. A shim whose build chose the class-mean head ships two: the
centroids it uses and the linear head it falls back to.

**But the head is the smaller half.** Every shim also ships its examples as int8 vectors —
392 bytes each — and that runs from about the size of the head to sixteen times it. They are
not optional: familiarity, refusal, long-input chunk routing and the nearest-neighbour head
all read them. A size that counts only the head is the weights file minus the model.

The two large rows are the price of a particular kind of task. Where the input is a NAME —
a garment, a grocery item — what a shim lacks is vocabulary, the head that works is the one
that remembers, and a lookup is as big as what it looks up (measured: 43% → 83% and 35% → 67%
for 3× and 2.3× the weights). A shim that reads what someone *means* stays in the first two
rows. These are runtime bytes; the JSON on disk is about 1.4× larger.

Latency is measured in headless Chrome on an M-series Mac, WASM, single thread. It has not
been measured on mid-range hardware.

## Authoring

A shim's source is a JSON file you edit by hand:

```json
{
  "name": "urgency",
  "type": "classify",
  "question": "How quickly does this need attention?",
  "labels": ["now", "soon", "whenever"],
  "examples": [
    { "text": "Production is down, every request returns 503.", "label": "now" },
    { "text": "Typo on the pricing page.", "label": "whenever" }
  ]
}
```

`shim-compile` turns that into `urgency.weights.json`. **Both files are committed.** The
examples are reviewable in a pull request — a teammate can argue with a label — and the
weights are a build artifact you do not read, like a lockfile. The same source file rebuilds to
the same heads, gates and report; the [compiler's page](compiler.md) says what that
does and does not cover.

The report the build hands back carries an interval, not just a number:

```
accuracy 88% (95%: 69%–96%)   balanced 88%   24 held out · linear head
```

Read the interval. Across 120 small shims, held-out accuracy sat a standard deviation of
7.5–8.8 points from the reported figure whenever the answers were confusable. A 24-example
shim reporting 88% is telling you something between 69% and 96%.

## Types

- `classify` — one answer of K. Implemented.
- `tags` — any of K, independent binary heads. Implemented. Confidence is the weakest tag
  raised, and the build fits the gate on whole tag sets, folding over items and refitting every
  head per fold. Measured on CLINC built as a 60-tag task, it acts on 64% of inputs at 95%
  precision, where acting on everything scored 72%. A tags shim built from prototypes alone has
  nothing to fit on, so its gate ships withheld and the report says so. `refitGate()` and
  `recalibrate()` decline on a tags shim and say why, and long input is always decided on its most
  familiar piece: a set of tags has no distribution to pool.
- `schema` — several fields off one encode, mixed types. Format reserved, not built.
- `extract` — spans lifted from the text, using per-token vectors. Not built; the
  encoder already computes those vectors and mean-pooling discards them.

## Watching real traffic

Two things in this SDK need traffic and cannot be fixed without it.

```js
import { observe, outcome, ring } from '@zerowidth/shims-sdk';

const log = ring(2000);
observe(log.push);                      // or a sink writing to IndexedDB, a file, a POST

const r = await shim.decide(text);      // r.id exists only while something is observing
if (userWentWithIt)  outcome(r.id, 'accepted');
if (userPickedOther) outcome(r.id, 'corrected', { label: theirChoice });

log.summary();
// { decisions, acted, suggested, refused, medianWords, medianMs, withOutcome }
```

Costs nothing when nobody is watching: no sink, no id, no allocation.

Every way of asking a shim records: `shim.decide()`, `Bank.decide()` (one row per shim, each with
its own `id`, tagged `via: 'bank'` and the field it filled), `System.run()` (one row per shim the
system asked, tagged with the system's name — and `run()`'s result carries the `id` of the step
whose answer became the system's, which is the one to report an `outcome` against), and
`AdaptiveBank.decide()` (what was *said*, with `memory: 'yes' | 'no'` when a remembered verdict
spoke). A long message is recorded with the vector of the piece it was decided on.

**`shim.refitFloor(vectors)`** — the familiarity floor the build ships is a percentile of
the shim's *own examples*, so it is calibrated to the distribution it was built from.
Measured against real search queries, floors aiming to wrongly refuse one request in
twenty refused between one in three and five in six. Refitting needs a few hundred vectors
of text **the app believes is in scope**; handing it raw traffic that is half out-of-scope
drags the floor to zero and disables refusal entirely.

**`shim.recalibrate(rows)`** — refits the temperature on real outcomes. Needs 30 per shim.

**`shim.refitGate(vectors)`** — the build's promise, *"above X this shim is 90% accurate"*, is read
off held-out examples in which every answer is about equally common. Uneven traffic alone does not
break it. What breaks it is the **hard answers being the busy ones**: measured on exactly that, a
gate promising 90% delivered 80–84%. Refitting needs about a hundred vectors of in-scope input and
**no labels** — the shim estimates its own answer mix from them and re-reads the gate for that mix
(80–84% → 89–92%). The cost is honest: it acts on less. It cannot see an answer the shim gets
*confidently wrong* — the mix is estimated from the shim's own predictions — so where the busy
answers are the shim's blind spots it helps (66% → 74%) and does not restore the promise; only real
outcomes do. It declines, and says why, on a shim whose gate the build withheld.

`log.summary()` is worth reading in the first week, and none of it needs a label. A high
`refused` means the floor is wrong rather than the model. A `withOutcome` of zero means
`recalibrate` will never become usable, however long it runs — `refitFloor` and `refitGate` need only
the vectors.

## Where the encoder comes from

Nothing a user types leaves the device. The first page load does download the encoder (32.4MB)
from the Hugging Face hub, and it is cached afterwards. The ONNX runtime is a `.wasm` file: a
bundler that follows `new URL(…, import.meta.url)`, as Vite does, ships it with your app's own
assets, and where none does, transformers.js fetches it from jsDelivr.

To serve both yourself, copy the model repository's files (and, if your bundler does not ship them,
the runtime's `.wasm` files) to your own host, then name that host once at startup:

```js
import { preload } from '@zerowidth/shims-sdk';

await preload({
  remoteHost: 'https://static.example.com/',
  remotePathTemplate: 'models/{model}/',          // → …/models/Xenova/bge-small-en-v1.5/onnx/model_quantized.onnx
  wasmPaths: 'https://static.example.com/ort/',
});
```

`preload()` forwards these to the SDK's worker, and they are read by whichever call loads the
encoder first, so make this call before the first `decide()`. The model is
[`Xenova/bge-small-en-v1.5`](https://huggingface.co/Xenova/bge-small-en-v1.5), an ONNX conversion of
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), which is MIT licensed. Weights
are compiled against one encoder and record which; serve the same model and quantisation the
compiler used, or the vectors will not match.

## What shims are bad at

**Input that leans on knowing things.** A shim reads meaning well and knows little, and the place
that shows is the search box. On real short queries — typed by a person, two to five words, complete
in themselves — a frontier model given only the answer names scores **100%** on 147 clothing queries
and 94% on 300 grocery ones; shims built for those tasks score **49%** and **32%**. "anorak",
"henley?", "bodycon" are not ambiguous and not short of information. They are vocabulary, and a 32MB
encoder with nine examples per answer does not have it. More examples per answer is the lever
(every example is a word the shim now knows); a bigger encoder is the other.

It is tempting to blame short input. Cut a sentence to its first two words and a shim falls from
89% to 13%. But that removes the information and then blames the reader: a frontier model on the
same truncated items scores **20% at two words** (shim 21%), 36% at three (37%), 77% at five (65%), 91% at eight (89%) — the shim is at the
ceiling the input allows except around five words, where it trails by twelve. Length as such costs
a shim almost nothing. Before choosing where a shim goes, look at what people type there and ask
whether answering it takes reading or takes knowing.

**Surface rather than meaning.** A head can only separate what the encoder already
distinguishes, and the encoder is trained for meaning. Questions, formatting, keyword
presence, word order — sentence embeddings deliberately discard these. Detecting a question
mark scores 97% with `text.includes('?')` and 56% with an encoder plus a trained head. The
compiler checks for this and fails the build with `not-learnable` when balanced accuracy
sits near chance.

**Answers that overlap.** Nobody can reliably separate `Wool` from `Merino Wool` either.
Answer count is cheap; confusable answers are not. Use the confusion matrix in the report —
merging four mutually confused pairs took one shim from 54% to 72%.

**Generating anything.** A shim cannot write. It can tell you a group of inputs resembles
each other and none of its answers fit; it cannot tell you what to call them.
