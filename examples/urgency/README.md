# urgency

One shim, start to finish: a source file, the weights compiled from it, and a script that decides.

```bash
pnpm install
pnpm --filter @zerowidth/shims-example-urgency decide
```

```
act      now       confidence 1.00  familiarity 0.81  the whole site is down for us
act      soon      confidence 0.57  familiarity 0.63  invoice total looks wrong, need it fixed by friday
act      whenever  confidence 0.99  familiarity 1.00  typo on the pricing page
refuse   —         confidence 0.84  familiarity 0.00  what is the capital of France
```

The last line is the one to look at. The head is 84% sure that a geography question is `whenever`,
because something has to win. Familiarity is 0: nothing in the shim's examples looks like this, so it
refuses. Confidence alone would have acted.

`urgency.shim.json` is 35 hand-written messages, about twelve per answer, all shaped like support and
business mail. Edit it, then rebuild:

```bash
pnpm --filter @zerowidth/shims-example-urgency build
```

```
✓ urgency classify · 35 examples · 4.6KB
   accuracy 97% (95%: 85%–99%)   balanced 97% (chance 33%)   35 held out · linear head
   90% accurate above confidence 0.494 — covers 100% of inputs
```

That score is measured on the shim's own held-out examples, which read easier than real traffic.
Treat it as a ceiling. Pass your own messages to see how it does on them:

```bash
node decide.mjs "our whole team is locked out" "no rush on this one"
```
