# Changelog

## 0.1.1 — 2026-09-22

- `compileShim` runs in a browser. The vector cache was the only part of the compiler that
  touched a filesystem; its Node modules now load on first use, and only when a caller passes
  `cacheDir`. Leave it off and pass `{ embed: encode }` to build on the encoder the page already has.
- `bytesToBase64` from `@zerowidth/shims-sdk/format`: the bytes of any typed array as base64, in
  Node or a browser.

## 0.1.0 — 2026-09-21

First public release: `@zerowidth/shims-sdk`, the runtime and the compiler in one package.

### The runtime — `@zerowidth/shims-sdk`

- `Shim`, `Bank`, `System`, `AdaptiveShim` and `AdaptiveBank`: decide, and report `act`, `suggest` or
  `refuse` from the confidence gate and familiarity floor fitted at build.
- Three heads, chosen per shim at build: a fitted linear head, nearest neighbours, a class mean.
  Trees when the build measured one beating the flat shim.
- Long input is split and decided in pieces past 420 characters; `decide(text, { reduce: 'pooled' })`
  weighs every piece where the default reads the most familiar one.
- `observe()`, `outcome()` and `ring()` capture decisions on every path. `refitFloor()`,
  `refitGate()` and `recalibrate()` refit a shim's gates on real traffic.
- The encoder runs in a worker the SDK starts itself. `preload()` warms it, `setEmbedder()` replaces it.
- `preload({ remoteHost, remotePathTemplate, wasmPaths })` serves the encoder and the ONNX runtime
  from your own host.
- Every text is embedded in a forward pass of its own, so a vector depends on its text and nothing
  else: the compiler and the runtime see the same one, and weights rebuild exactly from an empty cache.
- Hand-written type declarations for every entry point.

### The compiler — `@zerowidth/shims-sdk/compile` and the `shim-compile` CLI

- `shim-compile <dir|file…> [--check]` and `compileShim(source, { cacheDir })`.
- Deterministic: the same examples produce the same heads, gates and report.
- The report carries held-out accuracy with a 95% interval, recall per answer, the confusion matrix,
  the confidence gate and its coverage, the familiarity floor, and notes on the task itself.
- `compileSystem` checks `*.system.json` wiring against the shims it names.
