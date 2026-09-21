// The encoder, off the main thread. Shipped with the SDK so no app has to write this file:
// a single embed is ~20–30ms, and a batch on the UI thread is a visibly frozen page.
import { embed, loadEncoder } from './index.mjs';

self.onmessage = async ({ data: { id, type, texts, opts } }) => {
  try {
    if (type === 'warm') { await loadEncoder(opts); return self.postMessage({ id, ok: true }); }
    const vectors = await embed(texts, opts);
    self.postMessage({ id, ok: true, vectors: Array.isArray(texts) ? vectors : [vectors] });
  } catch (e) {
    self.postMessage({ id, ok: false, error: String(e?.message ?? e) });
  }
};
