// node examples/urgency/decide.mjs "the whole site is down for us"
//
// Loads the compiled shim and decides. The first run downloads the encoder (32.4MB) and caches it;
// after that a decision takes a few milliseconds and nothing leaves the machine.
import fs from 'node:fs/promises';
import { Shim } from '@zerowidth/shims-sdk';

const weights = JSON.parse(await fs.readFile(new URL('./urgency.weights.json', import.meta.url), 'utf8'));
const shim = new Shim(weights);

const messages = process.argv.slice(2).length ? process.argv.slice(2) : [
  'the whole site is down for us',
  'invoice total looks wrong, need it fixed by friday',
  'typo on the pricing page',
  'what is the capital of France',
];

for (const text of messages) {
  const r = await shim.decide(text);
  // act: clears the gate this shim earned at build. suggest: offer it, do not fire it.
  // refuse: nothing like this in its examples, so the honest answer is silence.
  const said = r.action === 'refuse' ? '—' : r.answer;
  console.log(`${r.action.padEnd(8)} ${String(said).padEnd(9)} confidence ${r.confidence.toFixed(2)}  familiarity ${r.familiarity.toFixed(2)}  ${text}`);
}
