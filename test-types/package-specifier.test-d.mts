// Type test: the package resolves by NAME through its `exports` map, the way a consumer imports
// it (here by self-reference), and every subpath finds its declarations.

import { Shim, Bank, observe, outcome, ring, preload, setEmbedder } from '@zerowidth/shims-sdk';
import type { ShimWeights, ShimSource, ShimReport, Decision, Embedder, Outcome } from '@zerowidth/shims-sdk';
import { fitHead } from '@zerowidth/shims-sdk/math';
import { validateSource, FORMAT_VERSION } from '@zerowidth/shims-sdk/format';
import type { ShimWeightsInput } from '@zerowidth/shims-sdk/format';
import { familiarityScore } from '@zerowidth/shims-sdk/familiarity';
import { surfaceFeatures } from '@zerowidth/shims-sdk/surface';
import { AdaptiveShim, AdaptiveBank } from '@zerowidth/shims-sdk/adapt';
import { predictTree } from '@zerowidth/shims-sdk/tree';
import { System } from '@zerowidth/shims-sdk/system';
import type { SystemSpec, SystemRunResult } from '@zerowidth/shims-sdk/system';
import { record } from '@zerowidth/shims-sdk/observe';
import { compileShim } from '@zerowidth/shims-sdk/compile';
import type { CompileOptions } from '@zerowidth/shims-sdk/compile';
import { compileSystem } from '@zerowidth/shims-sdk/compile/system';

declare const weights: ShimWeights;
const decision: Decision = await new Shim(weights).decide('hello');
const two: 2 = FORMAT_VERSION;

void [Bank, observe, outcome, ring, preload, setEmbedder, fitHead, validateSource, familiarityScore, surfaceFeatures,
  AdaptiveShim, AdaptiveBank, predictTree, System, record, compileShim, compileSystem, decision, two];
export type { ShimSource, ShimReport, Embedder, Outcome, ShimWeightsInput, SystemSpec, SystemRunResult, CompileOptions };
