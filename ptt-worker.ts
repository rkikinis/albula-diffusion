// A PTT worker (ptt.ts trackPttParallel): builds the FOD field once, then tracks the seeds it is sent.
import { FodField, pttStreamline, type PttOptions } from "./ptt.ts";
import type { FodVolume } from "./csd-volume.ts";

let field: FodField | undefined, opts: Required<PttOptions> | undefined;
self.onmessage = (e: MessageEvent) => {
  const d = e.data as { init?: { fod: FodVolume; opts: Required<PttOptions> }; seeds?: Float64Array; first?: number };
  if (d.init) { field = new FodField(d.init.fod); opts = d.init.opts; return; }
  const out: (Float32Array | null)[] = [], n = d.seeds!.length / 3;
  for (let i = 0; i < n; i++) {
    // each seed its own generator, from the run's seed and the seed's index: the result does not depend on the workers
    let a = (opts!.seed + 0x9e3779b9 * (d.first! + i + 1)) >>> 0;
    const rand = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    out.push(pttStreamline(field!, Array.from(d.seeds!.subarray(3 * i, 3 * i + 3)), opts!, rand) ?? null);
  }
  (self as unknown as Worker).postMessage(out);
};
