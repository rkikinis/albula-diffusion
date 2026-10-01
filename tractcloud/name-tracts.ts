// NAMING A WHOLE-BRAIN TRACTOGRAPHY WITH TRACTCLOUD: streamlines in (RAS mm), a tract per streamline out.
// Short streamlines (under 40 mm) are left out, both from naming and as context: the atlas the network learned from
// has none (measured on PAT16: keeping them doubles the share called an outlier). Several draws of the random context
// can vote; each draw is seeded, so the same streamlines always get the same names.
import { draw, localNeighbors, prepare, tractsOf, type TractCloudModel } from "./tractcloud.ts";
import { contexts, tractCloudGpu } from "./tractcloud-gpu.ts";

export const MIN_LENGTH_MM = 40;
/** A streamline's tract: an index into model.json.tracts, or SHORT. */
export const SHORT = -1;

/** side: +1 right, -1 left, 0 none (the commissures and the middle cerebellar peduncle cross the midline). */
export interface Named { tract: Int32Array; side: Int8Array; draws: number; seconds: number }
/** Tracts without a side. */
const NO_SIDE = new Set(["MCP"]);

const lengthOf = (s: Float32Array) => { let L = 0; for (let i = 3; i < s.length; i += 3) L += Math.hypot(s[i] - s[i - 3], s[i + 1] - s[i - 2], s[i + 2] - s[i - 1]); return L; };

export async function nameTracts(device: GPUDevice, model: TractCloudModel, streamlines: Float32Array[], opts: { draws?: number; seed?: number } = {}): Promise<Named> {
  const t0 = performance.now(), draws = opts.draws ?? 1, seed = opts.seed ?? 20260930;
  const keep = streamlines.map((s, i) => [s, i] as const).filter(([s]) => lengthOf(s) >= MIN_LENGTH_MM);
  const tract = new Int32Array(streamlines.length).fill(SHORT), side = new Int8Array(streamlines.length);
  if (keep.length < model.json.settings.k + 1) return { tract, side, draws, seconds: (performance.now() - t0) / 1000 };
  const feat = prepare(keep.map(([s]) => s), model), feat32 = Float32Array.from(feat), N = keep.length;
  const T = model.json.tracts.length, votes = new Int32Array(N * T), first = new Int32Array(N);
  const gpu = tractCloudGpu(device, model);
  try {
    for (let d = 0; d < draws; d++) {
      const { ds, glob } = draw(N, model, seed + d);
      const clusters = await gpu.classify(feat32, contexts(localNeighbors(feat, ds, model), ds, glob, model.json.settings.k));
      const t = tractsOf(model, clusters);
      for (let i = 0; i < N; i++) { votes[i * T + t[i]]++; if (d === 0) first[i] = t[i]; }
    }
  } finally { gpu.destroy(); }
  for (let i = 0; i < N; i++) {
    let best = first[i];                                  // a tie keeps the first draw's answer
    for (let c = 0; c < T; c++) if (votes[i * T + c] > votes[i * T + best]) best = c;
    tract[keep[i][1]] = best;
    // THE SIDE, from where the streamline lies once the brain is centered on the atlas (RAS: +x is the patient's right).
    const info = model.json.tracts[best];
    if (info.category !== "Commissural" && !NO_SIDE.has(info.abbr)) {
      let x = 0; for (let p = 0; p < model.P; p++) x += feat[(i * model.P + p) * 3];
      side[keep[i][1]] = x > 0 ? 1 : -1;
    }
  }
  return { tract, side, draws, seconds: (performance.now() - t0) / 1000 };
}
