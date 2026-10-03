// @full-tier (needs the graphics card and the PAT16 reference; the processor side runs one group, ~12 s)
// RAPIDPARC ON THE GRAPHICS CARD against the processor version (itself checked against RapidParc in PyTorch,
// rapidparc.test.ts) and against RapidParc's own clusters for every PAT16 streamline of the reference.
//   deno test -A --no-check extensions/diffusion/rapidparc/rapidparc-gpu.test.ts
import { assert } from "jsr:@std/assert@1";
import { ABSENT, testData } from "albula/testing";
import { CONTEXT, forwardGroupCpu, groupRows, loadRapidParc, normalizeCube, resampleByIndex } from "./rapidparc.ts";
import { rapidParcGpu } from "./rapidparc-gpu.ts";

const REF = testData("rapidparc-reference") ?? ABSENT;
const TC = testData("tractcloud-reference") ?? ABSENT;
const have = (() => { try { return Deno.statSync(REF + "reference.json").isFile && Deno.statSync(TC + "streamlines.f32").isFile; } catch { return false; } })();
const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
const adapter = gpu ? await gpu.requestAdapter().catch(() => null) : null;
const L = adapter?.limits;
const device = adapter && L ? await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } }).catch(() => null) : null;

Deno.test({ name: "RapidParc on the card: the processor's clusters and margins, RapidParc's clusters, and the time (PAT16)", ignore: !device || !have, fn: async () => {
  const model = loadRapidParc(Deno.readFileSync(new URL("./model/rapidparc.safetensors", import.meta.url)).buffer);
  const json = JSON.parse(Deno.readTextFileSync(new URL("../tractcloud/model/model.json", import.meta.url))) as { clusterToTract: number[] };
  const lut = Int32Array.from(json.clusterToTract);
  const raw = new Float32Array(Deno.readFileSync(TC + "streamlines.f32").buffer), all: Float32Array[] = []; let o = 1;
  for (let i = 0; i < raw[0]; i++) { const n = raw[o]; all.push(raw.slice(o + 1, o + 1 + 3 * n)); o += 1 + 3 * n; }
  const keep = new Int32Array(Deno.readFileSync(REF + "keep.i32").buffer), perm = new Int32Array(Deno.readFileSync(REF + "perm.i32").buffer);
  const refCluster = new Int32Array(Deno.readFileSync(REF + "cluster.i32").buffer);
  const pts = new Float32Array(keep.length * 45); keep.forEach((i, r) => pts.set(resampleByIndex(all[i]), r * 45));
  const { rows, groups } = groupRows(normalizeCube(pts), perm);
  const g = rapidParcGpu(device!, model, lut);
  try {
    await g.classify(rows, groups);                                   // compiles and warms the pipelines
    const t0 = performance.now();
    const { cluster, margin } = await g.classify(rows, groups);
    const seconds = (performance.now() - t0) / 1000;
    // Against RapidParc's own clusters, every real row (row r is kept streamline perm[r]).
    let same = 0; for (let r = 0; r < perm.length; r++) if (cluster[r] === refCluster[perm[r]]) same++;
    // Against the processor version, group 0: clusters, and margins computed the same way from its logits.
    const lg = forwardGroupCpu(model, rows.subarray(0, CONTEXT * 45));
    let worstM = 0, sameCpu = 0;
    for (let r = 0; r < CONTEXT; r++) {
      const row = lg.subarray(r * 1600, r * 1600 + 1600); let best = 0; for (let c = 1; c < 1600; c++) if (row[c] > row[best]) best = c;
      if (best === cluster[r]) sameCpu++;
      const pt = new Float64Array(43); let total = 0; for (let c = 0; c < 1600; c++) { const e = Math.exp(row[c] - row[best]); pt[lut[c]] += e; total += e; }
      let other = 0; for (let t = 0; t < 43; t++) if (t !== lut[best]) other = Math.max(other, pt[t]);
      worstM = Math.max(worstM, Math.abs((pt[lut[best]] - other) / total - margin[r]));
    }
    console.log(`  ${same} of ${perm.length} clusters as RapidParc's own; group 0: ${sameCpu} of ${CONTEXT} as the processor's, margins within ${worstM.toExponential(2)}; ${groups} groups in ${seconds.toFixed(2)} s on the card`);
    assert(same >= perm.length - 2, `${perm.length - same} clusters differ from RapidParc's`);
    assert(sameCpu >= CONTEXT - 1 && worstM < 1e-4);
  } finally { g.destroy(); }
} });
