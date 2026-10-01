// THE TRACTCLOUD PORT AGAINST THE ORIGINAL: every layer, for 32 streamlines of PAT16, against TractCloud's own network in
// PyTorch (the "tractcloud-reference" test data; Contents/tools/tractcloud-reference.py), then the clusters of more.
//   deno test -A --no-check extensions/diffusion/tractcloud/tractcloud.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { ABSENT, testData } from "albula/testing";
import { classifyCpu, draw, loadModel, localNeighbors, prepare, resample, tractsOf, type ModelJson, type Recorded } from "./tractcloud.ts";

const REF = testData("tractcloud-reference") ?? ABSENT;
const have = (() => { try { return Deno.statSync(REF + "reference.json").isFile; } catch { return false; } })();
const HERE = new URL("./model/", import.meta.url);
const model = loadModel(Deno.readFileSync(new URL("weights.f32", HERE)).buffer, JSON.parse(Deno.readTextFileSync(new URL("model.json", HERE))) as ModelJson);
const f32 = (n: string) => new Float32Array(Deno.readFileSync(REF + n).buffer);
const i32 = (n: string) => new Int32Array(Deno.readFileSync(REF + n).buffer);
function streamlines(): Float32Array[] {
  const raw = f32("streamlines.f32"), out: Float32Array[] = []; let o = 1;
  for (let i = 0; i < raw[0]; i++) { const n = raw[o]; out.push(raw.slice(o + 1, o + 1 + 3 * n)); o += 1 + 3 * n; }
  return out;
}
/** Largest difference relative to the largest value, over a whole layer. */
function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0, m = 0;
  for (let i = 0; i < a.length; i++) { d = Math.max(d, Math.abs(a[i] - b[i])); m = Math.max(m, Math.abs(b[i])); }
  return d / (m || 1);
}

Deno.test("resampling: equal arc length, ends kept", () => {
  const r = resample(new Float32Array([0, 0, 0, 1, 0, 0, 3, 0, 0]), 4);
  assertEquals(Array.from(r).map((v) => +v.toFixed(6)), [0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0]);
});

Deno.test("draws: seeded, sized as the original's", () => {
  const a = draw(5120, model, 7), b = draw(5120, model, 7);
  assertEquals(a.ds, b.ds); assertEquals(a.glob, b.glob);
  assertEquals(a.ds.length, 512); assertEquals(new Set(a.ds).size, 512); assertEquals(a.glob.length, 80);
});

Deno.test({ name: "the port against TractCloud in PyTorch, layer by layer (PAT16)", ignore: !have, fn: () => {
  const feat = prepare(streamlines(), model), ref = f32("feat.f32");
  assert(relErr(feat, ref) < 1e-6, `resampled and recentered streamlines differ: ${relErr(feat, ref)}`);
  const ds = i32("ds.i32"), glob = i32("glob.i32"), topk = i32("topk.i32"), pred = i32("pred.i32");
  // Local neighbors: the same sets (order may differ where two are equally near).
  const mine = localNeighbors(feat, ds, model), k = model.json.settings.k, N = pred.length;
  let same = 0;
  for (let i = 0; i < N; i++) { const s = new Set(topk.subarray(i * k, i * k + k)); if (mine.subarray(i * k, i * k + k).every((j) => s.has(j))) same++; }
  console.log(`  local neighbors: ${same} of ${N} streamlines with the same 20`);
  assert(same / N > 0.99);
  // Every layer, 32 streamlines, the reference's own neighbors.
  const rec: Recorded[] = [];
  for (let f = 0; f < 32; f++) classifyCpu(model, feat, ds, glob, topk, f, f + 1, rec);
  const P = model.P, sizes: [keyof Recorded, string, number][] = [["conv1", "conv1.f32", 64 * P], ["conv2", "conv2.f32", 64 * P],
    ["conv3", "conv3.f32", 128 * P], ["conv4", "conv4.f32", 256 * P], ["conv5", "conv5.f32", 1024 * P], ["lin1", "linear1.f32", 512], ["lin2", "linear2.f32", 256], ["lin3", "linear3.f32", 1600]];
  for (const [key, file, n] of sizes) {
    const r = f32(file); let worst = 0;
    for (let f = 0; f < 32; f++) worst = Math.max(worst, relErr(rec[f][key], r.subarray(f * n, f * n + n)));
    console.log(`  ${key}: largest difference ${worst.toExponential(1)} of the layer's range`);
    assert(worst < 1e-4, `${key} differs by ${worst}`);
  }
  // Clusters and tracts for 512 streamlines.
  const t0 = performance.now(), got = classifyCpu(model, feat, ds, glob, topk, 0, 512);
  const ms = (performance.now() - t0) / 512;
  let c = 0, t = 0; const tr = i32("tract.i32"), gt = tractsOf(model, got);
  for (let i = 0; i < 512; i++) { if (got[i] === pred[i]) c++; if (gt[i] === tr[i]) t++; }
  console.log(`  512 streamlines: ${c} same cluster, ${t} same tract; ${ms.toFixed(1)} ms a streamline on the processor`);
  assert(c >= 510 && t >= 511);
} });

Deno.test({ name: "the graphics-card version against the processor version (PAT16, all streamlines)", ignore: !have || !("gpu" in navigator), fn: async () => {
  const { tractCloudGpu, contexts } = await import("./tractcloud-gpu.ts");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { console.log("  (no graphics card)"); return; }
  const device = await adapter.requestDevice();
  const feat = prepare(streamlines(), model), ds = i32("ds.i32"), glob = i32("glob.i32"), topk = i32("topk.i32"), pred = i32("pred.i32");
  const t0 = performance.now(); localNeighbors(feat, ds, model); const tn = performance.now() - t0;
  const gpu = tractCloudGpu(device, model);
  await gpu.classify(Float32Array.from(feat.subarray(0, 64 * 45)), contexts(topk.subarray(0, 64 * 20), ds, glob, 20).map((v) => Math.min(v, 63)));   // warm up
  const t1 = performance.now(), got = await gpu.classify(Float32Array.from(feat), contexts(topk, ds, glob, model.json.settings.k));
  const tg = performance.now() - t1;
  gpu.destroy(); device.destroy();
  const N = pred.length, tr = i32("tract.i32"), gt = tractsOf(model, got);
  let c = 0, t = 0; for (let i = 0; i < N; i++) { if (got[i] === pred[i]) c++; if (gt[i] === tr[i]) t++; }
  console.log(`  ${N} streamlines: ${c} same cluster as the original, ${t} same tract; network ${(tg / 1000).toFixed(2)} s on the card, neighbor search ${(tn / 1000).toFixed(2)} s on the processor`);
  assert(c / N > 0.998 && t / N > 0.999);
} });
