// @full-tier (the comparison runs a group of 2,000 on the processor, ~12 s)
// THE RAPIDPARC PORT AGAINST THE ORIGINAL: RapidParc's own network in PyTorch on PAT16's streamlines (the
// "rapidparc-reference" test data; Contents/tools/rapidparc-reference.py), layer by layer for 32 streamlines, then every
// streamline's cluster in a group of 2,000.
//   deno test -A --no-check extensions/diffusion/rapidparc/rapidparc.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { ABSENT, testData } from "albula/testing";
import { CONTEXT, forwardGroupCpu, groupRows, lengthMm, loadRapidParc, MIN_LENGTH_MM, normalizeCube, readSafetensors, resampleByIndex, shuffle, type Recorded } from "./rapidparc.ts";

const REF = testData("rapidparc-reference") ?? ABSENT;
const TC = testData("tractcloud-reference") ?? ABSENT;
const have = (() => { try { return Deno.statSync(REF + "reference.json").isFile && Deno.statSync(TC + "streamlines.f32").isFile; } catch { return false; } })();
const weights = Deno.readFileSync(new URL("./model/rapidparc.safetensors", import.meta.url));
const model = loadRapidParc(weights.buffer);
const f32 = (n: string) => new Float32Array(Deno.readFileSync(REF + n).buffer);
const i32 = (n: string) => new Int32Array(Deno.readFileSync(REF + n).buffer);
function streamlines(): Float32Array[] {
  const raw = new Float32Array(Deno.readFileSync(TC + "streamlines.f32").buffer), out: Float32Array[] = []; let o = 1;
  for (let i = 0; i < raw[0]; i++) { const n = raw[o]; out.push(raw.slice(o + 1, o + 1 + 3 * n)); o += 1 + 3 * n; }
  return out;
}
/** Largest difference relative to the largest value, over a whole layer. */
function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0, m = 0;
  for (let i = 0; i < a.length; i++) { d = Math.max(d, Math.abs(a[i] - b[i])); m = Math.max(m, Math.abs(b[i])); }
  return d / (m || 1);
}

Deno.test("the weights: 8 encoder layers, 128 wide, 1,600 classes, and the second (hemiaug) set reads too", () => {
  assertEquals([model.layers.length, model.emb.out, model.emb.inp, model.cls2.out], [8, 83, 45, 1600]);
  const t = readSafetensors(Deno.readFileSync(new URL("./model/hemiaug.safetensors", import.meta.url)).buffer);
  assertEquals(t.get("classifier.3.weight")?.shape, [1600, 256]);
});

Deno.test("15 points by index, halves to the even neighbor (numpy's round)", () => {
  const s = Float32Array.from({ length: 5 * 3 }, (_, i) => Math.floor(i / 3));   // points 0..4
  // linspace(0, 4, 15) * … : index 0, 0.2857 → 0, …, 1.714 → 2, 2 → 2 … ; spot-check the ends and a half
  const r = resampleByIndex(s);
  assertEquals([r[0], r[14 * 3]], [0, 4]);
  const t = Float32Array.from({ length: 3 * 3 }, (_, i) => Math.floor(i / 3));   // 3 points: linspace(0, 2, 15)[7] = 1.0
  assertEquals(resampleByIndex(t)[7 * 3], 1);
});

Deno.test("the shuffle is seeded and a permutation; groups are padded with the first shuffled rows", () => {
  const p = shuffle(4174, 5), q = shuffle(4174, 5);
  assertEquals(p, q);
  assertEquals(new Set(p).size, 4174);
  const x = Float32Array.from({ length: 3 * 45 }, (_, i) => Math.floor(i / 45));
  const g = groupRows(x, Int32Array.from([2, 0, 1]));
  assertEquals(g.groups, 1);
  assertEquals([g.rows[0], g.rows[45], g.rows[90], g.rows[135]], [2, 0, 1, 2]);
  assertEquals(g.rows.length, CONTEXT * 45);
});

Deno.test({ name: "the port against RapidParc in PyTorch, layer by layer and cluster by cluster (PAT16)", ignore: !have, fn: () => {
  const all = streamlines();
  const keep = i32("keep.i32");
  const mine = all.map((s, i) => [s, i] as const).filter(([s]) => lengthMm(s) >= MIN_LENGTH_MM).map(([, i]) => i);
  assertEquals(Int32Array.from(mine), keep, "the same streamlines are 40 mm or more");
  const raw = new Float32Array(keep.length * 45);
  keep.forEach((i, r) => raw.set(resampleByIndex(all[i]), r * 45));
  const x = normalizeCube(raw), ref = f32("feat.f32");
  assert(relErr(x, ref) < 1e-6, `the scaled points differ: ${relErr(x, ref)}`);
  // Group 0 in RapidParc's own shuffle (written by the reference).
  const perm = i32("perm.i32"), { rows } = groupRows(x, perm);
  const rec: Partial<Recorded> = {};
  const t0 = performance.now();
  const logits = forwardGroupCpu(model, rows.subarray(0, CONTEXT * 45), rec);
  const seconds = (performance.now() - t0) / 1000;
  for (const name of ["emb", "layer0", "enc", "hidden"] as const) {
    const e = relErr(rec[name]!.subarray(0, 32 * (name === "hidden" ? 256 : 128)), f32(`${name}.f32`));
    console.log(`  ${name}: largest difference ${e.toExponential(2)} of the largest value`);
    assert(e < 1e-4, `${name} differs: ${e}`);
  }
  // Every streamline of group 0: the reference wrote logits in the kept order; group row r is kept streamline perm[r].
  const L = f32("logits.f32"), C = i32("cluster.i32");
  let worst = 0, same = 0;
  for (let r = 0; r < CONTEXT; r++) {
    const k = perm[r], mineRow = logits.subarray(r * 1600, r * 1600 + 1600), refRow = L.subarray(k * 1600, k * 1600 + 1600);
    worst = Math.max(worst, relErr(mineRow, refRow));
    let best = 0; for (let c = 1; c < 1600; c++) if (mineRow[c] > mineRow[best]) best = c;
    if (best === C[k]) same++;
  }
  console.log(`  group 0: ${same} of ${CONTEXT} clusters as RapidParc's, logits within ${worst.toExponential(2)}; ${seconds.toFixed(1)} s on the processor`);
  assert(worst < 1e-4, `logits differ: ${worst}`);
  assert(same >= CONTEXT - 1, `${CONTEXT - same} clusters differ`);
} });
