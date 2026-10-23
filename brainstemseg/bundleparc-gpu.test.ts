// @full-tier (needs the graphics card and the BrainstemSeg reference, Contents/data/test-data.json "brainstemseg-reference")
// BRAINSTEMSEG'S NETWORK ON THE GRAPHICS CARD against PyTorch in double precision, block by block, on a 32-voxel piece of
// Prisma's prepared input (Contents/tools/brainstemseg-dump.py): the encoder's every level, the bottleneck, and the
// decoder's every level for two tracts (the corticospinal tract and the medial lemniscus, left), the prompts on the
// processor side, and the head. Each block's worst difference is printed (absolute, and relative to the block's largest
// value), so a fault shows where it starts.
//   deno test -A --no-check --unstable-webgpu --config ../../../src/SlicerLive/deno.jsonc bundleparc-gpu.test.ts
import { assert } from "jsr:@std/assert@1";
import { ABSENT, testData } from "albula/testing";
import { loadBundleParc } from "./bundleparc.ts";
import { bundleParcGpu } from "./bundleparc-gpu.ts";

const REF = (testData("brainstemseg-reference") ?? ABSENT) + "small32/";
const have = (() => { try { return Deno.statSync(REF + "manifest.json").isFile; } catch { return false; } })();
const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
const adapter = gpu ? await gpu.requestAdapter().catch(() => null) : null;
const L = adapter?.limits;
const device = adapter && L ? await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize } }).catch(() => null) : null;
const model = () => loadBundleParc(Deno.readFileSync(new URL("./model/bundleparc-brainstemseg.safetensors", import.meta.url)).buffer,
  JSON.parse(Deno.readTextFileSync(new URL("./model/tracts.json", import.meta.url))));

Deno.test({ name: "BrainstemSeg's network on the card: every block as PyTorch's, 32-voxel cube", ignore: !device || !have, fn: async () => {
  const man = JSON.parse(Deno.readTextFileSync(REF + "manifest.json")) as { D: number; tracts: Record<string, number>; tensors: Record<string, number[]> };
  const ref = (name: string) => new Float32Array(Deno.readFileSync(REF + name + ".f32").buffer);
  const m = model(), g = bundleParcGpu(device!, m, { keep: true });
  const rows: string[] = []; let worst = 0, worstAt = "";
  const cmp = (name: string, got: ArrayLike<number>, want: Float32Array) => {
    assert(got.length === want.length, `${name}: ${got.length} numbers, PyTorch ${want.length}`);
    let d = 0, top = 0; for (let i = 0; i < want.length; i++) { d = Math.max(d, Math.abs(got[i] - want[i])); top = Math.max(top, Math.abs(want[i])); }
    // A block of zeros (top 0) is judged by its absolute difference: d / 0 was NaN, which never counted as worse (critic finding 9).
    const rel = top > 0 ? d / top : d; if (rel > worst) { worst = rel; worstAt = name; }
    rows.push(`${name.padEnd(28)} ${d.toExponential(2)}  of ${top.toFixed(3).padStart(9)}  (${rel.toExponential(2)})`);
  };
  try {
    await g.encode(ref("input"), man.D);
    for (const n of ["stem", "enc0.conv1.dw", "enc0.conv1.expand", "enc0.conv1.out", ...[0, 1, 2, 3].flatMap((l) => [`enc${l}.conv1`, `enc${l}.res`, `enc${l}.down`]), "bottleneck"]) cmp(n, await g.read(n), ref(n));
    for (const [tract, i] of Object.entries(man.tracts)) {
      await g.decode(i);
      cmp(`${tract}.prompt`, g.prompts.get(`${tract}.prompt`)!, ref(`${tract}.prompt`));
      for (let j = 0; j < 4; j++) {
        cmp(`${tract}.dec${j}.up`, await g.read(`${tract}.dec${j}.up`), ref(`${tract}.dec${j}.up`));
        cmp(`${tract}.dec${j}.conv1`, await g.read(`${tract}.dec${j}.conv1`), ref(`${tract}.dec${j}.conv1`));
        cmp(`${tract}.dec${j}.promptIn`, g.prompts.get(`${tract}.dec${j}.promptIn`)!, ref(`${tract}.dec${j}.promptIn`));
        cmp(`${tract}.dec${j}.attn`, await g.read(`${tract}.dec${j}.attn`), ref(`${tract}.dec${j}.attn`));
        cmp(`${tract}.dec${j}.promptOut`, g.prompts.get(`${tract}.dec${j}.promptOut`)!, ref(`${tract}.dec${j}.promptOut`));
        cmp(`${tract}.dec${j}.out`, await g.read(`${tract}.dec${j}.out`), ref(`${tract}.dec${j}.out`));
      }
      cmp(`${tract}.head`, await g.read(`${tract}.head`), ref(`${tract}.head`));
    }
    console.log("  block                         worst difference (relative to the block's largest value)\n  " + rows.join("\n  "));
    console.log(`  worst: ${worst.toExponential(2)} at ${worstAt}`);
    // Measured 2026-10-22: 1.8e-6 at worst (single precision's rounding; double precision on PyTorch's side).
    assert(worst < 1e-5, `a block differs from PyTorch by ${worst.toExponential(2)} of its largest value (${worstAt})`);
  } finally { g.destroy(); }
} });

// THE WHOLE SCAN: Prisma's FODs (MRtrix3, as BrainstemSeg's pipeline makes them) through our input preparation and the
// card, every tract against the PyTorch run (Contents/tools/brainstemseg-reference.py, PyTorch's single precision on the
// Mac's graphics chip): per tract the voxels in both, in one only, Dice, and the positions that differ; the time.
const PRISMA = testData("brainstemseg-prisma") ?? ABSENT;
const havePrisma = (() => { try { return Deno.statSync(PRISMA + "wmfod.nii.gz").isFile && Deno.statSync(PRISMA + "brainstemseg/reference.json").isFile; } catch { return false; } })();
Deno.test({ name: "BrainstemSeg's network on the card: Prisma's 97 tracts as PyTorch's, and the time", ignore: !device || !havePrisma, fn: async () => {
  const { parseNiftiVolumes } = await import("albula");
  const { prepareInput, toFodGrid } = await import("./bundleparc.ts");
  const t0 = performance.now();
  const vols = await parseNiftiVolumes(Deno.readFileSync(PRISMA + "wmfod.nii.gz"));
  const dims = vols[0].dims as [number, number, number], n = dims[0] * dims[1] * dims[2], C = vols.length;
  // Our reader gives each volume with i fastest; the network's layout is x slowest, channel fastest.
  const fod = new Float32Array(n * C);
  for (let c = 0; c < C; c++) { const v = vols[c].data as ArrayLike<number>;
    for (let x = 0; x < dims[0]; x++) for (let y = 0; y < dims[1]; y++) for (let z = 0; z < dims[2]; z++) fod[((x * dims[1] + y) * dims[2] + z) * C + c] = v[x + dims[0] * (y + dims[1] * z)]; }
  const tRead = performance.now();
  const prep = prepareInput(fod, dims, C);
  const tPrep = performance.now();
  // Our prepared cube against Python's (the dump's input is its central 32-voxel piece).
  if (have) {
    const man = JSON.parse(Deno.readTextFileSync(REF + "manifest.json")) as { D: number; cubeStart: number };
    const want = new Float32Array(Deno.readFileSync(REF + "input.f32").buffer), s0 = man.cubeStart, S = 176; let d = 0, top = 0;
    for (let x = 0; x < man.D; x++) for (let y = 0; y < man.D; y++) for (let z = 0; z < man.D; z++) for (let c = 0; c < 45; c++) {
      const w = want[((x * man.D + y) * man.D + z) * 45 + c]; top = Math.max(top, Math.abs(w));
      d = Math.max(d, Math.abs(prep.cube[(((x + s0) * S + y + s0) * S + z + s0) * 45 + c] - w));
    }
    console.log(`  prepared input against Python's: ${d.toExponential(2)} at worst of ${top.toFixed(2)} (${(d / top).toExponential(2)}; fit ${prep.fit.join(" × ")}, offset ${prep.offset.join(", ")})`);
    // Python's standard deviation of the cube is numpy's in float32 (245 million numbers summed in float32): 7.8e-5 below
    // the exact one on Prisma (measured 2026-10-22); ours is in double precision. So the inputs differ by that uniform
    // scale, nothing else.
    assert(d / top < 1.5e-4, `the prepared input differs from Python's by ${d} of ${top}`);
  }
  const m = model(), g = bundleParcGpu(device!, m);
  try {
    const tE = performance.now(); await g.encode(prep.cube, 176); const encodeS = (performance.now() - tE) / 1000;
    const rows: string[] = []; let worstDice = 1, worstAt = "", decodeS = 0;
    for (let i = 0; i < m.tracts.length; i++) {
      const name = m.tracts[i], { labels, seconds } = await g.decode(i); decodeS += seconds;
      const ours = toFodGrid(labels, prep, dims);
      const refV = (await parseNiftiVolumes(Deno.readFileSync(`${PRISMA}brainstemseg/${name}.nii.gz`)))[0].data as ArrayLike<number>;
      let both = 0, onlyOurs = 0, onlyRef = 0, pos = 0;
      for (let x = 0; x < dims[0]; x++) for (let y = 0; y < dims[1]; y++) for (let z = 0; z < dims[2]; z++) {
        const a = ours[(x * dims[1] + y) * dims[2] + z], b = refV[x + dims[0] * (y + dims[1] * z)];
        if (a && b) { both++; if (a !== b) pos++; } else if (a) onlyOurs++; else if (b) onlyRef++;
      }
      const dice = both + onlyOurs + onlyRef ? 2 * both / (2 * both + onlyOurs + onlyRef) : 1;
      if (dice < worstDice) { worstDice = dice; worstAt = name; }
      if (/^(CST|ML|MLc|STT|STTc)-/.test(name) || dice < 0.99) rows.push(`${name.padEnd(22)} both ${String(both).padStart(6)}  ours only ${String(onlyOurs).padStart(5)}  PyTorch only ${String(onlyRef).padStart(5)}  Dice ${dice.toFixed(4)}  positions differ ${pos}`);
    }
    console.log(`  read ${((tRead - t0) / 1000).toFixed(1)} s, prepared ${((tPrep - tRead) / 1000).toFixed(1)} s, encoder ${encodeS.toFixed(1)} s, 97 decoders ${decodeS.toFixed(1)} s (${(decodeS / 97).toFixed(2)} s each)`);
    console.log("  " + rows.join("\n  "));
    console.log(`  lowest Dice: ${worstDice.toFixed(4)} (${worstAt})`);
    assert(worstDice > 0.98, `a tract's mask differs from PyTorch's: Dice ${worstDice} (${worstAt})`);
  } finally { g.destroy(); }
} });
