// The diffusion scan onto the T1 (registration.ts): a known rigid move is found again on a synthetic head, and the
// resampling with no move and no field gives the scan back. The comparisons with Mike Halle's code and BRAINSFit on the
// patients are Contents/tools/registration-check.ts (a check tool; 2026-10-03: 0.03-0.09 and 0.18-0.34 mm apart).
//   deno test -A --no-check registration.test.ts
import { assert, assertAlmostEquals } from "jsr:@std/assert@1";
import { alignToT1, applyRigid, brainOnlyT1, inv4, rigidSize, rigidToT1, rigidToT1Wide, resampleOntoT1, rotation, sampleAt, scanBrain, spacingOf, type Grid3, type Rigid } from "./registration.ts";
import { parseNiftiVolumes } from "albula";
import { ABSENT, testData } from "albula/testing";
import { fromFsl } from "./dwi.ts";
import type { DiffusionSeries } from "./dwi.ts";

/** A smooth "head": an ellipsoid with three blobs of different brightness, sampled on a grid. */
function head(dims: [number, number, number], ijkToRAS: number[], move?: Rigid): Float32Array {
  const [nx, ny, nz] = dims, out = new Float32Array(nx * ny * nz), M = ijkToRAS;
  // Inverse of the move: the moved head at y is the head at x = Rᵀ(y − c − t) + c.
  const back = (y: number[]) => { if (!move) return y; const R = move.R, d = [y[0] - move.c[0] - move.t[0], y[1] - move.c[1] - move.t[1], y[2] - move.c[2] - move.t[2]]; return [R[0] * d[0] + R[3] * d[1] + R[6] * d[2] + move.c[0], R[1] * d[0] + R[4] * d[1] + R[7] * d[2] + move.c[1], R[2] * d[0] + R[5] * d[1] + R[8] * d[2] + move.c[2]]; };
  const g = (p: number[], c: number[], s: number) => Math.exp(-((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2) / (2 * s * s));
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const y = [0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]), p = back(y);
    const e = (p[0] / 60) ** 2 + (p[1] / 75) ** 2 + (p[2] / 55) ** 2;
    out[(k * ny + j) * nx + i] = (e < 1 ? 60 : 0) + 120 * g(p, [20, 10, 5], 9) + 80 * g(p, [-25, -20, 15], 7) + 140 * g(p, [5, 35, -20], 11);
  }
  return out;
}

Deno.test("a known rigid move (2 mm, 3°) is found again on a synthetic head, within 0.3 mm in the brain", async () => {
  const t1Dims: [number, number, number] = [80, 96, 72], t1M = [2, 0, 0, -80, 0, 2, 0, -96, 0, 0, 2, -72, 0, 0, 0, 1];
  const truth: Rigid = { R: rotation(0.03, -0.02, 0.04), t: [1.5, -1.0, 0.8], c: [0, 0, 0] };
  const t1: Grid3 = { dims: t1Dims, ijkToRAS: t1M, data: head(t1Dims, t1M) };
  // The "b = 0": a coarser grid, the head moved by the inverse of the truth (so aligning it to the T1 is the truth).
  const bDims: [number, number, number] = [56, 66, 50], bM = [2.5, 0, 0, -70, 0, 2.5, 0, -82, 0, 0, 2.5, -62, 0, 0, 0, 1];
  const inv: Rigid = { R: [truth.R[0], truth.R[3], truth.R[6], truth.R[1], truth.R[4], truth.R[7], truth.R[2], truth.R[5], truth.R[8]], t: [0, 0, 0], c: [0, 0, 0] };
  const tt = applyRigid(inv, truth.t); inv.t = [-tt[0], -tt[1], -tt[2]];
  const b0: Grid3 = { dims: bDims, ijkToRAS: bM, data: head(bDims, bM, inv) };
  const mask = new Uint8Array(bDims[0] * bDims[1] * bDims[2]); for (let v = 0; v < mask.length; v++) mask[v] = b0.data[v] > 30 ? 1 : 0;
  const r = await rigidToT1(b0, t1, mask, { fixedMm: 2 });
  // Distance between where the found move and the truth put the brain's points.
  let worst = 0;
  for (let v = 0; v < mask.length; v += 13) if (mask[v]) {
    const i = v % bDims[0], j = Math.floor(v / bDims[0]) % bDims[1], k = Math.floor(v / (bDims[0] * bDims[1]));
    const x = [0, 1, 2].map((q) => bM[4 * q] * i + bM[4 * q + 1] * j + bM[4 * q + 2] * k + bM[4 * q + 3]);
    const a = applyRigid(r.T, x), b = applyRigid(truth, x);
    worst = Math.max(worst, Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
  }
  const s = rigidSize(r.T);
  console.log(`  found ${s.degrees.toFixed(2)}°, ${s.mm.toFixed(2)} mm (truth ${rigidSize(truth).degrees.toFixed(2)}°); worst point ${worst.toFixed(2)} mm; cost ${r.costAtStart.toFixed(4)} -> ${r.cost.toFixed(4)}`);
  assert(worst < 0.3, `the found move puts brain points up to ${worst.toFixed(2)} mm from the truth`);
  assert(r.cost < r.costAtStart);
});

Deno.test("resampling with no move and no field onto the scan's own axes gives the scan back", async () => {
  const dims: [number, number, number] = [12, 10, 8], M = [2, 0, 0, -10, 0, 2, 0, -8, 0, 0, 2, -6, 0, 0, 0, 1];
  const data = Float32Array.from({ length: 960 }, (_, v) => (v * 37) % 101);
  const dwi = { volumes: [{ dims, ijkToRAS: M, data, dtype: "<f4" }], bValues: [1000], gradients: [[1, 0, 0]], ijkToRAS: M, source: "test", convention: 1 } as unknown as DiffusionSeries;
  const out = await resampleOntoT1(dwi, { R: rotation(0, 0, 0), t: [0, 0, 0], c: [0, 0, 0] }, { dims: [50, 50, 50], ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
  const v = out.volumes[0];
  assert(v.dims.join() === dims.join(), `dims ${v.dims}`);
  for (let i = 0; i < data.length; i++) assertAlmostEquals(Number(v.data[i]), data[i], 1e-4);
  assertAlmostEquals(out.gradients[0][0], 1, 1e-12);
});


Deno.test("rule 3's target keeps the T1 inside the brain only, on the T1's grid or through world coordinates", () => {
  const dims: [number, number, number] = [8, 6, 4], M = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const t1: Grid3 = { dims, ijkToRAS: M, data: Float32Array.from({ length: 192 }, (_, v) => v + 1) };
  const inside = (i: number, j: number, k: number) => i >= 2 && i < 6 && j >= 2 && k >= 1;
  const brain = new Uint8Array(192); for (let v = 0; v < 192; v++) brain[v] = inside(v % 8, Math.floor(v / 8) % 6, Math.floor(v / 48)) ? 1 : 0;
  const a = brainOnlyT1(t1, { dims, ijkToRAS: M, data: brain });
  // The same brain on a grid twice as fine, shifted by half a voxel of it: the same voxels come out.
  const fd: [number, number, number] = [16, 12, 8], FM = [0.5, 0, 0, -0.25, 0, 0.5, 0, -0.25, 0, 0, 0.5, -0.25, 0, 0, 0, 1], fine = new Uint8Array(16 * 12 * 8);
  for (let v = 0; v < fine.length; v++) { const i = v % 16, j = Math.floor(v / 16) % 12, k = Math.floor(v / 192); fine[v] = inside(Math.floor(i / 2), Math.floor(j / 2), Math.floor(k / 2)) ? 1 : 0; }
  const b = brainOnlyT1(t1, { dims: fd, ijkToRAS: FM, data: fine });
  for (let v = 0; v < 192; v++) {
    const want = brain[v] ? v + 1 : 0;
    assert(a.data[v] === want && b.data[v] === want, `voxel ${v}: ${a.data[v]} / ${b.data[v]}, want ${want}`);
  }
});

Deno.test("rule 3 (the default) finds a large move (25°, 15 mm) again against a T1 with a bright scalp, within 0.5 mm in the brain", async () => {
  const t1Dims: [number, number, number] = [80, 96, 72], t1M = [2, 0, 0, -80, 0, 2, 0, -96, 0, 0, 2, -72, 0, 0, 0, 1], n1 = 80 * 96 * 72;
  const truth: Rigid = { R: rotation(0.25, -0.2, 0.3), t: [9, -8, 8], c: [0, 0, 0] };
  // The T1 is the head plus a bright shell outside the brain (the scalp the b = 0 does not show); SynthStrip's brain is
  // the ellipsoid.
  const data = head(t1Dims, t1M), brain = new Uint8Array(n1);
  for (let v = 0; v < n1; v++) {
    const i = v % 80, j = Math.floor(v / 80) % 96, k = Math.floor(v / (80 * 96)), p = [2 * i - 80, 2 * j - 96, 2 * k - 72];
    const e = (p[0] / 60) ** 2 + (p[1] / 75) ** 2 + (p[2] / 55) ** 2;
    if (e < 1) brain[v] = 1; else if (e < 1.35) data[v] = 200;
  }
  const t1: Grid3 = { dims: t1Dims, ijkToRAS: t1M, data };
  const bDims: [number, number, number] = [56, 66, 50], bM = [2.5, 0, 0, -70, 0, 2.5, 0, -82, 0, 0, 2.5, -62, 0, 0, 0, 1];
  const inv: Rigid = { R: [truth.R[0], truth.R[3], truth.R[6], truth.R[1], truth.R[4], truth.R[7], truth.R[2], truth.R[5], truth.R[8]], t: [0, 0, 0], c: [0, 0, 0] };
  const tt = applyRigid(inv, truth.t); inv.t = [-tt[0], -tt[1], -tt[2]];
  const b0 = head(bDims, bM, inv);
  const dwi = { volumes: [{ dims: bDims, ijkToRAS: bM, data: b0, dtype: "<f4" }], bValues: [0], gradients: [[0, 0, 0]], ijkToRAS: bM, source: "test", convention: 1 } as unknown as DiffusionSeries;
  const mask = new Uint8Array(b0.length); for (let v = 0; v < mask.length; v++) mask[v] = b0[v] > 30 ? 1 : 0;
  const a = await alignToT1(dwi, t1, undefined, undefined, { moves: [], b0, mask }, { brainT1: { dims: t1Dims, ijkToRAS: t1M, data: brain } });
  let worst = 0;
  for (let v = 0; v < mask.length; v += 13) if (mask[v]) {
    const i = v % bDims[0], j = Math.floor(v / bDims[0]) % bDims[1], k = Math.floor(v / (bDims[0] * bDims[1]));
    const x = [0, 1, 2].map((q) => bM[4 * q] * i + bM[4 * q + 1] * j + bM[4 * q + 2] * k + bM[4 * q + 3]);
    const p = applyRigid(a.T, x), q = applyRigid(truth, x);
    worst = Math.max(worst, Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]));
  }
  console.log(`  rule ${a.rule}: ${a.said}; worst point ${worst.toFixed(2)} mm; ${a.result.candidates?.length} candidates`);
  assert(a.rule === 3, `rule ${a.rule}`);
  assert(!a.doubt, a.doubt);
  assert(worst < 0.5, `the found move puts brain points up to ${worst.toFixed(2)} mm from the truth`);
});

// A REAL SCAN MOVED BY A KNOWN MOVE (critic 2026-10-25, registration rule 3, finding 6): PAT16's b = 0 turned 30° about its
// brain's center and shifted 30 mm (the known-move test's trial "30-0", where rule 1 against the whole T1 landed 26 mm
// off), resampled onto a grid with a 45 mm margin. Rule 3 must find the move again: its answer on the moved image, taken
// back through the move, lands within 1 mm of its answer on the image as acquired. The synthetic head above is too smooth
// for this -- rule 1 finds even 40° on it.
const DS = (testData("openneuro-ds001226", "") ?? ABSENT).replace(/\/$/, ""), SS = (testData("synthstrip-ds001226", "") ?? ABSENT).replace(/\/$/, "");
const have = (p: string) => { try { Deno.statSync(p); return true; } catch { return false; } };
const PAT16 = { dwi: `${DS}/sub-PAT16/ses-preop/dwi/sub-PAT16_ses-preop_acq-AP_dwi`, t1: `${DS}/sub-PAT16/ses-preop/anat/sub-PAT16_ses-preop_T1w.nii.gz`, brain: `${SS}/sub-PAT16/anat/sub-PAT16_ses-preop_desc-synthstrip_mask.nii.gz` };
Deno.test({ name: "rule 3 finds PAT16 again after a known 30°, 30 mm move (rule 1 against the whole T1 did not), within 1 mm", ignore: !have(`${PAT16.dwi}.nii.gz`) || !have(PAT16.brain), fn: async () => {
  const g = (v: { dims: number[]; ijkToRAS: number[]; data: unknown }): Grid3 => ({ dims: v.dims.slice(0, 3) as [number, number, number], ijkToRAS: v.ijkToRAS, data: v.data as ArrayLike<number> });
  const dwi = fromFsl(await parseNiftiVolumes(Deno.readFileSync(`${PAT16.dwi}.nii.gz`)), Deno.readTextFileSync(`${PAT16.dwi}.bval`), Deno.readTextFileSync(`${PAT16.dwi}.bvec`));
  const t1 = g((await parseNiftiVolumes(Deno.readFileSync(PAT16.t1)))[0]), brain = g((await parseNiftiVolumes(Deno.readFileSync(PAT16.brain)))[0]), only = brainOnlyT1(t1, brain);
  const { b0, mask } = scanBrain(dwi), src = dwi.volumes[0], M = src.ijkToRAS, [nx, ny, nz] = src.dims as [number, number, number];
  const world = (i: number, j: number, k: number, A = M) => [0, 1, 2].map((r) => A[4 * r] * i + A[4 * r + 1] * j + A[4 * r + 2] * k + A[4 * r + 3]);
  // The known-move test's trial "30-0" exactly (seed 20261025): y = R x + t in world mm.
  const D = [0.873327, -0.427963, 0.232695, -24.089168, 0.395628, 0.901823, 0.173764, -12.27849, -0.284215, -0.059693, 0.956901, 2.200481];
  const move: Rigid = { R: [D[0], D[1], D[2], D[4], D[5], D[6], D[8], D[9], D[10]], t: [D[3], D[7], D[11]], c: [0, 0, 0] };
  const Rt = [D[0], D[4], D[8], D[1], D[5], D[9], D[2], D[6], D[10]], tb = [0, 1, 2].map((r) => -(Rt[3 * r] * D[3] + Rt[3 * r + 1] * D[7] + Rt[3 * r + 2] * D[11]));
  const back: Rigid = { R: Rt, t: tb as [number, number, number], c: [0, 0, 0] };
  const sp = spacingOf(M), pad = sp.map((s) => Math.ceil(45 / s)), pd = [nx + 2 * pad[0], ny + 2 * pad[1], nz + 2 * pad[2]] as [number, number, number];
  const o = world(-pad[0], -pad[1], -pad[2]), MP = [M[0], M[1], M[2], o[0], M[4], M[5], M[6], o[1], M[8], M[9], M[10], o[2], 0, 0, 0, 1], Mi = inv4(M), tmp = new Float64Array(1);
  const img = new Float32Array(pd[0] * pd[1] * pd[2]), msk = new Uint8Array(img.length);
  for (let k = 0; k < pd[2]; k++) for (let j = 0; j < pd[1]; j++) for (let i = 0; i < pd[0]; i++) {
    const x = applyRigid(back, world(i, j, k, MP)), v = (k * pd[1] + j) * pd[0] + i;
    const X = Mi[0] * x[0] + Mi[1] * x[1] + Mi[2] * x[2] + Mi[3], Y = Mi[4] * x[0] + Mi[5] * x[1] + Mi[6] * x[2] + Mi[7], Z = Mi[8] * x[0] + Mi[9] * x[1] + Mi[10] * x[2] + Mi[11];
    if (sampleAt(b0, [nx, ny, nz], 1, X, Y, Z, tmp)) img[v] = tmp[0];
    const I = Math.round(X), J = Math.round(Y), K = Math.round(Z);
    if (I >= 0 && J >= 0 && K >= 0 && I < nx && J < ny && K < nz) msk[v] = mask[(K * ny + J) * nx + I];
  }
  const asAcquired = await rigidToT1Wide({ dims: [nx, ny, nz], ijkToRAS: M, data: b0 }, only, mask, brain);
  const movedGrid: Grid3 = { dims: pd, ijkToRAS: MP, data: img }, moved = await rigidToT1Wide(movedGrid, only, msk, brain), rule1 = await rigidToT1(movedGrid, t1, msk);
  const off = (T: Rigid) => { let sum = 0, n = 0; for (let v = 0; v < mask.length; v += 17) if (mask[v]) { const x = world(v % nx, Math.floor(v / nx) % ny, Math.floor(v / (nx * ny))), p = applyRigid(T, applyRigid(move, x)), q = applyRigid(asAcquired.T, x); sum += Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]); n++; } return sum / n; };
  console.log(`  rule 3 found it again within ${off(moved.T).toFixed(2)} mm (mean over the brain; ${moved.candidates.length} candidates, ${(moved.ms / 1000).toFixed(1)} s); rule 1 against the whole T1 ${off(rule1.T).toFixed(1)} mm off`);
  assert(off(moved.T) < 1, `rule 3 lands ${off(moved.T).toFixed(2)} mm from its own answer on the scan as acquired`);
  assert(off(rule1.T) > 2, `rule 1 found this move too (${off(rule1.T).toFixed(2)} mm): the test no longer shows what rule 3 adds`);
} });

Deno.test("without SynthStrip's brain, rule 3 is rule 2 against the whole MRI, and the record says so", async () => {
  const t1Dims: [number, number, number] = [40, 48, 36], t1M = [4, 0, 0, -80, 0, 4, 0, -96, 0, 0, 4, -72, 0, 0, 0, 1];
  const t1: Grid3 = { dims: t1Dims, ijkToRAS: t1M, data: head(t1Dims, t1M) };
  const bDims: [number, number, number] = [32, 38, 28], bM = [5, 0, 0, -80, 0, 5, 0, -95, 0, 0, 5, -70, 0, 0, 0, 1], b0 = head(bDims, bM);
  const dwi = { volumes: [{ dims: bDims, ijkToRAS: bM, data: b0, dtype: "<f4" }], bValues: [0], gradients: [[0, 0, 0]], ijkToRAS: bM, source: "test", convention: 1 } as unknown as DiffusionSeries;
  const mask = new Uint8Array(b0.length); for (let v = 0; v < mask.length; v++) mask[v] = b0[v] > 30 ? 1 : 0;
  const a = await alignToT1(dwi, t1, undefined, undefined, { moves: [], b0, mask }, { rule: 3 });
  assert(a.rule === 2, `rule ${a.rule}`);
  assert(/its brain was not available/.test(a.said), a.said);
  assert(/registration rule 2/.test(a.dwi.source), a.dwi.source);
});

Deno.test("check 2, the mirror test: a right pair passes; the same scan flipped left to right is doubted", async () => {
  const t1Dims: [number, number, number] = [80, 96, 72], t1M = [2, 0, 0, -80, 0, 2, 0, -96, 0, 0, 2, -72, 0, 0, 0, 1], n1 = 80 * 96 * 72;
  const data = head(t1Dims, t1M), brain = new Uint8Array(n1);
  for (let v = 0; v < n1; v++) { const i = v % 80, j = Math.floor(v / 80) % 96, k = Math.floor(v / 7680), p = [2 * i - 80, 2 * j - 96, 2 * k - 72]; if ((p[0] / 60) ** 2 + (p[1] / 75) ** 2 + (p[2] / 55) ** 2 < 1) brain[v] = 1; }
  const t1: Grid3 = { dims: t1Dims, ijkToRAS: t1M, data }, brainT1: Grid3 = { dims: t1Dims, ijkToRAS: t1M, data: brain };
  const bDims: [number, number, number] = [56, 66, 50], bM = [2.5, 0, 0, -70, 0, 2.5, 0, -82, 0, 0, 2.5, -62, 0, 0, 0, 1];
  const scan = (flip: boolean) => {
    const b0 = head(bDims, bM), out = new Float32Array(b0.length);
    // Flipped: the image's left and right swapped (x index reversed), the header unchanged -- as a mislabeled scan.
    for (let v = 0; v < b0.length; v++) { const i = v % bDims[0], r = v - i; out[v] = flip ? b0[r + bDims[0] - 1 - i] : b0[v]; }
    const mask = new Uint8Array(out.length); for (let v = 0; v < mask.length; v++) mask[v] = out[v] > 30 ? 1 : 0;
    return { dwi: { volumes: [{ dims: bDims, ijkToRAS: bM, data: out, dtype: "<f4" }], bValues: [0], gradients: [[0, 0, 0]], ijkToRAS: bM, source: "test", convention: 1 } as unknown as DiffusionSeries, motion: { moves: [], b0: out, mask } };
  };
  const right = scan(false), flipped = scan(true);
  const a = await alignToT1(right.dwi, t1, undefined, undefined, right.motion, { brainT1, check: 2 });
  const b = await alignToT1(flipped.dwi, t1, undefined, undefined, flipped.motion, { brainT1, check: 2 });
  console.log(`  right pair: ratio ${a.mirrorRatio?.toFixed(3)}${a.doubt ? ` (doubt: ${a.doubt})` : ""}; flipped: ratio ${b.mirrorRatio?.toFixed(3)}${b.doubt ? " (doubted)" : ""}`);
  assert(!a.doubt && a.mirrorRatio! < 0.93, `right pair: ${a.mirrorRatio} ${a.doubt ?? ""}`);
  assert(/flipped left to right/.test(b.doubt ?? ""), `flipped scan not doubted as flipped: ratio ${b.mirrorRatio}, ${b.doubt}`);
});

Deno.test("an empty brain in the diffusion scan is a doubt, not a pass (critic 2026-10-26, finding 9)", async () => {
  const t1Dims: [number, number, number] = [40, 48, 36], t1M = [4, 0, 0, -80, 0, 4, 0, -96, 0, 0, 4, -72, 0, 0, 0, 1];
  const t1: Grid3 = { dims: t1Dims, ijkToRAS: t1M, data: head(t1Dims, t1M) };
  const bDims: [number, number, number] = [32, 38, 28], bM = [5, 0, 0, -80, 0, 5, 0, -95, 0, 0, 5, -70, 0, 0, 0, 1], b0 = head(bDims, bM);
  const dwi = { volumes: [{ dims: bDims, ijkToRAS: bM, data: b0, dtype: "<f4" }], bValues: [0], gradients: [[0, 0, 0]], ijkToRAS: bM, source: "test", convention: 1 } as unknown as DiffusionSeries;
  const a = await alignToT1(dwi, t1, undefined, undefined, { moves: [], b0, mask: new Uint8Array(b0.length) }, { rule: 1 });
  assert(/could not be measured/.test(a.doubt ?? ""), `doubt: ${a.doubt}`);
});

Deno.test("the import's alignment, given, is used as recorded: no new alignment, the same scan on the T1's axes; a recorded doubt keeps the scan in place (2026-10-26)", async () => {
  const t1Dims: [number, number, number] = [40, 48, 36], t1M = [4, 0, 0, -80, 0, 4, 0, -96, 0, 0, 4, -72, 0, 0, 0, 1];
  const t1: Grid3 = { dims: t1Dims, ijkToRAS: t1M, data: head(t1Dims, t1M) };
  const bDims: [number, number, number] = [32, 38, 28], bM = [5, 0, 0, -80, 0, 5, 0, -95, 0, 0, 5, -70, 0, 0, 0, 1], b0 = head(bDims, bM);
  const dwi = () => ({ volumes: [{ dims: bDims, ijkToRAS: bM, data: Float32Array.from(b0), dtype: "<f4" }], bValues: [0], gradients: [[0, 0, 0]], ijkToRAS: bM, source: "test", convention: 1 }) as unknown as DiffusionSeries;
  const mask = new Uint8Array(b0.length); for (let v = 0; v < mask.length; v++) mask[v] = b0[v] > 30 ? 1 : 0;
  const motion = { moves: [], b0, mask };
  const made = await alignToT1(dwi(), t1, undefined, undefined, motion, { rule: 2, check: 1 });
  const times: { register?: number; resample?: number } = {};
  const given = await alignToT1(dwi(), t1, undefined, times, motion, { given: { T: made.T, rule: made.rule, mirrorRatio: 0.7, cost: made.result.cost } });
  assert(times.register === 0, "no alignment was made");
  assert(given.T === made.T && given.mirrorRatio === 0.7 && !given.doubt);
  assert(/as the import did/.test(given.said), given.said);
  const a = made.dwi.volumes[0].data, b = given.dwi.volumes[0].data;
  assert(a.length === b.length);
  for (let i = 0; i < a.length; i++) assertAlmostEquals(Number(b[i]), Number(a[i]), 1e-5);
  const doubted = await alignToT1(dwi(), t1, undefined, undefined, motion, { given: { T: made.T, rule: 3, doubt: "the scan matches its own mirror image about as well (0.99): the MRI of the anatomy may be another person's" } });
  assert(/another person's/.test(doubted.doubt ?? ""), `doubt kept: ${doubted.doubt}`);
  assert(doubted.dwi.volumes[0].dims.join() === bDims.join(), "a doubted alignment leaves the scan on its own grid");
});
