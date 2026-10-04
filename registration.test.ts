// The diffusion scan onto the T1 (registration.ts): a known rigid move is found again on a synthetic head, and the
// resampling with no move and no field gives the scan back. The comparisons with Mike Halle's code and BRAINSFit on the
// patients are Contents/tools/registration-check.ts (a check tool; 2026-10-03: 0.03-0.09 and 0.18-0.34 mm apart).
//   deno test -A --no-check registration.test.ts
import { assert, assertAlmostEquals } from "jsr:@std/assert@1";
import { applyRigid, rigidSize, rigidToT1, resampleOntoT1, rotation, type Grid3, type Rigid } from "./registration.ts";
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

Deno.test("the resampled box holds the T1-space points it is given besides the scan's covered voxels (rule 3's brain)", async () => {
  const dims: [number, number, number] = [12, 10, 8], M = [2, 0, 0, -10, 0, 2, 0, -8, 0, 0, 2, -6, 0, 0, 0, 1];
  const data = Float32Array.from({ length: 960 }, (_, v) => v % 7);
  const dwi = { volumes: [{ dims, ijkToRAS: M, data, dtype: "<f4" }], bValues: [0], gradients: [[0, 0, 0]], ijkToRAS: M, source: "test", convention: 1 } as unknown as DiffusionSeries;
  const cover = new Uint8Array(960); cover[(4 * 10 + 5) * 12 + 6] = 1;                     // one voxel, at (2, 2, 2) mm
  const I = { R: rotation(0, 0, 0), t: [0, 0, 0], c: [0, 0, 0] }, t1 = { dims: [50, 50, 50] as [number, number, number], ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
  const small = await resampleOntoT1(dwi, I, t1, undefined, { cover, marginMm: 0 });
  const big = await resampleOntoT1(dwi, I, t1, undefined, { cover, marginMm: 0, alsoCover: [[-6, -4, -2]] });
  assert(small.volumes[0].dims.join() === "1,1,1", `small ${small.volumes[0].dims}`);
  assert(big.volumes[0].dims.join() === "5,4,3", `big ${big.volumes[0].dims}`);
  assertAlmostEquals(big.ijkToRAS[3], -6, 1e-9);
});
