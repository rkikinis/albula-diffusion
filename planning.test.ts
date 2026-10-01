import { assert, assertEquals } from "jsr:@std/assert@1";
import { denseSeeds, otherSide, sortByDistance, wholeBrainSeeds } from "./planning.ts";
import type { TractCloudModel } from "./tractcloud/tractcloud.ts";
import { SHORT, type Named } from "./tractcloud/name-tracts.ts";
import type { TensorFit } from "./tensor.ts";

Deno.test("whole-brain seeds do not depend on the order the slices are stored in", () => {
  const [nx, ny, nz] = [7, 6, 5], n = nx * ny * nz;
  const fa = Float32Array.from({ length: n }, (_, v) => 0.1 + ((v * 37) % 11) / 11);
  const up = { dims: [nx, ny, nz], ijkToRAS: [2, 0, 0, -5, 0, 2, 0, -6, 0, 0, 2, 1, 0, 0, 0, 1], mask: new Uint8Array(n).fill(1), fa } as unknown as TensorFit;
  // The same grid, slices stored the other way round.
  const fa2 = new Float32Array(n);
  for (let k = 0; k < nz; k++) for (let v = 0; v < nx * ny; v++) fa2[(nz - 1 - k) * nx * ny + v] = fa[k * nx * ny + v];
  const down = { ...up, fa: fa2, ijkToRAS: [2, 0, 0, -5, 0, 2, 0, -6, 0, 0, -2, 1 + 2 * (nz - 1), 0, 0, 0, 1] } as unknown as TensorFit;
  const a = wholeBrainSeeds(up), b = wholeBrainSeeds(down);
  assert(a.length === b.length && a.length > 0);
  for (let i = 0; i < a.length; i++) for (let c = 0; c < 3; c++) assert(Math.abs(a[i][c] - b[i][c]) < 1e-9, `seed ${i} differs`);
});

// "MORE FIBERS" (Ron, 2026-10-01): dense starting points in the voxels a tract passes through.
Deno.test("dense seeds: every voxel a streamline passes through, 20 points inside each, the same every time (oblique grid)", () => {
  const c = Math.cos(0.4), s = Math.sin(0.4);
  const grid = { dims: [20, 20, 10], ijkToRAS: [2 * c, -2 * s, 0, 3, 2 * s, 2 * c, 0, -4, 0, 0, 2.5, 7, 0, 0, 0, 1] };
  const M = grid.ijkToRAS, at = (i: number, j: number, k: number) => [M[0] * i + M[1] * j + M[2] * k + M[3], M[4] * i + M[5] * j + M[6] * k + M[7], M[8] * i + M[9] * j + M[10] * k + M[11]];
  // A streamline through voxels (2..9, 5, 3), sampled twice per voxel, and one point off the grid.
  const pts: number[] = []; for (let i = 2; i <= 9; i += 0.5) pts.push(...at(i, 5, 3)); pts.push(...at(-9, 5, 3));
  const seeds = denseSeeds([Float32Array.from(pts)], grid);
  assertEquals(seeds.length, 8 * 20);
  // Each seed back in voxel space: inside one of those voxels, within half a voxel of its center.
  for (const p of seeds) {
    const x = p[0] - M[3], y = p[1] - M[7], z = p[2] - M[11];
    const i = (c * x + s * y) / 2, j = (-s * x + c * y) / 2, k = z / 2.5;
    assert(Math.round(j) === 5 && Math.round(k) === 3 && Math.round(i) >= 2 && Math.round(i) <= 9, `seed outside: ${i} ${j} ${k}`);
  }
  assertEquals(denseSeeds([Float32Array.from(pts)], grid), seeds);
});

Deno.test("dense seeds: past the cap, fewer per voxel; past that, a sample of the voxels, one each", () => {
  const grid = { dims: [100, 4, 4], ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
  const line = Float32Array.from(Array.from({ length: 100 }, (_, i) => [i, 1, 1]).flat());
  assertEquals(denseSeeds([line], grid, 20, 1000).length, 1000);          // 100 voxels x 10
  assertEquals(denseSeeds([line], grid, 20, 50).length, 50);              // 50 of the 100 voxels, one each
  assertEquals(denseSeeds([], grid).length, 0);
});

Deno.test("faint tracts: within reach with fewer than the minimum; every side's total for comparing", () => {
  const model = { json: { tracts: [{}, {}, {}] } } as unknown as TractCloudModel;   // tracts 0, 1; 2 is Other
  // Tract 0 right: 6 streamlines, 5 near (near). Tract 1 right: 4, 3 near (faint). Tract 1 left: 30, none near (far).
  const spec: [number, number, number][] = [...Array(5).fill([0, 1, 2]), [0, 1, 50], ...Array(3).fill([1, 1, 4]), [1, 1, 40], ...Array(30).fill([1, -1, 30]), [SHORT, 0, 1], [2, 0, 60]];
  const named = { tract: Int32Array.from(spec.map((x) => x[0])), side: Int8Array.from(spec.map((x) => x[1])), draws: 1, seconds: 0 } as Named;
  const r = sortByDistance(model, named, Float64Array.from(spec.map((x) => x[2])), 8);
  assertEquals(r.near.map((e) => [e.tract, e.side, e.within]), [[0, 1, 5]]);
  assertEquals(r.faint.map((e) => [e.tract, e.side, e.within]), [[1, 1, 3]]);
  assertEquals(r.far.length, 2);                                          // the faint one is still among the far
  assertEquals([r.total(1, 1), r.total(1, otherSide(1)), r.total(0, -1)], [4, 30, 0]);
  assertEquals(r.unnamedNear.length, 1);
});
