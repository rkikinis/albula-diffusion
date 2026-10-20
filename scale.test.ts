// Sizes in voxels of the scan (scale.ts), at three scales: 1 nm voxels (1e-6 mm), a 2 mm diffusion grid, 1 parsec
// voxels (3.086e19 mm). On clinical grids every size is what it was in mm, bit for bit; far outside, it scales with the
// voxel (Ron, 2026-10-20: "did you forget about scales? from astrophysics to electron microscope").
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc scale.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { drawnRadius, finestVoxel, lengthText, probeReach, raySlack, seedRadius, tuned, voxelSizes } from "./scale.ts";

const NM = 1e-6, PC = 3.086e19;

Deno.test("tuned: the clinical value unchanged in its range of voxels, held to it outside, at 1 nm, 2 mm and 1 pc", () => {
  assertEquals(tuned(0.3, 2, 0.05, 0.5), 0.3);
  assertEquals(tuned(0.3, 2.5, 0.05, 0.5), 0.3);
  assertEquals(tuned(0.3, NM, 0.05, 0.5), 0.5 * NM, "half a voxel at most");
  assertEquals(tuned(0.3, PC, 0.05, 0.5), 0.05 * PC, "a twentieth of a voxel at least");
  assertEquals(tuned(0.3, undefined, 0.05, 0.5), 0.3, "no grid: the published value");
});

Deno.test("the module's sizes: today's on 2 and 2.5 mm diffusion grids, a sane number of voxels at 1 nm and 1 pc", () => {
  for (const v of [2, 2.5]) {
    assertEquals([drawnRadius("tubes", v), drawnRadius("lines", v), raySlack(v), probeReach(v), seedRadius(v)], [0.5, 0.12, 0.5, 2, 5], `${v} mm grid`);
  }
  // At 1 nm every size is at the top of its range of voxels, at 1 pc at the bottom: a sane number of voxels at both.
  const inVoxels = (v: number) => [drawnRadius("tubes", v), drawnRadius("lines", v), raySlack(v), probeReach(v), seedRadius(v)].map((x) => +(x / v).toPrecision(12));
  assertEquals(inVoxels(NM), [0.5, 0.12, 0.5, 2, 5]);
  assertEquals(inVoxels(PC), [0.1, 0.024, 0.1, 0.5, 1]);
});

Deno.test("voxel sizes from a grid's matrix, at any scale", () => {
  for (const v of [NM, 2, PC]) {
    assertEquals(finestVoxel([v, 0, 0, 0, 0, 2 * v, 0, 0, 0, 0, 3 * v, 0, 0, 0, 0, 1]), v);
    assertEquals(voxelSizes([0, 0, v, 0, v, 0, 0, 0, 0, v, 0, 0, 0, 0, 0, 1]), [v, v, v]);
  }
  assertEquals(finestVoxel(undefined), undefined);
  assertEquals(finestVoxel([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), undefined, "a degenerate grid has no voxel");
});

Deno.test("a length in words: as the module wrote it on clinical grids, readable at 1 nm and 1 pc", () => {
  assertEquals(lengthText(2), "2 mm");
  assertEquals(lengthText(1.3, true), "1.3 mm");
  assertEquals(lengthText(1, true), "1.0 mm");
  assertEquals(lengthText(2 * NM), "2.0e-6 mm");
  assertEquals(lengthText(2 * PC), "6.2e+19 mm");
});
