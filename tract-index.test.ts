// tractsNear on sets whose answer is known: two tracts crossing near a point, one far away.
//   deno test -A --no-check tract-index.test.ts   (with core's config)
import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert";
import { buildTractIndex, firstAlongRay, tractsNear } from "./tract-index.ts";

const line = (from: number[], step: number[], n: number) => Float32Array.from({ length: n * 3 }, (_, i) => from[i % 3] + step[i % 3] * Math.floor(i / 3));

Deno.test("the tracts near a point: which, how many of their streamlines, how close; far ones not", () => {
  const along = [line([-20, 0, 0], [1, 0, 0], 41), line([-20, 1, 0], [1, 0, 0], 41)];   // two streamlines along x
  const across = [line([0.5, -20, 0], [0, 1, 0], 41)];                                    // one along y, 0.5 mm off
  const far = [line([50, 50, 50], [1, 0, 0], 10)];
  const ix = buildTractIndex([along, across, far]);
  const r = tractsNear(ix, [0, 0, 0], 2);
  assertEquals(r.map((x) => x.set), [0, 1]);
  assertEquals(r[0].streamlines, 2);
  assertEquals(r[0].closestMm, 0);
  assertEquals(r[1].closestMm, 0.5);
  assertEquals(tractsNear(ix, [0, 0, 30], 2), [], "nothing near a point away from all of them");
});

// THE SAME TRACTS AT 1 nm, 2 mm AND 1 pc VOXELS (Ron, 2026-10-20: "did you forget about scales?"): the set above with
// every length times a voxel, the cell one voxel: the same answers, a sane number of cells, the ray march bounded.
Deno.test("the same answers at 1 nm, 2 mm and 1 pc voxels: near a point, along a ray; the march bounded", () => {
  for (const v of [1e-6, 2, 3.086e19]) {
    const s = (f: Float32Array) => f.map((x) => x * v);
    const along = [line([-20, 0, 0], [1, 0, 0], 41), line([-20, 1, 0], [1, 0, 0], 41)].map(s), across = [s(line([0.5, -20, 0], [0, 1, 0], 41))];
    const far = [s(line([50, 50, 50], [1, 0, 0], 10))];
    const ix = buildTractIndex([along, across, far], v);
    assert(ix.cells.size <= 133, `${ix.cells.size} cells for 133 points`);
    const r = tractsNear(ix, [0, 0, 0], 2 * v);
    assertEquals(r.map((x) => x.set), [0, 1], `${v} mm voxels`);
    assertEquals(r[0].streamlines, 2);
    assertAlmostEquals(r[1].closestMm / v, 0.5, 1e-5);
    assertEquals(tractsNear(ix, [0, 0, 30 * v], 2 * v), []);
    // A ray down the z axis from above meets the tracts at the origin's level (z = 0, 40 voxels down), within a voxel.
    const t = firstAlongRay(ix, [0, 0, 40 * v], [0, 0, -1], v, 0.5 * v);
    assert(t !== null && Math.abs(t / v - 39) <= 1, `hit at ${t === null ? "none" : t / v} voxels`);
    // A step a billion times too fine (as 0.5 mm was on a parsec grid) still ends at once: at most 4,096 steps.
    const t0 = performance.now();
    assertEquals(firstAlongRay(ix, [200 * v, 200 * v, 40 * v], [0, 0, -1], v, 1e-9 * v), null);
    assert(performance.now() - t0 < 2000, "the march is bounded");
  }
});

Deno.test("a radius far wider than the cell: every point looked at once, not more cells than points", () => {
  const ix = buildTractIndex([[line([0, 0, 0], [1, 0, 0], 10)]], 1e-6);
  const t0 = performance.now();
  const r = tractsNear(ix, [0, 0, 0], 100);
  assertEquals(r.length, 1); assertEquals(r[0].streamlines, 1);
  assert(performance.now() - t0 < 500, "no loop over 1e24 cells");
  // Without a cell, the streamlines' own point spacing is used (here 1).
  assertEquals(buildTractIndex([[line([0, 0, 0], [1, 0, 0], 10)]]).cell, 1);
  assertAlmostEquals(buildTractIndex([[line([0, 0, 0], [3e19, 0, 0], 10)]]).cell / 3e19, 1, 1e-6);
});
