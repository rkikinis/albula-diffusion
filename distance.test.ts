import { assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { distanceMap } from "./distance.ts";

Deno.test("distance to a segment: exact, anisotropic spacing", () => {
  const dims = [9, 7, 5], M = [0.5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const inside = (i: number) => i === (2 * 7 + 3) * 9 + 4;             // one voxel at (4,3,2)
  const d = distanceMap(inside, dims, M);
  for (let k = 0; k < 5; k++) for (let j = 0; j < 7; j++) for (let i = 0; i < 9; i++) {
    assertAlmostEquals(d[(k * 7 + j) * 9 + i], Math.hypot(0.5 * (i - 4), 2 * (j - 3), k - 2), 1e-5);
  }
  assertEquals(distanceMap(() => false, [3, 3, 3], M)[0], Infinity);
});
