// Streamlines outside the brain (outside-brain.ts): one that crosses the fluid at the brain's edge is marked; one that
// only ends in it is not.
//   deno test -A --no-check outside-brain.test.ts
import { assertEquals } from "jsr:@std/assert@1";
import { outsideBrain } from "./outside-brain.ts";

Deno.test("a streamline crossing the edge fluid is outside the brain; one ending in it is not", () => {
  // A row of 12 voxels of 1 mm along x; voxels 4-8 are the fluid.
  const grid = { dims: [12, 1, 1], ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] };
  const region = Uint8Array.from([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]);
  const line = (from: number, to: number) => Float32Array.from(Array.from({ length: to - from + 1 }, (_, i) => [from + i, 0, 0]).flat());
  const r = outsideBrain([line(0, 11), line(0, 8), line(4, 11), line(2, 10), line(0, 6)], grid, region, 4, 2);
  // 0-11: crosses (4 brain points before, 3 after). 0-8: ends in it. 4-11: starts in it. 2-10: 2 before, 2 after: crosses.
  // 0-6: only 3 fluid points, and at its end.
  assertEquals([...r], [1, 0, 0, 1, 0]);
});
