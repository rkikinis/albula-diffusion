// tractsNear on sets whose answer is known: two tracts crossing near a point, one far away.
//   deno test -A --no-check tract-index.test.ts   (with core's config)
import { assertEquals } from "jsr:@std/assert";
import { buildTractIndex, tractsNear } from "./tract-index.ts";

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
