// DIPY's median_otsu, written from its source (median-otsu.ts): the fast filter against the plain one; the threshold; the mask.
import { assert, assertEquals } from "jsr:@std/assert@1";
import { medianFilter, medianFilterPasses, otsuThreshold } from "./median-otsu.ts";
import { rng } from "./tractcloud/tractcloud.ts";

Deno.test("the sliding-histogram filter gives exactly the plain filter's answer, pass after pass, edges mirrored", () => {
  const r = rng(7), dims = [13, 11, 9], n = 13 * 11 * 9;
  for (const kind of ["floats", "few values"]) {
    const data = Float64Array.from({ length: n }, () => kind === "floats" ? r() * 1000 : Math.floor(r() * 7) / 6);
    let slow = data;
    for (let p = 0; p < 3; p++) slow = medianFilter(slow, dims, 2);
    const fast = medianFilterPasses(data, dims, 2, 3);
    assertEquals(fast, slow, kind);
  }
});

Deno.test("Otsu's threshold splits two groups between them, at a bin center", () => {
  const v = Float64Array.from([...Array(500).fill(10), ...Array(500).fill(90)].map((x, i) => x + (i % 5)));
  const t = otsuThreshold(v);
  assert(t > 14 && t < 90, String(t));
  assertEquals(otsuThreshold(Float64Array.from([3, 3, 3])), 3);
});

Deno.test("as many distinct values as voxels (a corrected scan): still exact, with the block jumps", () => {
  const r = rng(11), dims = [17, 15, 12], n = 17 * 15 * 12;
  const data = Float64Array.from({ length: n }, () => r() * 1e4);
  let slow = data; for (let p = 0; p < 2; p++) slow = medianFilter(slow, dims, 3);
  assertEquals(medianFilterPasses(data, dims, 3, 2), slow);
});
