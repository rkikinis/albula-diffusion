// selection.ts: by name, through a box, the whole-tract numbers (each voxel once), laterality, the CSV, the opening way.
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc selection.test.ts
import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { byName, laterality, measure, openingWay, throughBox, throughVoxels, toCsv, voxelsOf } from "./selection.ts";

const line = (...pts: number[][]) => Float32Array.from(pts.flat());

Deno.test("by name: the streamlines of each chosen tract and side, nothing else", () => {
  const names = { tract: Int32Array.from([3, 3, 5, 3, 7]), side: Int8Array.from([-1, 1, -1, -1, 0]) };
  const got = byName(names, [{ tract: 3, side: -1 }, { tract: 7, side: 0 }, { tract: 9, side: 1 }]);
  assertEquals([...got.entries()], [["3:-1", [0, 3]], ["7:0", [4]], ["9:1", []]]);
});

Deno.test("through a box: a streamline counts when any point is inside, in the box's own axes", () => {
  const a = line([0, 0, 0], [10, 0, 0]), b = line([20, 20, 20], [30, 30, 30]), c = line([4, 4, 0], [6, 6, 0]);
  assertEquals(throughBox([a, b, c], { center: [0, 0, 0], size: [2, 2, 2] }), [0]);
  // A box turned 45 degrees about z, long along the diagonal: it holds c's points, not a's far end only.
  const r = Math.SQRT1_2;
  assertEquals(throughBox([a, b, c], { center: [5, 5, 0], size: [4, 1, 1], orientation: [r, -r, 0, r, r, 0, 0, 0, 1] }), [2]);
});

Deno.test("the numbers: each voxel once, volume from the grid, median length, mean maps", () => {
  const grid = { dims: [10, 10, 10], ijkToRAS: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1] };   // 2 mm voxels: 8 µL each
  // Two streamlines over the same two voxels (0,0,0) and (1,0,0): counted once each.
  const s1 = line([0, 0, 0], [2, 0, 0]), s2 = line([0, 0, 0], [2, 0, 0], [2, 0, 0]);
  const fa = new Float32Array(1000); fa[0] = 0.2; fa[1] = 0.6;
  const md = new Float32Array(1000).fill(0.0008);
  const m = measure([s1, s2], { ...grid, fa, md }, 1);
  assertEquals([m.lines, m.voxels], [2, 2]);
  assertAlmostEquals(m.volumeMl, 0.016, 1e-9);
  assertAlmostEquals(m.fa!, 0.4, 1e-6);   // rule 1: (0.2 + 0.6) / 2, each voxel once
  // Rule 2, the default: every point counts -- s1 reads (0,0,0),(1,0,0); s2 (0,0,0),(1,0,0),(1,0,0).
  assertAlmostEquals(measure([s1, s2], { ...grid, fa, md }).fa!, (2 * 0.2 + 3 * 0.6) / 5, 1e-6);
  assertAlmostEquals(m.md!, 0.0008, 1e-9);
  assertEquals(m.ad, undefined);
  assertAlmostEquals(m.lengthMm, 2, 1e-6);
  assertEquals(voxelsOf([line([-5, 0, 0])], grid).size, 0);   // outside the grid: no voxel
});

Deno.test("laterality: (left - right) / (left + right), none when it cannot be said", () => {
  assertAlmostEquals(laterality(3, 1)!, 0.5, 1e-12);
  assertEquals(laterality(0, 0), undefined);
  assertEquals(laterality(undefined, 2), undefined);
});

Deno.test("the CSV: a header, a row per tract and side, quoted where needed", () => {
  const csv = toCsv([{ scan: 'PAT16 · DTI "AP"', tract: "Arcuate fasciculus", abbreviation: "AF", side: "left", madeBy: "tracking rule 3",
    m: { lines: 10, voxels: 4, volumeMl: 0.032, lengthMm: 88.25, fa: 0.4567, md: 0.00081 } }]);
  const [head, row] = csv.trim().split("\n");
  assert(head.startsWith("scan,tract,abbreviation,side,lines"));
  assertEquals(row, '"PAT16 · DTI ""AP""",Arcuate fasciculus,AF,left,10,4,0.03,88.3,0.4567,0.0008100,,,tracking rule 3,,');
  // The grid and the rule the means were taken by travel with the numbers (critic 2026-10-15, finding 23).
  const r2 = toCsv([{ scan: "s", tract: "t", abbreviation: "T", side: "right", madeBy: "m", gridMm: [2.5, 2.5, 2.5], rule: 2, m: { lines: 1, voxels: 1, volumeMl: 0, lengthMm: 0 } }]).trim().split("\n")[1];
  assert(r2.includes(",m,2.50 x 2.50 x 2.50,") && r2.endsWith(",2: mean over the streamlines' points (as SlicerDMRI's tractography measurements)"), r2);
});

Deno.test("the face opens near the tumor when there is one, else by name", () => {
  assertEquals([openingWay(true), openingWay(false)], ["near", "name"]);
});

Deno.test("through a structure: a line crossing a one-voxel-thick region counts, even stepping past it; one beside it does not", () => {
  // A 10 x 10 x 10 grid of 1 mm voxels; the region is the coronal layer j = 5, i 3..6, k 3..6.
  const dims = [10, 10, 10], ijk = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const inside = (v: number) => { const i = v % 10, j = Math.floor(v / 10) % 10, k = Math.floor(v / 100); return j === 5 && i >= 3 && i <= 6 && k >= 3 && k <= 6; };
  const across = new Float32Array([4, 2, 4, 4, 8, 4]);   // one 6 mm step from j 2 to j 8, through the layer
  const beside = new Float32Array([8, 2, 4, 8, 8, 4]);   // the same, at i = 8, outside the region
  const ends = new Float32Array([5, 5, 5]);              // a single point in it
  assertEquals(throughVoxels([across, beside, ends], { dims, ijkToRAS: ijk, inside }), [0, 2]);
});

Deno.test("through a structure: an empty segment passes nothing", () => {
  assertEquals(throughVoxels([new Float32Array([1, 1, 1, 2, 2, 2])], { dims: [4, 4, 4], ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], inside: () => false }), []);
});
