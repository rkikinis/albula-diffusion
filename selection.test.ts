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

Deno.test("through a structure: every line crossing a one-voxel layer drawn on a slice tilted 10-40 degrees counts (critic 2026-10-20, finding 1)", () => {
  // A 64^3 grid of 2 mm voxels, not at the origin. The region is a disk of radius 15 voxels filled the way the Draw tool
  // fills (core logic/fill-polygon.ts, copied, not imported): a voxel is in when its center lies within half a voxel of
  // the plane along the grid axis most aligned with the plane's normal -- one voxel per column, a staircase whose steps
  // touch only along an edge. The half-voxel samples missed 1-17% of these lines.
  const dims = [64, 64, 64], h = 2, ijk = [h, 0, 0, -61, 0, h, 0, -67, 0, 0, h, -59, 0, 0, 0, 1];
  const toRAS = (i: number, j: number, k: number) => [h * i - 61, h * j - 67, h * k - 59];
  let seed = 7; const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const unit = (a: number[]) => { const l = Math.hypot(a[0], a[1], a[2]); return a.map((x) => x / l); };
  const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const c = toRAS(31.3, 32.2, 30.7);   // the disk's center, off the voxel centers (no exact ties)
  for (const deg of [10, 20, 30, 40]) {
    // The plane's normal: tilted `deg` from the j axis (a coronal slice) about an oblique axis, so it steps in i and k.
    const t = deg * Math.PI / 180, n = unit([Math.sin(t) * 0.8, Math.cos(t), Math.sin(t) * 0.6]);
    let halfThick = 0;
    for (let a = 0; a < 3; a++) halfThick = Math.max(halfThick, Math.abs(ijk[a] * n[0] + ijk[4 + a] * n[1] + ijk[8 + a] * n[2]));
    halfThick *= 0.5;
    const d0 = dot(c, n), lab = new Uint8Array(64 * 64 * 64);
    let filled = 0;
    for (let k = 0; k < 64; k++) for (let j = 0; j < 64; j++) for (let i = 0; i < 64; i++) {
      const p = toRAS(i, j, k), off = [p[0] - c[0], p[1] - c[1], p[2] - c[2]], along = dot(off, n);
      if (Math.abs(dot(p, n) - d0) > halfThick) continue;
      if (Math.hypot(off[0] - along * n[0], off[1] - along * n[1], off[2] - along * n[2]) > 15 * h) continue;
      lab[(k * 64 + j) * 64 + i] = 1; filled++;
    }
    assert(filled > 500, `${deg}: ${filled} voxels`);
    // 2,000 straight lines crossing the disk within its inner 10 voxels, nearly along its normal, points 3 mm apart
    // (more than a voxel); and 200 crossing the plane 20-25 voxels out, beside the disk.
    const e1 = unit(cross(n, [1, 0, 0])), e2 = cross(n, e1);
    const lines: Float32Array[] = [], across = 2000;
    for (let s = 0; s < across + 200; s++) {
      const r = (s < across ? 10 * Math.sqrt(rnd()) : 20 + 5 * rnd()) * h, a = 2 * Math.PI * rnd();
      const at = [0, 1, 2].map((q) => c[q] + r * (Math.cos(a) * e1[q] + Math.sin(a) * e2[q]));
      const dir = unit([0, 1, 2].map((q) => n[q] + 0.3 * (rnd() - 0.5) * e1[q] + 0.3 * (rnd() - 0.5) * e2[q]));
      const start = -10 * h - 3 * rnd(), pts: number[] = [];
      for (let u = start; u < 10 * h; u += 3) pts.push(at[0] + u * dir[0], at[1] + u * dir[1], at[2] + u * dir[2]);
      lines.push(Float32Array.from(pts));
    }
    const got = throughVoxels(lines, { dims, ijkToRAS: ijk, inside: (v) => lab[v] === 1 });
    assertEquals(got.filter((x) => x < across).length, across, `${deg} degrees: lines crossing the disk`);
    assertEquals(got.filter((x) => x >= across).length, 0, `${deg} degrees: lines beside it`);
  }
});

Deno.test("through a structure: a line through only the corner of a voxel counts; one missing it by a hair does not", () => {
  const dims = [10, 10, 10], ijk = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const inside = (v: number) => v === (5 * 10 + 5) * 10 + 5;   // voxel (5, 5, 5): [4.5, 5.5) on each axis
  // Two diagonal pieces whose ends lie in the neighbors (4, 5, 5) and (5, 4, 5): x + y = 9.02 cuts the voxel's corner
  // (x and y from 4.5 to 4.52); x + y = 8.98 passes just beside it.
  const clip = new Float32Array([4, 5.02, 5, 5.02, 4, 5]), miss = new Float32Array([4, 4.98, 5, 4.98, 4, 5]);
  assertEquals(throughVoxels([clip, miss], { dims, ijkToRAS: ijk, inside }), [0]);
});
