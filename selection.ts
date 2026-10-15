// CHOOSING TRACTS AND MEASURING THEM -- the Diffusion module's v3 face (Ron, 2026-10-15: "v3: 1: yes, 2: yes"; the
// workspace's Contents/docs/DMRI-WORKFLOW-RETHINK.md and mockups/diffusion-generic-v3-2026-10-14.html). Plain functions on
// a whole-brain run (its streamlines, each with a tract and a side), so they are tested (selection.test.ts); module.ts
// calls them and holds no copy of the rules.
//   - by name: the streamlines of chosen tracts and sides;
//   - through a box: the streamlines with a point inside a box (a Markups Box: center, size, its own axes);
//   - the numbers of a set of streamlines: lines, the mean FA / MD / AD / RD, the volume (the voxels they pass through,
//     each once) and the median length. HOW THE MEAN IS TAKEN is a numbered rule (MEASURE_RULES; the export records it):
//     2, the default since 2026-10-15, the mean over the streamlines' points, each read from the voxel it falls in, as
//     SlicerDMRI's tractography measurements take it -- a core voxel hundreds of lines pass through counts as often as
//     they do; 1, the first version, the mean over the voxels passed through, each once, which the critic found about a
//     third lower (an edge voxel one line grazes weighs as much as the core: qa/2026-10-15-diffusion-v3-face.md, finding
//     4). Jones, Knosche and Turner 2013 on why the number of lines is no measure of the brain;
//   - laterality (left - right) / (left + right);
//   - the export: one CSV row per tract and side.

/** A whole-brain run's names: per streamline its tract (an index into the atlas's list, or the "not named" index) and
 *  its side (+1 right, -1 left, 0 none). */
export interface Names { tract: Int32Array; side: Int8Array }

/** The streamlines of each chosen tract and side, keyed "tract:side". */
export function byName(names: Names, chosen: { tract: number; side: number }[]): Map<string, number[]> {
  const want = new Set(chosen.map((c) => `${c.tract}:${c.side}`)), out = new Map<string, number[]>();
  for (const k of want) out.set(k, []);
  for (let i = 0; i < names.tract.length; i++) {
    const k = `${names.tract[i]}:${names.side[i]}`;
    if (want.has(k)) out.get(k)!.push(i);
  }
  return out;
}

/** A box as Markups keeps one: its center and full size in mm (RAS), and its own axes (columns, unit; none = RAS). */
export interface Box { center: number[]; size: number[]; orientation?: number[] }

/** The streamlines with at least one point inside the box (RAS points, three per point). */
export function throughBox(sl: Float32Array[], box: Box): number[] {
  const [cx, cy, cz] = box.center, h = box.size.map((x) => Math.abs(x) / 2);
  // The box's axes as rows of a 3x3 (its orientation's columns), so a point's offset is read in the box's own frame.
  const o = box.orientation && box.orientation.length >= 9 ? box.orientation : [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const ax = [[o[0], o[3], o[6]], [o[1], o[4], o[7]], [o[2], o[5], o[8]]];
  const out: number[] = [];
  for (let i = 0; i < sl.length; i++) {
    const p = sl[i];
    for (let k = 0; k < p.length; k += 3) {
      const dx = p[k] - cx, dy = p[k + 1] - cy, dz = p[k + 2] - cz;
      if (Math.abs(dx * ax[0][0] + dy * ax[0][1] + dz * ax[0][2]) <= h[0] && Math.abs(dx * ax[1][0] + dy * ax[1][1] + dz * ax[1][2]) <= h[1]
        && Math.abs(dx * ax[2][0] + dy * ax[2][1] + dz * ax[2][2]) <= h[2]) { out.push(i); break; }
    }
  }
  return out;
}

/** The maps measured: on one grid (the stored maps' grid, after every correction). Any may be missing. */
export interface Maps { dims: number[]; ijkToRAS: number[]; fa?: ArrayLike<number>; md?: ArrayLike<number>; ad?: ArrayLike<number>; rd?: ArrayLike<number> }

export interface Measures { lines: number; voxels: number; volumeMl: number; lengthMm: number; fa?: number; md?: number; ad?: number; rd?: number }

/** The inverse of a 4x4 affine (row-major), for RAS -> voxel. */
function inverse(m: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  const R = [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((x) => x / det);
  const t = [m[3], m[7], m[11]];
  return [R[0], R[1], R[2], -(R[0] * t[0] + R[1] * t[1] + R[2] * t[2]), R[3], R[4], R[5], -(R[3] * t[0] + R[4] * t[1] + R[5] * t[2]), R[6], R[7], R[8], -(R[6] * t[0] + R[7] * t[1] + R[8] * t[2]), 0, 0, 0, 1];
}

/** The voxels (flat indices on the maps' grid) the streamlines pass through, each once. */
export function voxelsOf(sl: Float32Array[], grid: { dims: number[]; ijkToRAS: number[] }): Set<number> {
  const inv = inverse(grid.ijkToRAS), [nx, ny, nz] = grid.dims, out = new Set<number>();
  for (const p of sl) for (let k = 0; k < p.length; k += 3) {
    const x = p[k], y = p[k + 1], z = p[k + 2];
    const i = Math.round(inv[0] * x + inv[1] * y + inv[2] * z + inv[3]), j = Math.round(inv[4] * x + inv[5] * y + inv[6] * z + inv[7]), l = Math.round(inv[8] * x + inv[9] * y + inv[10] * z + inv[11]);
    if (i >= 0 && j >= 0 && l >= 0 && i < nx && j < ny && l < nz) out.add(i + nx * (j + ny * l));
  }
  return out;
}

const lengthOf = (p: Float32Array) => { let L = 0; for (let k = 3; k < p.length; k += 3) L += Math.hypot(p[k] - p[k - 3], p[k + 1] - p[k - 2], p[k + 2] - p[k - 1]); return L; };

/** How a tract's mean is taken: the numbered rules (Ron, 2026-09-25: "modular … and also versioned"). */
// Rule 2 is SlicerDMRI's (Ron, 2026-10-15: "use and reference the slicer approach"): FiberTractMeasurements.cxx sums a
// measure over every point of the bundle and divides by the points (references.ts). Albula reads each point's value from
// the map's voxel under it; SlicerDMRI the value stored on the point.
export const MEASURE_RULES = { 1: "mean over the voxels passed through, each once", 2: "mean over the streamlines' points (as SlicerDMRI's tractography measurements)" } as const;
export type MeasureRule = keyof typeof MEASURE_RULES;
export const MEASURE_RULE: MeasureRule = 2;

/** The flat voxel index of every streamline point on the grid, repeats kept (rule 2's samples). */
function pointVoxels(sl: Float32Array[], grid: { dims: number[]; ijkToRAS: number[] }): number[] {
  const inv = inverse(grid.ijkToRAS), [nx, ny, nz] = grid.dims, out: number[] = [];
  for (const p of sl) for (let k = 0; k < p.length; k += 3) {
    const x = p[k], y = p[k + 1], z = p[k + 2];
    const i = Math.round(inv[0] * x + inv[1] * y + inv[2] * z + inv[3]), j = Math.round(inv[4] * x + inv[5] * y + inv[6] * z + inv[7]), l = Math.round(inv[8] * x + inv[9] * y + inv[10] * z + inv[11]);
    if (i >= 0 && j >= 0 && l >= 0 && i < nx && j < ny && l < nz) out.push(i + nx * (j + ny * l));
  }
  return out;
}

/** The whole-tract numbers of a set of streamlines on the maps; the mean taken by `rule` (MEASURE_RULES). */
export function measure(sl: Float32Array[], maps: Maps, rule: MeasureRule = MEASURE_RULE): Measures {
  const vox = voxelsOf(sl, maps);
  const m = maps.ijkToRAS, voxelMl = Math.abs(m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8])) / 1000;
  const samples: number[] = rule === 1 ? [...vox] : pointVoxels(sl, maps);
  const mean = (a?: ArrayLike<number>) => { if (!a || !samples.length) return undefined; let s = 0; for (const v of samples) s += Number(a[v]); return s / samples.length; };
  const lengths = sl.map(lengthOf).sort((a, b) => a - b);
  const median = lengths.length ? (lengths.length % 2 ? lengths[(lengths.length - 1) / 2] : (lengths[lengths.length / 2 - 1] + lengths[lengths.length / 2]) / 2) : 0;
  const out: Measures = { lines: sl.length, voxels: vox.size, volumeMl: vox.size * voxelMl, lengthMm: median };
  const fa = mean(maps.fa), md = mean(maps.md), ad = mean(maps.ad), rd = mean(maps.rd);
  if (fa !== undefined) out.fa = fa; if (md !== undefined) out.md = md; if (ad !== undefined) out.ad = ad; if (rd !== undefined) out.rd = rd;
  return out;
}

/** How lopsided: (left - right) / (left + right); undefined when both are 0 or one is missing. */
export function laterality(left: number | undefined, right: number | undefined): number | undefined {
  if (left === undefined || right === undefined || left + right === 0) return undefined;
  return (left - right) / (left + right);
}

/** One table row per tract and side, as a CSV a spreadsheet opens (RFC 4180 quoting). */
export interface ExportRow { scan: string; tract: string; abbreviation: string; side: string; m: Measures; madeBy: string;
  /** The maps' voxel size in mm (the volume depends on it), and the rule the means were taken by. */ gridMm?: number[]; rule?: MeasureRule }
export function toCsv(rows: ExportRow[]): string {
  const q = (s: string) => /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  const n = (x: number | undefined, d: number) => x === undefined ? "" : x.toFixed(d);
  const head = ["scan", "tract", "abbreviation", "side", "lines", "voxels", "volume_ml", "median_length_mm", "fa", "md_mm2_per_s", "ad_mm2_per_s", "rd_mm2_per_s", "made_by", "map_grid_mm", "mean_rule"];
  const lines = rows.map((r) => [q(r.scan), q(r.tract), q(r.abbreviation), r.side, String(r.m.lines), String(r.m.voxels), n(r.m.volumeMl, 2), n(r.m.lengthMm, 1),
    n(r.m.fa, 4), n(r.m.md, 7), n(r.m.ad, 7), n(r.m.rd, 7), q(r.madeBy), r.gridMm ? r.gridMm.map((x) => x.toFixed(2)).join(" x ") : "", r.rule ? q(`${r.rule}: ${MEASURE_RULES[r.rule]}`) : ""].join(","));
  return [head.join(","), ...lines].join("\n") + "\n";
}

/** Which way the face opens on (Ron, 2026-10-14: near a structure the default; "Any other brain opens switched to By
 *  name"): near the tumor when the case has a tumor outline, else by name. */
export type Way = "near" | "name" | "box";
export const openingWay = (hasTumor: boolean): Way => hasTumor ? "near" : "name";
