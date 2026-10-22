// THE CORTICOSPINAL TRACT, CHECKED AGAINST THE THALAMUS (Ron, 2026-10-21, on Steve's Prisma scan: "Clearly in violation of
// anatomy. growing into the thalamus and outside the crus"; his anatomy of 2026-10-06, TRACT-REVIEW.md: the tract runs in
// the posterior limb of the internal capsule, between the thalamus medially and the lentiform nucleus laterally; a fiber
// in or through the thalamus is errant -- most likely the sensory radiation, thalamus to postcentral gyrus).
//
// RULE 1: a streamline the tract-naming network called the corticospinal tract that has at least `minPoints` points
// inside a thalamus -- inside meaning in it AND with its six neighbors in it too, one voxel in from its edge, so a fiber
// grazing the posterior limb's border is not taken -- is not the corticospinal tract: it is moved to the unnamed
// streamlines. The thalamus is FastSurfer's (the brain's structures stored at import), on the T1's grid, found by its
// code (SNOMED 42695009, Thalamus), either side. The crus test (in front of the substantia nigra) is not in this rule:
// FastSurfer has no nigra, and the ventral diencephalon holds the crus and what lies behind it alike.
// On the Prisma scan before this rule: 324 of 572 left and 316 of 886 right corticospinal streamlines passed through.

export const CST_THALAMUS_RULE = 1;
/** SNOMED CT: Thalamus (structure). */
export const THALAMUS_CODE = "42695009";

export interface LabelGrid { dims: readonly [number, number, number] | readonly number[]; ijkToRAS: readonly number[]; lab: ArrayLike<number> }

/** True for a corticospinal tract's set label, as TractCloud's table names it. */
export const isCorticospinal = (label: string) => /cortico-?spinal/i.test(label);

/** A function giving, for a point in RAS mm, its voxel index on the grid (-1 outside). */
export function voxelIndexer(g: LabelGrid): (x: number, y: number, z: number) => number {
  const M = g.ijkToRAS, [nx, ny, nz] = g.dims as number[];
  const a = [M[0], M[1], M[2], M[4], M[5], M[6], M[8], M[9], M[10]];
  const det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
  const inv = [(a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det, (a[1] * a[5] - a[2] * a[4]) / det,
    (a[5] * a[6] - a[3] * a[8]) / det, (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
    (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det, (a[0] * a[4] - a[1] * a[3]) / det];
  return (x, y, z) => {
    const p0 = x - M[3], p1 = y - M[7], p2 = z - M[11];
    const i = Math.round(inv[0] * p0 + inv[1] * p1 + inv[2] * p2), j = Math.round(inv[3] * p0 + inv[4] * p1 + inv[5] * p2), k = Math.round(inv[6] * p0 + inv[7] * p1 + inv[8] * p2);
    return i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz ? -1 : (k * ny + j) * nx + i;
  };
}

/** The voxels of the given labels one voxel in from their edge (in the label, and all six neighbors in the same label). */
export function coreOf(g: LabelGrid, values: ReadonlySet<number>): Uint8Array {
  const [nx, ny, nz] = g.dims as number[], lab = g.lab, out = new Uint8Array(nx * ny * nz), s = nx * ny;
  for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const v = (k * ny + j) * nx + i, l = lab[v];
    if (!values.has(l)) continue;
    if (lab[v - 1] === l && lab[v + 1] === l && lab[v - nx] === l && lab[v + nx] === l && lab[v - s] === l && lab[v + s] === l) out[v] = 1;
  }
  return out;
}

export interface GateResult<S> { sets: S[]; moved: { label: string; side: number; of: number; moved: number }[] }

/**
 * Rule 1 over a run's tract sets: each corticospinal set keeps the streamlines that do not pass through a thalamus; the
 * others join the unnamed set (made when there is none). Other sets are returned as they were.
 */
export function gateCorticospinal<S extends { label: string; side: number; streamlines: Float32Array[] }>(
  sets: S[], brain: LabelGrid, thalamusValues: ReadonlySet<number>, unnamed: (moved: Float32Array[]) => S, opts: { minPoints?: number } = {},
): GateResult<S> {
  const minPoints = opts.minPoints ?? 2, at = voxelIndexer(brain), core = coreOf(brain, thalamusValues);
  const through = (f: Float32Array) => { let n = 0; for (let q = 0; q < f.length; q += 3) { const v = at(f[q], f[q + 1], f[q + 2]); if (v >= 0 && core[v] && ++n >= minPoints) return true; } return false; };
  const moved: GateResult<S>["moved"] = [], out: S[] = [], taken: Float32Array[] = [];
  for (const s of sets) {
    if (!isCorticospinal(s.label)) { out.push(s); continue; }
    const keep: Float32Array[] = [];
    for (const f of s.streamlines) (through(f) ? taken : keep).push(f);
    moved.push({ label: s.label, side: s.side, of: s.streamlines.length, moved: s.streamlines.length - keep.length });
    if (keep.length) out.push({ ...s, streamlines: keep });
  }
  if (taken.length) {
    const u = out.findIndex((s) => s.label === unnamed([]).label);
    if (u >= 0) out[u] = { ...out[u], streamlines: [...out[u].streamlines, ...taken] };
    else out.push(unnamed(taken));
  }
  return { sets: out, moved };
}

// ── RULE 2 (2026-10-21, Ron on the Prisma scan after rule 1: "internal capsule: mostly ok. A few stray tubes in the
// putamen. Crus cerebri: false positives in the hipocampus and brainstem parcels"; "Many fibers outside the precentral
// gyrus. Those are all false positives"). Rule 1's thalamus, and three more tests, each on FastSurfer's labels:
//   THE LENTIFORM NUCLEUS: `lentiformMinPoints` points or more in the putamen or the pallidum (in the label, not one voxel
//     in: the strays Ron saw run along its edge as well);
//   THE CRUS: where the streamline crosses the crus plane (the brainstem frame's plane at the height where the two tracts
//     are 11 mm from the midline, review.ts levelsFromPair), the crossing is not in the brainstem or the hippocampus --
//     at that height FastSurfer labels the peduncles ventral diencephalon and the tegmentum behind them brainstem;
//   THE END: walking in from the streamline's upper end up to `endReachMm`, the first cortical parcel at a point or one of
//     its six neighbors is an allowed one (the precentral gyrus; the paracentral lobule, which holds the leg's motor
//     area in this parcellation, only when asked).
// A streamline that fails any test is moved to the unnamed ones; the first test it failed is counted.
export const CST_RULE_2 = { id: 2, lentiformMinPoints: 2, endReachMm: 6 } as const;
// ── RULE 3 (the critic, 2026-10-21, finding 9): rule 2 with two classification choices made sturdier.
//   THE END'S PARCEL by majority: the cortical labels of the point and its 26 neighbors are counted and the most frequent
//     wins (a tie goes to the nearer voxel). Rule 2 took the point's own label, else the first cortical neighbor in the
//     axis order +i, -i, +j, -j, +k, -k -- at the central sulcus that let the grid's orientation decide precentral or
//     postcentral.
//   THE CRUS TEST ONLY WHERE THE PLANE IS IN THE VENTRAL DIENCEPHALON: FastSurfer's ventral diencephalon ends in the upper
//     midbrain and the peduncles below it are labeled brainstem. The plane's own cut is measured: when it passes through
//     fewer than `crusMinVentralVoxels` voxels of the ventral diencephalon (both sides, within half a voxel of the plane),
//     the plane is below it and every true crus crossing would read "brainstem": the crus test is skipped and the result
//     says so. (The share of the tract's own crossings was tried first and is no measure: the false positives the test is
//     for pull it down -- 34% on the Prisma scan, whose plane does cut the ventral diencephalon.)
export const CST_RULE_3 = { id: 3, lentiformMinPoints: 2, endReachMm: 6, endBy: "majority", crusMinVentralVoxels: 50 } as const;
export type CstRule = typeof CST_RULE_2 | typeof CST_RULE_3;
export const CODES = { precentral: "465006", paracentral: "9898001", brainstem: "15926001", hippocampus: "5366008", putamen: "89278009", pallidum: "14738005", ventralDC: "110700" } as const;
export type Failed = "thalamus" | "lentiform" | "crus" | "end";
export interface CstAnatomy {
  brain: LabelGrid; thalamus: ReadonlySet<number>; lentiform: ReadonlySet<number>;
  /** Every cortical parcel's label value, and those an end may be in. */
  cortex: ReadonlySet<number>; allowedEnd: ReadonlySet<number>;
  /** The crus plane and the labels a crossing of it may not be in; absent: the crus is not tested. */
  crus?: { origin: [number, number, number]; normal: [number, number, number]; errant: ReadonlySet<number>;
    /** Rule 3: the ventral diencephalon's labels, for the check that the plane lies in it. */ ventral?: ReadonlySet<number> };
}
/** The label values where a streamline crosses the crus plane (every crossing). */
function crusCrossings(f: Float32Array, crus: NonNullable<CstAnatomy["crus"]>, lab: ArrayLike<number>, at: (x: number, y: number, z: number) => number): number[] {
  const { origin: o, normal: m } = crus, out: number[] = [];
  for (let k = 1; k < f.length / 3; k++) {
    const a = 3 * (k - 1), b = 3 * k;
    const da = (f[a] - o[0]) * m[0] + (f[a + 1] - o[1]) * m[1] + (f[a + 2] - o[2]) * m[2], db = (f[b] - o[0]) * m[0] + (f[b + 1] - o[1]) * m[1] + (f[b + 2] - o[2]) * m[2];
    if (da === db || (da > 0) === (db > 0)) continue;
    const s = da / (da - db), v = at(f[a] + s * (f[b] - f[a]), f[a + 1] + s * (f[b + 1] - f[a + 1]), f[a + 2] + s * (f[b + 2] - f[a + 2]));
    if (v >= 0) out.push(lab[v]);
  }
  return out;
}
/** The labels the plane passes through: voxels whose center lies within half a voxel of it, counted by label. */
function planeCut(g: LabelGrid, o: readonly number[], m: readonly number[]): Map<number, number> {
  const [nx, ny, nz] = g.dims as number[], M = g.ijkToRAS, out = new Map<number, number>();
  // The plane's distance per step along i, j, k, and half a voxel along the normal.
  const di = M[0] * m[0] + M[4] * m[1] + M[8] * m[2], dj = M[1] * m[0] + M[5] * m[1] + M[9] * m[2], dk = M[2] * m[0] + M[6] * m[1] + M[10] * m[2];
  const half = 0.5 * Math.max(Math.abs(di), Math.abs(dj), Math.abs(dk));
  const d0 = (M[3] - o[0]) * m[0] + (M[7] - o[1]) * m[1] + (M[11] - o[2]) * m[2];
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
    const base = d0 + j * dj + k * dk;
    for (let i = 0; i < nx; i++) {
      if (Math.abs(base + i * di) > half) continue;
      const l = g.lab[(k * ny + j) * nx + i];
      if (l) out.set(l, (out.get(l) ?? 0) + 1);
    }
  }
  return out;
}
/** Rule 3's end parcel: the most frequent cortical label in the point's 3 x 3 x 3 block; a tie goes to the nearer voxel. */
function majorityCortex(v: number, lab: ArrayLike<number>, cortex: ReadonlySet<number>, nx: number, ny: number, nz: number): number {
  const i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny));
  const count = new Map<number, number>(), nearest = new Map<number, number>();
  for (let dk = -1; dk <= 1; dk++) for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
    const a = i + di, b = j + dj, c = k + dk;
    if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz) continue;
    const l = lab[(c * ny + b) * nx + a];
    if (!cortex.has(l)) continue;
    const d = di * di + dj * dj + dk * dk;
    count.set(l, (count.get(l) ?? 0) + 1);
    nearest.set(l, Math.min(nearest.get(l) ?? 9, d));
  }
  let best = 0, bc = 0, bd = 9;
  for (const [l, c] of count) { const d = nearest.get(l)!; if (c > bc || (c === bc && d < bd)) { best = l; bc = c; bd = d; } }
  return best;
}
/** The first test a streamline fails, or undefined when it passes all. */
export function cstFails(f: Float32Array, A: CstAnatomy, at: (x: number, y: number, z: number) => number, thalCore: Uint8Array, rule: CstRule = CST_RULE_2): Failed | undefined {
  const lab = A.brain.lab, [nx, ny, nz] = A.brain.dims as number[], n = f.length / 3;
  let t = 0, l = 0;
  for (let q = 0; q < f.length; q += 3) { const v = at(f[q], f[q + 1], f[q + 2]); if (v < 0) continue; if (thalCore[v]) t++; if (A.lentiform.has(lab[v])) l++; }
  if (t >= 2) return "thalamus";
  if (l >= rule.lentiformMinPoints) return "lentiform";
  if (A.crus) {
    const { origin: o, normal: m, errant } = A.crus;
    for (let k = 1; k < n; k++) {
      const a = 3 * (k - 1), b = 3 * k;
      const da = (f[a] - o[0]) * m[0] + (f[a + 1] - o[1]) * m[1] + (f[a + 2] - o[2]) * m[2], db = (f[b] - o[0]) * m[0] + (f[b + 1] - o[1]) * m[1] + (f[b + 2] - o[2]) * m[2];
      if (da === db || (da > 0) === (db > 0)) continue;
      const s = da / (da - db), v = at(f[a] + s * (f[b] - f[a]), f[a + 1] + s * (f[b + 1] - f[a + 1]), f[a + 2] + s * (f[b + 2] - f[a + 2]));
      if (v >= 0 && errant.has(lab[v])) return "crus";
    }
  }
  const upFirst = f[2] > f[3 * (n - 1) + 2], near = "endBy" in rule ? (v: number) => majorityCortex(v, lab, A.cortex, nx, ny, nz) : (v: number) => { if (A.cortex.has(lab[v])) return lab[v]; for (const d of [1, -1, nx, -nx, nx * ny, -nx * ny]) { const w = v + d; if (w >= 0 && w < lab.length && A.cortex.has(lab[w])) return lab[w]; } return 0; };
  let walked = 0, hit = 0;
  for (let k = 0; k < n && walked <= rule.endReachMm; k++) {
    const p = 3 * (upFirst ? k : n - 1 - k);
    if (k > 0) { const r = upFirst ? p - 3 : p + 3; walked += Math.hypot(f[p] - f[r], f[p + 1] - f[r + 1], f[p + 2] - f[r + 2]); }
    const v = at(f[p], f[p + 1], f[p + 2]); if (v >= 0 && (hit = near(v))) break;
  }
  return hit && A.allowedEnd.has(hit) ? undefined : "end";
}
export interface GateResult2<S> { sets: S[]; moved: ({ label: string; side: number; of: number; kept: number } & Record<Failed, number>)[];
  /** Rule 3: the crus plane's check -- the share of corticospinal crossings in the ventral diencephalon, and whether the test ran. */
  crus?: { ventralVoxels: number; brainstemVoxels: number; ventralShareOfCrossings: number; tested: boolean } }
/** Rule 2 over a run's tract sets, as gateCorticospinal does for rule 1. */
export function gateCorticospinal2<S extends { label: string; side: number; streamlines: Float32Array[] }>(sets: S[], A: CstAnatomy, unnamed: (moved: Float32Array[]) => S, rule: CstRule = CST_RULE_2): GateResult2<S> {
  const at = voxelIndexer(A.brain), thalCore = coreOf(A.brain, A.thalamus);
  let crusCheck: GateResult2<S>["crus"];
  if ("crusMinVentralVoxels" in rule && A.crus) {
    const ventral = A.crus.ventral ?? new Set<number>();
    const cut = planeCut(A.brain, A.crus.origin, A.crus.normal);
    let vv = 0, bv = 0;
    for (const [l, c] of cut) { if (ventral.has(l)) vv += c; else if (A.crus.errant.has(l)) bv += c; }
    let all = 0, inVentral = 0;
    for (const s of sets) if (isCorticospinal(s.label)) for (const f of s.streamlines) for (const l of crusCrossings(f, A.crus, A.brain.lab, at)) { all++; if (ventral.has(l)) inVentral++; }
    const tested = vv >= rule.crusMinVentralVoxels;
    crusCheck = { ventralVoxels: vv, brainstemVoxels: bv, ventralShareOfCrossings: all ? +(inVentral / all).toFixed(3) : 0, tested };
    if (!tested) A = { ...A, crus: undefined };
  }
  const moved: GateResult2<S>["moved"] = [], out: S[] = [], taken: Float32Array[] = [];
  for (const s of sets) {
    if (!isCorticospinal(s.label)) { out.push(s); continue; }
    const keep: Float32Array[] = [], why: Record<Failed, number> = { thalamus: 0, lentiform: 0, crus: 0, end: 0 };
    for (const f of s.streamlines) { const w = cstFails(f, A, at, thalCore, rule); if (w) { why[w]++; taken.push(f); } else keep.push(f); }
    moved.push({ label: s.label, side: s.side, of: s.streamlines.length, kept: keep.length, ...why });
    if (keep.length) out.push({ ...s, streamlines: keep });
  }
  if (taken.length) {
    const u = out.findIndex((s) => s.label === unnamed([]).label);
    if (u >= 0) out[u] = { ...out[u], streamlines: [...out[u].streamlines, ...taken] }; else out.push(unnamed(taken));
  }
  return { sets: out, moved, ...(crusCheck ? { crus: crusCheck } : {}) };
}
