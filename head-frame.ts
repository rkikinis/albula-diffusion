// THE HEAD'S OWN FRAME (Ron, 2026-10-06: "You must use anatomic orientation, not scanner orientation. Left right and up
// down need to be anatomically correct"; "We are in the midbrain for the crus and in the diencephalon for the internal
// capsule. Those are the areas that we need aligned, not the neocortex"; on the convention: "Talairach is better known
// ... I trust the brainstem more"). Survey: Contents/docs/head-frame-alignment-survey-2026-10-06.md (Albula workspace).
//
// The MNI ICBM 2009c asymmetric template (Fonov et al. 2009, 2011; Collins, MNI/McGill, permissive notice) is aligned
// RIGIDLY to the patient's T1 in two stages -- the template's brain to get close, then only a fixed box around the
// diencephalon and the midbrain for the answer (a cortical tumor then barely counts; Schönecker et al. 2009 report
// deep-focused registration at 1.29 mm landmark error) -- with Albula's own registration (registration.ts). Two frames
// follow from the template's landmarks (the AFIDs consensus placements on this template; Lau et al. 2019, Taha et al.
// 2023): the TALAIRACH frame (its axial plane through the AC's upper edge and the PC's lower edge) for the internal
// capsule, and the BRAINSTEM frame (its axial plane parallel to the pontomesencephalic junction plane, from the
// infracollicular sulcus to the front of the junction, as Coulombe et al. 2021 cut the brainstem) for the crus.

import { inv4, rigidToT1, type Grid3, type Rigid } from "./registration.ts";

export const HEAD_FRAME_RULE = 1;

/** Template landmarks (MNI152NLin2009cAsym, mm, RAS): the AFIDs consensus placements (several raters). */
export const AFIDS = {
  AC: [-0.2049, 2.7229, -4.8811], PC: [-0.0073, -25.0451, -2.2107],
  infracollicularSulcus: [0.0260, -37.6919, -11.0111], PMJ: [-0.0892, -23.1886, -21.5288],
  superiorInterpeduncularFossa: [-0.0565, -13.7677, -10.9500], intermammillarySulcus: [-0.0743, -8.3603, -15.9355],
  rightSuperiorLMS: [13.3113, -25.9511, -9.7489], leftSuperiorLMS: [-13.6075, -26.4451, -9.7076],
} as const;

/** The deep box (template mm): the diencephalon and the midbrain with a margin -- thalami, internal capsules, the
 *  commissures, the third ventricle, the cerebral peduncles down to the pontomesencephalic junction. */
export const DEEP_BOX = { lo: [-32, -45, -30], hi: [32, 18, 22] } as const;

/** The Talairach line on the template, measured once (2026-10-06, half-way intensity crossings on the template's T1 at
 *  x = 0): the AC's upper edge at z -2.88 mm (2.0 mm above its center), the PC's lower edge at z -3.71 mm (1.5 mm below
 *  its center); the line from the PC's edge to the AC's rises 1.71 degrees toward the front (the centers' line,
 *  Schaltenbrand's: -5.5 degrees -- 7.2 degrees apart, as the literature's ~8). */
export const TALAIRACH_EDGES = { acUpper: [-0.2049, 2.7229, -2.88], pcLower: [-0.0073, -25.0451, -3.71] } as const;
export const TALAIRACH_RISE_DEG = 1.71;

/** The starting pitches tried (degrees; positive: the front up). */
export const PITCH_STARTS = [0, 15, 30, -15];

export type M4 = number[];
export const I4: M4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
export function mul4(A: M4, B: M4): M4 {
  const C = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) for (let k = 0; k < 4; k++) C[4 * r + c] += A[4 * r + k] * B[4 * k + c];
  return C;
}
export const apply4 = (M: M4, p: ArrayLike<number>): [number, number, number] =>
  [0, 1, 2].map((r) => M[4 * r] * p[0] + M[4 * r + 1] * p[1] + M[4 * r + 2] * p[2] + M[4 * r + 3]) as [number, number, number];
/** A rigid move y = R (x − c) + c + t as a matrix. */
export function rigidM4(T: Rigid): M4 {
  const R = T.R, o = [0, 1, 2].map((r) => T.c[r] + T.t[r] - (R[3 * r] * T.c[0] + R[3 * r + 1] * T.c[1] + R[3 * r + 2] * T.c[2]));
  return [R[0], R[1], R[2], o[0], R[3], R[4], R[5], o[1], R[6], R[7], R[8], o[2], 0, 0, 0, 1];
}
const translate4 = (d: ArrayLike<number>): M4 => [1, 0, 0, d[0], 0, 1, 0, d[1], 0, 0, 1, d[2], 0, 0, 0, 1];
/** A rotation about the template's left-right (x) axis, by `deg` degrees (positive: the front goes up). */
export function pitch4(deg: number): M4 {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1];
}

/** The angle (degrees) of the line from `back` to `front` (template y-z plane) above the horizontal. */
export function riseDeg(back: ArrayLike<number>, front: ArrayLike<number>): number {
  return (Math.atan2(front[2] - back[2], front[1] - back[1]) * 180) / Math.PI;
}

/** The center of a volume's bright part (intensity above 25% of its 99th percentile), weighted by intensity (mm). */
export function brightCenter(g: Grid3, mask?: Uint8Array): [number, number, number] {
  const n = g.data.length, sample: number[] = [];
  for (let v = 0; v < n; v += 13) if (!mask || mask[v]) sample.push(Number(g.data[v]));
  sample.sort((a, b) => a - b);
  const thr = 0.25 * (sample[Math.floor(0.99 * (sample.length - 1))] ?? 0), [nx, ny] = g.dims, M = g.ijkToRAS;
  let w = 0; const c = [0, 0, 0];
  for (let v = 0; v < n; v++) {
    if (mask && !mask[v]) continue;
    const x = Number(g.data[v]); if (!(x > thr)) continue;
    const i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny));
    for (let r = 0; r < 3; r++) c[r] += x * (M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]);
    w += x;
  }
  return c.map((v) => v / Math.max(w, 1e-9)) as [number, number, number];
}

/** The deep box as a mask on the template's grid, inside the template's brain. */
export function deepMask(template: Grid3, brain: Uint8Array, box: { lo: readonly number[]; hi: readonly number[] } = DEEP_BOX): Uint8Array {
  const [nx, ny] = template.dims, M = template.ijkToRAS, out = new Uint8Array(brain.length);
  for (let v = 0; v < brain.length; v++) {
    if (!brain[v]) continue;
    const i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny));
    const p = [0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]);
    if (p.every((x, r) => x >= box.lo[r] && x <= box.hi[r])) out[v] = 1;
  }
  return out;
}

export interface HeadFrameFit {
  /** Template mm -> the patient's RAS mm. */
  templateToPatient: M4;
  stages: { name: string; cost: number; costAtStart: number; ms: number }[];
}

/**
 * ALIGN THE TEMPLATE TO THE PATIENT'S T1 (rigid). The template is first put at the T1's bright center (the scanner's
 * origin need not be near the head's), then aligned on its whole brain, then on the deep box alone. `exclude` (on the
 * template's grid) leaves voxels out of both -- a tumor carried into template space, when known.
 */
export async function alignTemplate(t1: Grid3, template: Grid3, templateBrain: Uint8Array, exclude?: Uint8Array): Promise<HeadFrameFit> {
  const keep = (m: Uint8Array) => exclude ? m.map((x, v) => (x && !exclude[v] ? 1 : 0)) : m;
  const tc = brightCenter(template, templateBrain), pc = brightCenter(t1);
  const moved = (M: M4): Grid3 => ({ ...template, ijkToRAS: mul4(M, template.ijkToRAS) });
  // SEVERAL STARTING PITCHES (CON07, 2026-10-06: from the scanner's placement alone the optimizer settled 26 degrees off;
  // heads lie pitched 4-33 degrees in the scanner on these cases): the template, centered on the T1's bright center, is
  // tipped by each start about its own center; the coarse whole-brain stage runs from each, and the best cost wins.
  let best: { cost: number; M: M4; s: Awaited<ReturnType<typeof rigidToT1>> } | undefined;
  for (const tilt of PITCH_STARTS) {
    const start = mul4(translate4(pc), mul4(pitch4(tilt), translate4(tc.map((x) => -x))));
    const s = await rigidToT1(moved(start), t1, keep(templateBrain), { fwhmMm: [12, 8, 4], fixedMm: 2 });
    if (!best || s.cost < best.cost) best = { cost: s.cost, M: mul4(rigidM4(s.T), start), s };
  }
  const s1 = best!.s, after1 = best!.M;
  const s2 = await rigidToT1(moved(after1), t1, keep(deepMask(template, templateBrain)), { fwhmMm: [6, 4, 3], fixedMm: 1.5, samples: 60000 });
  return {
    templateToPatient: mul4(rigidM4(s2.T), after1),
    stages: [{ name: "whole brain", cost: s1.cost, costAtStart: s1.costAtStart, ms: s1.ms }, { name: "diencephalon and midbrain", cost: s2.cost, costAtStart: s2.costAtStart, ms: s2.ms }],
  };
}

/** A frame: its origin and its three axes (the patient's left-right, back-front, down-up; unit vectors in the patient's
 *  RAS), as one matrix frame -> patient RAS. */
export interface Frame { name: string; toPatient: M4 }

/**
 * THE TWO FRAMES from a fit. `talairachRiseDeg` is the rise of the template's Talairach line from the PC's lower edge to
 * the AC's upper edge (riseDeg; positive: the front higher), measured once on the template; each frame's front-back axis
 * lies along its line (pitch4 of the rise turns the template's front-back axis onto it). The brainstem frame's plane is the one through
 * the infracollicular sulcus and the front of the pontomesencephalic junction (AFIDs). Both frames have their origin at
 * the AC and keep the template's left-right axis.
 */
export function frames(fit: HeadFrameFit, talairachRiseDeg: number): { talairach: Frame; brainstem: Frame } {
  const at = (rise: number): M4 => mul4(fit.templateToPatient, mul4(translate4(AFIDS.AC), pitch4(rise)));
  const bsRise = riseDeg(AFIDS.infracollicularSulcus, AFIDS.PMJ);   // negative: the front of the plane is lower
  return { talairach: { name: "Talairach (AC-PC)", toPatient: at(talairachRiseDeg) }, brainstem: { name: "brainstem (PMJ plane)", toPatient: at(bsRise) } };
}

export { inv4 };
