// THE DIFFUSION DIRECTIONS CHECKED AGAINST THE IMAGES THEMSELVES (Ron, 2026-10-06: "Our pipeline should be robust as much
// as is reasonable" -- a random site, a random scanner, nobody to look -- and, on what to do when the check disagrees with
// the scanner's record: "use the table the data prefer and say so").
//
// THE FAILURE IT CATCHES: the scanner's record of the gradient directions read in the wrong convention -- two axes
// swapped, one flipped, the patient's axes taken for the image's, or the scan's tilt ignored. The tensors then point the
// wrong way and every tract runs wrong, while the FA map looks normal. Our reader is checked against dcm2niix
// (second-opinion.ts), but that catches OUR reading errors, not a record that is wrong in the scanner's own convention.
//
// THE METHOD, after Jeurissen, Leemans & Sijbers, Medical Image Analysis 18:953-962, 2014 (read in full 2026-10-06), from
// its description: with the right directions, fibers run on through the brain; with wrong ones, many stop early. So
// whole-brain tracking from random brain seeds is done for each candidate table, scored by the FA-weighted fiber length
// per seed. Differences from the paper: no Gaussian smoothing of the images first (thinned and noisy versions of PAT16,
// down to a b = 0 signal-to-noise of about 5, still came out right -- critic 2026-10-06); no search over ANY rotation.
//
// RULE 2 (2026-10-06, after the critic: Contents/docs/qa/2026-10-06-dmri-gradient-check.md; rule 1, the same morning,
// never shipped):
//   - THE CANDIDATES: every swap and flip of the axes in BOTH frames a record can be in -- the PATIENT's axes (a DICOM
//     record) and the IMAGE's axes (an FSL record) -- 24 each, the same 24 on an untilted scan; and, for a scan tilted at
//     least GRADIENT_CHECK.tiltMinDeg, the two ways its tilt can be lost. Rule 1 swapped the image's axes only, and on a
//     tilted scan a patient-axis error was then no candidate at all: PAT11 kept a table 89° wrong (critic, finding 1).
//     MRtrix3's dwigradcheck, after the same paper, tests both frames too. Two errors at once (a swap AND a lost tilt)
//     are not candidates.
//   - ONE CONTEST: each candidate's tensors are tracked from the same fresh random seeds, round after round. The winner
//     must win EVERY round AND beat every other candidate by GRADIENT_CHECK.margin in mean score (the record only needs
//     to win every round against a lost tilt: keeping it is the safe side; see checkGradientTable). Rule 1 ran the tilts
//     in a second contest gated on the first, and a large lost tilt then failed the first and was never tried (finding 3).
//   - SMALL TILTS ARE NOT CORRECTED: below tiltMinDeg a lost tilt and the truth score within a few percent of each other,
//     and which wins depends on the seeds and on the step order (PAT05: a false 6° correction with seed 2, and after the
//     movement correction; finding 2). A tilt lost below 10° bends the directions by less than that.
//   - One tensor fit; a table changed by an orthogonal Q gives exactly the tensors Q D Qᵀ (gradient-check.test.ts), so each
//     candidate's tensors are turned, into ONE reused buffer (finding 8: rule 1 held 24 copies, up to +1.8 GB).
import type { DiffusionSeries } from "./dwi.ts";
import { isotropicVolumes } from "./dwi.ts";
import { fitTensors, type TensorFit } from "./tensor.ts";
import { trackFromSeeds } from "./tracking.ts";

/** The check's numbered rule (Ron, 2026-09-25: "as modular as possible and also versioned"). */
export const DIRECTION_CHECK_RULE = 2;

/** rounds, seeds, tries: the contest. maxB: the tensor's shells (the lowest shell when none is at or below it).
 *  tiltMinDeg: a scan tilted less is not offered a lost-tilt correction. margin: a winner's mean score over every other's. */
export const GRADIENT_CHECK = { rounds: 5, seeds: 300, tries: 3, maxB: 1500, tiltMinDeg: 10, margin: 1.2 } as const;

export interface Candidate {
  /** The error this candidate undoes, technically ("as recorded", "patient axes: left-right flipped", "image axes: …",
   *  "the scan's 22° tilt lost one way") -- for the record and Advanced, not the person's line. */
  label: string;
  kind: "record" | "patient" | "image" | "tilt";
  /** Patient RAS, row-major: the directions the images fit are Q · recorded. */
  Q: number[];
  /** The same swap or flip in the other frame (patient ↔ image), on a tilted scan a few degrees away: the winner need
   *  not beat its twin by the margin, only every other candidate. */
  twin?: number;
}

export interface GradientCheck {
  rule: number;
  /** "as recorded": the record wins clearly. "corrected": another table wins clearly (`best`, used). "unconfirmed": no
   *  clear winner and the record is among the top (record used). "uncertain": no clear winner and the record is clearly
   *  beaten (the best used, with a warning). "not checked": the check could not run (record used). */
  verdict: "as recorded" | "corrected" | "unconfirmed" | "uncertain" | "not checked";
  /** The table used: the record unless "corrected" or "uncertain". */
  best: Candidate;
  /** The top candidate's mean score over the next one's (whichever won). */
  topOverNext: number;
  /** The record's mean score over the best other candidate's (above 1: the record leads). */
  recordOverBest: number;
  seeds: number; rounds: number; candidates: number;
  /** Mean score per candidate (mm of FA-weighted fiber per seed), best first. */
  scores: { label: string; score: number }[];
  /** For the person, plain words. */
  said: string;
  /** Why "not checked". */
  why?: string;
  ms: number;
}

const mul = (A: number[], B: number[]) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c]));
const tr = (A: number[]) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
const det = (Q: number[]) => Q[0] * (Q[4] * Q[8] - Q[5] * Q[7]) - Q[1] * (Q[3] * Q[8] - Q[5] * Q[6]) + Q[2] * (Q[3] * Q[7] - Q[4] * Q[6]);
/** Two tables are the same if Q or -Q (a diffusion direction has no sign). */
const sameTable = (A: number[], B: number[]) => A.every((x, i) => Math.abs(x - B[i]) < 1e-6) || A.every((x, i) => Math.abs(x + B[i]) < 1e-6);
const AXIS = ["left-right", "front-back", "up-down"];

/** The 48 signed permutations (row-major): P[r][p[r]] = s[r]. */
function signedPermutations(): number[][] {
  const perms = [[0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 1, 0], [1, 2, 0], [2, 0, 1]], out: number[][] = [];
  for (const p of perms) for (let s = 0; s < 8; s++) { const P = [0, 0, 0, 0, 0, 0, 0, 0, 0]; for (let r = 0; r < 3; r++) P[3 * r + p[r]] = s & (1 << r) ? -1 : 1; out.push(P); }
  return out;
}

/** The image's axes as unit columns in patient RAS (row-major 3×3), and its tilt: the angle between R and the nearest
 *  axis-aligned frame S (the signed permutation with det(S) = det(R) closest to R), with T = R Sᵀ the tilt alone. */
function imageFrame(M: number[]): { R: number[]; S: number[]; T: number[]; tiltDeg: number; names: string[] } {
  const cols = [0, 1, 2].map((c) => { const v = [M[c], M[4 + c], M[8 + c]], l = Math.hypot(v[0], v[1], v[2]); return v.map((x) => x / l); });
  const R = [0, 1, 2].flatMap((r) => [cols[0][r], cols[1][r], cols[2][r]]);
  // Nearest S: the largest trace(Sᵀ R) among the signed permutations with R's handedness (so T is a proper rotation;
  // per-column rounding could make S singular at 45°, critic finding 12).
  let S = signedPermutations()[0], bestTr = -Infinity;
  for (const P of signedPermutations()) { if (Math.sign(det(P)) !== Math.sign(det(R))) continue; const t = P.reduce((s, x, i) => s + x * R[i], 0); if (t > bestTr) { bestTr = t; S = P; } }
  const T = mul(R, tr(S)), tiltDeg = Math.acos(Math.max(-1, Math.min(1, (T[0] + T[4] + T[8] - 1) / 2))) * 180 / Math.PI;
  // Each image axis named by the patient axis S sends it to (one name each, even at 45°).
  const names = [0, 1, 2].map((c) => AXIS[[0, 1, 2].find((r) => S[3 * r + c] !== 0)!]);
  return { R, S, T, tiltDeg, names };
}

function describe(P: number[], names: string[]): string {
  const p = [0, 1, 2].map((r) => [0, 1, 2].find((c) => P[3 * r + c] !== 0)!), s = [0, 1, 2].map((r) => P[3 * r + p[r]]);
  // A flip of all three is no flip (a direction has no sign): name the flips of the sign pattern with fewer minus signs.
  const flips = s.filter((x) => x < 0).length >= 2 ? [0, 1, 2].filter((r) => s[r] > 0) : [0, 1, 2].filter((r) => s[r] < 0);
  const words: string[] = [], moved = [0, 1, 2].filter((r) => p[r] !== r);
  if (moved.length === 2) words.push(`${names[moved[0]]} and ${names[moved[1]]} swapped`);
  else if (moved.length === 3) words.push(`all three axes mixed up (${names[p[0]]} read as ${names[0]}, ${names[p[1]]} as ${names[1]}, ${names[p[2]]} as ${names[2]})`);
  if (flips.length) words.push(`${flips.map((r) => names[r]).join(" and ")} flipped`);
  return words.join(", ");
}

/** The candidates: the record first, the swaps and flips of the patient's axes, those of the image's axes that differ,
 *  and the two lost tilts when the scan is tilted at least tiltMinDeg. */
export function candidates(M: number[]): Candidate[] {
  const { R, S, T, tiltDeg, names } = imageFrame(M), out: Candidate[] = [{ label: "as recorded", kind: "record", Q: [1, 0, 0, 0, 1, 0, 0, 0, 1] }];
  const add = (c: Candidate) => { if (!out.some((o) => sameTable(o.Q, c.Q))) out.push(c); };
  for (const P of signedPermutations()) add({ label: `patient axes: ${describe(P, AXIS)}`, kind: "patient", Q: P });
  for (const P of signedPermutations()) {
    const Q = mul(mul(R, P), tr(R));
    if (out.some((o) => sameTable(o.Q, Q))) continue;
    // Its twin: the same swap or flip of the patient axes the image axes lie nearest to (S P Sᵀ).
    const twin = out.findIndex((o) => o.kind === "patient" && sameTable(o.Q, mul(mul(S, P), tr(S))));
    out.push({ label: `image axes: ${describe(P, names)}`, kind: "image", Q, ...(twin >= 0 ? { twin } : {}) });
    if (twin >= 0) out[twin].twin = out.length - 1;
  }
  if (tiltDeg >= GRADIENT_CHECK.tiltMinDeg) {
    add({ label: `the scan's ${tiltDeg.toFixed(0)}° tilt lost one way`, kind: "tilt", Q: T });
    add({ label: `the scan's ${tiltDeg.toFixed(0)}° tilt lost the other way`, kind: "tilt", Q: tr(T) });
  }
  return out;
}

/** The tensor field turned by Q (D' = Q D Qᵀ per voxel) into `out` -- no allocation per voxel or per candidate. */
function turnInto(out: Float32Array, D: Float32Array, Q: number[]): void {
  const [a, b, c, d, e, f, g, h, k] = Q;
  for (let o = 0; o < D.length; o += 6) {
    const xx = D[o], xy = D[o + 1], xz = D[o + 2], yy = D[o + 3], yz = D[o + 4], zz = D[o + 5];
    // Rows of Q·S (S symmetric), then (Q S) Qᵀ's upper triangle.
    const m0 = a * xx + b * xy + c * xz, m1 = a * xy + b * yy + c * yz, m2 = a * xz + b * yz + c * zz;
    const n0 = d * xx + e * xy + f * xz, n1 = d * xy + e * yy + f * yz, n2 = d * xz + e * yz + f * zz;
    const p0 = g * xx + h * xy + k * xz, p1 = g * xy + h * yy + k * yz, p2 = g * xz + h * yz + k * zz;
    out[o] = m0 * a + m1 * b + m2 * c; out[o + 1] = m0 * d + m1 * e + m2 * f; out[o + 2] = m0 * g + m1 * h + m2 * k;
    out[o + 3] = n0 * d + n1 * e + n2 * f; out[o + 4] = n0 * g + n1 * h + n2 * k; out[o + 5] = p0 * g + p1 * h + p2 * k;
  }
}

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** `count` random points (patient RAS) inside the voxels of `mask`, each somewhere inside its voxel. */
function randomSeeds(fit: TensorFit, mask: Uint8Array, count: number, random: () => number): number[][] {
  const idx: number[] = []; for (let v = 0; v < mask.length; v++) if (mask[v]) idx.push(v);
  const [nx, ny] = fit.dims, M = fit.ijkToRAS, out: number[][] = [];
  for (let s = 0; s < count && idx.length; s++) {
    const v = idx[Math.floor(random() * idx.length)], i = v % nx + random() - 0.5, j = Math.floor(v / nx) % ny + random() - 0.5, k = Math.floor(v / (nx * ny)) + random() - 0.5;
    out.push([0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]));
  }
  return out;
}

/** The score: FA-weighted fiber length per seed (mm), each segment weighted by the FA of the voxel at its middle (FA is
 *  the same for every candidate: turning a tensor keeps its eigenvalues). */
function score(fit: TensorFit, seeds: number[][]): number {
  const sl = trackFromSeeds(fit, seeds), [nx, ny, nz] = fit.dims, M = fit.ijkToRAS;
  const a = M[0], b = M[1], c = M[2], d = M[4], e = M[5], f = M[6], g = M[8], h = M[9], k = M[10];
  const dt = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  const Mi = [(e * k - f * h) / dt, (c * h - b * k) / dt, (b * f - c * e) / dt, (f * g - d * k) / dt, (a * k - c * g) / dt, (c * d - a * f) / dt, (d * h - e * g) / dt, (b * g - a * h) / dt, (a * e - b * d) / dt];
  let total = 0;
  for (const s of sl) {
    const p = s.points;
    for (let q = 3; q < p.length; q += 3) {
      const len = Math.hypot(p[q] - p[q - 3], p[q + 1] - p[q - 2], p[q + 2] - p[q - 1]);
      const x = (p[q] + p[q - 3]) / 2 - M[3], y = (p[q + 1] + p[q - 2]) / 2 - M[7], z = (p[q + 2] + p[q - 1]) / 2 - M[11];
      const i = Math.round(Mi[0] * x + Mi[1] * y + Mi[2] * z), j = Math.round(Mi[3] * x + Mi[4] * y + Mi[5] * z), kk = Math.round(Mi[6] * x + Mi[7] * y + Mi[8] * z);
      if (i < 0 || j < 0 || kk < 0 || i >= nx || j >= ny || kk >= nz) continue;
      total += len * fit.fa[(kk * ny + j) * nx + i];
    }
  }
  return total / seeds.length;
}

/** The tensor's shells: up to GRADIENT_CHECK.maxB, or, when fewer than 6 directions are there, up to the lowest shell
 *  that has them (a scan with b = 2000 only; critic finding 5). */
function maxBFor(dwi: DiffusionSeries): number {
  const iso = new Set(isotropicVolumes(dwi)), bs = dwi.bValues.filter((b, i) => b > 50 && !iso.has(i)).sort((x, y) => x - y);
  return bs.filter((b) => b <= GRADIENT_CHECK.maxB).length >= 6 ? GRADIENT_CHECK.maxB : bs.length >= 6 ? bs[5] + 50 : GRADIENT_CHECK.maxB;
}

const SAID = {
  "as recorded": "the scanner's diffusion directions match the images",
  corrected: "the scanner's record of the diffusion directions did not match the images, so Albula used the directions that do; please tell the MRI team",
  unconfirmed: "Albula could not fully confirm the scanner's diffusion directions against the images and used them as recorded",
  uncertain: "the scanner's record of the diffusion directions does not match the images, and Albula could not be sure which directions do; it used the closest match -- treat these fiber tracts with caution and tell the MRI team",
  "not checked": "the diffusion directions could not be checked against the images and were used as recorded",
} as const;

/**
 * Check `dwi`'s recorded directions against its images (rule 2, header). Deterministic for a given `seed`. Never throws:
 * a scan the check cannot fit comes back "not checked" with the reason.
 */
export async function checkGradientTable(dwi: DiffusionSeries, opts: { seed?: number } = {}): Promise<GradientCheck> {
  const t0 = performance.now(), seed = opts.seed ?? 1;
  const record = candidates(dwi.volumes[0].ijkToRAS)[0];
  let fit: TensorFit;
  try { fit = fitTensors(dwi, { maxB: maxBFor(dwi) }); }
  catch (e) {
    const why = (e as Error).message;
    return { rule: DIRECTION_CHECK_RULE, verdict: "not checked", best: record, topOverNext: NaN, recordOverBest: NaN, seeds: 0, rounds: 0, candidates: 0, scores: [], said: `${SAID["not checked"]} (${why})`, why, ms: performance.now() - t0 };
  }
  const cands = candidates(fit.ijkToRAS), mask = fit.seedMask ?? fit.mask, buf = new Float32Array(fit.D.length), turnedFit = { ...fit, D: buf };
  let seeds: number = GRADIENT_CHECK.seeds, means: number[] = [], winners: number[] = [];
  for (let attempt = 0; attempt < GRADIENT_CHECK.tries; attempt++) {
    const sums = cands.map(() => 0); winners = [];
    for (let r = 0; r < GRADIENT_CHECK.rounds; r++) {
      // The same fresh seeds for every candidate in a round (a paired comparison), new seeds each round.
      const pts = randomSeeds(fit, mask, seeds, rng(seed * 1000 + attempt * 100 + r));
      const sc = cands.map((c) => { turnInto(buf, fit.D, c.Q); return score(turnedFit, pts); });
      sc.forEach((s, i) => (sums[i] += s / GRADIENT_CHECK.rounds));
      winners.push(sc.indexOf(Math.max(...sc)));
      await new Promise((res) => setTimeout(res, 0));
    }
    means = sums;
    const lead = sums.indexOf(Math.max(...sums));
    if (winners.every((w) => w === lead || w === cands[lead].twin)) break;
    if (attempt < GRADIENT_CHECK.tries - 1) seeds *= 2;
  }
  const order = means.map((_, i) => i).sort((x, y) => means[y] - means[x]), top = order[0], twin = cands[top].twin;
  // Clear: the top won every round (its twin may have taken some) and beats every candidate but its twin by the margin --
  // except that the RECORD need not beat a lost tilt by the margin, only win every round. Asymmetric on purpose (the case
  // library, 2026-10-06): on scans tilted 10-16° the lost tilt scores within 10-18% of a right record, so the margin made
  // 19 of 87 right records "unconfirmed"; keeping the record needs no strong evidence, changing it does.
  const consistent = winners.every((w) => w === top || w === twin);
  const clear = consistent && order.slice(1).every((i) => i === twin || (top === 0 && cands[i].kind === "tilt") || means[top] >= GRADIENT_CHECK.margin * means[i]);
  const bestOther = order.find((i) => i !== 0)!, recordOverBest = means[0] / means[bestOther];
  const verdict: GradientCheck["verdict"] = clear ? (top === 0 ? "as recorded" : "corrected") : recordOverBest * GRADIENT_CHECK.margin >= 1 ? "unconfirmed" : "uncertain";
  const best = verdict === "corrected" || verdict === "uncertain" ? cands[top] : cands[0];
  return { rule: DIRECTION_CHECK_RULE, verdict, best, topOverNext: +(means[top] / means[order[1]]).toFixed(3), recordOverBest: +recordOverBest.toFixed(3),
    seeds, rounds: GRADIENT_CHECK.rounds, candidates: cands.length, scores: order.map((i) => ({ label: cands[i].label, score: +means[i].toFixed(2) })),
    said: SAID[verdict], ms: performance.now() - t0 };
}

/** `dwi` with its directions as the check found them (changed only for "corrected" and "uncertain"). */
export function withCheckedDirections(dwi: DiffusionSeries, check: GradientCheck): DiffusionSeries {
  if (check.best.kind === "record") return dwi;
  const Q = check.best.Q;
  return { ...dwi, gradients: dwi.gradients.map((g) => [Q[0] * g[0] + Q[1] * g[1] + Q[2] * g[2], Q[3] * g[0] + Q[4] * g[1] + Q[5] * g[2], Q[6] * g[0] + Q[7] * g[1] + Q[8] * g[2]] as [number, number, number]) };
}
