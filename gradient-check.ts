// THE DIFFUSION DIRECTIONS CHECKED AGAINST THE IMAGES THEMSELVES (Ron, 2026-10-06: "Our pipeline should be robust as much
// as is reasonable" -- a random site, a random scanner, nobody to look -- and, on what to do when the check disagrees with
// the scanner's record: "use the table the data prefer and say so").
//
// THE FAILURE IT CATCHES: the scanner's record of the gradient directions read in the wrong convention -- two axes
// swapped, or one flipped. The tensors then point the wrong way everywhere and every tract runs wrong, while the maps of
// FA look normal. Our reader is checked against dcm2niix (second-opinion.ts), but that catches OUR reading errors, not a
// record that is wrong in the scanner's own convention (both readers read it the same way).
//
// THE METHOD, after Jeurissen, Leemans & Sijbers, Medical Image Analysis 18:953-962, 2014 (read in full 2026-10-06), from
// its description: with the right directions, fibers run on through the brain; with wrong ones, many stop early. So
// whole-brain tracking from random seeds is done for each candidate table, and the candidate with the longest fibers
// (each step's length weighted by the FA there) wins -- but only if it wins EVERY round of fresh random seeds; otherwise
// the seeds are doubled, and after three tries the answer is "undecided" (the record is then kept, and that is said).
//   - The candidates: the 24 ways the image axes can be swapped and flipped (6 orders × 4 sign choices; flipping all three
//     is the same as none, a diffusion direction having no sign), and, for a tilted scan, the two ways the tilt can be
//     lost (the record read as if in the image's axes when it is in the patient's, or the reverse).
//   - One tensor fit: a table changed by an orthogonal Q gives exactly the tensors Q D Qᵀ (the fit treats every direction
//     alike), so each candidate's tensors are turned, not refitted (as the paper does).
//   - Tracking: the plain tensor tracker (tracking.ts), as the paper's DTI tracking; seeds random inside the brain.
// Not here: the paper's search over ANY rotation (27 min in their Matlab); the tilt candidates cover the rotation a
// random site is most likely to meet.
import type { DiffusionSeries } from "./dwi.ts";
import { fitTensors, type TensorFit } from "./tensor.ts";
import { trackFromSeeds } from "./tracking.ts";

export interface Candidate {
  /** Plain words: "as recorded", "left-right flipped", "front-back and up-down swapped", "tilt lost (…)". */
  label: string;
  /** Patient RAS, row-major: the directions the images fit are Q · recorded. */
  Q: number[];
}

export interface GradientCheck {
  /** "as recorded": the record fits the images; "corrected": another table fits clearly better (`best`); "undecided". */
  verdict: "as recorded" | "corrected" | "undecided";
  best: Candidate;
  /** The swaps-and-flips contest: the winner's score over the runner-up's. */
  overNext: number;
  /** The tilt contest, for a tilted scan: whether it was settled, the winner over the runner-up, the seeds used. */
  tilt?: { settled: boolean; overNext: number; overBase: number; seeds: number };
  /** Seeds per round at the decision, rounds, tries. */
  seeds: number; rounds: number;
  /** Mean score per candidate over the last try's rounds (mm of FA-weighted fiber per seed): the swaps and flips, then the tilts. */
  scores: { label: string; score: number }[];
  said: string;
  ms: number;
}

/** rounds, seeds, tries: the contests (see `contest`). maxB: the tensor's shells. tiltDeg: below this a scan is not tilted.
 *  MARGINS: a winner must also beat the runner-up (swaps and flips) or the base (a lost tilt) by this factor in mean
 *  score, besides winning every round -- consistency alone let a 6° tilt beat PAT05's correct record by 2.3% (2026-10-06). */
export const GRADIENT_CHECK = { rounds: 5, seeds: 300, tries: 3, maxB: 1500, tiltDeg: 1, axesMargin: 1.2, tiltMargin: 1.05 } as const;

const mul = (A: number[], B: number[]) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c]));
const tr = (A: number[]) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];

/** The image's axes as unit columns in patient RAS (row-major 3×3), and which anatomical axis each mostly runs along. */
function imageAxes(M: number[]): { R: number[]; names: string[]; tiltDeg: number } {
  const cols = [0, 1, 2].map((c) => { const v = [M[c], M[4 + c], M[8 + c]], l = Math.hypot(v[0], v[1], v[2]); return v.map((x) => x / l); });
  const R = [0, 1, 2].flatMap((r) => [cols[0][r], cols[1][r], cols[2][r]]);
  const names = cols.map((v) => ["left-right", "front-back", "up-down"][[0, 1, 2].reduce((a, b) => (Math.abs(v[b]) > Math.abs(v[a]) ? b : a), 0)]);
  // How far the image's axes are from the patient's (the largest angle between an axis and its nearest patient axis).
  const tiltDeg = Math.max(...cols.map((v) => Math.acos(Math.min(1, Math.max(...v.map(Math.abs)))) * 180 / Math.PI));
  return { R, names, tiltDeg };
}

/** The candidates: 24 swaps and flips along the image's axes, then the two lost tilts when the scan is tilted. */
export function candidates(M: number[]): Candidate[] {
  const { R, names, tiltDeg } = imageAxes(M), out: Candidate[] = [];
  const perms = [[0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 1, 0], [1, 2, 0], [2, 0, 1]];
  for (const p of perms) for (const flip of [-1, 0, 1, 2]) {
    // P sends the recorded component along image axis p[r] to image axis r; one sign flipped (or none).
    const P = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let r = 0; r < 3; r++) P[3 * r + p[r]] = r === flip ? -1 : 1;
    const words: string[] = [];
    const swapped = [0, 1, 2].filter((r) => p[r] !== r);
    if (swapped.length === 2) words.push(`${names[swapped[0]]} and ${names[swapped[1]]} swapped`);
    else if (swapped.length === 3) words.push(`all three axes rotated (${names[p[0]]} → ${names[0]}, ${names[p[1]]} → ${names[1]}, ${names[p[2]]} → ${names[2]})`);
    if (flip >= 0) words.push(`${names[flip]} flipped`);
    out.push({ label: words.length ? words.join(", ") : "as recorded", Q: mul(mul(R, P), tr(R)) });
  }
  if (tiltDeg > GRADIENT_CHECK.tiltDeg) {
    // The tilt alone: R against the axis-aligned frame S nearest to it (R's columns rounded to ±patient axes), T = R Sᵀ, a
    // small proper rotation. A record along the image's axes read as if the scan were not tilted gives S Rᵀ g: undone by T;
    // the reverse by Tᵀ.
    const S = R.map(() => 0);
    for (let c = 0; c < 3; c++) { const col = [R[c], R[3 + c], R[6 + c]], r = [0, 1, 2].reduce((a, b) => (Math.abs(col[b]) > Math.abs(col[a]) ? b : a), 0); S[3 * r + c] = Math.sign(col[r]); }
    const T = mul(R, tr(S));
    out.push({ label: `the scan's tilt (${tiltDeg.toFixed(0)}°) lost one way`, Q: T });
    out.push({ label: `the scan's tilt (${tiltDeg.toFixed(0)}°) lost the other way`, Q: tr(T) });
  }
  return out;
}

/** The tensor field turned by Q: D' = Q D Qᵀ per voxel (FA unchanged, directions turned). */
function turned(fit: TensorFit, Q: number[]): TensorFit {
  const D = new Float32Array(fit.D.length), n = fit.D.length / 6;
  for (let v = 0; v < n; v++) {
    const o = 6 * v, xx = fit.D[o], xy = fit.D[o + 1], xz = fit.D[o + 2], yy = fit.D[o + 3], yz = fit.D[o + 4], zz = fit.D[o + 5];
    const S = [xx, xy, xz, xy, yy, yz, xz, yz, zz], T = mul(mul(Q, S), tr(Q));
    D[o] = T[0]; D[o + 1] = T[1]; D[o + 2] = T[2]; D[o + 3] = T[4]; D[o + 4] = T[5]; D[o + 5] = T[8];
  }
  return { ...fit, D };
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

/** The score: FA-weighted fiber length per seed (mm), each segment weighted by the FA of the voxel at its middle. */
function score(fit: TensorFit, seeds: number[][]): number {
  const sl = trackFromSeeds(fit, seeds), [nx, ny, nz] = fit.dims, M = fit.ijkToRAS;
  const a = M[0], b = M[1], c = M[2], d = M[4], e = M[5], f = M[6], g = M[8], h = M[9], k = M[10];
  const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  const Mi = [(e * k - f * h) / det, (c * h - b * k) / det, (b * f - c * e) / det, (f * g - d * k) / det, (a * k - c * g) / det, (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det];
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

/** Check `dwi`'s recorded directions against its images. Deterministic (seeded); `fit` may be given to save the fit. */
/** One contest: each candidate's tensors tracked from the same fresh random seeds, round after round; a winner only if it
 *  wins every round, the seeds doubled when it does not, "undecided" after the last try. */
async function contest(fit: TensorFit, cands: Candidate[], seed: number): Promise<{ decided: boolean; winner: number; means: number[]; seeds: number }> {
  const fits = cands.map((c) => turned(fit, c.Q)), mask = fit.seedMask ?? fit.mask;
  let seeds: number = GRADIENT_CHECK.seeds, means: number[] = [], winners: number[] = [];
  for (let attempt = 0; attempt < GRADIENT_CHECK.tries; attempt++) {
    const sums = cands.map(() => 0); winners = [];
    for (let r = 0; r < GRADIENT_CHECK.rounds; r++) {
      // The same fresh seeds for every candidate in a round (a paired comparison), new seeds each round.
      const pts = randomSeeds(fit, mask, seeds, rng(seed * 1000 + attempt * 100 + r));
      const sc = fits.map((f) => score(f, pts));
      sc.forEach((s, i) => (sums[i] += s / GRADIENT_CHECK.rounds));
      winners.push(sc.indexOf(Math.max(...sc)));
      await new Promise((res) => setTimeout(res, 0));
    }
    means = sums;
    if (winners.every((w) => w === winners[0])) break;
    if (attempt < GRADIENT_CHECK.tries - 1) seeds *= 2;
  }
  const decided = winners.every((w) => w === winners[0]);
  return { decided, winner: decided ? winners[0] : 0, means, seeds };
}

/**
 * Check `dwi`'s recorded directions against its images, in two contests (2026-10-06, the case library: a lost tilt of
 * 5-8° costs the fibers only 0-5%, so on 4 of 29 correct scans it could not be told from the record, while a swap or flip
 * costs about half): FIRST the 24 swaps and flips -- the dangerous errors, always clear -- THEN, from that winner, whether
 * a lost tilt fits better. An unclear second contest leaves the first one's answer. Deterministic (seeded).
 */
export async function checkGradientTable(dwi: DiffusionSeries, opts: { fit?: TensorFit; seed?: number } = {}): Promise<GradientCheck> {
  const t0 = performance.now(), seed = opts.seed ?? 1;
  const fit = opts.fit ?? fitTensors(dwi, { maxB: GRADIENT_CHECK.maxB });
  const all = candidates(fit.ijkToRAS), axes = all.slice(0, 24), tilts = all.slice(24);
  const a = await contest(fit, axes, seed);
  const order = a.means.map((_, i) => i).sort((x, y) => a.means[y] - a.means[x]), next = order.find((i) => i !== a.winner)!;
  const scores = order.map((i) => ({ label: axes[i].label, score: +a.means[i].toFixed(2) }));
  const axesDecided = a.decided && a.means[a.winner] >= GRADIENT_CHECK.axesMargin * a.means[next];
  let best = axes[axesDecided ? a.winner : 0], tilt: GradientCheck["tilt"];
  if (axesDecided && tilts.length) {
    const base = axes[a.winner], join = (l: string) => (base.label === "as recorded" ? l : `${base.label}; ${l}`);
    const tc: Candidate[] = [base, ...tilts.map((c) => ({ label: join(c.label), Q: mul(c.Q, base.Q) }))];
    const b = await contest(fit, tc, seed + 7);
    const bo = b.means.map((_, i) => i).sort((x, y) => b.means[y] - b.means[x]);
    const overBase = b.means[b.winner] / b.means[0], settled = b.decided && (b.winner === 0 || overBase >= GRADIENT_CHECK.tiltMargin);
    tilt = { settled, overNext: +(b.means[bo[0]] / b.means[bo[1]]).toFixed(3), overBase: +overBase.toFixed(3), seeds: b.seeds };
    if (settled) best = tc[b.winner];
    for (const i of bo) if (i > 0) scores.push({ label: tc[i].label, score: +b.means[i].toFixed(2) });
  }
  const verdict = !axesDecided ? "undecided" : best.label === "as recorded" ? "as recorded" : "corrected";
  const said = verdict === "as recorded" ? "the scanner's diffusion directions fit the images"
    : verdict === "corrected" ? `the scanner's record of the diffusion directions did not fit the images (${best.label}); the directions the images fit were used`
    : "the diffusion directions could not be checked against the images (no clear answer); the scanner's record was used";
  return { verdict, best, overNext: +(a.means[a.winner] / a.means[next]).toFixed(3), seeds: a.seeds, rounds: GRADIENT_CHECK.rounds, ...(tilt ? { tilt } : {}),
    scores, said, ms: performance.now() - t0 };
}

/** `dwi` with its directions as the check found them (unchanged unless the verdict is "corrected"). */
export function withCheckedDirections(dwi: DiffusionSeries, check: GradientCheck): DiffusionSeries {
  if (check.verdict !== "corrected") return dwi;
  const Q = check.best.Q;
  return { ...dwi, gradients: dwi.gradients.map((g) => [Q[0] * g[0] + Q[1] * g[1] + Q[2] * g[2], Q[3] * g[0] + Q[4] * g[1] + Q[5] * g[2], Q[6] * g[0] + Q[7] * g[1] + Q[8] * g[2]] as [number, number, number]) };
}
