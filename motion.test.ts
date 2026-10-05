// Head movement (motion.ts): on a synthetic head scanned as a diffusion series while it moves -- every image made exactly
// from a continuous model, with the tissue moved and the gradient turned with it, no interpolation -- the moves are found
// again, the images agree with their predictions better afterwards, and the put-back scan matches the still one.
//   deno test -A --no-check motion.test.ts
import { assert } from "jsr:@std/assert@1";
import type { DiffusionSeries } from "./dwi.ts";
import { applyMotion, estimateMotion, logRot, rotVec, shellsOf, type Move } from "./motion.ts";
import type { Rigid } from "./registration.ts";

const DIMS: [number, number, number] = [48, 52, 38], H = 2.5;
const M = [-H, 0, 0, 58.75, 0, H, 0, -63.75, 0, 0, H, -46.25, 0, 0, 0, 1];   // a scan in LAS-like orientation; the head inside it

/** The head at a point (RAS mm): its b = 0 brightness and its diffusion tensor (mm²/s), smooth and not symmetric. */
function tissue(x: number, y: number, z: number): { s0: number; D: number[] } {
  const e = (x / 42) ** 2 + (y / 50) ** 2 + (z / 34) ** 2;
  if (e > 1.3) return { s0: 0, D: [0, 0, 0, 0, 0, 0] };
  const edge = 1 / (1 + Math.exp((e - 1) * 25));
  let s0 = 600 * edge + 500 * edge * Math.exp(-((x - 12) ** 2 + (y + 8) ** 2 + z * z) / 200) + 350 * edge * Math.exp(-((x + 15) ** 2 + (y - 18) ** 2 + (z - 8) ** 2) / 150);
  // Folds: a texture of ridges, as a cortex gives a real scan its edges (without it a smooth head barely shows a turn).
  s0 *= 1 + 0.25 * Math.sin(x / 5 + 0.5 * Math.sin(z / 8)) * Math.sin(y / 6 - z / 9);
  // Fibers: around an axis off the middle and tilted, along it further out; a "ventricle" of free water.
  const X = x - 8, Y = y + 6 - 0.3 * z, r = Math.hypot(X, Y) || 1, w = Math.exp(-((r - 18) ** 2) / 120);
  const d = [(-Y / r) * w + 0.3 * (1 - w), (X / r) * w, 1 - w], dl = Math.hypot(d[0], d[1], d[2]);
  const u = d.map((v) => v / dl), l1 = 1.7e-3, l2 = 0.3e-3, fw = Math.exp(-((x - 4) ** 2 + (y - 4) ** 2 + (z + 6) ** 2) / 80);
  const D = [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => (1 - fw) * ((l1 - l2) * u[i] * u[j] + (i === j ? l2 : 0)) + fw * (i === j ? 3e-3 : 0)));
  s0 += 400 * fw;
  return { s0, D: [D[0], D[4], D[8], D[1], D[2], D[5]] };
}

/** A move about a center: reference point → where it was in that image. */
const move = (w: number[], t: number[], c: [number, number, number]): Rigid => ({ R: rotVec(w.map((a) => a * Math.PI / 180)), t: t as [number, number, number], c });

/** Eddy currents in the phantom: per image a shift along e (RAS unit vector) of d0 + g·p/100 mm at scanner point p. */
interface Eddy { e: [number, number, number]; g: number[][]; d0: number[] }

function scan(moves: Rigid[], bValues: number[], gradients: [number, number, number][], noise = 0, eddy?: Eddy): DiffusionSeries {
  const [nx, ny, nz] = DIMS;
  let seed = 7;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const volumes = moves.map((m, v) => {
    const data = new Float32Array(nx * ny * nz), R = m.R, g0 = gradients[v];
    // The gradient as the tissue felt it: Rᵀ g. The tissue at scanner point p came from the reference point R⁻¹(p − c − t) + c.
    const g = [R[0] * g0[0] + R[3] * g0[1] + R[6] * g0[2], R[1] * g0[0] + R[4] * g0[1] + R[7] * g0[2], R[2] * g0[0] + R[5] * g0[1] + R[8] * g0[2]];
    const eg = eddy?.g[v] ?? [0, 0, 0], ee = eddy?.e ?? [0, 0, 0], ed0 = eddy?.d0[v] ?? 0, ge = (eg[0] * ee[0] + eg[1] * ee[1] + eg[2] * ee[2]) / 100;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const pa = [0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]);
      // The signal from scanner point p appears at p + (d0 + g·p/100) e: solved for p, and its brightness spread by the stretch.
      const dd = (ed0 + (eg[0] * pa[0] + eg[1] * pa[1] + eg[2] * pa[2]) / 100) / (1 + ge), p = [pa[0] - dd * ee[0], pa[1] - dd * ee[1], pa[2] - dd * ee[2]];
      const d = [p[0] - m.c[0] - m.t[0], p[1] - m.c[1] - m.t[1], p[2] - m.c[2] - m.t[2]];
      const x = [R[0] * d[0] + R[3] * d[1] + R[6] * d[2] + m.c[0], R[1] * d[0] + R[4] * d[1] + R[7] * d[2] + m.c[1], R[2] * d[0] + R[5] * d[1] + R[8] * d[2] + m.c[2]];
      const { s0, D } = tissue(x[0], x[1], x[2]);
      const q = D[0] * g[0] * g[0] + D[1] * g[1] * g[1] + D[2] * g[2] * g[2] + 2 * (D[3] * g[0] * g[1] + D[4] * g[0] * g[2] + D[5] * g[1] * g[2]);
      data[(k * ny + j) * nx + i] = Math.max(0, s0 * Math.exp(-bValues[v] * q) / (1 + ge) + noise * gauss());
    }
    return { dims: DIMS, ijkToRAS: M, data, dtype: "<f4" };
  });
  return { volumes, bValues, gradients, ijkToRAS: M, source: "synthetic", convention: 1 } as unknown as DiffusionSeries;
}

/** ds001226's layout: 102 images, b = 0 at 0, 1, 26, 51, 76 and 101, three shells interleaved (48 at b 2800, 32 at
 *  1200, 16 at 700), each shell's directions spread evenly (golden spiral) over the whole sphere. */
function protocol() {
  const bValues: number[] = [], gradients: [number, number, number][] = [], B0 = new Set([0, 1, 26, 51, 76, 101]);
  const cycle = [2800, 1200, 2800, 700, 2800, 1200], count = new Map([[2800, 48], [1200, 32], [700, 16]]), used = new Map<number, number>();
  let k = 0;
  for (let v = 0; v < 102; v++) {
    if (B0.has(v)) { bValues.push(0); gradients.push([0, 0, 0]); continue; }
    const b = cycle[k++ % cycle.length], n = count.get(b)!, i = (used.get(b) ?? 0) + 0.5; used.set(b, i + 0.5);
    // Every other direction turned to its opposite: the same diffusion measurement, but the directions then cover the whole
    // sphere, as ds001226's do -- which is what lets eddy currents (opposite for opposite directions) be told apart from
    // the signal (the same for both). On a half sphere they cannot be.
    const z = 1 - i / n, r = Math.sqrt(1 - z * z), ph = Math.PI * (3 - Math.sqrt(5)) * i + b / 1000, sg = Math.floor(i) % 2 ? -1 : 1;
    bValues.push(b); gradients.push([sg * r * Math.cos(ph), sg * r * Math.sin(ph), sg * z]);
  }
  return { bValues, gradients };
}

Deno.test("shells are grouped by b, b = 0 and directionless images left out", () => {
  const s = shellsOf({ bValues: [0, 1000, 2500, 1005, 0, 2480, 1000], gradients: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0]] });
  assert(JSON.stringify(s) === JSON.stringify([[1, 3], [2, 5]]), JSON.stringify(s));
});

Deno.test("a moving head: each image's move is found again (root mean square within 0.25 mm and 0.3°)", async () => {
  const { bValues, gradients } = protocol(), c: [number, number, number] = [0, 0, 0];
  // A slow drift (up to 2 mm and 2° by the end) and a jitter of a few tenths per image.
  let seed = 3; const rnd = () => { seed = (seed * 69069 + 1) % 4294967296; return seed / 4294967296 - 0.5; };
  const truth = bValues.map((_, v) => { const u = v / 101; return move([1.5 * u + 0.4 * rnd(), -1.0 * u + 0.4 * rnd(), 2.0 * u * u + 0.4 * rnd()], [1.2 * u + 0.4 * rnd(), 0.8 * u * u + 0.4 * rnd(), -2.0 * u + 0.4 * rnd()], c); });
  const dwi = scan(truth, bValues, gradients, 4);
  const t0 = performance.now();
  const est = await estimateMotion(dwi);
  const ms = performance.now() - t0;
  // The reference is the b = 0 images' mean position, not the scanner's: compare after taking out the common offset.
  const vec = (m: Rigid) => { const w = logRot(m.R).map((a) => a * 180 / Math.PI); const d = [m.c[0] - c[0], m.c[1] - c[1], m.c[2] - c[2]]; const Rd = [0, 1, 2].map((r) => m.R[3 * r] * d[0] + m.R[3 * r + 1] * d[1] + m.R[3 * r + 2] * d[2]); return [...w, ...[0, 1, 2].map((r) => m.t[r] + d[r] - Rd[r])]; };
  const diff = truth.map((m, v) => { const a = vec(est.moves[v]), b = vec(m); return a.map((x, q) => x - b[q]); });
  const b0 = bValues.map((b, i) => (b === 0 ? i : -1)).filter((i) => i >= 0), off = [0, 1, 2, 3, 4, 5].map((q) => b0.reduce((s, i) => s + diff[i][q], 0) / b0.length);
  let worstDeg = 0, worstMm = 0, sDeg = 0, sMm = 0;
  for (const d of diff) {
    const deg = Math.hypot(d[0] - off[0], d[1] - off[1], d[2] - off[2]), mm = Math.hypot(d[3] - off[3], d[4] - off[4], d[5] - off[5]);
    worstDeg = Math.max(worstDeg, deg); worstMm = Math.max(worstMm, mm); sDeg += deg * deg; sMm += mm * mm;
  }
  const rmsDeg = Math.sqrt(sDeg / diff.length), rmsMm = Math.sqrt(sMm / diff.length);
  if (Deno.env.get("MOTION_DEBUG")) diff.forEach((d, v) => console.log(v, bValues[v], d.map((x, q) => (x - off[q]).toFixed(2)).join(" "), vec(truth[v]).map((x) => x.toFixed(2)).join(" ")));
  console.log(`  ${bValues.length} images, ${DIMS.join("×")}: root mean square ${rmsMm.toFixed(3)} mm, ${rmsDeg.toFixed(3)}°, worst ${worstMm.toFixed(3)} mm, ${worstDeg.toFixed(3)}°; residual ${est.residual.before.toFixed(4)} → ${est.residual.after.toFixed(4)}; ${(ms / 1000).toFixed(1)} s`);
  // What the method achieves (2026-10-05): about 0.17 mm and 0.2°, most of it at b 2800 and in movement patterns that
  // follow the gradient directions, which no prediction from the other images can see (FSL's eddy shares the limit).
  assert(rmsMm < 0.25, `translation off by ${rmsMm.toFixed(3)} mm (root mean square)`);
  assert(rmsDeg < 0.3, `rotation off by ${rmsDeg.toFixed(3)}° (root mean square)`);
  assert(worstMm < 0.6 && worstDeg < 1.2, `worst image off by ${worstMm.toFixed(3)} mm, ${worstDeg.toFixed(3)}°`);
  assert(est.residual.after < 0.7 * est.residual.before, "the images do not agree better with their predictions");

  // Put back, the first and the last b = 0 image (2 mm and 2° apart as taken) are the same picture again.
  const fixed = await applyMotion(dwi, est);
  const a0 = fixed.volumes[0].data as Float32Array, a1 = fixed.volumes[101].data as Float32Array, r0 = dwi.volumes[0].data as Float32Array, r1 = dwi.volumes[101].data as Float32Array;
  let e = 0, eRaw = 0, n = 0;
  for (let i = 0; i < a0.length; i++) if (r0[i] > 300) { e += (a1[i] - a0[i]) ** 2; eRaw += (r1[i] - r0[i]) ** 2; n++; }
  console.log(`  put back: first and last b = 0 differ by ${Math.sqrt(e / n).toFixed(1)} (as taken ${Math.sqrt(eRaw / n).toFixed(1)}; the noise alone ${(4 * Math.SQRT2).toFixed(1)})`);
  assert(e < 0.2 * eRaw, "putting the images back did not bring the first and last b = 0 together");
});

Deno.test("eddy currents: with rule 2 every brain point is read where it was, far closer than with movement alone", async () => {
  const { bValues, gradients } = protocol(), c: [number, number, number] = [0, 0, 0];
  let seed = 11; const rnd = () => { seed = (seed * 69069 + 1) % 4294967296; return seed / 4294967296 - 0.5; };
  const truth = bValues.map((_, v) => { const u = v / 101; return move([1.0 * u + 0.2 * rnd(), -0.6 * u + 0.2 * rnd(), 0.8 * u + 0.2 * rnd()], [0.8 * u + 0.2 * rnd(), 0.5 * u + 0.2 * rnd(), -1.0 * u + 0.2 * rnd()], c); });
  // Eddy currents grow with b and follow the gradient's direction (a fixed 3×3 coupling, mm of shift per 100 mm), along the
  // phase-encoding axis j (+A here), with a constant part too; none at b = 0.
  const K = [[1.2, 0.3, 0], [0.2, 1.5, 0.1], [0, 0.4, 0.8]], e: [number, number, number] = [0, 1, 0];
  const eddy: Eddy = { e, g: gradients.map((d, v) => K.map((row) => (bValues[v] / 2800) * (row[0] * d[0] + row[1] * d[1] + row[2] * d[2]))), d0: gradients.map((d, v) => (bValues[v] / 2800) * 0.8 * d[1]) };
  const dwi = scan(truth, bValues, gradients, 4, eddy);
  // Where each brain point of the reference was read from, by the truth and by an estimate (gauge-free: the common offset
  // of the b = 0 images taken out point by point).
  const pts: number[][] = [];
  for (let z = -24; z <= 24; z += 8) for (let y = -40; y <= 40; y += 8) for (let x = -32; x <= 32; x += 8) if ((x / 42) ** 2 + (y / 50) ** 2 + (z / 34) ** 2 < 0.8) pts.push([x, y, z]);
  const posTrue = (v: number, x: number[]) => { const m = truth[v], p = [0, 1, 2].map((r) => m.R[3 * r] * x[0] + m.R[3 * r + 1] * x[1] + m.R[3 * r + 2] * x[2] + m.t[r]); const d = eddy.d0[v] + (eddy.g[v][0] * p[0] + eddy.g[v][1] * p[1] + eddy.g[v][2] * p[2]) / 100; return p.map((a, r) => a + d * e[r]); };
  const posEst = (m: Move, x: number[]) => { const r = [0, 1, 2].map((q) => m.R[3 * q] * (x[0] - m.c[0]) + m.R[3 * q + 1] * (x[1] - m.c[1]) + m.R[3 * q + 2] * (x[2] - m.c[2]) + m.t[q]); const g = m.ec?.g ?? [0, 0, 0], ee = m.ec?.e ?? [0, 0, 0], d = (g[0] * r[0] + g[1] * r[1] + g[2] * r[2]) / 100; return r.map((a, q) => a + m.c[q] + d * ee[q]); };
  const b0 = bValues.map((b, i) => (b === 0 ? i : -1)).filter((i) => i >= 0), dw = bValues.map((b, i) => (b > 0 ? i : -1)).filter((i) => i >= 0);
  const errorOf = (moves: Move[]) => {
    const off = pts.map((x) => { const o = [0, 0, 0]; for (const i of b0) { const a = posEst(moves[i], x), t = posTrue(i, x); for (let q = 0; q < 3; q++) o[q] += (a[q] - t[q]) / b0.length; } return o; });
    let s = 0, n = 0;
    for (const i of dw) pts.forEach((x, k) => { const a = posEst(moves[i], x), t = posTrue(i, x); s += (a[0] - t[0] - off[k][0]) ** 2 + (a[1] - t[1] - off[k][1]) ** 2 + (a[2] - t[2] - off[k][2]) ** 2; n++; });
    return Math.sqrt(s / n);
  };
  const r1 = await estimateMotion(dwi, undefined, { rule: 1 });
  const t0 = performance.now();
  const r2 = await estimateMotion(dwi, undefined, { rule: 2, peAxis: 1 });
  const ms = performance.now() - t0;
  const e1 = errorOf(r1.moves), e2 = errorOf(r2.moves);
  // The stretch part only (the constant part is a movement along e and is reported as one).
  let trueMax = 0; for (const v of dw) for (const x of pts) { const g = eddy.g[v]; trueMax = Math.max(trueMax, Math.abs((g[0] * x[0] + g[1] * x[1] + g[2] * x[2]) / 100)); }
  console.log(`  where brain points are read from, root mean square error: movement only ${e1.toFixed(3)} mm, with eddy currents ${e2.toFixed(3)} mm; eddy shift up to ${r2.eddyMm} mm found (${trueMax.toFixed(2)} put in); residual ${r2.residual.before.toFixed(4)} → ${r2.residual.after.toFixed(4)} (movement only ${r1.residual.after.toFixed(4)}); ${(ms / 1000).toFixed(1)} s`);
  assert(r2.rule === 2 && r2.moves.every((m, i) => !!m.ec === (bValues[i] > 0)), "eddy currents estimated for exactly the diffusion-weighted images");
  assert(e2 < 0.25, `with eddy currents, points read ${e2.toFixed(3)} mm from where they were`);
  assert(e2 < 0.6 * e1, `the eddy-current model did not help: ${e2.toFixed(3)} against ${e1.toFixed(3)} mm`);
  assert(r2.residual.after < r1.residual.after, "the images do not agree better with the eddy-current model");
});
