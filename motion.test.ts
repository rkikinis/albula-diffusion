// Head movement (motion.ts): on a synthetic head scanned as a diffusion series while it moves -- every image made exactly
// from a continuous model, with the tissue moved and the gradient turned with it, no interpolation -- the moves are found
// again, the images agree with their predictions better afterwards, and the put-back scan matches the still one.
//   deno test -A --no-check motion.test.ts
import { assert } from "jsr:@std/assert@1";
import { applyMotion, estimateMotion, logRot, shellsOf, type Move } from "./motion.ts";
import { DIMS, M, move, protocol, scan, type Eddy } from "./test/phantom.ts";
import { resampleOntoT1, type Rigid } from "./registration.ts";
import { applyField, fieldFromCenters } from "./distortion.ts";

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

Deno.test("eddy currents: with rules 2 and 3 every brain point is read where it was, far closer than with movement alone; rule 3's slopes nearer the truth", async () => {
  const { bValues, gradients } = protocol(), c: [number, number, number] = [0, 0, 0];
  let seed = 11; const rnd = () => { seed = (seed * 69069 + 1) % 4294967296; return seed / 4294967296 - 0.5; };
  const truth = bValues.map((_, v) => { const u = v / 101; return move([1.0 * u + 0.2 * rnd(), -0.6 * u + 0.2 * rnd(), 0.8 * u + 0.2 * rnd()], [0.8 * u + 0.2 * rnd(), 0.5 * u + 0.2 * rnd(), -1.0 * u + 0.2 * rnd()], c); });
  // Eddy currents follow the gradient pulse: its direction through a fixed 3×3 coupling (mm of shift per 100 mm), its
  // strength as √b (fixed timing); along the phase-encoding axis j (+A here), with a constant part too; none at b = 0.
  const K = [[1.2, 0.3, 0], [0.2, 1.5, 0.1], [0, 0.4, 0.8]], e: [number, number, number] = [0, 1, 0];
  const eddy: Eddy = { e, g: gradients.map((d, v) => K.map((row) => Math.sqrt(bValues[v] / 2800) * (row[0] * d[0] + row[1] * d[1] + row[2] * d[2]))), d0: gradients.map((d, v) => Math.sqrt(bValues[v] / 2800) * 0.8 * d[1]) };
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
  const r3 = await estimateMotion(dwi, undefined, { rule: 3, peAxis: 1 });
  const e1 = errorOf(r1.moves), e2 = errorOf(r2.moves), e3 = errorOf(r3.moves);
  // The slopes themselves against the truth (root mean square over the diffusion-weighted images, mm per 100 mm).
  const slopeError = (moves: Move[]) => Math.sqrt(dw.reduce((s, v) => { const g = moves[v].ec?.g ?? [0, 0, 0]; return s + (g[0] - eddy.g[v][0]) ** 2 + (g[1] - eddy.g[v][1]) ** 2 + (g[2] - eddy.g[v][2]) ** 2; }, 0) / dw.length);
  const s2 = slopeError(r2.moves), s3 = slopeError(r3.moves);
  // The stretch part only (the constant part is a movement along e and is reported as one).
  let trueMax = 0; for (const v of dw) for (const x of pts) { const g = eddy.g[v]; trueMax = Math.max(trueMax, Math.abs((g[0] * x[0] + g[1] * x[1] + g[2] * x[2]) / 100)); }
  console.log(`  where brain points are read from, root mean square error: movement only ${e1.toFixed(3)} mm, with eddy currents ${e2.toFixed(3)} mm; eddy shift up to ${r2.eddyMm} mm found (${trueMax.toFixed(2)} put in); residual ${r2.residual.before.toFixed(4)} → ${r2.residual.after.toFixed(4)} (movement only ${r1.residual.after.toFixed(4)}); ${(ms / 1000).toFixed(1)} s`);
  assert(r2.rule === 2 && r2.moves.every((m, i) => !!m.ec === (bValues[i] > 0)), "eddy currents estimated for exactly the diffusion-weighted images");
  assert(e2 < 0.25, `with eddy currents, points read ${e2.toFixed(3)} mm from where they were`);
  assert(e2 < 0.6 * e1, `the eddy-current model did not help: ${e2.toFixed(3)} against ${e1.toFixed(3)} mm`);
  assert(r2.residual.after < r1.residual.after, "the images do not agree better with the eddy-current model");
  // Rule 3 (the slopes tied to the gradient): the slopes nearer the truth than rule 2's, the points read no worse.
  console.log(`  rule 3: points ${e3.toFixed(3)} mm; slopes off by ${s3.toFixed(3)} (rule 2 ${s2.toFixed(3)}) mm per 100 mm`);
  assert(r3.rule === 3 && s3 < s2, `tying the slopes to the gradient did not bring them nearer the truth: ${s3.toFixed(3)} against ${s2.toFixed(3)}`);
  assert(e3 < e2 + 0.02, `rule 3 reads points worse than rule 2: ${e3.toFixed(3)} against ${e2.toFixed(3)} mm`);
});

/** The largest common move of the diffusion-weighted images (mean of their six numbers), in mm and degrees. */
function commonMove(moves: Move[], bValues: number[]): { mm: number; degrees: number } {
  const dw = bValues.map((b, i) => (b > 50 ? i : -1)).filter((i) => i >= 0), v = [0, 0, 0, 0, 0, 0];
  for (const i of dw) { const w = logRot(moves[i].R); for (let q = 0; q < 3; q++) { v[q] += w[q] / dw.length; v[3 + q] += moves[i].t[q] / dw.length; } }
  return { mm: Math.hypot(v[3], v[4], v[5]), degrees: Math.hypot(v[0], v[1], v[2]) * 180 / Math.PI };
}

Deno.test("with a distortion field, a still head stays still: the field is not applied twice (critic 2026-10-05, finding 1)", async () => {
  const { bValues, gradients } = protocol(), c: [number, number, number] = [0, 0, 0];
  const still = scan(bValues.map(() => move([0, 0, 0], [0, 0, 0], c)), bValues, gradients, 2);
  // A smooth field along j of up to 2 voxels; the scan is distorted by it, and the correction is given it.
  const [nx, ny, nz] = DIMS, cen = new Float32Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) cen[(k * ny + j) * nx + i] = 2 * Math.exp(-(((i - nx / 2) / 12) ** 2 + ((j - ny / 2 - 6) / 14) ** 2 + ((k - nz / 3) / 9) ** 2));
  const fit = fieldFromCenters(DIMS, 1, cen);
  for (const v of still.volumes) v.data = applyField(fit, v.data as ArrayLike<number>, 1);
  const est = await estimateMotion(still, { fit, sign: -1 });
  const cm = commonMove(est.moves, bValues);
  console.log(`  common move of the diffusion-weighted images: ${cm.mm.toFixed(3)} mm, ${cm.degrees.toFixed(3)}° (before the fix: 2.2 mm); largest ${est.largest.mm} mm`);
  assert(cm.mm < 0.3 && cm.degrees < 0.3, `the diffusion-weighted images were moved together by ${cm.mm.toFixed(2)} mm and ${cm.degrees.toFixed(2)}°`);
});

Deno.test("two shells: a still head stays still (the extrapolation to b = 0 needs three; critic 2026-10-05, finding 2)", async () => {
  const c: [number, number, number] = [0, 0, 0], { bValues: b3, gradients } = protocol();
  const bValues = b3.map((b) => (b === 0 ? 0 : b === 2800 ? 2000 : 1000));        // b 1000 and 2000 only
  const still = scan(bValues.map(() => move([0, 0, 0], [0, 0, 0], c)), bValues, gradients, 4);
  const est = await estimateMotion(still);
  console.log(`  largest move ${est.largest.mm} mm, ${est.largest.degrees}° (before the fix: 7.3 mm, 19.3°)`);
  assert(est.largest.mm < 1 && est.largest.degrees < 1, `a still head was moved ${est.largest.mm} mm and ${est.largest.degrees}°`);
});

Deno.test("the single resampling reads exactly what the estimate read: moves, field and eddy shifts, and the gradients turned (critic 2026-10-05, finding 11)", async () => {
  const { bValues, gradients } = protocol(), c: [number, number, number] = [3, -5, 2];
  const n = bValues.length, dwi = scan(bValues.map(() => move([0, 0, 0], [0, 0, 0], [0, 0, 0])), bValues, gradients, 0);
  // Known moves about an off-center point, eddy slopes on the diffusion-weighted images, and a field.
  const moves: Move[] = bValues.map((b, v) => ({ ...move([0.8 * Math.sin(v), -0.5 * Math.cos(v), 0.3], [0.6 * Math.cos(v), 0.4, -0.7 * Math.sin(v)], c), ...(b > 0 ? { ec: { g: [0.5 * Math.sin(v), 0.8, -0.3] as [number, number, number], e: [0, 1, 0] as [number, number, number] } } : {}) }));
  const [nx, ny, nz] = DIMS, cen = new Float32Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) cen[(k * ny + j) * nx + i] = 1.5 * Math.sin(i / 9) * Math.cos(k / 7);
  const field = { fit: fieldFromCenters(DIMS, 1, cen), sign: -1 as const };
  const a = await applyMotion(dwi, { moves }, field);
  const b = await resampleOntoT1(dwi, { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0], c: [0, 0, 0] }, { dims: DIMS, ijkToRAS: M }, field, { motion: moves });
  let worst = 0, mean = 0, cnt = 0;
  for (let v = 0; v < n; v += 7) { const x = a.volumes[v].data as Float32Array, y = b.volumes[v].data as Float32Array; assert(x.length === y.length, "the two grids differ"); for (let i = 0; i < x.length; i++) { worst = Math.max(worst, Math.abs(x[i] - y[i])); mean += Math.abs(x[i]); cnt++; } }
  console.log(`  largest voxel difference ${worst.toExponential(2)} on a mean signal of ${(mean / cnt).toFixed(1)}`);
  assert(worst < 1e-2, `the resampling and the estimate's reading differ by ${worst}`);
  // Each gradient turned into the reference by its own move: Rᵥᵀ g (both paths).
  for (let v = 0; v < n; v++) {
    const R = moves[v].R, g = gradients[v], want = [R[0] * g[0] + R[3] * g[1] + R[6] * g[2], R[1] * g[0] + R[4] * g[1] + R[7] * g[2], R[2] * g[0] + R[5] * g[1] + R[8] * g[2]];
    for (let q = 0; q < 3; q++) { assert(Math.abs(a.gradients[v][q] - want[q]) < 1e-9, "applyMotion's gradient not Rᵀg"); assert(Math.abs(b.gradients[v][q] - want[q]) < 1e-9, "resampleOntoT1's gradient not Rᵀg"); }
  }
});

Deno.test("rule 3 always has its free round (critic 2026-10-05, finding 6); a six-direction scan says it was not corrected (finding 7)", async () => {
  const c: [number, number, number] = [0, 0, 0], { bValues, gradients } = protocol();
  const K = [[1.2, 0.3, 0], [0.2, 1.5, 0.1], [0, 0.4, 0.8]];
  const eddy: Eddy = { e: [0, 1, 0], g: gradients.map((d, v) => K.map((row) => Math.sqrt(bValues[v] / 2800) * (row[0] * d[0] + row[1] * d[1] + row[2] * d[2]))), d0: bValues.map(() => 0) };
  const dwi = scan(bValues.map(() => move([0, 0, 0], [0, 0, 0], c)), bValues, gradients, 2, eddy);
  const one = await estimateMotion(dwi, undefined, { rule: 3, peAxis: 1, rounds: 1 });
  assert(one.rule === 3 && (one.eddyMm ?? 0) > 0.2, `rule 3 with one round found no eddy currents (${one.eddyMm} mm)`);
  const six = scan([0, 1000, 1000, 1000, 1000, 1000, 1000].map(() => move([0, 0, 0], [0, 0, 0], c)), [0, 1000, 1000, 1000, 1000, 1000, 1000], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.7071, 0.7071, 0], [0.7071, 0, 0.7071], [0, 0.7071, 0.7071]], 2);
  const r6 = await estimateMotion(six);
  assert(r6.rule === 0 && /not corrected/.test(r6.said) && !/NaN/.test(r6.said), `a six-direction scan: "${r6.said}"`);
});
