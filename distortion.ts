// SUSCEPTIBILITY DISTORTION CORRECTION FROM A REVERSED PHASE-ENCODING PAIR -- written from the papers (Ron, 2026-09-29:
// "fix it from paper derived code"; no FSL). The method is the reversed-gradient principle of Chang & Fitzpatrick
// (IEEE TMI 1992) in the variational form of Ruthotto et al. (HySCO; Phys Med Biol 2012) as written out in full by
// Macdonald & Ruthotto, "Improved susceptibility artifact correction of echo-planar MRI using the alternating direction
// method of multipliers", arXiv:1607.00531 (J Math Imaging Vis 2018) -- their Gauss-Newton formulation, eqs. and
// parameters as published. Nothing was read from any implementation.
//
// THE MODEL. An EPI image is displaced only along its phase-encoding direction v, by b(x) (in voxels here), and its
// intensity is modulated by the Jacobian of that displacement. Two images taken with opposite phase-encoding (+v, -v)
// see the same field with opposite signs, so the undistorted image I satisfies
//     I(x) = I₊(x + b(x)v)·(1 + ∂ᵥb(x)) = I₋(x − b(x)v)·(1 − ∂ᵥb(x)).
// b is found by minimizing
//     J(b) = ½‖r(b)‖² + (α/2)(‖D₁b‖² + ‖D₂b‖² + ‖D₃b‖²) + β Σ φ(D₁b),   r(b) = I₊(x + Ab)⊙(1 + D₁b) − I₋(x − Ab)⊙(1 − D₁b),
// φ(t) = t⁴/(1 − t²) on (−1, 1), infinite outside -- the barrier that keeps both Jacobians positive (the transformation
// invertible, "diffeomorphic"). D₁ differences along v, D₂ and D₃ across it; A averages the face-staggered b to cell
// centers (b lives on the faces between voxels along v: m + 1 values per line of m voxels).
//
// THE SOLVER (as published): Gauss-Newton; the step from preconditioned conjugate gradients (forcing η = 0.1) on
//     H = JᵣᵀJᵣ + α(D₁ᵀD₁ + D₂ᵀD₂ + D₃ᵀD₃) + β D₁ᵀ diag(φ''(D₁b)) D₁ + γI,   γ = 1e-3,
// preconditioned by its block-Jacobi part -- one tridiagonal (m+1)×(m+1) system per phase-encoding line (JᵣᵀJᵣ, αD₁ᵀD₁
// and the barrier are exactly tridiagonal along the line; the cross-line smoothness is kept as its diagonal); a
// backtracking line search that stays inside |D₁b| < 1; at most 10 iterations per level, the paper's stopping rules;
// three levels, coarse to fine. α = 200, β = 10 (the paper's values).
//
// UNITS AND ASSUMPTIONS (ours, where the paper's domain is the unit cube): b and all differences are in voxels of the
// input grid (Ron: "work in voxels"), which leaves the objective unchanged but for a constant factor; the paper's images
// were in the usual 0..255 range, so both images are scaled together so their 99th percentile is 255 before fitting
// (the α it recommends is relative to that). Along v the image is interpolated linearly (the paper does not say which
// interpolation); its derivative is the linear interpolant of centered differences. Motion between the two scans: rule 1
// does not model it (the field absorbs it); rule 2 (2026-10-03, the default) estimates it with the field -- see "Motion
// between the two scans" below.

/** 1: the field alone, α = 200 (2026-09-29). 2: the field with the reversed scan's movement, α = 500 (2026-10-03). */
export const DISTORTION_RULE = 2;

export interface EpiPair {
  dims: [number, number, number];
  /** The image taken with phase encoding along +axis (voxel data, x fastest). */
  plus: Float32Array;
  /** The image taken with phase encoding along -axis. */
  minus: Float32Array;
  /** The voxel axis of phase encoding: 0 (i), 1 (j) or 2 (k). */
  axis: 0 | 1 | 2;
  /** Voxel size along i, j, k in mm, for the movement between the scans (rule 2); default 1. */
  voxel?: [number, number, number];
}

export interface FieldFit {
  /** b on the faces along the axis: lines × (m + 1), in voxels; line order: the other two axes, lower one fastest. */
  b: Float32Array;
  dims: [number, number, number];
  axis: 0 | 1 | 2;
  /** Per level: Gauss-Newton iterations, and the relative residual ‖r(b)‖/‖r(0)‖ at the end. */
  levels: { dims: [number, number, number]; iterations: number; residual: number }[];
  ms: number;
}

// ── Lines along the phase-encoding axis ──────────────────────────────────────────────────────────────────────────

interface Grid { dims: [number, number, number]; axis: 0 | 1 | 2; m: number; lines: number; start: Int32Array; stride: number; u: number; w: number }

function grid(dims: [number, number, number], axis: 0 | 1 | 2): Grid {
  const [nx, ny, nz] = dims, stride = [1, nx, nx * ny][axis], m = dims[axis];
  const others = ([0, 1, 2] as const).filter((a) => a !== axis);
  const u = dims[others[0]], w = dims[others[1]];
  const start = new Int32Array(u * w);
  const s = [1, nx, nx * ny];
  for (let q = 0; q < w; q++) for (let p = 0; p < u; p++) start[q * u + p] = p * s[others[0]] + q * s[others[1]];
  return { dims, axis, m, lines: u * w, start, stride, u, w };
}

/** Linear interpolation of line values at position t (cell centers at 0..m-1); zero beyond the ends. */
function lerp(line: Float32Array, t: number): number {
  const m = line.length;
  if (t <= -1 || t >= m) return 0;
  const i0 = Math.floor(t), f = t - i0;
  const a = i0 >= 0 ? line[i0] : 0, c = i0 + 1 < m ? line[i0 + 1] : 0;
  return a + (c - a) * f;
}

const phi = (t: number) => (t * t * t * t) / (1 - t * t);
const phi1 = (t: number) => (4 * t ** 3 - 2 * t ** 5) / (1 - t * t) ** 2;
const phi2 = (t: number) => (2 * t * t * (t ** 4 - 3 * t * t + 6)) / (1 - t * t) ** 3;

/** Fit b on one level, starting from b0 (or zero). */
function fitLevel(g: Grid, plus: Float32Array, minus: Float32Array, b0: Float32Array | null, alpha: number, beta: number): { b: Float32Array; iterations: number; residual: number } {
  const { m, lines, start, stride, u, w } = g, F = m + 1, nb = lines * F, gamma = 1e-3;
  // Lines of both images and their centered derivatives along the axis.
  const L1: Float32Array[] = [], L2: Float32Array[] = [], G1: Float32Array[] = [], G2: Float32Array[] = [];
  for (let l = 0; l < lines; l++) {
    const a = new Float32Array(m), c = new Float32Array(m);
    for (let q = 0; q < m; q++) { a[q] = plus[start[l] + q * stride]; c[q] = minus[start[l] + q * stride]; }
    const da = new Float32Array(m), dc = new Float32Array(m);
    for (let q = 0; q < m; q++) {
      const lo = Math.max(0, q - 1), hi = Math.min(m - 1, q + 1), h = hi - lo || 1;
      da[q] = (a[hi] - a[lo]) / h; dc[q] = (c[hi] - c[lo]) / h;
    }
    L1.push(a); L2.push(c); G1.push(da); G2.push(dc);
  }
  const b = b0 ? Float32Array.from(b0) : new Float32Array(nb);
  const r = new Float32Array(lines * m), U = new Float32Array(lines * m), W = new Float32Array(lines * m), Jd = new Float32Array(lines * m);

  /** Residual, its linearization coefficients, and the objective at b. Infinity if |D₁b| >= 1 anywhere. */
  const evaluate = (bb: Float32Array, keep: boolean): number => {
    let E = 0, S = 0, P = 0;
    for (let l = 0; l < lines; l++) {
      const o = l * F;
      for (let q = 0; q < m; q++) {
        const j = bb[o + q + 1] - bb[o + q];
        if (!(j > -1 && j < 1)) return Infinity;
        const s = 0.5 * (bb[o + q] + bb[o + q + 1]);
        const i1 = lerp(L1[l], q + s), i2 = lerp(L2[l], q - s);
        const rr = i1 * (1 + j) - i2 * (1 - j);
        E += rr * rr; S += j * j; P += phi(j);
        if (keep) {
          const k = l * m + q;
          r[k] = rr;
          U[k] = lerp(G1[l], q + s) * (1 + j) + lerp(G2[l], q - s) * (1 - j);
          W[k] = i1 + i2;
        }
      }
    }
    // Smoothness across lines (D₂, D₃): differences between neighboring lines at the same face.
    for (let q = 0; q < w; q++) for (let p = 0; p < u; p++) {
      const l = q * u + p;
      for (let f = 0; f < F; f++) {
        const v = bb[l * F + f];
        if (p + 1 < u) { const d = bb[(l + 1) * F + f] - v; S += d * d; }
        if (q + 1 < w) { const d = bb[(l + u) * F + f] - v; S += d * d; }
      }
    }
    return 0.5 * E + 0.5 * alpha * S + beta * P;
  };
  /** The Laplacian-type smoothness operator (D₁ᵀD₁ + D₂ᵀD₂ + D₃ᵀD₃) applied to x, added into y times c. */
  const smooth = (x: Float32Array, y: Float32Array, c: number) => {
    for (let q = 0; q < w; q++) for (let p = 0; p < u; p++) {
      const l = q * u + p, o = l * F;
      for (let f = 0; f < F; f++) {
        let s = 0;
        if (f > 0) s += x[o + f] - x[o + f - 1];
        if (f < m) s += x[o + f] - x[o + f + 1];
        if (p > 0) s += x[o + f] - x[o - F + f];
        if (p + 1 < u) s += x[o + f] - x[o + F + f];
        if (q > 0) s += x[o + f] - x[o - u * F + f];
        if (q + 1 < w) s += x[o + f] - x[o + u * F + f];
        y[o + f] += c * s;
      }
    }
  };

  const r0 = (() => { const z = new Float32Array(nb); evaluate(z, true); let s = 0; for (const v of r) s += v * v; return Math.sqrt(s) || 1; })();
  let J = evaluate(b, true), J0 = J, iterations = 0;
  const grad = new Float32Array(nb), d = new Float32Array(nb);
  for (let it = 0; it < 10; it++) {
    iterations = it + 1;
    // Gradient: Jᵣᵀr + α·smooth(b) + β D₁ᵀ φ'(D₁b).
    grad.fill(0);
    const ph1 = new Float32Array(lines * m), ph2 = new Float32Array(lines * m);
    for (let l = 0; l < lines; l++) {
      const o = l * F;
      for (let q = 0; q < m; q++) {
        const k = l * m + q, j = b[o + q + 1] - b[o + q];
        ph1[k] = phi1(j); ph2[k] = phi2(j);
        const a0 = 0.5 * U[k] - W[k], a1 = 0.5 * U[k] + W[k];      // row of Jᵣ on faces q, q+1
        grad[o + q] += a0 * r[k] + beta * -ph1[k];
        grad[o + q + 1] += a1 * r[k] + beta * ph1[k];
      }
    }
    smooth(b, grad, alpha);
    let gnorm = 0; for (const v of grad) gnorm += v * v; gnorm = Math.sqrt(gnorm);
    // Hessian-vector product and the block-Jacobi (per-line tridiagonal) preconditioner.
    const Hv = (x: Float32Array, y: Float32Array) => {
      y.fill(0);
      for (let l = 0; l < lines; l++) {
        const o = l * F;
        for (let q = 0; q < m; q++) {
          const k = l * m + q, a0 = 0.5 * U[k] - W[k], a1 = 0.5 * U[k] + W[k];
          const jd = a0 * x[o + q] + a1 * x[o + q + 1];
          const dd = x[o + q + 1] - x[o + q];
          y[o + q] += a0 * jd - beta * ph2[k] * dd;
          y[o + q + 1] += a1 * jd + beta * ph2[k] * dd;
        }
      }
      smooth(x, y, alpha);
      for (let t = 0; t < nb; t++) y[t] += gamma * x[t];
    };
    const diag = new Float64Array(nb), off = new Float64Array(nb);     // off[f] couples f and f+1 within a line
    for (let l = 0; l < lines; l++) {
      const o = l * F, p = l % u, q2 = Math.floor(l / u);
      const cross = (p > 0 ? 1 : 0) + (p + 1 < u ? 1 : 0) + (q2 > 0 ? 1 : 0) + (q2 + 1 < w ? 1 : 0);
      for (let f = 0; f < F; f++) diag[o + f] = gamma + alpha * cross;
      for (let q = 0; q < m; q++) {
        const k = l * m + q, a0 = 0.5 * U[k] - W[k], a1 = 0.5 * U[k] + W[k], c = alpha + beta * ph2[k];
        diag[o + q] += a0 * a0 + c; diag[o + q + 1] += a1 * a1 + c; off[o + q] += a0 * a1 - c;
      }
    }
    const precond = (x: Float32Array, y: Float32Array) => {         // Thomas algorithm, one line at a time
      const cp = new Float64Array(F), dp = new Float64Array(F);
      for (let l = 0; l < lines; l++) {
        const o = l * F;
        cp[0] = off[o] / diag[o]; dp[0] = x[o] / diag[o];
        for (let f = 1; f < F; f++) {
          const den = diag[o + f] - off[o + f - 1] * cp[f - 1];
          cp[f] = f < m ? off[o + f] / den : 0;
          dp[f] = (x[o + f] - off[o + f - 1] * dp[f - 1]) / den;
        }
        y[o + m] = dp[m];
        for (let f = m - 1; f >= 0; f--) y[o + f] = dp[f] - cp[f] * y[o + f + 1];
      }
    };
    // PCG on H d = -grad, to ‖H d + grad‖ <= η‖grad‖.
    const res = new Float32Array(nb), z = new Float32Array(nb), pv = new Float32Array(nb), Hp = new Float32Array(nb);
    d.fill(0); for (let t = 0; t < nb; t++) res[t] = -grad[t];
    precond(res, z); pv.set(z);
    let rz = 0; for (let t = 0; t < nb; t++) rz += res[t] * z[t];
    for (let cg = 0; cg < 50; cg++) {
      Hv(pv, Hp);
      let pHp = 0; for (let t = 0; t < nb; t++) pHp += pv[t] * Hp[t];
      if (!(pHp > 0)) break;
      const a = rz / pHp;
      let rn = 0;
      for (let t = 0; t < nb; t++) { d[t] += a * pv[t]; res[t] -= a * Hp[t]; rn += res[t] * res[t]; }
      if (Math.sqrt(rn) <= 0.1 * gnorm) break;
      precond(res, z);
      let rz2 = 0; for (let t = 0; t < nb; t++) rz2 += res[t] * z[t];
      const beta2 = rz2 / rz; rz = rz2;
      for (let t = 0; t < nb; t++) pv[t] = z[t] + beta2 * pv[t];
    }
    // Backtracking line search (Armijo), staying inside |D₁b| < 1.
    let slope = 0; for (let t = 0; t < nb; t++) slope += grad[t] * d[t];
    let lambda = 1, bNew = new Float32Array(nb), Jn = Infinity;
    for (let ls = 0; ls < 12; ls++) {
      for (let t = 0; t < nb; t++) bNew[t] = b[t] + lambda * d[t];
      Jn = evaluate(bNew, false);
      if (Jn <= J + 1e-4 * lambda * slope) break;
      lambda /= 2;
    }
    if (!(Jn < J)) break;
    let step = 0, bn = 0; for (let t = 0; t < nb; t++) { step += (bNew[t] - b[t]) ** 2; bn += b[t] * b[t]; }
    b.set(bNew);
    const Jprev = J;
    J = evaluate(b, true);
    // The paper's stopping rules: objective change, step and gradient, relative.
    if (Math.abs(Jprev - J) <= 1e-3 * (1 + Math.abs(J0)) && Math.sqrt(step) <= 1e-2 * (1 + Math.sqrt(bn)) && gnorm <= 1e-2 * (1 + Math.abs(J0))) break;
  }
  let rs = 0; for (const v of r) rs += v * v;
  return { b, iterations, residual: Math.sqrt(rs) / r0 };
}

/** Half-size images (2×2×2 averages; an odd last voxel is averaged with itself). */
function restrict(img: Float32Array, dims: [number, number, number]): { img: Float32Array; dims: [number, number, number] } {
  const [nx, ny, nz] = dims, cd: [number, number, number] = [Math.ceil(nx / 2), Math.ceil(ny / 2), Math.ceil(nz / 2)];
  const out = new Float32Array(cd[0] * cd[1] * cd[2]);
  for (let k = 0; k < cd[2]; k++) for (let j = 0; j < cd[1]; j++) for (let i = 0; i < cd[0]; i++) {
    let s = 0, c = 0;
    for (let dk = 0; dk < 2; dk++) for (let dj = 0; dj < 2; dj++) for (let di = 0; di < 2; di++) {
      const x = Math.min(2 * i + di, nx - 1), y = Math.min(2 * j + dj, ny - 1), z = Math.min(2 * k + dk, nz - 1);
      s += img[(z * ny + y) * nx + x]; c++;
    }
    out[(k * cd[1] + j) * cd[0] + i] = s / c;
  }
  return { img: out, dims: cd };
}

/** A coarse b prolonged to the next finer grid: linear along the axis (and doubled -- the voxels halved), nearest across. */
function prolong(bc: Float32Array, gc: Grid, gf: Grid): Float32Array {
  const Fc = gc.m + 1, Ff = gf.m + 1, out = new Float32Array(gf.lines * Ff);
  for (let q = 0; q < gf.w; q++) for (let p = 0; p < gf.u; p++) {
    const lc = Math.min(q >> 1, gc.w - 1) * gc.u + Math.min(p >> 1, gc.u - 1), lf = q * gf.u + p;
    for (let f = 0; f < Ff; f++) {
      const t = Math.min(f / 2, gc.m), f0 = Math.floor(t), a = t - f0;
      const v0 = bc[lc * Fc + f0], v1 = bc[lc * Fc + Math.min(f0 + 1, gc.m)];
      out[lf * Ff + f] = 2 * (v0 + (v1 - v0) * a);
    }
  }
  return out;
}

/** Estimate the displacement field of a reversed phase-encoding pair (three levels, coarse to fine). */
export function estimateField(pair: EpiPair, opts: { alpha?: number; beta?: number; levels?: number; /** A field to start from, on this grid: then only the finest level is fitted (rule 2's rounds). */ start?: Float32Array } = {}): FieldFit {
  const t0 = performance.now();
  const alpha = opts.alpha ?? 200, beta = opts.beta ?? 10, nLevels = opts.start ? 1 : opts.levels ?? 3;
  // Both images scaled together: the 99th percentile of their mean to 255 (see UNITS above).
  const sample: number[] = [];
  for (let v = 0; v < pair.plus.length; v += 7) sample.push(0.5 * (pair.plus[v] + pair.minus[v]));
  sample.sort((a, b) => a - b);
  const scale = 255 / (sample[Math.floor(sample.length * 0.99)] || 1);
  const pyr: { plus: Float32Array; minus: Float32Array; dims: [number, number, number] }[] = [
    { plus: pair.plus.map((v) => v * scale), minus: pair.minus.map((v) => v * scale), dims: pair.dims },
  ];
  for (let lv = 1; lv < nLevels; lv++) {
    const top = pyr[pyr.length - 1];
    if (Math.min(...top.dims) < 8) break;
    const a = restrict(top.plus, top.dims), c = restrict(top.minus, top.dims);
    pyr.push({ plus: a.img, minus: c.img, dims: a.dims });
  }
  const levels: FieldFit["levels"] = [];
  let b: Float32Array | null = null, gPrev: Grid | null = null;
  for (let lv = pyr.length - 1; lv >= 0; lv--) {
    const L = pyr[lv], g = grid(L.dims, pair.axis);
    const start = b && gPrev ? prolong(b, gPrev, g) : opts.start ?? null;
    const fit = fitLevel(g, L.plus, L.minus, start, alpha, beta);
    levels.push({ dims: L.dims, iterations: fit.iterations, residual: fit.residual });
    b = fit.b; gPrev = g;
  }
  return { b: b!, dims: pair.dims, axis: pair.axis, levels, ms: performance.now() - t0 };
}

// ── Motion between the two scans (rule 2, 2026-10-03) ─────────────────────────────────────────────────────────────
// WHY: the field of rule 1 also absorbed any movement of the head between the two scans. On the split-half check
// (Contents/tools/dmri-field-repeatability.ts; validation step 4) two independent halves of PAT16's b = 0 images gave fields
// 0.30 mm apart (median), 2.3 mm (99th), against topup's 0.09 / 0.57 mm; topup found the PA image 0.65 mm apart between the
// halves and took it out as movement -- its model (Andersson, Skare & Ashburner, NeuroImage 20:870, 2003) estimates the
// field together with each scan's rigid movement. Ron: "Build".
// HOW (ours, from that model; nothing read from topup's code): alternate -- the field with the reversed image held where
// it is; then, with the field held, the reversed image's rigid movement that makes the two corrected images agree best
// (least squares, Gauss-Newton, the six derivatives by differences); then the field again on the moved image; until the
// movement settles. The movement is applied to the reversed image as acquired, before the reversed-gradient model -- the
// order is approximate (the distortion happened in the scanner's frame after the head moved); with movements of a
// millimeter and a degree and a field of tens of millimeters at most the error is about a field times the angle: 0.3 mm
// at 20 mm and 1°. FIVE parameters, not six: a shift ALONG the phase-encoding axis cannot be told from a constant field
// (both slide the two images apart along it, and the smoothness penalty does not see a constant), so it stays in the
// field, as in rule 1; rotations and the two shifts across the axis are estimated. Rotations are about the volume's
// center, in millimeters (EpiPair.voxel).

/** The reversed image's rigid movement: shifts (mm) along the grid's axes and rotations (radians) about them, about the
 *  volume's center; y = Rz·Ry·Rx·(x − c) + c + t, where the moved image at x samples the acquired one at y. */
export interface Motion { t: [number, number, number]; r: [number, number, number] }

/** The image moved rigidly (Motion), trilinear, the edge value beyond the edges. */
export function moveRigid(img: ArrayLike<number>, dims: [number, number, number], voxel: [number, number, number], mo: Motion): Float32Array {
  const [nx, ny, nz] = dims, out = new Float32Array(nx * ny * nz);
  const [a, b, c] = mo.r, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cc = Math.cos(c), sc = Math.sin(c);
  // R = Rz(c)·Ry(b)·Rx(a)
  const R = [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa, sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa, -sb, cb * sa, cb * ca];
  const ctr = [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2];
  const at = (i: number, j: number, k: number) => img[(Math.min(nz - 1, Math.max(0, k)) * ny + Math.min(ny - 1, Math.max(0, j))) * nx + Math.min(nx - 1, Math.max(0, i))];
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = (i - ctr[0]) * voxel[0], y = (j - ctr[1]) * voxel[1], z = (k - ctr[2]) * voxel[2];
    const X = (R[0] * x + R[1] * y + R[2] * z + mo.t[0]) / voxel[0] + ctr[0];
    const Y = (R[3] * x + R[4] * y + R[5] * z + mo.t[1]) / voxel[1] + ctr[1];
    const Z = (R[6] * x + R[7] * y + R[8] * z + mo.t[2]) / voxel[2] + ctr[2];
    const i0 = Math.floor(X), j0 = Math.floor(Y), k0 = Math.floor(Z), fx = X - i0, fy = Y - j0, fz = Z - k0;
    const c00 = at(i0, j0, k0) * (1 - fx) + at(i0 + 1, j0, k0) * fx, c10 = at(i0, j0 + 1, k0) * (1 - fx) + at(i0 + 1, j0 + 1, k0) * fx;
    const c01 = at(i0, j0, k0 + 1) * (1 - fx) + at(i0 + 1, j0, k0 + 1) * fx, c11 = at(i0, j0 + 1, k0 + 1) * (1 - fx) + at(i0 + 1, j0 + 1, k0 + 1) * fx;
    out[(k * ny + j) * nx + i] = (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
  }
  return out;
}

/** A Gaussian blur of σ voxels (separable, the edge value beyond the edges). */
export function gaussSmooth(img: ArrayLike<number>, dims: [number, number, number], sigma: number): Float32Array {
  const [nx, ny, nz] = dims, R = Math.ceil(3 * sigma), w: number[] = [];
  let sum = 0; for (let d = -R; d <= R; d++) { const x = Math.exp(-d * d / (2 * sigma * sigma)); w.push(x); sum += x; }
  for (let i = 0; i < w.length; i++) w[i] /= sum;
  let a = Float32Array.from(img);
  const n = [nx, ny, nz], st = [1, nx, nx * ny];
  for (let ax = 0; ax < 3; ax++) {
    const b = new Float32Array(a.length);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const at = [i, j, k][ax], v = (k * ny + j) * nx + i; let acc = 0;
      for (let d = -R; d <= R; d++) acc += w[d + R] * a[v + (Math.min(n[ax] - 1, Math.max(0, at + d)) - at) * st[ax]];
      b[v] = acc;
    }
    a = b;
  }
  return a;
}

/** Both images' blur before the movement is measured, in voxels: none, by measurement. WHY IT WAS TRIED: every
 *  resampling of the moved image blurs it a little (linear interpolation), so the plain difference is smallest where it
 *  needs the least resampling, which pulls the estimate toward no movement -- on a realistic phantom (PAT16's corrected
 *  b = 0, a known movement of 0.4/0.3 mm and 0.3-0.5°) a full-size fit with the true field held recovered half of it, and
 *  with a 1.5-voxel blur of both nearly all. WHY NOT: with the field estimated too, the blurred fit was unstable (the
 *  rotations the field can mimic -- the two that move points partly along the phase-encoding axis -- wandered: −0.85°
 *  for +0.3°) and its fields were worse (phantom field error 0.27-0.46 mm median against 0.23 without). Fitting the
 *  movement on HALF-SIZE images (fitMotion's halfSize) averages both images alike, takes most of the bias out (PAT16's
 *  and PAT08's movements then close to topup's own estimates), and is eight times cheaper. */
const MOTION_BLUR = 0;

/** The reversed image's movement with the field held (Gauss-Newton on the corrected images' difference, five parameters). */
export function fitMotion(fit: FieldFit, pair: EpiPair, voxel: [number, number, number], start: Motion, opts: { blur?: number; rotations?: "all" | "about-axis"; /** Measure on half-size images (default; the movement is in mm, the same at any size). */ halfSize?: boolean } = {}): Motion {
  if (opts.halfSize ?? true) {
    // Both images and the field at half size: the field's values halve with the voxels doubled.
    const a = restrict(Float32Array.from(pair.plus), pair.dims), c = restrict(Float32Array.from(pair.minus), pair.dims);
    const fc = restrict(fieldAtCenters(fit), fit.dims);
    for (let v = 0; v < fc.img.length; v++) fc.img[v] /= 2;
    return fitMotion(fieldFromCenters(a.dims, pair.axis, fc.img), { ...pair, dims: a.dims, plus: a.img, minus: c.img }, voxel.map((x) => 2 * x) as [number, number, number], start, { ...opts, halfSize: false });
  }
  const free: { kind: "t" | "r"; a: number; h: number }[] = [0, 1, 2].filter((a) => a !== pair.axis).map((a) => ({ kind: "t" as const, a, h: 0.05 }) as { kind: "t" | "r"; a: number; h: number })
    .concat([0, 1, 2].filter((a) => opts.rotations !== "about-axis" || a === pair.axis).map((a) => ({ kind: "r" as const, a, h: 1e-3 })));
  const blur = opts.blur ?? MOTION_BLUR;
  const corrMinus = applyField(fit, blur > 0 ? gaussSmooth(pair.minus, pair.dims, blur) : pair.minus, -1), plusB = blur > 0 ? gaussSmooth(pair.plus, pair.dims, blur) : Float32Array.from(pair.plus);
  const resid = (mo: Motion) => { const c = applyField(fit, moveRigid(plusB, pair.dims, voxel, mo), 1); for (let v = 0; v < c.length; v++) c[v] -= corrMinus[v]; return c; };
  const ssd = (r: Float32Array) => { let s = 0; for (const x of r) s += x * x; return s; };
  const with_ = (mo: Motion, d: number[]) => { const n: Motion = { t: [...mo.t], r: [...mo.r] }; free.forEach((f, k) => { n[f.kind][f.a] += d[k]; }); return n; };
  let mo: Motion = { t: [...start.t], r: [...start.r] }, r0 = resid(mo), E = ssd(r0), lambda = 1e-3;
  for (let it = 0; it < 12; it++) {
    const P = free.length, J = free.map((f, k) => { const d = new Array(P).fill(0); d[k] = f.h; const r = resid(with_(mo, d)); for (let v = 0; v < r.length; v++) r[v] = (r[v] - r0[v]) / f.h; return r; });
    const A = new Float64Array(P * P), g = new Float64Array(P);
    for (let p = 0; p < P; p++) { for (let q = p; q < P; q++) { let s = 0; const a = J[p], b = J[q]; for (let v = 0; v < a.length; v++) s += a[v] * b[v]; A[p * P + q] = A[q * P + p] = s; } let s = 0; for (let v = 0; v < r0.length; v++) s += J[p][v] * r0[v]; g[p] = -s; }
    let accepted = false;
    for (let tries = 0; tries < 6 && !accepted; tries++) {
      const M = Float64Array.from(A); for (let p = 0; p < P; p++) M[p * P + p] *= 1 + lambda;
      const d = solveSmall(M, g, P), cand = with_(mo, d), rc = resid(cand), Ec = ssd(rc);
      if (Ec < E) {
        accepted = true; mo = cand; r0 = rc; lambda = Math.max(1e-6, lambda / 10);
        const small = free.every((f, k) => Math.abs(d[k]) < (f.kind === "t" ? 0.005 : 5e-5)), gain = (E - Ec) / E; E = Ec;
        if (small || gain < 1e-5) return mo;
      } else lambda *= 10;
    }
    if (!accepted) break;
  }
  return mo;
}

/** A small dense system by Gaussian elimination with partial pivoting. */
function solveSmall(A: Float64Array, b: Float64Array, n: number): number[] {
  const M = Float64Array.from(A), x = Array.from(b);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[p * n + c])) p = r;
    if (p !== c) { for (let k = 0; k < n; k++) { const t = M[c * n + k]; M[c * n + k] = M[p * n + k]; M[p * n + k] = t; } const t = x[c]; x[c] = x[p]; x[p] = t; }
    const d = M[c * n + c] || 1e-30;
    for (let r = c + 1; r < n; r++) { const f = M[r * n + c] / d; for (let k = c; k < n; k++) M[r * n + k] -= f * M[c * n + k]; x[r] -= f * x[c]; }
  }
  for (let c = n - 1; c >= 0; c--) { let s = x[c]; for (let k = c + 1; k < n; k++) s -= M[c * n + k] * x[k]; x[c] = s / (M[c * n + c] || 1e-30); }
  return x;
}

/** RULE 2: the field and the reversed image's movement together (see "Motion between the two scans" above). */
export function estimateFieldWithMotion(pair: EpiPair, options: { alpha?: number; beta?: number; levels?: number; rounds?: number; alphaMotion?: number; blur?: number; rotations?: "all" | "about-axis" } = {}): FieldFit & { motion: Motion; rounds: number } {
  const t0 = performance.now(), voxel = pair.voxel ?? [1, 1, 1];
  // α = 500, not rule 1's 200: on the split halves and against topup (PAT16, PAT05, PAT08) 500 with the movement was better
  // than 200 with it, than 500 without it, and than 1000 (which lost topup's sharp field near the sinuses, 99th 2.4-2.9 mm).
  const opts = { alpha: 500, rounds: 6, ...options };
  // alphaMotion: the field's smoothness while the movement is measured (default the same). A much stiffer one (20,000) was
  // tried so that the field could not bend to absorb the movement; on the phantom it gave the same field (0.22 against
  // 0.23 mm median error) in 40% more time, so it is not the default.
  const stiff = { ...opts, alpha: opts.alphaMotion ?? opts.alpha };
  let mo: Motion = { t: [0, 0, 0], r: [0, 0, 0] }, fit = estimateField(pair, stiff), rounds = 0;
  for (let round = 0; round < (opts.rounds ?? 4); round++) {
    const next = fitMotion(fit, pair, voxel, mo, opts);
    // Settled: under 0.05 mm and 0.03° of change (a fiftieth of a 2.5 mm voxel; 0.03° moves a point 0.05 mm at 100 mm).
    const moved = Math.max(...[0, 1, 2].map((a) => Math.abs(next.t[a] - mo.t[a]))) > 0.05 || Math.max(...[0, 1, 2].map((a) => Math.abs(next.r[a] - mo.r[a]))) > 5e-4;
    mo = next; rounds = round + 1;
    if (!moved) break;
    fit = estimateField({ ...pair, plus: moveRigid(pair.plus, pair.dims, voxel, mo) }, { ...stiff, start: fit.b });
  }
  fit = estimateField({ ...pair, plus: moveRigid(pair.plus, pair.dims, voxel, mo) }, { ...opts, start: fit.b });
  return { ...fit, motion: mo, rounds, ms: performance.now() - t0 };
}

/** Correct one image taken with phase encoding along +axis (sign +1) or -axis (sign -1): I(x) = I±(x ± b v)(1 ± ∂ᵥb). */
export function applyField(fit: FieldFit, image: ArrayLike<number>, sign: 1 | -1): Float32Array {
  const g = grid(fit.dims, fit.axis), F = g.m + 1, out = new Float32Array(image.length), line = new Float32Array(g.m);
  for (let l = 0; l < g.lines; l++) {
    for (let q = 0; q < g.m; q++) line[q] = image[g.start[l] + q * g.stride];
    for (let q = 0; q < g.m; q++) {
      const b0 = fit.b[l * F + q], b1 = fit.b[l * F + q + 1];
      out[g.start[l] + q * g.stride] = lerp(line, q + sign * 0.5 * (b0 + b1)) * (1 + sign * (b1 - b0));
    }
  }
  return out;
}

/** A field given at cell centers (voxels along the axis, x fastest) as a FieldFit, so applyField can apply a field made
 *  elsewhere -- topup's, to check the applying against applytopup (distortion-apply.test.ts). Each inner face takes the
 *  mean of its two cells, an end face its one cell; fieldAtCenters of the result is that field smoothed by one face. */
export function fieldFromCenters(dims: [number, number, number], axis: 0 | 1 | 2, c: ArrayLike<number>): FieldFit {
  const g = grid(dims, axis), F = g.m + 1, b = new Float32Array(g.lines * F);
  for (let l = 0; l < g.lines; l++) {
    const at = (q: number) => c[g.start[l] + q * g.stride];
    b[l * F] = at(0); b[l * F + g.m] = at(g.m - 1);
    for (let q = 1; q < g.m; q++) b[l * F + q] = 0.5 * (at(q - 1) + at(q));
  }
  return { b, dims, axis, levels: [], ms: 0 };
}

/** The field at cell centers (voxels along the axis), x fastest -- for display and for reporting in Hz or mm. */
export function fieldAtCenters(fit: FieldFit): Float32Array {
  const g = grid(fit.dims, fit.axis), F = g.m + 1, out = new Float32Array(fit.dims[0] * fit.dims[1] * fit.dims[2]);
  for (let l = 0; l < g.lines; l++) for (let q = 0; q < g.m; q++) out[g.start[l] + q * g.stride] = 0.5 * (fit.b[l * F + q] + fit.b[l * F + q + 1]);
  return out;
}
