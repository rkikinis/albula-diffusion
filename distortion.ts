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
// interpolation); its derivative is the linear interpolant of centered differences. Motion between the two scans is not
// modeled (the pair is taken back to back).

export const DISTORTION_RULE = 1;

export interface EpiPair {
  dims: [number, number, number];
  /** The image taken with phase encoding along +axis (voxel data, x fastest). */
  plus: Float32Array;
  /** The image taken with phase encoding along -axis. */
  minus: Float32Array;
  /** The voxel axis of phase encoding: 0 (i), 1 (j) or 2 (k). */
  axis: 0 | 1 | 2;
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
export function estimateField(pair: EpiPair, opts: { alpha?: number; beta?: number; levels?: number } = {}): FieldFit {
  const t0 = performance.now();
  const alpha = opts.alpha ?? 200, beta = opts.beta ?? 10, nLevels = opts.levels ?? 3;
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
    const start = b && gPrev ? prolong(b, gPrev, g) : null;
    const fit = fitLevel(g, L.plus, L.minus, start, alpha, beta);
    levels.push({ dims: L.dims, iterations: fit.iterations, residual: fit.residual });
    b = fit.b; gPrev = g;
  }
  return { b: b!, dims: pair.dims, axis: pair.axis, levels, ms: performance.now() - t0 };
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

/** The field at cell centers (voxels along the axis), x fastest -- for display and for reporting in Hz or mm. */
export function fieldAtCenters(fit: FieldFit): Float32Array {
  const g = grid(fit.dims, fit.axis), F = g.m + 1, out = new Float32Array(fit.dims[0] * fit.dims[1] * fit.dims[2]);
  for (let l = 0; l < g.lines; l++) for (let q = 0; q < g.m; q++) out[g.start[l] + q * g.stride] = 0.5 * (fit.b[l * F + q] + fit.b[l * F + q + 1]);
  return out;
}
