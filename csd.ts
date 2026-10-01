// CONSTRAINED SPHERICAL DECONVOLUTION, MULTI-SHELL MULTI-TISSUE -- on the processor, in double precision: the reference
// the graphics-card version is checked against, itself checked against DIPY (csd.test.ts; Contents/tools/csd-reference.py
// in the workspace). Plan: Contents/docs/csd-ptt-plan-2026-10-01.md (workspace).
//
// THE MODEL (Jeurissen et al., NeuroImage 103:411, 2014; Tournier et al., NeuroImage 35:1459, 2007). In a voxel, the
// signal of gradient g at b-value b is
//     S(b, g) = Σ_t f_t · K_t(b)  +  Σ_{l even, m} c_lm · K_wm(b, l) / Y_l0(pole) · Y_lm(g)
// -- isotropic tissues t (fluid, gray matter) with a fraction each, and the white-matter fiber orientation distribution
// (FOD) as real spherical harmonics to order 8 (45 numbers), blurred order by order by the white-matter response. The
// kernel K (per shell: the isotropic tissues' signals, then the white matter's per order l = 0, 2, …, 8) is the response
// function's, as DIPY's `multi_shell_fiber_response` gives it. b = 0 images see only order 0.
// The fit: least squares, with the fractions >= 0 and the FOD >= 0 on a set of directions (the CONSTRAINT).
//
// THE SOLVER, exact: with P = XᵀX = LLᵀ and z = Lᵀx the problem is the projection of c = L⁻¹Xᵀs onto the cone
// {z : M z >= 0}, M = G L⁻ᵀ (G: the constraint rows). By Moreau's decomposition z = c + Mᵀλ, where λ solves the
// non-negative least-squares problem min_{λ>=0} |Mᵀλ + c|² (Lawson & Hanson, "Solving Least Squares Problems", 1974,
// ch. 23). So: one Cholesky per scan (X and G do not change from voxel to voxel), then one small NNLS per voxel (47 rows,
// one column per constraint; at most 47 columns active). DIPY solves the same problem with a general solver (cvxpy).

/** Real, symmetric spherical harmonics to order `lmax` (even orders), orthonormal on the sphere: m < 0 → √2·N·P_l^|m|·
 *  sin(|m|φ), m = 0 → N·P_l^0, m > 0 → √2·N·P_l^m·cos(mφ). The order of the 45 coefficients: l = 0, 2, …; m = −l … l. */
export function shBasis(lmax: number, dirs: ArrayLike<number>): Float64Array {
  const n = dirs.length / 3, nc = shCount(lmax), out = new Float64Array(n * nc);
  const fact = (k: number) => { let f = 1; for (let i = 2; i <= k; i++) f *= i; return f; };
  for (let d = 0; d < n; d++) {
    const x = dirs[3 * d], y = dirs[3 * d + 1], z = dirs[3 * d + 2], r = Math.hypot(x, y, z) || 1;
    const ct = z / r, st = Math.sqrt(Math.max(0, 1 - ct * ct)), phi = Math.atan2(y, x);
    // Associated Legendre P_l^m(ct) for all l <= lmax, m <= l (Condon-Shortley phase omitted: it cancels in a real basis).
    const P: number[][] = [];
    for (let m = 0; m <= lmax; m++) {
      P[m] = new Array(lmax + 1).fill(0);
      let pmm = 1; for (let i = 1; i <= m; i++) pmm *= (2 * i - 1) * st;
      P[m][m] = pmm;
      if (m < lmax) P[m][m + 1] = ct * (2 * m + 1) * pmm;
      for (let l = m + 2; l <= lmax; l++) P[m][l] = ((2 * l - 1) * ct * P[m][l - 1] - (l + m - 1) * P[m][l - 2]) / (l - m);
    }
    let c = 0;
    for (let l = 0; l <= lmax; l += 2) for (let m = -l; m <= l; m++) {
      const am = Math.abs(m), N = Math.sqrt((2 * l + 1) / (4 * Math.PI) * fact(l - am) / fact(l + am));
      out[d * nc + c++] = m === 0 ? N * P[0][l] : Math.SQRT2 * N * P[am][l] * (m < 0 ? Math.sin(am * phi) : Math.cos(am * phi));
    }
  }
  return out;
}
export const shCount = (lmax: number) => (lmax + 1) * (lmax + 2) / 2;

/** The response kernel: per shell (b-value), the isotropic tissues' signals, then the white matter's per even order. */
export interface Kernel { shells: number[]; response: number[][]; iso: number; lmax: number }

export interface CsdModel {
  lmax: number; iso: number; nc: number; nx: number;
  /** Lower Cholesky factor of XᵀX (nx × nx, row-major), X itself (rows × nx), and M = G L⁻ᵀ (constraints × nx). */
  L: Float64Array; X: Float64Array; M: Float64Array; rows: number; constraints: number;
  /** For each gradient (row), its shell's index. */
  shellOf: Int32Array;
}

/**
 * The model for one scan: `gradients` unit vectors (any frame; the FOD comes out in the same frame), `bValues` per
 * gradient, the kernel, and the directions where the FOD must not be negative (a hemisphere suffices: the FOD is
 * antipodally symmetric).
 */
export function csdModel(gradients: ArrayLike<number>, bValues: ArrayLike<number>, k: Kernel, constraintDirs: ArrayLike<number>, b0Threshold = 50): CsdModel {
  const { lmax, iso } = k, nc = shCount(lmax), nx = iso + nc, rows = bValues.length;
  const B = shBasis(lmax, gradients);
  const Y0 = (l: number) => Math.sqrt((2 * l + 1) / (4 * Math.PI));
  const shellOf = new Int32Array(rows);
  for (let r = 0; r < rows; r++) {
    let best = 0;
    for (let s = 1; s < k.shells.length; s++) if (Math.abs(k.shells[s] - bValues[r]) < Math.abs(k.shells[best] - bValues[r])) best = s;
    shellOf[r] = best;
  }
  const X = new Float64Array(rows * nx);
  for (let r = 0; r < rows; r++) {
    const K = k.response[shellOf[r]], b0 = bValues[r] <= b0Threshold;
    for (let t = 0; t < iso; t++) X[r * nx + t] = K[t];
    let c = 0;
    for (let l = 0; l <= lmax; l += 2) for (let m = -l; m <= l; m++, c++) {
      X[r * nx + iso + c] = b0 && l > 0 ? 0 : B[r * nc + c] * K[iso + l / 2] / Y0(l);
    }
  }
  // P = XᵀX and its Cholesky factor.
  const P = new Float64Array(nx * nx);
  for (let i = 0; i < nx; i++) for (let j = 0; j <= i; j++) {
    let s = 0; for (let r = 0; r < rows; r++) s += X[r * nx + i] * X[r * nx + j];
    P[i * nx + j] = P[j * nx + i] = s;
  }
  const L = cholesky(P, nx)!;
  // G: the fractions (identity), then the FOD's value at each constraint direction.
  const nd = constraintDirs.length / 3, constraints = iso + nd, Bc = shBasis(lmax, constraintDirs);
  const G = new Float64Array(constraints * nx);
  for (let t = 0; t < iso; t++) G[t * nx + t] = 1;
  for (let d = 0; d < nd; d++) for (let c = 0; c < nc; c++) G[(iso + d) * nx + iso + c] = Bc[d * nc + c];
  // M = G L⁻ᵀ: each row g of G becomes the solution y of L y = g (since (G L⁻ᵀ)ᵀ = L⁻¹ Gᵀ).
  const M = new Float64Array(constraints * nx);
  for (let i = 0; i < constraints; i++) {
    for (let a = 0; a < nx; a++) {
      let s = G[i * nx + a];
      for (let b = 0; b < a; b++) s -= L[a * nx + b] * M[i * nx + b];
      M[i * nx + a] = s / L[a * nx + a];
    }
  }
  return { lmax, iso, nc, nx, L, X, M, rows, constraints, shellOf };
}

function cholesky(A: Float64Array, n: number, soft = false): Float64Array | undefined {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * n + j];
    for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
    if (i === j) {
      if (s <= (soft ? 1e-12 * Math.max(1, A[i * n + i]) : 0)) { if (soft) return undefined; throw new Error("CSD: the design is singular (too few gradients or shells for the model)"); }
      L[i * n + i] = Math.sqrt(s);
    }
    else L[i * n + j] = s / L[j * n + j];
  }
  return L;
}

/**
 * NON-NEGATIVE LEAST SQUARES, Lawson & Hanson's active-set algorithm: min_{λ>=0} |Aλ − b|, A given by its COLUMNS
 * (here the rows of M: `col(j)` is column j, length m). Returns λ (most entries 0). `tol` on the gradient.
 */
export function nnls(m: number, ncols: number, col: (j: number) => Float64Array, b: Float64Array, maxIter = 2000, tol = 1e-10): Float64Array {
  const x = new Float64Array(ncols), passive: number[] = [], inP = new Uint8Array(ncols);
  const r = Float64Array.from(b);                                 // residual b − Ax
  const w = new Float64Array(ncols);
  const solveP = (): Float64Array | undefined => {
    // least squares on the passive columns: normal equations, Cholesky (k <= m columns)
    const k = passive.length, N = new Float64Array(k * k), rhs = new Float64Array(k);
    const C = passive.map((j) => col(j));
    for (let a = 0; a < k; a++) {
      for (let c = 0; c <= a; c++) { let s = 0; for (let i = 0; i < m; i++) s += C[a][i] * C[c][i]; N[a * k + c] = N[c * k + a] = s; }
      let s = 0; for (let i = 0; i < m; i++) s += C[a][i] * b[i]; rhs[a] = s;
    }
    const Lk = cholesky(N, k, true);
    if (!Lk) return undefined;                                   // the passive columns are (nearly) dependent
    const y = new Float64Array(k), z = new Float64Array(k);
    for (let a = 0; a < k; a++) { let s = rhs[a]; for (let c = 0; c < a; c++) s -= Lk[a * k + c] * y[c]; y[a] = s / Lk[a * k + a]; }
    for (let a = k - 1; a >= 0; a--) { let s = y[a]; for (let c = a + 1; c < k; c++) s -= Lk[c * k + a] * z[c]; z[a] = s / Lk[a * k + a]; }
    return z;
  };
  const residual = () => {
    r.set(b);
    for (const j of passive) { const c = col(j), v = x[j]; if (v) for (let i = 0; i < m; i++) r[i] -= c[i] * v; }
  };
  // A column that would make the passive set (nearly) dependent is set aside until the set changes otherwise.
  const blocked = new Uint8Array(ncols);
  for (let it = 0; it < maxIter; it++) {
    // the gradient Aᵀr; the most promising inactive column
    let best = -1, bw = tol;
    for (let j = 0; j < ncols; j++) {
      if (inP[j] || blocked[j]) continue;
      const c = col(j); let s = 0; for (let i = 0; i < m; i++) s += c[i] * r[i];
      w[j] = s; if (s > bw) { bw = s; best = j; }
    }
    if (best < 0 || passive.length >= m) break;
    passive.push(best); inP[best] = 1;
    const first = solveP();
    if (!first || first[first.length - 1] <= 0) {                // no progress along it: set it aside
      passive.pop(); inP[best] = 0; blocked[best] = 1; continue;
    }
    for (let z: Float64Array | undefined = first; ; z = solveP()) {
      if (!z) { passive.pop(); break; }
      if (z.every((v) => v > 0)) { passive.forEach((j, a) => { x[j] = z![a]; }); break; }
      // step back toward the old x until a passive variable reaches 0, and drop it
      let alpha = Infinity;
      passive.forEach((j, a) => { if (z![a] <= 0) alpha = Math.min(alpha, x[j] / (x[j] - z![a])); });
      passive.forEach((j, a) => { x[j] += alpha * (z![a] - x[j]); });
      for (let a = passive.length - 1; a >= 0; a--) if (x[passive[a]] <= 1e-14) { x[passive[a]] = 0; inP[passive[a]] = 0; passive.splice(a, 1); blocked.fill(0); }
      if (!passive.length) break;
    }
    residual();
  }
  return x;
}

/** The fit of one voxel: x = [fractions…, FOD coefficients…] (nx numbers). */
export function csdFit(model: CsdModel, signal: ArrayLike<number>): Float64Array {
  const { nx, rows, X, L, M, constraints } = model;
  const q = new Float64Array(nx);
  for (let a = 0; a < nx; a++) { let s = 0; for (let r = 0; r < rows; r++) s += X[r * nx + a] * signal[r]; q[a] = s; }
  // c = L⁻¹ q
  const c = new Float64Array(nx);
  for (let a = 0; a < nx; a++) { let s = q[a]; for (let b = 0; b < a; b++) s -= L[a * nx + b] * c[b]; c[a] = s / L[a * nx + a]; }
  // λ = argmin_{λ>=0} |Mᵀλ + c|²: NNLS with A = Mᵀ (columns = rows of M), b = −c
  const negc = c.map((v) => -v);
  const lam = nnls(nx, constraints, (j) => M.subarray(j * nx, j * nx + nx), negc);
  const z = Float64Array.from(c);
  for (let j = 0; j < constraints; j++) if (lam[j]) for (let a = 0; a < nx; a++) z[a] += lam[j] * M[j * nx + a];
  // x = L⁻ᵀ z
  const x = new Float64Array(nx);
  for (let a = nx - 1; a >= 0; a--) { let s = z[a]; for (let b = a + 1; b < nx; b++) s -= L[b * nx + a] * x[b]; x[a] = s / L[a * nx + a]; }
  return x;
}
