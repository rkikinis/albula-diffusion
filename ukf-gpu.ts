// UKF TRACTOGRAPHY ON THE GRAPHICS CARD -- the two-tensor free-water filter of ukf.ts (read its header: the port of
// UKFTractography and its license notice, which apply here too), run as one WORKGROUP OF 64 THREADS PER HALF FIBER.
// Ron, 2026-09-29: "GPU and parallelize as much as possible." Design: Contents/docs/ukf-review-2026-09-29.md.
//
// Every half fiber advances in lockstep inside its workgroup; the 64 threads split each step's work:
//   - the signal at the current point (Gaussian 3×3×3 average): threads over gradients;
//   - Cholesky of P (column by column), the 23 sigma points, the constraint projection of each (one thread per point),
//     the direction normalization (F);
//   - the predicted signal of every sigma point, its mean, the state-signal cross-covariance, the innovation and
//     Ht = Yk·Pxz: threads over gradients (each holds its gradient's 23 predictions);
//   - W = Yk + R·Ht·Htᵀ and i: threads over the 66 entries of the symmetric matrix;
//   - the two 11×11 inversions: Gauss-Jordan, rows in parallel (the matrices are symmetric positive, no pivoting);
//   - the tensor ordering, the step and the stops (Step2T / Follow2T): one thread, then broadcast.
// K steps per dispatch; the host re-dispatches until every fiber of the batch has stopped (short command buffers also
// keep clear of macOS's GPU watchdog).
//
// SAME FILTER, SCALED UNITS: the eigenvalues are carried in 1e-3 mm²/s here (1.7, not the original's 1700 in 1e-6),
// with Ql and p0 scaled to match, because single precision needs the smaller spread (the covariance's condition number
// is ~40 in these units, ~1e6 in the original's). The filter itself is unchanged by a linear change of coordinates --
// BUT NOT the sigma-point projection, whose metric is the covariance: it has to be scaled back explicitly (metricScale
// below; critic, 2026-09-30, finding 1 -- without it the card solved a different projection and its fibers parted from
// the processor's). The signal is stored in 32 bits when the device allows a buffer that large, as rounded 16-bit
// pairs otherwise, cropped to the brain's bounding box. The constraint projection uses the INVERSE of its metric (the covariance for a sigma point,
// P_new for the state -- both already at hand): y = V[:,A]·V[A,A]⁻¹·(bound − x)_A, which equals the active-set solve of
// ukf.ts. The free-water model (state of 11, the default) or the plain two-tensor model (state of 10, `freeWater: false`;
// 2026-10-03, for the training-conditions check: the ORG atlas behind the tract names was tracked with it). The plain
// model as ukf.ts has it from the original: w absent (1 in the signal), no constraint projection, every eigenvalue
// floored at 100 in the original's 1e-6 units (0.1 here) in F, H and the step. At most 192 gradients.

import type { UkfData, UkfOptions } from "./ukf.ts";
import { seedTensor, signalAt } from "./ukf.ts";

// The STATE LAYOUT keeps the free-water model's slots for both models (x at 0, P at 11, P⁻¹ at 132); the plain model's
// 10×10 matrices use the first 100 floats of their slot.
const STRIDE = 264, MAXG = 192;
/** The plain model's eigenvalue floor in the card's 1e-3 units (ukf.ts LAMBDA_MIN_PLAIN = 100 in 1e-6). */
const LMIN_PLAIN = 0.1;

/** The shader; `sig32`: the signal as 32-bit floats (the processor's precision) or as rounded 16-bit pairs. */
export const wgsl = (sig32: boolean, chol1 = false, pre = false, wgInv = false, onePass = false, parInv = false, fw = true) => /* wgsl */ `
const N: u32 = ${fw ? 11 : 10}u;
const S: u32 = ${fw ? 23 : 21}u;
const NN: u32 = ${fw ? 121 : 100}u;
const LMIN: f32 = ${fw ? "0.0" : LMIN_PLAIN.toFixed(1)};   // eigenvalue floor in the signal model and the step (plain model only)
const MAXG: u32 = ${MAXG}u;
const STRIDE: u32 = ${STRIDE}u;
struct Params {
  dims: vec3<u32>, G: u32,            // cropped grid
  origin: vec3<i32>, Gp: u32,         // crop origin (full-grid voxel), gradients padded to even
  voxel: vec3<f32>, sigma: f32,
  step: f32, R: f32, kappa: f32, stopFA: f32,
  stopSignal: f32, maxSteps: u32, perRecord: u32, cap: u32,
  count: u32, K: u32, qm: f32, ql: f32,
  qw: f32, _p0: f32, _p1: f32, _p2: f32,
};
@group(0) @binding(0) var<uniform> P_: Params;
@group(0) @binding(1) var<storage, read> sig: array<${sig32 ? "f32" : "u32"}>;   // voxel-major (cropped): 32-bit, or f16 pairs
@group(0) @binding(2) var<storage, read> mask: array<u32>;         // one u32 per cropped voxel
@group(0) @binding(3) var<storage, read> grad: array<vec4<f32>>;   // x, y, z (voxel frame), b
@group(0) @binding(4) var<storage, read_write> st: array<f32>;     // per half fiber: see STATE LAYOUT
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;   // per half fiber: cap × (x, y, z, fa, w)
@group(0) @binding(6) var<storage, read_write> alive: atomic<u32>;

// STATE LAYOUT (floats): 0..10 x | 11..131 P | 132..252 Pi (= P⁻¹) | 253..255 pos | 256..258 m1 | 259 stepnr |
// 260 alive (1/0) | 261 records | 262 fa | 263 spare
var<workgroup> X: array<f32, 253>;     // sigma points, s-major
var<workgroup> L: array<f32, 121>;
var<workgroup> A: array<f32, 121>;     // work matrix: Pn -> Yk
var<workgroup> Wm: array<f32, 121>;    // W = Yk + I -> (kept) and its inverse goes to B
var<workgroup> B: array<f32, 121>;     // P_new
var<workgroup> Pc: array<f32, 121>;    // P (current)
var<workgroup> Pic: array<f32, 121>;   // P⁻¹ (current)
var<workgroup> Ht: array<f32, ${11 * MAXG}>;
var<workgroup> innov: array<f32, ${MAXG}>;
var<workgroup> z: array<f32, ${MAXG}>;
var<workgroup> xh: array<f32, 11>;
var<workgroup> yh: array<f32, 11>;
var<workgroup> iv: array<f32, 11>;
var<workgroup> xs: array<f32, 11>;     // the state
var<workgroup> red: array<f32, 64>;
// Per sigma point, what the signal model needs, computed once a step instead of once per gradient (prePredict):
// the two unit directions with the free-water weight, and the four eigenvalues floored at zero.
var<workgroup> spA: array<vec4<f32>, 23>;   // m1 (unit), w
var<workgroup> spB: array<vec4<f32>, 23>;   // m2 (unit), unused
var<workgroup> spL: array<vec4<f32>, 23>;   // l1a, l1p, l2a, l2p
${wgInv ? `// invertA's scratch in workgroup memory, not in the one thread's private memory (wgInv): two arrays idle while it runs
// -- L (the Cholesky factor, used up by the sigma points; the swap below rewrites it) and B (P_new, written only after
// the second inversion) -- since two more arrays would pass the standard 16 KB of workgroup memory.` : ""}
var<workgroup> flag: array<u32, 4>;    // 0 alive, 1 swap, 2 record, 3 fail
var<workgroup> pos: vec3<f32>;
var<workgroup> m1: vec3<f32>;
var<workgroup> fa: f32;

fn wgt(s: u32) -> f32 { let n = f32(N); if (s == 0u) { return P_.kappa / (n + P_.kappa); } return 0.5 / (n + P_.kappa); }

fn sigAt(v: u32, q: u32) -> f32 {
  ${sig32 ? "return sig[v * P_.Gp + q];" : `let idx = v * P_.Gp + q;
  let pr = unpack2x16float(sig[idx >> 1u]);
  return select(pr.x, pr.y, (idx & 1u) == 1u);`}
}

// Predicted normalized signal of state row s (in X, as Xd + xh when centered = true) for gradient q.
fn predict(xv: array<f32, 11>, q: u32) -> f32 {
  let g = grad[q];
  let bq = g.w * 1e-3;                         // eigenvalues in 1e-3 mm²/s
  let m1v = normalize(vec3<f32>(xv[0], xv[1], xv[2]));
  let m2v = normalize(vec3<f32>(xv[5], xv[6], xv[7]));
  let c1 = dot(g.xyz, m1v); let c2 = dot(g.xyz, m2v);
  let l1a = max(xv[3], LMIN); let l1p = max(xv[4], LMIN); let l2a = max(xv[8], LMIN); let l2p = max(xv[9], LMIN);
  let w = ${fw ? "clamp(xv[10], 0.0, 1.0)" : "1.0"};
  let t = 0.5 * exp(-bq * (l1p + (l1a - l1p) * c1 * c1)) + 0.5 * exp(-bq * (l2p + (l2a - l2p) * c2 * c2));
  return w * t + (1.0 - w) * exp(-g.w * 0.003);
}

// The same model from the per-sigma-point values (spA, spB, spL): the same operations in the same order as predict.
// The gradient and the free-water term (which does not depend on the sigma point) come from the caller, once per
// gradient instead of once per sigma point -- the same values (2026-10-01 night).
fn predictPre(t: u32, g: vec4<f32>, fw: f32) -> f32 {
  let bq = g.w * 1e-3;
  let a = spA[t]; let b = spB[t]; let l = spL[t];
  let c1 = dot(g.xyz, a.xyz); let c2 = dot(g.xyz, b.xyz);
  let w = a.w;
  let tt = 0.5 * exp(-bq * (l.y + (l.x - l.y) * c1 * c1)) + 0.5 * exp(-bq * (l.w + (l.z - l.w) * c2 * c2));
  return w * tt + (1.0 - w) * fw;
}

// SYMMETRIC POSITIVE INVERSE THROUGH CHOLESKY (A <- A⁻¹), as the processor's spdInverse (ukf.ts) -- one thread, in its
// own memory (11×11: a few hundred multiplications), the others wait. It replaced an in-place Gauss-Jordan without
// pivoting (2026-09-30) as the standard stable method for symmetric positive matrices; the card/processor difference
// then being chased turned out to be the projection's metric (metricScale), not the inversion. All 64 threads must call it.
${parInv ? `fn invertA(lid: u32) {
  // IN PARALLEL (parInv, 2026-10-01 night): the factor in one thread (as below), then one column of its inverse per
  // thread and the product's entries over all threads -- each entry computed by the same operations in the same order
  // as the one-thread version, while 63 threads no longer wait through two 11×11 inversions a step.
  if (lid == 0u) {
    for (var j = 0u; j < N; j++) {
      var s = A[j * N + j];
      for (var k = 0u; k < j; k++) { s -= L[j * N + k] * L[j * N + k]; }
      let d = sqrt(max(s, 1e-30));
      L[j * N + j] = d;
      for (var i = j + 1u; i < N; i++) {
        var t = A[i * N + j];
        for (var k = 0u; k < j; k++) { t -= L[i * N + k] * L[j * N + k]; }
        L[i * N + j] = t / d;
      }
    }
  }
  workgroupBarrier();
  if (lid < N) {
    let j = lid;
    B[j * N + j] = 1.0 / L[j * N + j];
    for (var i = j + 1u; i < N; i++) {
      var s = 0.0;
      for (var k = j; k < i; k++) { s -= L[i * N + k] * B[k * N + j]; }
      B[i * N + j] = s / L[i * N + i];
    }
  }
  workgroupBarrier();
  for (var e = lid; e < NN; e += 64u) {
    let i = e / N; let j = e % N;
    if (j <= i) {
      var s = 0.0;
      for (var k = i; k < N; k++) { s += B[k * N + i] * B[k * N + j]; }
      A[i * N + j] = s; A[j * N + i] = s;
    }
  }
  workgroupBarrier();
}` : `fn invertA(lid: u32) {
  if (lid == 0u) {
    ${wgInv ? "" : `var Lf: array<f32, 121>;
    var Li: array<f32, 121>;`}
    for (var j = 0u; j < N; j++) {
      var s = A[j * N + j];
      for (var k = 0u; k < j; k++) { s -= ${wgInv ? "L[" : "Lf["}j * N + k] * ${wgInv ? "L[" : "Lf["}j * N + k]; }
      let d = sqrt(max(s, 1e-30));
      ${wgInv ? "L[" : "Lf["}j * N + j] = d;
      for (var i = j + 1u; i < N; i++) {
        var t = A[i * N + j];
        for (var k = 0u; k < j; k++) { t -= ${wgInv ? "L[" : "Lf["}i * N + k] * ${wgInv ? "L[" : "Lf["}j * N + k]; }
        ${wgInv ? "L[" : "Lf["}i * N + j] = t / d;
      }
    }
    for (var j = 0u; j < N; j++) {
      ${wgInv ? "B[" : "Li["}j * N + j] = 1.0 / ${wgInv ? "L[" : "Lf["}j * N + j];
      for (var i = j + 1u; i < N; i++) {
        var s = 0.0;
        for (var k = j; k < i; k++) { s -= ${wgInv ? "L[" : "Lf["}i * N + k] * ${wgInv ? "B[" : "Li["}k * N + j]; }
        ${wgInv ? "B[" : "Li["}i * N + j] = s / ${wgInv ? "L[" : "Lf["}i * N + i];
      }
    }
    for (var i = 0u; i < N; i++) {
      for (var j = 0u; j <= i; j++) {
        var s = 0.0;
        for (var k = i; k < N; k++) { s += ${wgInv ? "B[" : "Li["}k * N + i] * ${wgInv ? "B[" : "Li["}k * N + j]; }
        A[i * N + j] = s; A[j * N + i] = s;
      }
    }
  }
  workgroupBarrier();
}`}

// THE METRIC'S UNITS (critic, 2026-09-30, qa/2026-09-30-ukf-gpu-numerics.md, finding 1). The sigma points are projected in
// the metric P (the original's Constrain(X, p)), and a covariance used as a metric does NOT survive a change of units:
// P scales as D·P·D, a metric must scale as D⁻¹·M·D⁻¹. With the eigenvalues carried here in 1e-3 (not 1e-6) units, the
// processor's problem in these coordinates has the inverse metric V_ij = P⁻¹_ij·s_i·s_j, s = 1e-6 on the four eigenvalue
// indices. Without it the card solved a different projection: 94 of 1,443 steps differed by 0.01-1.5 degrees and 309 of
// 546 half fibers parted from the processor's (PAT16); with it, 0 and 73. The state's own projection (metric W, the
// information, useB) transforms correctly and is unchanged.
fn metricScale(i: u32) -> f32 { return select(1.0, 1e-6, i == 3u || i == 4u || i == 8u || i == 9u); }
fn vInv(i: u32, j: u32, useB: bool) -> f32 {
  let e = i * N + j;
  return select(Pic[e] * metricScale(i) * metricScale(j), B[e], useB);
}

// Project state vector v (private) onto the box constraints in the metric whose INVERSE is V (vInv).
// THE EXACT PROJECTION (2026-10-02; the processor's ukf.ts has the same): min (x' − x)ᵀW(x' − x) over the box, in the dual
// form -- with the bounds A held, x' = x + V[:,A] t where V[A,A] t = c (V = W⁻¹, c = bound − x on A), and t IS the vector
// of Lagrange multipliers: the answer is the minimizer exactly when every held lower bound has t ≥ 0 and every held upper
// bound t ≤ 0, and nothing free is out of its box. The grow-only active set below usually lands there; where it does not
// (it keeps a bound the minimizer releases: critic, 2026-10-01, finding 1), the 48 faces of the box are tried (the four
// eigenvalues free or at 0; w free, at 0 or at 1) and the feasible one of least objective, cᵀt, is kept.
fn solveFace(ai: array<u32, 5>, cv: array<f32, 5>, na: u32, useB: bool) -> array<f32, 5> {
  var M: array<f32, 25>;
  for (var a = 0u; a < na; a++) { for (var b = 0u; b < na; b++) { M[a * 5u + b] = vInv(ai[a], ai[b], useB); } }
  var t = cv;
  for (var p = 0u; p < na; p++) {
    let d = M[p * 5u + p];
    for (var r = p + 1u; r < na; r++) {
      let f = M[r * 5u + p] / d;
      for (var c2 = p; c2 < na; c2++) { M[r * 5u + c2] -= f * M[p * 5u + c2]; }
      t[r] -= f * t[p];
    }
  }
  for (var pp = 0u; pp < na; pp++) {
    let p = na - 1u - pp;
    var s = t[p];
    for (var c2 = p + 1u; c2 < na; c2++) { s -= M[p * 5u + c2] * t[c2]; }
    t[p] = s / M[p * 5u + p];
  }
  return t;
}

fn projectInv(v: ptr<function, array<f32, 11>>, useB: bool) -> bool {
  var lo = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
  var hi = array<f32, 5>(1e30, 1e30, 1e30, 1e30, 1.0);
  let idx = array<u32, 5>(3u, 4u, 8u, 9u, 10u);
  var any = false;
  for (var c = 0u; c < 5u; c++) { let x = (*v)[idx[c]]; if (x < lo[c] - 1e-7 || x > hi[c] + 1e-7) { any = true; } }
  if (!any) { return false; }
  let orig = *v;
  var act = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
  var bnd = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
  var tLast = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
  var cLast = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
  var naLast = 0u;
  var still = false;
  for (var round = 0u; round < 3u; round++) {
    var na = 0u;
    var ai = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
    var cvec = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
    for (var c = 0u; c < 5u; c++) {
      let x = (*v)[idx[c]];
      if (act[c] == 1u || x < lo[c] - 1e-7 || x > hi[c] + 1e-7) {
        if (act[c] == 0u) { act[c] = 1u; bnd[c] = clamp(x, lo[c], hi[c]); }
        ai[na] = idx[c]; cvec[na] = bnd[c] - orig[idx[c]]; cLast[na] = c; na++;
      }
    }
    let t = solveFace(ai, cvec, na, useB);
    tLast = t; naLast = na;
    for (var k = 0u; k < N; k++) {
      var y = 0.0;
      for (var a = 0u; a < na; a++) { y += vInv(k, ai[a], useB) * t[a]; }
      (*v)[k] = orig[k] + y;
    }
    still = false;
    for (var c = 0u; c < 5u; c++) { let x = (*v)[idx[c]]; if (act[c] == 0u && (x < lo[c] - 1e-7 || x > hi[c] + 1e-7)) { still = true; } }
    if (!still) { break; }
  }
  // Optimal? Every held bound's multiplier of the right sign (lower: t ≥ 0; upper: t ≤ 0), nothing free outside.
  var ok = !still;
  for (var a = 0u; a < naLast; a++) {
    let c = cLast[a];
    let tol = 1e-6 * (1.0 + abs(tLast[a]));
    if (bnd[c] == hi[c]) { if (tLast[a] > tol) { ok = false; } } else { if (tLast[a] < -tol) { ok = false; } }
  }
  if (!ok) {
    var bestObj = 1e30;
    var bestCode = 0xffffffffu;
    for (var code = 0u; code < 48u; code++) {
      var na = 0u;
      var ai = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
      var cvec = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
      var held = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
      for (var c = 0u; c < 4u; c++) { if (((code >> c) & 1u) == 1u) { held[c] = 1u; ai[na] = idx[c]; cvec[na] = lo[c] - orig[idx[c]]; na++; } }
      let wch = code / 16u;                                   // 0 free, 1 at 0, 2 at 1
      if (wch == 1u) { held[4] = 1u; ai[na] = idx[4]; cvec[na] = lo[4] - orig[idx[4]]; na++; }
      if (wch == 2u) { held[4] = 2u; ai[na] = idx[4]; cvec[na] = hi[4] - orig[idx[4]]; na++; }
      var t = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
      if (na > 0u) { t = solveFace(ai, cvec, na, useB); }
      var feasible = true;
      for (var c = 0u; c < 5u; c++) {
        if (held[c] != 0u) { continue; }
        var y = 0.0;
        for (var a = 0u; a < na; a++) { y += vInv(idx[c], ai[a], useB) * t[a]; }
        let x = orig[idx[c]] + y;
        if (x < lo[c] - 1e-7 || x > hi[c] + 1e-7) { feasible = false; }
      }
      if (!feasible) { continue; }
      var obj = 0.0;
      for (var a = 0u; a < na; a++) { obj += cvec[a] * t[a]; }
      if (obj < bestObj) { bestObj = obj; bestCode = code; }
    }
    if (bestCode != 0xffffffffu) {
      var na = 0u;
      var ai = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
      var cvec = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
      for (var c = 0u; c < 4u; c++) { if (((bestCode >> c) & 1u) == 1u) { ai[na] = idx[c]; cvec[na] = lo[c] - orig[idx[c]]; na++; } }
      let wch = bestCode / 16u;
      if (wch == 1u) { ai[na] = idx[4]; cvec[na] = lo[4] - orig[idx[4]]; na++; }
      if (wch == 2u) { ai[na] = idx[4]; cvec[na] = hi[4] - orig[idx[4]]; na++; }
      var t = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
      if (na > 0u) { t = solveFace(ai, cvec, na, useB); }
      for (var k = 0u; k < N; k++) {
        var y = 0.0;
        for (var a = 0u; a < na; a++) { y += vInv(k, ai[a], useB) * t[a]; }
        (*v)[k] = orig[k] + y;
      }
    }
  }
  for (var c = 0u; c < 5u; c++) { (*v)[idx[c]] = clamp((*v)[idx[c]], lo[c], hi[c]); }
  return true;
}

fn l2fa(a: f32, p: f32) -> f32 { let d = sqrt(a * a + 2.0 * p * p); if (d <= 0.0) { return 0.0; } return abs(a - p) / d; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let f = wg.x;
  if (f >= P_.count) { return; }
  let base = f * STRIDE;
  let G = P_.G;
  let n3 = P_.dims.x * P_.dims.y * P_.dims.z;
  // Load the fiber's state.
  if (lid == 0u) {
    flag[0] = select(0u, 1u, st[base + 260u] > 0.5);
    pos = vec3<f32>(st[base + 253u], st[base + 254u], st[base + 255u]);
    m1 = vec3<f32>(st[base + 256u], st[base + 257u], st[base + 258u]);
    fa = st[base + 262u];
  }
  for (var e = lid; e < NN; e += 64u) { Pc[e] = st[base + 11u + e]; Pic[e] = st[base + 132u + e]; }
  if (lid < N) { xs[lid] = st[base + lid]; }
  workgroupBarrier();
  if (workgroupUniformLoad(&flag[0]) == 0u) { return; }

  for (var kstep = 0u; kstep < P_.K; kstep++) {
    // ── the signal at pos (threads over gradients) ──
    let ci = vec3<i32>(round(pos)) ;
    for (var q = lid; q < G; q += 64u) {
      var s = 0.0; var ws = 1e-16;
      for (var a = -1; a <= 1; a++) { for (var b = -1; b <= 1; b++) { for (var c = -1; c <= 1; c++) {
        let vx = ci + vec3<i32>(a, b, c);
        let lc = vx - P_.origin;
        if (any(lc < vec3<i32>(0)) || any(lc >= vec3<i32>(P_.dims))) { continue; }
        let d = (vec3<f32>(vx) - pos) * P_.voxel;
        let w = exp(-dot(d, d) / P_.sigma);
        let v = (u32(lc.z) * P_.dims.y + u32(lc.y)) * P_.dims.x + u32(lc.x);
        s += w * sigAt(v, q); ws += w;
      } } }
      z[q] = s / ws;
    }
    // ── Cholesky of P ──
    for (var e = lid; e < NN; e += 64u) { L[e] = 0.0; }
    if (lid == 0u) { flag[3] = 0u; }
    workgroupBarrier();
    ${chol1 ? `// ONE THREAD, ONE BARRIER (2026-10-01, Safari benchmark): the same arithmetic in the same order as the column-by-column
    // form, which spent two workgroup barriers per column (22 a step) on a matrix of 11.
    if (lid == 0u) {
      for (var j = 0u; j < N; j++) {
        var s = Pc[j * N + j];
        for (var k = 0u; k < j; k++) { s -= L[j * N + k] * L[j * N + k]; }
        if (!(s > 0.0)) { flag[3] = 1u; s = 1e-12; }
        L[j * N + j] = sqrt(s);
        for (var i = j + 1u; i < N; i++) {
          var t = Pc[i * N + j];
          for (var k = 0u; k < j; k++) { t -= L[i * N + k] * L[j * N + k]; }
          L[i * N + j] = t / L[j * N + j];
        }
      }
    }
    workgroupBarrier();` : `for (var j = 0u; j < N; j++) {
      if (lid == 0u) {
        var s = Pc[j * N + j];
        for (var k = 0u; k < j; k++) { s -= L[j * N + k] * L[j * N + k]; }
        if (!(s > 0.0)) { flag[3] = 1u; s = 1e-12; }
        L[j * N + j] = sqrt(s);
      }
      workgroupBarrier();
      for (var i = j + 1u + lid; i < N; i += 64u) {
        var t = Pc[i * N + j];
        for (var k = 0u; k < j; k++) { t -= L[i * N + k] * L[j * N + k]; }
        L[i * N + j] = t / L[j * N + j];
      }
      workgroupBarrier();
    }`}
    // ── sigma points, projection (metric P, inverse Pi), F ──
    let sc = sqrt(f32(N) + P_.kappa);
    for (var e = lid; e < S * N; e += 64u) {
      let s = e / N; let k = e % N;
      var v = xs[k];
      if (s >= 1u && s <= N) { v += sc * L[k * N + (s - 1u)]; }
      if (s > N) { v -= sc * L[k * N + (s - 1u - N)]; }
      X[e] = v;
    }
    workgroupBarrier();
    if (lid < S) {
      var v: array<f32, 11>;
      for (var k = 0u; k < N; k++) { v[k] = X[lid * N + k]; }
      ${fw ? "_ = projectInv(&v, false);" : "// the plain model is unconstrained in the original: no projection, the floor below instead"}
      let a = normalize(vec3<f32>(v[0], v[1], v[2])); let b = normalize(vec3<f32>(v[5], v[6], v[7]));
      v[0] = a.x; v[1] = a.y; v[2] = a.z; v[5] = b.x; v[6] = b.y; v[7] = b.z;
      ${fw ? "" : "v[3] = max(v[3], LMIN); v[4] = max(v[4], LMIN); v[8] = max(v[8], LMIN); v[9] = max(v[9], LMIN);"}
      for (var k = 0u; k < N; k++) { X[lid * N + k] = v[k]; }
    }
    workgroupBarrier();
    // ── mean, centered points, Pn = Q + Σ w Xd Xdᵀ ──
    if (lid < N) { var s = 0.0; for (var t = 0u; t < S; t++) { s += wgt(t) * X[t * N + lid]; } xh[lid] = s; }
    workgroupBarrier();
    for (var e = lid; e < S * N; e += 64u) { X[e] -= xh[e % N]; }
    workgroupBarrier();
    for (var e = lid; e < NN; e += 64u) {
      let i = e / N; let j = e % N;
      var s = 0.0;
      for (var t = 0u; t < S; t++) { s += wgt(t) * X[t * N + i] * X[t * N + j]; }
      if (i == j) {
        if (i == 0u || i == 1u || i == 2u || i == 5u || i == 6u || i == 7u) { s += P_.qm; }
        else if (i == 10u) { s += P_.qw; } else { s += P_.ql; }   // i = 10 only in the free-water model
      }
      A[e] = s;
    }
    workgroupBarrier();
    invertA(lid);                                   // A <- Yk
    if (lid < N) { var s = 0.0; for (var k = 0u; k < N; k++) { s += A[lid * N + k] * xh[k]; } yh[lid] = s; }
    ${pre ? `// THE SIGMA POINTS' MODEL VALUES ONCE A STEP (2026-10-01 night, Safari benchmark): each of the 23 threads takes one.
    if (lid < S) {
      let t = lid;
      let m1v = normalize(vec3<f32>(X[t * N + 0u] + xh[0], X[t * N + 1u] + xh[1], X[t * N + 2u] + xh[2]));
      let m2v = normalize(vec3<f32>(X[t * N + 5u] + xh[5], X[t * N + 6u] + xh[6], X[t * N + 7u] + xh[7]));
      spA[t] = vec4<f32>(m1v, ${fw ? "clamp(X[t * N + 10u] + xh[10], 0.0, 1.0)" : "1.0"});
      spB[t] = vec4<f32>(m2v, 0.0);
      spL[t] = vec4<f32>(max(X[t * N + 3u] + xh[3], LMIN), max(X[t * N + 4u] + xh[4], LMIN), max(X[t * N + 8u] + xh[8], LMIN), max(X[t * N + 9u] + xh[9], LMIN));
    }` : ""}
    workgroupBarrier();
    // ── per gradient: predictions, mean, Pxz column, innovation, Ht column ──
    ${onePass ? `// ONE PASS, NO PER-THREAD ARRAYS (onePass, 2026-10-01 night: WebKit's Metal is slow with them). The cross-covariance
    // column Σ w·Xd·(Z − zm) equals Σ w·Xd·Z, since the weights sum to one and the points are centered (Σ w·Xd = 0), so
    // the predictions are not kept: they are summed as they come, into three vec4 accumulators.
    for (var q = lid; q < G; q += 64u) {
      let gq = grad[q]; let fwq = exp(-gq.w * 0.003);
      var zm = 0.0; var p0 = vec4<f32>(0.0); var p1 = vec4<f32>(0.0); var p2 = vec4<f32>(0.0);
      for (var t = 0u; t < S; t++) {
        let wz = wgt(t) * predictPre(t, gq, fwq);
        zm += wz;
        let b = t * N;
        p0 += wz * vec4<f32>(X[b], X[b + 1u], X[b + 2u], X[b + 3u]);
        p1 += wz * vec4<f32>(X[b + 4u], X[b + 5u], X[b + 6u], X[b + 7u]);
        p2 += wz * vec4<f32>(X[b + 8u], X[b + 9u], X[b + 10u], 0.0);
      }
      innov[q] = z[q] - zm + dot(p0, vec4<f32>(yh[0], yh[1], yh[2], yh[3])) + dot(p1, vec4<f32>(yh[4], yh[5], yh[6], yh[7])) + dot(p2, vec4<f32>(yh[8], yh[9], yh[10], 0.0));
      for (var i = 0u; i < N; i++) {
        let r = i * N;
        Ht[i * MAXG + q] = dot(vec4<f32>(A[r], A[r + 1u], A[r + 2u], A[r + 3u]), p0) + dot(vec4<f32>(A[r + 4u], A[r + 5u], A[r + 6u], A[r + 7u]), p1) + dot(vec4<f32>(A[r + 8u], A[r + 9u], A[r + 10u], 0.0), p2);
      }
    }` : `    for (var q = lid; q < G; q += 64u) {
      ${pre ? "let gq = grad[q]; let fwq = exp(-gq.w * 0.003);" : ""}
      var Zs: array<f32, 23>;
      var zm = 0.0;
      for (var t = 0u; t < S; t++) {
        ${pre ? `Zs[t] = predictPre(t, gq, fwq);` : `var xv: array<f32, 11>;
        for (var k = 0u; k < N; k++) { xv[k] = X[t * N + k] + xh[k]; }
        Zs[t] = predict(xv, q);`} zm += wgt(t) * Zs[t];
      }
      var pc: array<f32, 11>;
      for (var k = 0u; k < N; k++) { var s = 0.0; for (var t = 0u; t < S; t++) { s += wgt(t) * X[t * N + k] * (Zs[t] - zm); } pc[k] = s; }
      var inn = z[q] - zm;
      for (var k = 0u; k < N; k++) { inn += pc[k] * yh[k]; }
      innov[q] = inn;
      for (var i = 0u; i < N; i++) { var s = 0.0; for (var k = 0u; k < N; k++) { s += A[i * N + k] * pc[k]; } Ht[i * MAXG + q] = s; }
    }
`}
    workgroupBarrier();
    // ── W = Yk + R·Ht·Htᵀ, i = R·Ht·innov + yh ──
    for (var e = lid; e < NN; e += 64u) {
      let i = e / N; let j = e % N;
      var s = 0.0;
      for (var q = 0u; q < G; q++) { s += Ht[i * MAXG + q] * Ht[j * MAXG + q]; }
      Wm[e] = A[e] + P_.R * s;
    }
    if (lid < N) { var s = 0.0; for (var q = 0u; q < G; q++) { s += Ht[lid * MAXG + q] * innov[q]; } iv[lid] = P_.R * s + yh[lid]; }
    workgroupBarrier();
    for (var e = lid; e < NN; e += 64u) { A[e] = Wm[e]; }
    workgroupBarrier();
    invertA(lid);                                   // A <- P_new
    for (var e = lid; e < NN; e += 64u) { B[e] = A[e]; }
    workgroupBarrier();
    // ── the new state, projected in the metric W (its inverse is P_new = B); one thread does the fiber logic ──
    if (lid == 0u) {
      var v: array<f32, 11>;
      for (var i = 0u; i < N; i++) { var s = 0.0; for (var k = 0u; k < N; k++) { s += B[i * N + k] * iv[k]; } v[i] = s; }
      ${fw ? "_ = projectInv(&v, true);" : ""}
      // Step2T: orient both tensors to the previous direction, order them, move along the first.
      let old = m1;
      var a = normalize(vec3<f32>(v[0], v[1], v[2])); var c = normalize(vec3<f32>(v[5], v[6], v[7]));
      if (dot(a, old) < 0.0) { a = -a; } if (dot(c, old) < 0.0) { c = -c; }
      let angle = degrees(acos(clamp(dot(a, c), -1.0, 1.0)));
      var swap = dot(a, old) < dot(c, old);
      var fa1 = l2fa(max(v[3], LMIN), max(v[4], LMIN)); var fa2 = l2fa(max(v[8], LMIN), max(v[9], LMIN));
      if (swap) { let tt = fa1; fa1 = fa2; fa2 = tt; }
      var swap2 = false;
      if (angle <= 20.0 && min(fa1, fa2) <= 0.2 && !(fa1 > 0.2)) { swap2 = true; let tt = fa1; fa1 = fa2; fa2 = tt; }
      let doSwap = swap != swap2;
      if (doSwap) { for (var k = 0u; k < 5u; k++) { let tt = v[k]; v[k] = v[5u + k]; v[5u + k] = tt; } }
      let lead = select(a, c, swap);
      let dir = select(lead, select(c, a, swap), swap2);
      let la = max(v[3], LMIN); let lp = max(v[4], LMIN);
      fa = select(fa1, 0.0, la < lp);
      m1 = dir;
      pos = pos + dir / P_.voxel * P_.step;
      for (var k = 0u; k < N; k++) { xs[k] = v[k]; }
      flag[1] = select(0u, 1u, doSwap);
    }
    workgroupBarrier();
    // Swap the covariance blocks (and of its inverse) when the tensors were exchanged.
    if (workgroupUniformLoad(&flag[1]) == 1u) {
      for (var e = lid; e < NN; e += 64u) {
        let i = e / N; let j = e % N;
        let pi = select(select(i - 5u, i + 5u, i < 5u), i, i == 10u);
        let pj = select(select(j - 5u, j + 5u, j < 5u), j, j == 10u);
        L[e] = B[pi * N + pj]; A[e] = Wm[pi * N + pj];
      }
      workgroupBarrier();
      for (var e = lid; e < NN; e += 64u) { B[e] = L[e]; Wm[e] = A[e]; }
    }
    for (var e = lid; e < NN; e += 64u) { Pc[e] = B[e]; Pic[e] = Wm[e]; }
    // Predicted mean signal of the new state (threads over gradients).
    var part = 0.0;
    for (var q = lid; q < G; q += 64u) { var xv: array<f32, 11>; for (var k = 0u; k < N; k++) { xv[k] = xs[k]; } part += predict(xv, q); }
    red[lid] = part;
    workgroupBarrier();
    if (lid == 0u) {
      var mean = 0.0; for (var t = 0u; t < 64u; t++) { mean += red[t]; } mean /= f32(G);
      let stepnr = u32(st[base + 259u]) + 1u;
      st[base + 259u] = f32(stepnr);
      let ci2 = vec3<i32>(round(pos)) - P_.origin;
      var inBrain = false;
      if (all(ci2 >= vec3<i32>(0)) && all(ci2 < vec3<i32>(P_.dims))) { inBrain = mask[(u32(ci2.z) * P_.dims.y + u32(ci2.y)) * P_.dims.x + u32(ci2.x)] > 0u; }
      if (flag[3] == 1u || !inBrain || mean < P_.stopSignal || fa < P_.stopFA || stepnr > P_.maxSteps) {
        flag[0] = 0u;
      } else if ((stepnr + 1u) % P_.perRecord == 0u) {
        let r = u32(st[base + 261u]);
        if (r < P_.cap) {
          let o = (f * P_.cap + r) * 5u;
          outp[o] = pos.x; outp[o + 1u] = pos.y; outp[o + 2u] = pos.z; outp[o + 3u] = fa; outp[o + 4u] = ${fw ? "xs[10]" : "1.0"};
          st[base + 261u] = f32(r + 1u);
        } else { flag[0] = 0u; }
      }
    }
    workgroupBarrier();
    if (workgroupUniformLoad(&flag[0]) == 0u) { break; }
  }
  // Store the state back.
  for (var e = lid; e < NN; e += 64u) { st[base + 11u + e] = Pc[e]; st[base + 132u + e] = Pic[e]; }
  if (lid < N) { st[base + lid] = xs[lid]; }
  if (lid == 0u) {
    st[base + 253u] = pos.x; st[base + 254u] = pos.y; st[base + 255u] = pos.z;
    st[base + 256u] = m1.x; st[base + 257u] = m1.y; st[base + 258u] = m1.z;
    st[base + 260u] = f32(flag[0]); st[base + 262u] = fa;
    if (flag[0] == 1u) { atomicAdd(&alive, 1u); }
  }
}
`;

const packCache = new WeakMap<UkfData, Map<GPUDevice, { lo: number[]; cd: number[]; sig32: boolean; sigBuf: GPUBuffer; maskBuf: GPUBuffer; gBuf: GPUBuffer }>>();
const pipeCache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();
/** Steps per dispatch the card last handled within the time cap (see the dispatch loop). */
const stepMemo = new WeakMap<GPUDevice, number>();

export interface GpuUkfResult {
  fibers: { points: Float32Array; fa: Float32Array; freeWater: Float32Array; seed: number }[];
  seedsUsed: number; seedsRejected: number; dispatches: number;
  ms: { prepare: number; gpu: number; total: number };
  /** With `debugOneDispatch`: every half fiber's state (STATE LAYOUT) before and after the first dispatch. */
  debug?: { before: Float32Array[]; after: Float32Array[] };
}

/** Track from every seed (voxel coordinates) in both directions on the graphics card. */
export async function trackUkfGpu(device: GPUDevice, data: UkfData, seeds: number[][], opts: UkfOptions & { batch?: number; stepsPerDispatch?: number; /** Hold each dispatch near this many milliseconds (steps set from the last one's time; see the dispatch loop). */ targetMsPerDispatch?: number; /** Halve the steps after a dispatch longer than this (default 500 ms). */ maxMsPerDispatch?: number; /** The Cholesky in one thread (one barrier) instead of column by column (22). */ cholOneThread?: boolean; /** The signal model's per-sigma-point values computed once a step, not once per gradient. */ prePredict?: boolean; /** The inversions' scratch in workgroup memory. */ wgInverse?: boolean; /** The per-gradient sums in one pass, without per-thread arrays (needs prePredict). */ onePass?: boolean; /** The two inversions a step in parallel over the workgroup (needs wgInverse). */ parInverse?: boolean; /** Pack the signal and compile the shader every call, as before 2026-10-01 night (benchmarking only). */ noCache?: boolean; /** Checking only: stop after one dispatch and return the states (with stepsPerDispatch 1: one filter step). */ debugOneDispatch?: boolean } = {}): Promise<GpuUkfResult> {
  const t0 = performance.now();
  const fw = opts.freeWater ?? true, N = fw ? 11 : 10;
  if (!fw && opts.onePass) throw new Error("onePass is written for the free-water model's 11 state values");
  const G = data.G;
  if (G > MAXG) throw new Error(`${G} gradients: the graphics-card UKF holds at most ${MAXG}`);
  const step = opts.stepLength ?? 0.3, perRecord = Math.max(1, Math.round((opts.recordLength ?? 0.9) / step));
  const maxSteps = Math.ceil((opts.maxHalfFiberLength ?? 250) / step), cap = Math.ceil(maxSteps / perRecord) + 2;
  const sigma = opts.sigmaSignal ?? Math.min(...data.voxel), p0 = opts.p0 ?? 0.01, seedFA = opts.seedingThreshold ?? 0.18;
  const [nx, ny, nz] = data.dims;
  // THE PACKED SIGNAL AND THE COMPILED SHADER, ONCE PER SCAN AND DEVICE (2026-10-01 night): a whole brain is tracked in 13
  // batches, and each batch packed the whole signal again and compiled the shader again -- on the processor, and in
  // WebKit about three times slower than in Chrome (the "seeds" part of the status line: 5.3 s against 1.7 s).
  const packed = opts.noCache ? undefined : packCache.get(data)?.get(device);
  const { lo, cd, sig32, sigBuf, maskBuf, gBuf } = packed ?? pack();
  if (!packed && !opts.noCache) { const m = packCache.get(data) ?? new Map(); m.set(device, { lo, cd, sig32, sigBuf, maskBuf, gBuf }); packCache.set(data, m); }
  const Gp = G + (G & 1);
  function pack() {
  // Crop to the mask's bounding box (+1 voxel for the 3×3×3 average), pack the signal as f16 pairs.
  let lo = [nx, ny, nz], hi = [-1, -1, -1];
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (data.mask[(k * ny + j) * nx + i]) {
    lo = [Math.min(lo[0], i), Math.min(lo[1], j), Math.min(lo[2], k)]; hi = [Math.max(hi[0], i), Math.max(hi[1], j), Math.max(hi[2], k)];
  }
  lo = lo.map((v) => Math.max(0, v - 1)); hi = hi.map((v, a) => Math.min(data.dims[a] - 1, v + 1));
  const cd = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1], cn = cd[0] * cd[1] * cd[2], Gp = G + (G & 1);
  // THE SIGNAL IN 32 BITS WHEN THE DEVICE ALLOWS IT (critic, 2026-09-30, finding 3: after the metric fix, the 16-bit signal
  // was the largest remaining difference from the processor -- 73 partings of 546 half fibers against 34 in 32 bits).
  // A device whose single storage buffer cannot hold it gets the rounded 16-bit copy, half the size.
  const sig32 = cn * Gp * 4 <= device.limits.maxStorageBufferBindingSize && cn * Gp * 4 <= device.limits.maxBufferSize;
  const f16 = new Uint16Array(sig32 ? 0 : cn * Gp), f32s = new Float32Array(sig32 ? cn * Gp : 0), cmask = new Uint32Array(cn);
  const toHalf = (() => { const f = new Float32Array(1), u = new Uint32Array(f.buffer); return (x: number) => {
    f[0] = x; const b = u[0], s = (b >>> 16) & 0x8000, e = ((b >>> 23) & 0xff) - 127 + 15, m = b & 0x7fffff;
    // ROUNDED TO NEAREST (2026-09-30; it truncated, a small downward bias on every value). A carry into the exponent is
    // the next half-float up, which the plain addition gives.
    if (e <= 0) return s; if (e >= 31) return s | 0x7c00; const h = (e << 10) | (m >>> 13), rest = m & 0x1fff;
    return s | Math.min(0x7c00, h + (rest > 0x1000 || (rest === 0x1000 && (h & 1)) ? 1 : 0)); }; })();
  for (let k = 0; k < cd[2]; k++) for (let j = 0; j < cd[1]; j++) for (let i = 0; i < cd[0]; i++) {
    const c = (k * cd[1] + j) * cd[0] + i, v = ((k + lo[2]) * ny + (j + lo[1])) * nx + (i + lo[0]);
    cmask[c] = data.mask[v];
    if (sig32) for (let q = 0; q < G; q++) f32s[c * Gp + q] = data.signal[v * G + q];
    else for (let q = 0; q < G; q++) f16[c * Gp + q] = toHalf(data.signal[v * G + q]);
  }
  const grads = new Float32Array(4 * G);
  for (let q = 0; q < G; q++) grads.set([data.g[3 * q], data.g[3 * q + 1], data.g[3 * q + 2], data.b[q]], 4 * q);
  const mk0 = (d: ArrayBufferView, usage: number) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(d.byteLength / 4) * 4), usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, d.buffer, d.byteOffset, d.byteLength); return b; };
  return { lo, cd, sig32, sigBuf: mk0(sig32 ? f32s : new Uint32Array(f16.buffer), GPUBufferUsage.STORAGE), maskBuf: mk0(cmask, GPUBufferUsage.STORAGE), gBuf: mk0(grads, GPUBufferUsage.STORAGE) };
  }

  // Seed states on the processor (the original's UnpackTensor): both directions, eigenvalues in 1e-3 mm²/s.
  const z = new Float64Array(G), halves: { seed: number; state: Float32Array }[] = [];
  let used = 0, rejected = 0;
  const l2fa3 = (a: number, b: number, c: number) => Math.sqrt(0.5 * ((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2) / (a * a + b * b + c * c)) || 0;
  seeds.forEach((sd, idx) => {
    signalAt(data, sd, sigma, z);
    for (let q = 0; q < G; q++) if (!(z[q] >= 0) || !Number.isFinite(z[q])) { rejected++; return; }
    const t = seedTensor(data, z);
    // The seed's FA from the major eigenvalue and the mean of the minor two, as the original (ukf.ts, the same rule).
    const lm = t ? (t.l[1] + t.l[2]) / 2 : 0;
    if (!t || !(l2fa3(t.l[0], lm, lm) > seedFA)) { rejected++; return; }
    used++;
    const la = t.l[0] * 1e-3, lp = lm * 1e-3, fa = l2fa3(t.l[0], lm, lm);
    for (const sgn of [1, -1]) {
      const s = new Float32Array(STRIDE), m = t.m.map((v) => sgn * v);
      s.set([m[0], m[1], m[2], la, lp, t.m[0], t.m[1], t.m[2], la, lp, ...(fw ? [1] : [])], 0);
      // P0 = p0·I in the original's units; the eigenvalue coordinates scale by 1e-3, their variance by 1e-6.
      for (let k = 0; k < N; k++) { const pk = [3, 4, 8, 9].includes(k) ? p0 * 1e-6 : p0; s[11 + k * N + k] = pk; s[132 + k * N + k] = 1 / pk; }
      s.set(sd, 253); s.set(m, 256); s[259] = 0; s[260] = 1; s[261] = 1; s[262] = fa;
      halves.push({ seed: idx, state: s });
    }
  });

  const mk = (d: ArrayBufferView, usage: number) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(d.byteLength / 4) * 4), usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, d.buffer, d.byteOffset, d.byteLength); return b; };
  // THE DEFAULTS, from the Safari benchmark (2026-10-01 night; WebKit runs Albula's window): the signal model's values
  // once a step (its free-water term once a gradient), the inversions' scratch in workgroup memory, and the inversions in
  // parallel -- the same fibers bit for bit in Deno, the card about half the time in WebKit (PAT16's 2,000 reference
  // seeds: 2.68-2.98 s -> 1.40 s). One pass (1.04 s) changes the fibers at rounding level and stays off until it is
  // checked on more cases.
  const pre = (opts.prePredict ?? true) || !!opts.onePass, wgInv = opts.wgInverse ?? true;
  // The two inversions a step in parallel (2026-10-01 night, Safari: 2.01-2.07 s -> 1.40 s; the same fibers bit for bit
  // in Deno): on with the workgroup-memory scratch it needs.
  const parInv = wgInv && (opts.parInverse ?? true);
  const pkey = `${sig32}|${opts.cholOneThread ? 1 : 0}|${pre ? 1 : 0}|${wgInv ? 1 : 0}|${opts.onePass ? 1 : 0}|${parInv ? 1 : 0}|${fw ? 1 : 0}`;
  let pipeline = opts.noCache ? undefined : pipeCache.get(device)?.get(pkey);
  if (!pipeline) {
    pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: wgsl(sig32, !!opts.cholOneThread, pre, wgInv, !!opts.onePass, parInv, fw) }), entryPoint: "main" } });
    if (!opts.noCache) { const m = pipeCache.get(device) ?? new Map(); m.set(pkey, pipeline); pipeCache.set(device, m); }
  }
  const tPrep = performance.now();

  const batch = opts.batch ?? 4096, K = opts.stepsPerDispatch ?? 16;
  const R = 2 / (opts.Rs ?? 0.02), out: GpuUkfResult["fibers"] = [];
  const debug = { before: [] as Float32Array[], after: [] as Float32Array[] };
  const recorded: { pts: Float32Array; fa: Float32Array; w: Float32Array }[] = new Array(halves.length);
  let dispatches = 0;
  for (let b0 = 0; b0 < halves.length; b0 += batch) {
    const hs = halves.slice(b0, b0 + batch), count = hs.length;
    const stArr = new Float32Array(count * STRIDE);
    hs.forEach((h, i) => stArr.set(h.state, i * STRIDE));
    const outArr = new Float32Array(count * cap * 5);
    hs.forEach((h, i) => outArr.set([h.state[253], h.state[254], h.state[255], h.state[262], 1], i * cap * 5));   // the seed point
    const stBuf = mk(stArr, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC), outBuf = mk(outArr, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const aliveBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const aliveRead = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const params = new ArrayBuffer(128), dv = new DataView(params);
    const u32 = (o: number, v: number) => dv.setUint32(o, v, true), i32 = (o: number, v: number) => dv.setInt32(o, v, true), f32 = (o: number, v: number) => dv.setFloat32(o, v, true);
    u32(0, cd[0]); u32(4, cd[1]); u32(8, cd[2]); u32(12, G);
    i32(16, lo[0]); i32(20, lo[1]); i32(24, lo[2]); u32(28, Gp);
    f32(32, data.voxel[0]); f32(36, data.voxel[1]); f32(40, data.voxel[2]); f32(44, sigma);
    f32(48, step); f32(52, R); f32(56, 0.01); f32(60, opts.stoppingFA ?? 0.15);
    f32(64, opts.stoppingThreshold ?? 0.1); u32(68, maxSteps); u32(72, perRecord); u32(76, cap);
    u32(80, count); u32(84, K); f32(88, opts.Qm ?? 0.001); f32(92, (opts.Ql ?? 50) * 1e-6);
    f32(96, opts.Qw ?? 0.0015);
    const pBuf = mk(new Uint8Array(params), GPUBufferUsage.UNIFORM);
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [pBuf, sigBuf, maskBuf, gBuf, stBuf, outBuf, aliveBuf].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    // STEPS PER DISPATCH, held to a time (2026-10-03): Ron's window lost its card when tracking ran beside the 3D view's
    // solid anatomy -- macOS's watchdog aborted a command buffer ("Impacting Interactivity", the system log at the crash;
    // as on 2026-09-23). With `targetMsPerDispatch` each dispatch is timed and the next one's steps set to stay near it:
    // short while every fiber is alive, longer as they stop. The steps a dispatch takes do not change the fibers (the
    // same steps run either way; checked bit for bit, planning.ts trackUkfSeeds).
    const target = opts.targetMsPerDispatch, capMs = opts.maxMsPerDispatch ?? 500;
    // A SLOWER CARD (critic, 2026-10-03, finding 5): each batch starts at 16 steps (its first dispatch is its heaviest:
    // every fiber alive) and doubles up to K; a dispatch that took longer than capMs halves the next. On this M1 Max
    // nothing changes but a few more dispatches; on a smaller card the dispatches stay short of the watchdog.
    // The step count the card last handled is remembered per device, so only the first call ramps (a fast card), and a
    // slow card stays short on every call.
    let k = target ? Math.min(K, 2) : Math.min(K, stepMemo.get(device) ?? 16), stepsDone = 0;
    for (; stepsDone <= maxSteps + k; ) {
      device.queue.writeBuffer(pBuf, 84, Uint32Array.of(k));
      const ts = performance.now();
      const enc = device.createCommandEncoder();
      enc.clearBuffer(aliveBuf);
      const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(count); pass.end();
      enc.copyBufferToBuffer(aliveBuf, 0, aliveRead, 0, 4);
      device.queue.submit([enc.finish()]); dispatches++;
      await aliveRead.mapAsync(GPUMapMode.READ);
      const n = new Uint32Array(aliveRead.getMappedRange().slice(0))[0];
      aliveRead.unmap();
      stepsDone += k;
      if (n === 0 || opts.debugOneDispatch) break;
      const ms = performance.now() - ts;
      if (target) k = Math.max(1, Math.min(256, Math.round(k * Math.min(2, Math.max(0.25, target / Math.max(ms, 0.5))))));
      else if (ms > capMs) k = Math.max(4, k >> 1);
      else if (ms < capMs / 4 && k < K) k = Math.min(K, k * 2);
      if (!target) stepMemo.set(device, k);
    }
    // Read back the recorded points and the record counts.
    const read = async (buf: GPUBuffer, size: number) => { const r = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); const e = device.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, r, 0, size); device.queue.submit([e.finish()]); await r.mapAsync(GPUMapMode.READ); const a = new Float32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return a; };
    const stOut = await read(stBuf, count * STRIDE * 4), pts = await read(outBuf, count * cap * 5 * 4);
    if (opts.debugOneDispatch) { debug.before.push(...hs.map((h) => h.state)); debug.after.push(...hs.map((_, i) => stOut.slice(i * STRIDE, (i + 1) * STRIDE))); }
    hs.forEach((_, i) => {
      const nrec = Math.max(1, Math.round(stOut[i * STRIDE + 261]));
      const P = new Float32Array(3 * nrec), FA = new Float32Array(nrec), W = new Float32Array(nrec);
      for (let r = 0; r < nrec; r++) { const o = (i * cap + r) * 5; P[3 * r] = pts[o]; P[3 * r + 1] = pts[o + 1]; P[3 * r + 2] = pts[o + 2]; FA[r] = pts[o + 3]; W[r] = pts[o + 4]; }
      recorded[b0 + i] = { pts: P, fa: FA, w: W };
    });
    for (const b of [stBuf, outBuf, aliveBuf, aliveRead, pBuf]) b.destroy();
  }
  const tGpu = performance.now();
  // Join the two halves of every seed (backward reversed, then forward) and turn voxel positions into patient RAS.
  const M = data.ijkToRAS;
  for (let h = 0; h + 1 < halves.length; h += 2) {
    const fwd = recorded[h], bwd = recorded[h + 1], nb = bwd.fa.length - 1, nf = fwd.fa.length, n = nb + nf;
    const points = new Float32Array(3 * n), faA = new Float32Array(n), wA = new Float32Array(n);
    const put = (dst: number, src: Float32Array, si: number) => { const x = src[3 * si], y = src[3 * si + 1], zz = src[3 * si + 2]; points[3 * dst] = M[0] * x + M[1] * y + M[2] * zz + M[3]; points[3 * dst + 1] = M[4] * x + M[5] * y + M[6] * zz + M[7]; points[3 * dst + 2] = M[8] * x + M[9] * y + M[10] * zz + M[11]; };
    for (let r = 0; r < nb; r++) { const si = nb - r; put(r, bwd.pts, si); faA[r] = bwd.fa[si]; wA[r] = bwd.w[si]; }
    for (let r = 0; r < nf; r++) { put(nb + r, fwd.pts, r); faA[nb + r] = fwd.fa[r]; wA[nb + r] = fwd.w[r]; }
    out.push({ points, fa: faA, freeWater: wA, seed: halves[h].seed });
  }
  if (opts.noCache) for (const b of [sigBuf, maskBuf, gBuf]) b.destroy();   // cached ones live as long as the scan's data
  return { fibers: out, seedsUsed: used, seedsRejected: rejected, dispatches, ms: { prepare: tPrep - t0, gpu: tGpu - tPrep, total: performance.now() - t0 }, ...(opts.debugOneDispatch ? { debug } : {}) };
}
