// UKF TRACTOGRAPHY ON THE GRAPHICS CARD, LAID OUT AFTER MICHAEL HALLE'S KERNEL (tractline's ukf_metal.py, Apache-2.0:
// "LPF lanes of a SIMD group per half-fiber (default 8: four half-fibers to a SIMD group). Every lane carries the
// filter's small algebra ... redundantly; the gradients are split across the fiber's lanes"). SPEED-2026-10-28.md in the
// workspace, B1 (Ron, 2026-10-28: "do all"). The same filter as ukf-gpu.ts's plain two-tensor model (read its header:
// the port of UKFTractography, its license notice, its units) -- the same operations, but summed in another order, so the
// fibers differ at rounding level: a new tracking version, not a silent replacement.
//
// THE LAYOUT. A workgroup of 32 threads holds four half fibers, eight lanes each. Every lane keeps the whole filter state
// in its own registers -- the state, the covariance, the 21 sigma points, the two inverses -- and does the small algebra
// itself (the same numbers in every lane of a fiber); only the work over gradients is split: lane l takes gradients l,
// l + 8, .... Each gradient's contribution (its row of Ht, its innovation) is added straight into the lane's own sums of
// W and i, never stored (as Mike's kernel). The eight lanes' sums meet in workgroup memory, added in lane order (fixed, so
// the result does not depend on timing). Deno's WebGPU has no subgroup operations on this Mac (checked 2026-10-28:
// "subgroups" is not among the adapter's features), so the meeting is a workgroup barrier instead of Mike's shuffles.
// The four fibers of a group run until the last of them stops; a stopped fiber's lanes wait at the barriers.

import type { UkfData, UkfOptions } from "./ukf.ts";
import { seedTensor, signalAt, ukfLengths, ukfSigma } from "./ukf.ts";
import type { GpuUkfResult } from "./ukf-gpu.ts";

const STRIDE = 128, MAXG = 192, LPF = 8, FIBERS = 4;
/** Gradients a lane holds at most. */
const PER_LANE = MAXG / LPF;
const LMIN_PLAIN = 0.1;

// STATE LAYOUT per half fiber (floats): 0..9 x | 10..109 P | 110..112 pos | 113..115 m1 | 116 stepnr | 117 alive |
// 118 records | 119 fa
export const wgslLanes = (sig32: boolean, FIBERS = 4, LPF = 8) => /* wgsl */ `
const N: u32 = 10u;
const S: u32 = 21u;
const LPF: u32 = ${LPF}u;
const LMIN: f32 = ${LMIN_PLAIN.toFixed(1)};
const STRIDE: u32 = ${STRIDE}u;
struct Params {
  dims: vec3<u32>, G: u32,
  origin: vec3<i32>, Gp: u32,
  voxel: vec3<f32>, sigma: f32,
  step: f32, R: f32, kappa: f32, stopFA: f32,
  stopSignal: f32, maxSteps: u32, perRecord: u32, cap: u32,
  count: u32, K: u32, qm: f32, ql: f32,
};
@group(0) @binding(0) var<uniform> P_: Params;
@group(0) @binding(1) var<storage, read> sig: array<${sig32 ? "f32" : "u32"}>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<storage, read> grad: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> st: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;
@group(0) @binding(6) var<storage, read_write> alive: atomic<u32>;

// The lanes' partial sums: per thread the 55 entries of W's lower triangle, the 10 of i, and one spare.
var<workgroup> red: array<f32, ${FIBERS * LPF * 66}>;
var<workgroup> live: array<u32, ${FIBERS}>;
var<workgroup> liveAny: u32;

fn wgt(s: u32) -> f32 { let n = f32(N); if (s == 0u) { return P_.kappa / (n + P_.kappa); } return 0.5 / (n + P_.kappa); }

fn sigAt(v: u32, q: u32) -> f32 {
  ${sig32 ? "return sig[v * P_.Gp + q];" : `let idx = v * P_.Gp + q;
  let pr = unpack2x16float(sig[idx >> 1u]);
  return select(pr.x, pr.y, (idx & 1u) == 1u);`}
}

fn l2fa(a: f32, p: f32) -> f32 { let d = sqrt(a * a + 2.0 * p * p); if (d <= 0.0) { return 0.0; } return abs(a - p) / d; }

// The plain model's predicted signal of a state (the floor at LMIN, w = 1), as ukf-gpu.ts's predict.
fn predictState(x: ptr<function, array<f32, 10>>, q: u32) -> f32 {
  let g = grad[q];
  let bq = g.w * 1e-3;
  let m1v = normalize(vec3<f32>((*x)[0], (*x)[1], (*x)[2]));
  let m2v = normalize(vec3<f32>((*x)[5], (*x)[6], (*x)[7]));
  let c1 = dot(g.xyz, m1v); let c2 = dot(g.xyz, m2v);
  let l1a = max((*x)[3], LMIN); let l1p = max((*x)[4], LMIN); let l2a = max((*x)[8], LMIN); let l2p = max((*x)[9], LMIN);
  return 0.5 * exp(-bq * (l1p + (l1a - l1p) * c1 * c1)) + 0.5 * exp(-bq * (l2p + (l2a - l2p) * c2 * c2));
}

// A <- A⁻¹ for a symmetric positive 10 x 10, through Cholesky, as ukf-gpu.ts's invertA (the same operations per entry).
fn invert(A: ptr<function, array<f32, 100>>) {
  var Lc: array<f32, 100>;
  var Bi: array<f32, 100>;
  for (var j = 0u; j < N; j++) {
    var s = (*A)[j * N + j];
    for (var k = 0u; k < j; k++) { s -= Lc[j * N + k] * Lc[j * N + k]; }
    let d = sqrt(max(s, 1e-30));
    Lc[j * N + j] = d;
    for (var i = j + 1u; i < N; i++) {
      var t = (*A)[i * N + j];
      for (var k = 0u; k < j; k++) { t -= Lc[i * N + k] * Lc[j * N + k]; }
      Lc[i * N + j] = t / d;
    }
  }
  for (var j = 0u; j < N; j++) {
    Bi[j * N + j] = 1.0 / Lc[j * N + j];
    for (var i = j + 1u; i < N; i++) {
      var s = 0.0;
      for (var k = j; k < i; k++) { s -= Lc[i * N + k] * Bi[k * N + j]; }
      Bi[i * N + j] = s / Lc[i * N + i];
    }
  }
  for (var i = 0u; i < N; i++) {
    for (var j = 0u; j <= i; j++) {
      var s = 0.0;
      for (var k = i; k < N; k++) { s += Bi[k * N + i] * Bi[k * N + j]; }
      (*A)[i * N + j] = s; (*A)[j * N + i] = s;
    }
  }
}

@compute @workgroup_size(${FIBERS * LPF})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let slot = lid / LPF; let lane = lid % LPF;
  let f = wg.x * ${FIBERS}u + slot;
  let G = P_.G;
  let base = f * STRIDE;
  var on = f < P_.count;
  if (on) { on = st[base + 117u] > 0.5; }
  var x: array<f32, 10>;
  var P: array<f32, 100>;
  var pos = vec3<f32>(0.0);
  var m1 = vec3<f32>(0.0);
  var fa = 0.0;
  var stepnr = 0u; var records = 0u;
  if (on) {
    stepnr = u32(st[base + 116u]); records = u32(st[base + 118u]);
    for (var k = 0u; k < N; k++) { x[k] = st[base + k]; }
    for (var e = 0u; e < 100u; e++) { P[e] = st[base + 10u + e]; }
    pos = vec3<f32>(st[base + 110u], st[base + 111u], st[base + 112u]);
    m1 = vec3<f32>(st[base + 113u], st[base + 114u], st[base + 115u]);
    fa = st[base + 119u];
  }
  if (lane == 0u) { live[slot] = select(0u, 1u, on); }
  workgroupBarrier();
  if (lid == 0u) { var t = 0u; for (var q = 0u; q < ${FIBERS}u; q++) { t += live[q]; } liveAny = t; }
  if (workgroupUniformLoad(&liveAny) == 0u) { return; }

  for (var kstep = 0u; kstep < P_.K; kstep++) {
    var fail = false;
    // Zeroed explicitly every step: WGSL zeroes a variable declared in a loop on every pass, but the translation to Metal
    // was seen to zero it only once (the sums then grew from step to step: 2026-10-28, one step a dispatch against 16).
    var Wl: array<f32, 55>;
    var il: array<f32, 10>;
    for (var e = 0u; e < 55u; e++) { Wl[e] = 0.0; }
    for (var e = 0u; e < 10u; e++) { il[e] = 0.0; }
    var Yk: array<f32, 100>;
    var yh: array<f32, 10>;
    if (on) {
      // ── Cholesky of P ──
      var Lc: array<f32, 100>;
      for (var j = 0u; j < N; j++) {
        var s = P[j * N + j];
        for (var k = 0u; k < j; k++) { s -= Lc[j * N + k] * Lc[j * N + k]; }
        if (!(s > 0.0)) { fail = true; s = 1e-12; }
        Lc[j * N + j] = sqrt(s);
        for (var i = j + 1u; i < N; i++) {
          var t = P[i * N + j];
          for (var k = 0u; k < j; k++) { t -= Lc[i * N + k] * Lc[j * N + k]; }
          Lc[i * N + j] = t / Lc[j * N + j];
        }
      }
      // ── sigma points (plain model: directions normalized, eigenvalues floored), their mean, centered ──
      let sc = sqrt(f32(N) + P_.kappa);
      var X: array<f32, 210>;
      for (var s = 0u; s < S; s++) {
        for (var k = 0u; k < N; k++) {
          var v = x[k];
          if (s >= 1u && s <= N) { v += sc * Lc[k * N + (s - 1u)]; }
          if (s > N) { v -= sc * Lc[k * N + (s - 1u - N)]; }
          X[s * N + k] = v;
        }
        let a = normalize(vec3<f32>(X[s * N], X[s * N + 1u], X[s * N + 2u]));
        let b = normalize(vec3<f32>(X[s * N + 5u], X[s * N + 6u], X[s * N + 7u]));
        X[s * N] = a.x; X[s * N + 1u] = a.y; X[s * N + 2u] = a.z; X[s * N + 5u] = b.x; X[s * N + 6u] = b.y; X[s * N + 7u] = b.z;
        X[s * N + 3u] = max(X[s * N + 3u], LMIN); X[s * N + 4u] = max(X[s * N + 4u], LMIN);
        X[s * N + 8u] = max(X[s * N + 8u], LMIN); X[s * N + 9u] = max(X[s * N + 9u], LMIN);
      }
      var xh: array<f32, 10>;
      for (var k = 0u; k < N; k++) { var s = 0.0; for (var t = 0u; t < S; t++) { s += wgt(t) * X[t * N + k]; } xh[k] = s; }
      // The model's per-sigma-point values, from the points before centering (= centered + mean, as ukf-gpu.ts's pre).
      var m1s: array<vec3<f32>, 21>;
      var m2s: array<vec3<f32>, 21>;
      var ls: array<vec4<f32>, 21>;
      for (var t = 0u; t < S; t++) { for (var k = 0u; k < N; k++) { X[t * N + k] -= xh[k]; } }
      for (var t = 0u; t < S; t++) {
        m1s[t] = normalize(vec3<f32>(X[t * N] + xh[0], X[t * N + 1u] + xh[1], X[t * N + 2u] + xh[2]));
        m2s[t] = normalize(vec3<f32>(X[t * N + 5u] + xh[5], X[t * N + 6u] + xh[6], X[t * N + 7u] + xh[7]));
        ls[t] = vec4<f32>(max(X[t * N + 3u] + xh[3], LMIN), max(X[t * N + 4u] + xh[4], LMIN), max(X[t * N + 8u] + xh[8], LMIN), max(X[t * N + 9u] + xh[9], LMIN));
      }
      // ── Pn = Q + Σ w Xd Xdᵀ, its inverse Yk, yh = Yk·xh ──
      for (var e = 0u; e < 100u; e++) {
        let i = e / N; let j = e % N;
        var s = 0.0;
        for (var t = 0u; t < S; t++) { s += wgt(t) * X[t * N + i] * X[t * N + j]; }
        if (i == j) { if (i == 0u || i == 1u || i == 2u || i == 5u || i == 6u || i == 7u) { s += P_.qm; } else { s += P_.ql; } }
        Yk[e] = s;
      }
      invert(&Yk);
      for (var i = 0u; i < N; i++) { var s = 0.0; for (var k = 0u; k < N; k++) { s += Yk[i * N + k] * xh[k]; } yh[i] = s; }
      // ── this lane's gradients: the signal at pos, the predictions, Ht's row, added into W and i ──
      let ci = vec3<i32>(round(pos));
      for (var q = lane; q < G; q += LPF) {
        var zs = 0.0; var ws = 1e-16;
        for (var a = -1; a <= 1; a++) { for (var b = -1; b <= 1; b++) { for (var c = -1; c <= 1; c++) {
          let vx = ci + vec3<i32>(a, b, c);
          let lc = vx - P_.origin;
          if (any(lc < vec3<i32>(0)) || any(lc >= vec3<i32>(P_.dims))) { continue; }
          let d = (vec3<f32>(vx) - pos) * P_.voxel;
          let w = exp(-dot(d, d) / P_.sigma);
          let v = (u32(lc.z) * P_.dims.y + u32(lc.y)) * P_.dims.x + u32(lc.x);
          zs += w * sigAt(v, q); ws += w;
        } } }
        let zq = zs / ws;
        let g = grad[q]; let bq = g.w * 1e-3;
        var Zs: array<f32, 21>;
        var zm = 0.0;
        for (var t = 0u; t < S; t++) {
          let c1 = dot(g.xyz, m1s[t]); let c2 = dot(g.xyz, m2s[t]); let l = ls[t];
          Zs[t] = 0.5 * exp(-bq * (l.y + (l.x - l.y) * c1 * c1)) + 0.5 * exp(-bq * (l.w + (l.z - l.w) * c2 * c2));
          zm += wgt(t) * Zs[t];
        }
        var pc: array<f32, 10>;
        for (var k = 0u; k < N; k++) { var s = 0.0; for (var t = 0u; t < S; t++) { s += wgt(t) * X[t * N + k] * (Zs[t] - zm); } pc[k] = s; }
        var inn = zq - zm;
        for (var k = 0u; k < N; k++) { inn += pc[k] * yh[k]; }
        var ht: array<f32, 10>;
        for (var i = 0u; i < N; i++) { var s = 0.0; for (var k = 0u; k < N; k++) { s += Yk[i * N + k] * pc[k]; } ht[i] = s; }
        var e = 0u;
        for (var i = 0u; i < N; i++) { for (var j = 0u; j <= i; j++) { Wl[e] += ht[i] * ht[j]; e++; } il[i] += ht[i] * inn; }
      }
    }
    // ── the eight lanes' sums meet ──
    for (var e = 0u; e < 55u; e++) { red[lid * 66u + e] = Wl[e]; }
    for (var i = 0u; i < N; i++) { red[lid * 66u + 55u + i] = il[i]; }
    workgroupBarrier();
    var stopNow = false;
    if (on) {
      var W: array<f32, 100>;
      var iv: array<f32, 10>;
      let first = slot * LPF;
      var e = 0u;
      for (var i = 0u; i < N; i++) {
        for (var j = 0u; j <= i; j++) {
          var s = 0.0; for (var l = 0u; l < LPF; l++) { s += red[(first + l) * 66u + e]; }
          W[i * N + j] = Yk[i * N + j] + P_.R * s; W[j * N + i] = W[i * N + j]; e++;
        }
        var s2 = 0.0; for (var l = 0u; l < LPF; l++) { s2 += red[(first + l) * 66u + 55u + i]; }
        iv[i] = P_.R * s2 + yh[i];
      }
      var B = W;
      invert(&B);                                   // P_new
      var v: array<f32, 10>;
      for (var i = 0u; i < N; i++) { var s = 0.0; for (var k = 0u; k < N; k++) { s += B[i * N + k] * iv[k]; } v[i] = s; }
      // Step2T, as ukf-gpu.ts (orient, order, move along the first).
      let old = m1;
      var a = normalize(vec3<f32>(v[0], v[1], v[2])); var c = normalize(vec3<f32>(v[5], v[6], v[7]));
      if (dot(a, old) < 0.0) { a = -a; } if (dot(c, old) < 0.0) { c = -c; }
      let angle = degrees(acos(clamp(dot(a, c), -1.0, 1.0)));
      let swap = dot(a, old) < dot(c, old);
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
      x = v;
      if (doSwap) {
        for (var ee = 0u; ee < 100u; ee++) {
          let i = ee / N; let j = ee % N;
          let pi = select(i - 5u, i + 5u, i < 5u); let pj = select(j - 5u, j + 5u, j < 5u);
          P[ee] = B[pi * N + pj];
        }
      } else { P = B; }
      // The predicted mean signal of the new state, this lane's gradients.
      var part = 0.0;
      for (var q = lane; q < G; q += LPF) { part += predictState(&x, q); }
      red[lid * 66u + 65u] = part;
    }
    workgroupBarrier();
    if (on) {
      var mean = 0.0; for (var l = 0u; l < LPF; l++) { mean += red[(slot * LPF + l) * 66u + 65u]; } mean /= f32(G);
      stepnr = stepnr + 1u;
      let ci2 = vec3<i32>(round(pos)) - P_.origin;
      var inBrain = false;
      if (all(ci2 >= vec3<i32>(0)) && all(ci2 < vec3<i32>(P_.dims))) { inBrain = mask[(u32(ci2.z) * P_.dims.y + u32(ci2.y)) * P_.dims.x + u32(ci2.x)] > 0u; }
      if (fail || !inBrain || mean < P_.stopSignal || fa < P_.stopFA || stepnr > P_.maxSteps) {
        stopNow = true;
      } else if ((stepnr + 1u) % P_.perRecord == 0u) {
        if (records < P_.cap) {
          if (lane == 0u) {
            let o = (f * P_.cap + records) * 5u;
            outp[o] = pos.x; outp[o + 1u] = pos.y; outp[o + 2u] = pos.z; outp[o + 3u] = fa; outp[o + 4u] = 1.0;
          }
          records = records + 1u;
        } else { stopNow = true; }
      }
    }
    if (stopNow) { on = false; }
    if (lane == 0u) { live[slot] = select(0u, 1u, on); }
    workgroupBarrier();
    if (lid == 0u) { var t = 0u; for (var q = 0u; q < ${FIBERS}u; q++) { t += live[q]; } liveAny = t; }
    if (workgroupUniformLoad(&liveAny) == 0u) { break; }
  }
  // Store the state back (lane 0 of each fiber).
  if (f < P_.count && lane == 0u) {
    if (st[base + 117u] > 0.5) {
      for (var k = 0u; k < N; k++) { st[base + k] = x[k]; }
      for (var e = 0u; e < 100u; e++) { st[base + 10u + e] = P[e]; }
      st[base + 110u] = pos.x; st[base + 111u] = pos.y; st[base + 112u] = pos.z;
      st[base + 113u] = m1.x; st[base + 114u] = m1.y; st[base + 115u] = m1.z;
      st[base + 116u] = f32(stepnr); st[base + 118u] = f32(records);
      st[base + 117u] = select(0.0, 1.0, on); st[base + 119u] = fa;
      if (on) { atomicAdd(&alive, 1u); }
    }
  }
}
`;

const pipeCache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();
const packCache = new WeakMap<UkfData, Map<GPUDevice, { lo: number[]; cd: number[]; sig32: boolean; sigBuf: GPUBuffer; maskBuf: GPUBuffer; gBuf: GPUBuffer }>>();

/** Track from every seed (voxel coordinates) in both directions, plain two-tensor model, with the lane layout. The same
 *  options, seeding and output as ukf-gpu.ts trackUkfGpu with `freeWater: false`. */
export async function trackUkfGpuLanes(device: GPUDevice, data: UkfData, seeds: number[][], opts: UkfOptions & { batch?: number; stepsPerDispatch?: number; maxMsPerDispatch?: number; /** Half fibers a workgroup, lanes a half fiber. */ fibersPerGroup?: number; lanesPerFiber?: number; /** Checking only: one dispatch, the states before and after. */ debugOneDispatch?: boolean } = {}): Promise<GpuUkfResult> {
  const t0 = performance.now();
  if (opts.freeWater !== false) throw new Error("the lane layout is written for the plain two-tensor model (freeWater: false)");
  const N = 10, G = data.G;
  if (G > MAXG) throw new Error(`${G} gradients: the graphics-card UKF holds at most ${MAXG}`);
  const { step, perRecord, maxSteps } = ukfLengths(data.voxel, opts), cap = Math.ceil(maxSteps / perRecord) + 2;
  const sigma = ukfSigma(data.voxel, opts), p0 = opts.p0 ?? 0.01, seedFA = opts.seedingThreshold ?? 0.18;
  const [nx, ny, nz] = data.dims;
  let packed = packCache.get(data)?.get(device);
  if (!packed) {
    // The signal cropped to the mask's box (+1 voxel), 32-bit when the device allows it: as ukf-gpu.ts's pack.
    let lo = [nx, ny, nz], hi = [-1, -1, -1];
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (data.mask[(k * ny + j) * nx + i]) {
      lo = [Math.min(lo[0], i), Math.min(lo[1], j), Math.min(lo[2], k)]; hi = [Math.max(hi[0], i), Math.max(hi[1], j), Math.max(hi[2], k)];
    }
    lo = lo.map((v) => Math.max(0, v - 1)); hi = hi.map((v, a) => Math.min(data.dims[a] - 1, v + 1));
    const cd = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1], cn = cd[0] * cd[1] * cd[2], Gp = G + (G & 1);
    const sig32 = cn * Gp * 4 <= device.limits.maxStorageBufferBindingSize && cn * Gp * 4 <= device.limits.maxBufferSize;
    if (!sig32) throw new Error("the lane layout needs the signal in 32 bits (this device's buffers are too small)");
    const f32s = new Float32Array(cn * Gp), cmask = new Uint32Array(cn);
    for (let k = 0; k < cd[2]; k++) for (let j = 0; j < cd[1]; j++) for (let i = 0; i < cd[0]; i++) {
      const c = (k * cd[1] + j) * cd[0] + i, v = ((k + lo[2]) * ny + (j + lo[1])) * nx + (i + lo[0]);
      cmask[c] = data.mask[v];
      for (let q = 0; q < G; q++) f32s[c * Gp + q] = data.signal[v * G + q];
    }
    const grads = new Float32Array(4 * G);
    for (let q = 0; q < G; q++) grads.set([data.g[3 * q], data.g[3 * q + 1], data.g[3 * q + 2], data.b[q]], 4 * q);
    const mk0 = (d: ArrayBufferView) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(d.byteLength / 4) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, d.buffer, d.byteOffset, d.byteLength); return b; };
    packed = { lo, cd, sig32, sigBuf: mk0(f32s), maskBuf: mk0(cmask), gBuf: mk0(grads) };
    const m = packCache.get(data) ?? new Map(); m.set(device, packed); packCache.set(data, m);
  }
  const { lo, cd, sigBuf, maskBuf, gBuf } = packed, Gp = G + (G & 1);
  // Seed states, as ukf-gpu.ts (both directions, eigenvalues in 1e-3 mm²/s), in this layout.
  const z = new Float64Array(G), halves: { seed: number; state: Float32Array }[] = [];
  let used = 0, rejected = 0;
  const l2fa3 = (a: number, b: number, c: number) => Math.sqrt(0.5 * ((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2) / (a * a + b * b + c * c)) || 0;
  seeds.forEach((sd, idx) => {
    signalAt(data, sd, sigma, z);
    for (let q = 0; q < G; q++) if (!(z[q] >= 0) || !Number.isFinite(z[q])) { rejected++; return; }
    const t = seedTensor(data, z);
    const lm = t ? (t.l[1] + t.l[2]) / 2 : 0;
    if (!t || !(l2fa3(t.l[0], lm, lm) > seedFA)) { rejected++; return; }
    used++;
    const la = t.l[0] * 1e-3, lp = lm * 1e-3, fa = l2fa3(t.l[0], lm, lm);
    for (const sgn of [1, -1]) {
      const s = new Float32Array(STRIDE), m = t.m.map((v) => sgn * v);
      s.set([m[0], m[1], m[2], la, lp, t.m[0], t.m[1], t.m[2], la, lp], 0);
      for (let k = 0; k < N; k++) s[10 + k * N + k] = [3, 4, 8, 9].includes(k) ? p0 * 1e-6 : p0;
      s.set(sd, 110); s.set(m, 113); s[116] = 0; s[117] = 1; s[118] = 1; s[119] = fa;
      halves.push({ seed: idx, state: s });
    }
  });
  const FPG = opts.fibersPerGroup ?? FIBERS, LANES = opts.lanesPerFiber ?? LPF, pkey = `lanes|${FPG}|${LANES}`;
  let pipeline = pipeCache.get(device)?.get(pkey);
  if (!pipeline) {
    pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: wgslLanes(true, FPG, LANES) }), entryPoint: "main" } });
    const m = pipeCache.get(device) ?? new Map(); m.set(pkey, pipeline); pipeCache.set(device, m);
  }
  const tPrep = performance.now();
  const mk = (d: ArrayBufferView, usage: number) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(d.byteLength / 4) * 4), usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, d.buffer, d.byteOffset, d.byteLength); return b; };
  const batch = opts.batch ?? 4096, K = opts.stepsPerDispatch ?? 16, capMs = opts.maxMsPerDispatch ?? 500;
  const R = 2 / (opts.Rs ?? 0.02), recorded: { pts: Float32Array; fa: Float32Array }[] = new Array(halves.length);
  let dispatches = 0;
  const debug = { before: [] as Float32Array[], after: [] as Float32Array[] };
  for (let b0 = 0; b0 < halves.length; b0 += batch) {
    const hs = halves.slice(b0, b0 + batch), count = hs.length;
    const stArr = new Float32Array(count * STRIDE); hs.forEach((h, i) => stArr.set(h.state, i * STRIDE));
    const outArr = new Float32Array(count * cap * 5);
    hs.forEach((h, i) => outArr.set([h.state[110], h.state[111], h.state[112], h.state[119], 1], i * cap * 5));
    const stBuf = mk(stArr, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC), outBuf = mk(outArr, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const aliveBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const aliveRead = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const params = new ArrayBuffer(96), dv = new DataView(params);
    const u32 = (o: number, v: number) => dv.setUint32(o, v, true), i32 = (o: number, v: number) => dv.setInt32(o, v, true), f32 = (o: number, v: number) => dv.setFloat32(o, v, true);
    u32(0, cd[0]); u32(4, cd[1]); u32(8, cd[2]); u32(12, G);
    i32(16, lo[0]); i32(20, lo[1]); i32(24, lo[2]); u32(28, Gp);
    f32(32, data.voxel[0]); f32(36, data.voxel[1]); f32(40, data.voxel[2]); f32(44, sigma);
    f32(48, step); f32(52, R); f32(56, 0.01); f32(60, opts.stoppingFA ?? 0.15);
    f32(64, opts.stoppingThreshold ?? 0.1); u32(68, maxSteps); u32(72, perRecord); u32(76, cap);
    u32(80, count); u32(84, K); f32(88, opts.Qm ?? 0.001); f32(92, (opts.Ql ?? 50) * 1e-6);
    const pBuf = mk(new Uint8Array(params), GPUBufferUsage.UNIFORM);
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [pBuf, sigBuf, maskBuf, gBuf, stBuf, outBuf, aliveBuf].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    let k = Math.min(K, 16), stepsDone = 0;
    for (; stepsDone <= maxSteps + k; ) {
      device.queue.writeBuffer(pBuf, 84, Uint32Array.of(k));
      const ts = performance.now(), enc = device.createCommandEncoder();
      enc.clearBuffer(aliveBuf);
      const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(Math.ceil(count / FPG)); pass.end();
      enc.copyBufferToBuffer(aliveBuf, 0, aliveRead, 0, 4);
      device.queue.submit([enc.finish()]); dispatches++;
      await aliveRead.mapAsync(GPUMapMode.READ);
      const n = new Uint32Array(aliveRead.getMappedRange().slice(0))[0];
      aliveRead.unmap();
      stepsDone += k;
      if (n === 0 || opts.debugOneDispatch) break;
      const ms = performance.now() - ts;
      if (ms > capMs) k = Math.max(4, k >> 1); else if (ms < capMs / 4 && k < K) k = Math.min(K, k * 2);
    }
    const read = async (buf: GPUBuffer, size: number) => { const r = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); const e = device.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, r, 0, size); device.queue.submit([e.finish()]); await r.mapAsync(GPUMapMode.READ); const a = new Float32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return a; };
    const stOut = await read(stBuf, count * STRIDE * 4), pts = await read(outBuf, count * cap * 5 * 4);
    if (opts.debugOneDispatch) { debug.before.push(...hs.map((h) => h.state)); debug.after.push(...hs.map((_, i) => stOut.slice(i * STRIDE, (i + 1) * STRIDE))); }
    hs.forEach((_, i) => {
      const nrec = Math.max(1, Math.round(stOut[i * STRIDE + 118]));
      const P = new Float32Array(3 * nrec), FA = new Float32Array(nrec);
      for (let r = 0; r < nrec; r++) { const o = (i * cap + r) * 5; P[3 * r] = pts[o]; P[3 * r + 1] = pts[o + 1]; P[3 * r + 2] = pts[o + 2]; FA[r] = pts[o + 3]; }
      recorded[b0 + i] = { pts: P, fa: FA };
    });
    for (const b of [stBuf, outBuf, aliveBuf, aliveRead, pBuf]) b.destroy();
  }
  const tGpu = performance.now();
  const M = data.ijkToRAS, out: GpuUkfResult["fibers"] = [];
  for (let h = 0; h + 1 < halves.length; h += 2) {
    const fwd = recorded[h], bwd = recorded[h + 1], nb = bwd.fa.length - 1, nf = fwd.fa.length, n = nb + nf;
    const points = new Float32Array(3 * n), faA = new Float32Array(n), wA = new Float32Array(n).fill(1);
    const put = (dst: number, src: Float32Array, si: number) => { const x = src[3 * si], y = src[3 * si + 1], zz = src[3 * si + 2]; points[3 * dst] = M[0] * x + M[1] * y + M[2] * zz + M[3]; points[3 * dst + 1] = M[4] * x + M[5] * y + M[6] * zz + M[7]; points[3 * dst + 2] = M[8] * x + M[9] * y + M[10] * zz + M[11]; };
    for (let r = 0; r < nb; r++) { const si = nb - r; put(r, bwd.pts, si); faA[r] = bwd.fa[si]; }
    for (let r = 0; r < nf; r++) { put(nb + r, fwd.pts, r); faA[nb + r] = fwd.fa[r]; }
    out.push({ points, fa: faA, freeWater: wA, seed: halves[h].seed });
  }
  return { fibers: out, seedsUsed: used, seedsRejected: rejected, dispatches, ms: { prepare: tPrep - t0, gpu: tGpu - tPrep, total: performance.now() - t0 }, ...(opts.debugOneDispatch ? { debug } : {}) };
}
