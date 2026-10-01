// THE DIFFUSION TENSOR ON THE GRAPHICS CARD -- milestone 1, step 2 (Contents/docs/dmri-review-2026-09-28.md): FA and
// color FA computed on the card. The same model, fit and conventions as the processor reference
// extensions/diffusion/tensor.ts (read its header): log-linear least squares, one weighted pass (weights = fitted signal²),
// b ≤ maxB, a floor for zero signals, gradients in patient RAS; eigenvalues in closed form, the principal eigenvector
// from the largest cross product of two rows. Checked against that reference in extensions/diffusion/tensor-gpu.test.ts.
//
// One thread per voxel. The signals go up voxel-major (a voxel's m values side by side) in chunks of voxels, so a large
// scan never needs one storage binding past the browser's default 128 MB (Chrome's maxStorageBufferBindingSize). The
// design matrix and the least-squares pseudo-inverse are the same for every voxel and computed once on the processor.
//
// Output per voxel: FA, MD, the principal eigenvector (x, y, z) -- five f32, read back or kept on the card for drawing.
// f32 throughout: FA agrees with the f64 reference to about 1e-3 (the test's bound), which is below anything a color
// map shows.
import { type DiffusionSeries, isotropicVolumes } from "./dwi.ts";
import { brainMask } from "./tensor.ts";

export interface GpuTensorResult {
  fa: Float32Array;
  md: Float32Array;
  /** Three per voxel, patient RAS, unit (sign arbitrary). */
  v1: Float32Array;
  mask: Uint8Array;
  used: number[];
  ms: { prepare: number; gpu: number; total: number };
}

const WGSL = /* wgsl */ `
struct Params { m: u32, count: u32, first: u32, floorLog: f32, weighted: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> X: array<f32>;       // m x 7, row-major
@group(0) @binding(2) var<storage, read> pinv: array<f32>;    // 7 x m
@group(0) @binding(3) var<storage, read> sig: array<f32>;     // count x m, voxel-major (this chunk)
@group(0) @binding(4) var<storage, read> mask: array<u32>;    // count (this chunk)
@group(0) @binding(5) var<storage, read_write> outv: array<f32>; // count x 5: fa, md, v1.xyz

fn eigvals(a: f32, d: f32, e: f32, b: f32, f: f32, c: f32) -> vec3<f32> {
  let p1 = d * d + e * e + f * f;
  let q = (a + b + c) / 3.0;
  if (p1 <= 1e-30 * (a * a + b * b + c * c) + 1e-38) {
    let hi = max(a, max(b, c));
    let lo = min(a, min(b, c));
    return vec3<f32>(hi, a + b + c - hi - lo, lo);
  }
  let p2 = (a - q) * (a - q) + (b - q) * (b - q) + (c - q) * (c - q) + 2.0 * p1;
  let p = sqrt(p2 / 6.0);
  let B0 = (a - q) / p; let B4 = (b - q) / p; let B8 = (c - q) / p;
  let B1 = d / p; let B2 = e / p; let B5 = f / p;
  let detB = B0 * (B4 * B8 - B5 * B5) - B1 * (B1 * B8 - B5 * B2) + B2 * (B1 * B5 - B4 * B2);
  let r = clamp(detB / 2.0, -1.0, 1.0);
  let phi = acos(r) / 3.0;
  let l1 = q + 2.0 * p * cos(phi);
  let l3 = q + 2.0 * p * cos(phi + 2.0943951023931953);
  return vec3<f32>(l1, 3.0 * q - l1 - l3, l3);
}

fn eigvec(a: f32, d: f32, e: f32, b: f32, f: f32, c: f32, l: f32) -> vec3<f32> {
  let r0 = vec3<f32>(a - l, d, e);
  let r1 = vec3<f32>(d, b - l, f);
  let r2 = vec3<f32>(e, f, c - l);
  let c0 = cross(r0, r1); let c1 = cross(r0, r2); let c2 = cross(r1, r2);
  let n0 = dot(c0, c0); let n1 = dot(c1, c1); let n2 = dot(c2, c2);
  var v = c0; var n = n0;
  if (n1 > n) { v = c1; n = n1; }
  if (n2 > n) { v = c2; n = n2; }
  if (n <= 0.0) { return vec3<f32>(1.0, 0.0, 0.0); }
  return v / sqrt(n);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= P.count) { return; }
  let o = v * 5u;
  if (mask[v] == 0u) { outv[o] = 0.0; outv[o + 1u] = 0.0; outv[o + 2u] = 0.0; outv[o + 3u] = 0.0; outv[o + 4u] = 0.0; return; }
  let m = P.m;
  let base = v * m;
  var beta: array<f32, 7>;
  for (var c = 0u; c < 7u; c++) {
    var s = 0.0;
    for (var r = 0u; r < m; r++) { s += pinv[c * m + r] * max(log(max(sig[base + r], 1e-30)), P.floorLog); }
    beta[c] = s;
  }
  if (P.weighted == 1u) {
    var A: array<f32, 49>;
    var y: array<f32, 7>;
    for (var r = 0u; r < m; r++) {
      var pred = 0.0;
      for (var c = 0u; c < 7u; c++) { pred += X[r * 7u + c] * beta[c]; }
      let w = exp(2.0 * min(pred, 40.0));
      let ls = max(log(max(sig[base + r], 1e-30)), P.floorLog);
      for (var i = 0u; i < 7u; i++) {
        let wxi = w * X[r * 7u + i];
        y[i] += wxi * ls;
        for (var j = 0u; j <= i; j++) { A[i * 7u + j] += wxi * X[r * 7u + j]; }
      }
    }
    // Cholesky, lower triangle in place; on failure the least-squares answer stands.
    var ok = true;
    for (var j = 0u; j < 7u; j++) {
      var s = A[j * 7u + j];
      for (var k = 0u; k < j; k++) { s -= A[j * 7u + k] * A[j * 7u + k]; }
      if (!(s > 0.0)) { ok = false; break; }
      let ljj = sqrt(s);
      A[j * 7u + j] = ljj;
      for (var i = j + 1u; i < 7u; i++) {
        var t = A[i * 7u + j];
        for (var k = 0u; k < j; k++) { t -= A[i * 7u + k] * A[j * 7u + k]; }
        A[i * 7u + j] = t / ljj;
      }
    }
    if (ok) {
      var x: array<f32, 7>;
      for (var i = 0u; i < 7u; i++) { var t = y[i]; for (var k = 0u; k < i; k++) { t -= A[i * 7u + k] * x[k]; } x[i] = t / A[i * 7u + i]; }
      for (var ii = 0u; ii < 7u; ii++) {
        let i = 6u - ii;
        var t = x[i];
        for (var k = i + 1u; k < 7u; k++) { t -= A[k * 7u + i] * x[k]; }
        x[i] = t / A[i * 7u + i];
      }
      for (var c = 0u; c < 7u; c++) { beta[c] = x[c]; }
    }
  }
  let ev = eigvals(beta[0], beta[1], beta[2], beta[3], beta[4], beta[5]);
  let e1 = eigvec(beta[0], beta[1], beta[2], beta[3], beta[4], beta[5], ev.x);
  let l = max(ev, vec3<f32>(0.0));
  let den = dot(l, l);
  var fa = 0.0;
  if (den > 0.0) {
    fa = min(1.0, sqrt(0.5 * ((l.x - l.y) * (l.x - l.y) + (l.y - l.z) * (l.y - l.z) + (l.z - l.x) * (l.z - l.x)) / den));
  }
  outv[o] = fa;
  outv[o + 1u] = (ev.x + ev.y + ev.z) / 3.0;
  outv[o + 2u] = e1.x; outv[o + 3u] = e1.y; outv[o + 4u] = e1.z;
}
`;

/** Cholesky solve of a 7x7 symmetric positive system (f64, processor): the pseudo-inverse's columns. */
function solve7(M: Float64Array, y: Float64Array): Float64Array | null {
  const A = M.slice(), x = new Float64Array(7);
  for (let j = 0; j < 7; j++) {
    let s = A[j * 7 + j];
    for (let k = 0; k < j; k++) s -= A[j * 7 + k] ** 2;
    if (!(s > 1e-300)) return null;
    const l = Math.sqrt(s); A[j * 7 + j] = l;
    for (let i = j + 1; i < 7; i++) { let t = A[i * 7 + j]; for (let k = 0; k < j; k++) t -= A[i * 7 + k] * A[j * 7 + k]; A[i * 7 + j] = t / l; }
  }
  for (let i = 0; i < 7; i++) { let t = y[i]; for (let k = 0; k < i; k++) t -= A[i * 7 + k] * x[k]; x[i] = t / A[i * 7 + i]; }
  for (let i = 6; i >= 0; i--) { let t = x[i]; for (let k = i + 1; k < 7; k++) t -= A[k * 7 + i] * x[k]; x[i] = t / A[i * 7 + i]; }
  return x;
}

export async function fitTensorsGpu(
  device: GPUDevice,
  dwi: DiffusionSeries,
  opts: { maxB?: number; mask?: Uint8Array; weighted?: boolean; minSignal?: number; chunkVoxels?: number } = {},
): Promise<GpuTensorResult> {
  const t0 = performance.now();
  const maxB = opts.maxB ?? 1500;
  // Volumes up to maxB; a b > 0 volume with no direction (a trace image) says nothing about direction and is left out.
  const iso = new Set(isotropicVolumes(dwi));
  const used = dwi.bValues.map((b, i) => (b <= maxB && !iso.has(i) ? i : -1)).filter((i) => i >= 0);
  const b0s = used.filter((i) => dwi.bValues[i] < 50);
  if (!b0s.length) throw new Error("no b=0 volume: the tensor needs a reference signal");
  if (used.length - b0s.length < 6) throw new Error(`${used.length - b0s.length} diffusion-weighted volumes up to b=${maxB}: the tensor needs at least 6 directions`);
  const [nx, ny, nz] = dwi.volumes[0].dims;
  const n = nx * ny * nz, m = used.length;

  // The mask: the processor reference's own rule (one function, so the two cannot drift apart).
  const mask = opts.mask ?? brainMask(dwi, b0s).mask;

  const X = new Float64Array(m * 7);
  used.forEach((vi, r) => {
    const b = dwi.bValues[vi], g = dwi.gradients[vi];
    [-b * g[0] * g[0], -2 * b * g[0] * g[1], -2 * b * g[0] * g[2], -b * g[1] * g[1], -2 * b * g[1] * g[2], -b * g[2] * g[2], 1].forEach((x, c) => (X[r * 7 + c] = x));
  });
  const XtX = new Float64Array(49);
  for (let r = 0; r < m; r++) for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) XtX[i * 7 + j] += X[r * 7 + i] * X[r * 7 + j];
  const pinv = new Float32Array(7 * m);
  for (let r = 0; r < m; r++) {
    const x = solve7(XtX, X.slice(r * 7, r * 7 + 7));
    if (!x) throw new Error("the gradient directions do not determine a tensor (too few, or all in one plane)");
    for (let c = 0; c < 7; c++) pinv[c * m + r] = x[c];
  }

  const mk = (data: ArrayBufferView, usage: number) => {
    const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  };
  const xBuf = mk(new Float32Array(X), GPUBufferUsage.STORAGE);
  const pBuf = mk(pinv, GPUBufferUsage.STORAGE);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: WGSL }), entryPoint: "main" } });

  // Chunks: at most 64 MB of signals each (and a multiple of the workgroup).
  const chunk = Math.max(64, Math.min(opts.chunkVoxels ?? Math.floor((64 << 20) / (4 * m)), n));
  const fa = new Float32Array(n), md = new Float32Array(n), v1 = new Float32Array(3 * n);
  const vols = used.map((i) => dwi.volumes[i].data);
  const floorLog = Math.log(opts.minSignal ?? 1);
  const tPrep = performance.now();
  let gpuMs = 0;
  for (let first = 0; first < n; first += chunk) {
    const count = Math.min(chunk, n - first);
    const sig = new Float32Array(count * m);
    for (let v = 0; v < count; v++) for (let r = 0; r < m; r++) sig[v * m + r] = vols[r][first + v];
    const mk32 = new Uint32Array(count);
    for (let v = 0; v < count; v++) mk32[v] = mask[first + v];
    const tg = performance.now();
    const params = new ArrayBuffer(32), pv = new DataView(params);
    pv.setUint32(0, m, true); pv.setUint32(4, count, true); pv.setUint32(8, first, true); pv.setFloat32(12, floorLog, true);
    pv.setUint32(16, (opts.weighted ?? true) ? 1 : 0, true);
    const uBuf = mk(new Uint8Array(params), GPUBufferUsage.UNIFORM);
    const sBuf = mk(sig, GPUBufferUsage.STORAGE);
    const mBuf = mk(mk32, GPUBufferUsage.STORAGE);
    const oBuf = device.createBuffer({ size: count * 5 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rBuf = device.createBuffer({ size: count * 5 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [uBuf, xBuf, pBuf, sBuf, mBuf, oBuf].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    enc.copyBufferToBuffer(oBuf, 0, rBuf, 0, count * 5 * 4);
    device.queue.submit([enc.finish()]);
    await rBuf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rBuf.getMappedRange().slice(0));
    rBuf.unmap();
    for (const b of [uBuf, sBuf, mBuf, oBuf, rBuf]) b.destroy();
    gpuMs += performance.now() - tg;
    for (let v = 0; v < count; v++) {
      fa[first + v] = out[5 * v]; md[first + v] = out[5 * v + 1];
      v1[3 * (first + v)] = out[5 * v + 2]; v1[3 * (first + v) + 1] = out[5 * v + 3]; v1[3 * (first + v) + 2] = out[5 * v + 4];
    }
  }
  xBuf.destroy(); pBuf.destroy();
  const total = performance.now() - t0;
  return { fa, md, v1, mask, used, ms: { prepare: tPrep - t0, gpu: gpuMs, total } };
}
