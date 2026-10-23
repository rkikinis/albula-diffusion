// BRAINSTEMSEG'S NETWORK ON THE GRAPHICS CARD -- the same network as bundleparc.ts describes (BundleParc's, MIT; the
// weights BrainstemSeg's, CC BY 4.0), as a chain of compute steps over a cube of D voxels a side (176 in use; any multiple
// of 16 for the tests). Single precision, as the original; checked block by block against PyTorch in double precision on
// a 32-voxel cube, and on a whole scan against the PyTorch run of Contents/tools/brainstemseg-reference.py
// (bundleparc-gpu.test.ts).
//
// Layout: CHANNEL LAST, voxel (x, y, z) of a level's grid (side d) at ((x·d + y)·d + z)·C -- the tests' layout.
// The steps:
//   gemm     one tiled multiply (64 × 64 tiles, 4 × 4 per thread) for every 1×1×1 convolution and linear layer:
//            out = act(A'·Wᵀ + b) (+ R), where A' is the input or, for the group norm, its channels scaled and shifted
//            on the way in; the rows can read every second voxel (the halving residual) or write the (a, b, c) corner of
//            each 2×2×2 block (the doubling residual: a transposed 2×2×2 convolution is 8 such multiplies).
//   dw3      the depthwise 3×3×3 convolution, stride 1 or 2, zero outside;   dwT2  the depthwise transposed 2×2×2 one.
//   stats    the group norm (one group per channel): partial sums per workgroup, then per channel the mean, then the
//            spread about the mean (two passes, not a sum of squares), and the scale and shift the next gemm applies.
//   scores · max · sums   the attention of the prompt over the voxels: per voxel and head (token + position code)·u,
//            the maximum, then per workgroup Σ e and Σ e·token; the processor adds the partial sums (bundleparc.ts).
//   tokens   after the MLP: layer norm, + the vector every voxel gets, layer norm.
//   label    the head's two channels: mask = sigmoid > 0.5, position = ceil(10·sigmoid) inside it; four voxels a word.
import { CHANNELS, DEC_RATIO, ENC_RATIO, HEADS, N_COEFS, attnScores, attnUpdate, positionTables, promptEmbedding, promptIn, tensor, type BundleParcModel } from "./bundleparc.ts";

export interface GpuBundleParc {
  /** The prepared cube (channel last, D³ × 45): the encoder, once; its outputs are kept for every tract. */
  encode(cube: Float32Array, D: number): Promise<void>;
  /** One tract (its index in model.tracts): per voxel of the cube, 0 outside the mask, else the position 1..10. */
  decode(tract: number): Promise<{ labels: Uint8Array; seconds: number }>;
  /** Tests: the named intermediate (as Contents/tools/brainstemseg-dump.py names them), kept when `keep` is on. */
  read(name: string): Promise<Float32Array>;
  /** Tests: the prompt after each decoder level of the last decode (processor side). */
  prompts: Map<string, Float64Array>;
  destroy(): void;
}

const ERF = `
fn erf_(x: f32) -> f32 {   // Abramowitz & Stegun 7.1.26: absolute error < 1.5e-7
  let a = abs(x); let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return select(-y, y, x >= 0.0);
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.70710678118654752)); }`;

const GEMM = `
struct P { N: u32, K: u32, M: u32, act: u32, woff: u32, boff: u32, hasAff: u32, hasRes: u32,
  inMode: u32, outMode: u32, d: u32, od: u32, oa: u32, ob: u32, oc: u32, nx: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> aff: array<f32>;
@group(0) @binding(4) var<storage, read> R: array<f32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
var<workgroup> ta: array<array<f32, 64>, 16>;
var<workgroup> tb: array<array<f32, 64>, 16>;
${ERF}
fn inRow(r: u32) -> u32 {            // inMode 1: row r of the half grid (side d) reads voxel (2x, 2y, 2z) of the full one (od)
  if (p.inMode == 1u) { let x = r / (p.d * p.d); let y = (r / p.d) % p.d; let z = r % p.d; return ((2u * x) * p.od + 2u * y) * p.od + 2u * z; }
  return r;
}
fn outRow(r: u32) -> u32 {           // outMode 1: row r of the half grid (d) writes voxel (2x+a, 2y+b, 2z+c) of the full one (od)
  if (p.outMode == 1u) { let x = r / (p.d * p.d); let y = (r / p.d) % p.d; let z = r % p.d; return ((2u * x + p.oa) * p.od + 2u * y + p.ob) * p.od + 2u * z + p.oc; }
  return r;
}
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let r0 = (w.y * p.nx + w.x) * 64u; let c0 = w.z * 64u;
  var acc: array<f32, 16>;
  for (var t = 0u; t < p.K; t = t + 16u) {
    for (var i = 0u; i < 4u; i = i + 1u) {
      let r = r0 + l.y * 4u + i; let k = t + l.x; var v = 0.0;
      if (r < p.N && k < p.K) { v = A[inRow(r) * p.K + k]; if (p.hasAff == 1u) { v = v * aff[k] + aff[p.K + k]; } }
      ta[l.x][l.y * 4u + i] = v;
    }
    for (var j = 0u; j < 4u; j = j + 1u) {
      let c = c0 + l.x * 4u + j; let k = t + l.y; var v = 0.0;
      if (c < p.M && k < p.K) { v = W[p.woff + c * p.K + k]; }
      tb[l.y][l.x * 4u + j] = v;
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k = k + 1u) {
      for (var i = 0u; i < 4u; i = i + 1u) {
        let a = ta[k][l.y * 4u + i];
        for (var j = 0u; j < 4u; j = j + 1u) { acc[i * 4u + j] = acc[i * 4u + j] + a * tb[k][l.x * 4u + j]; }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i = i + 1u) {
    let r = r0 + l.y * 4u + i; if (r >= p.N) { continue; }
    for (var j = 0u; j < 4u; j = j + 1u) {
      let c = c0 + l.x * 4u + j; if (c >= p.M) { continue; }
      var v = acc[i * 4u + j] + W[p.boff + c];
      if (p.act == 1u) { v = gelu(v); } else if (p.act == 2u) { v = max(v, 0.0); }
      let o = outRow(r) * p.M + c;
      if (p.hasRes == 1u) { v = v + R[o]; }
      O[o] = v;
    }
  }
}`;

const DW3 = `
struct P { d: u32, id: u32, C: u32, stride: u32, woff: u32, boff: u32, nx: u32, total: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let g = (w.y * p.nx + w.x) * 256u + l.x; if (g >= p.total) { return; }
  let c = g % p.C; let v = g / p.C; let x = v / (p.d * p.d); let y = (v / p.d) % p.d; let z = v % p.d;
  var s = W[p.boff + c]; let n = i32(p.id);
  for (var a = 0; a < 3; a = a + 1) { let ix = i32(x * p.stride) + a - 1; if (ix < 0 || ix >= n) { continue; }
    for (var b = 0; b < 3; b = b + 1) { let iy = i32(y * p.stride) + b - 1; if (iy < 0 || iy >= n) { continue; }
      for (var e = 0; e < 3; e = e + 1) { let iz = i32(z * p.stride) + e - 1; if (iz < 0 || iz >= n) { continue; }
        s = s + A[((u32(ix) * p.id + u32(iy)) * p.id + u32(iz)) * p.C + c] * W[p.woff + c * 27u + u32(a * 9 + b * 3 + e)]; } } }
  O[g] = s;
}`;

const DWT2 = `
struct P { d: u32, id: u32, C: u32, _s: u32, woff: u32, boff: u32, nx: u32, total: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let g = (w.y * p.nx + w.x) * 256u + l.x; if (g >= p.total) { return; }
  let c = g % p.C; let v = g / p.C; let x = v / (p.d * p.d); let y = (v / p.d) % p.d; let z = v % p.d;
  let k = (x % 2u) * 4u + (y % 2u) * 2u + (z % 2u);
  O[g] = A[(((x / 2u) * p.id + y / 2u) * p.id + z / 2u) * p.C + c] * W[p.woff + c * 8u + k] + W[p.boff + c];
}`;

const STATS = `
struct P { N: u32, C: u32, chunk: u32, pass_: u32, lanes: u32, cp: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let t = l.x; let lane = t / p.cp; let r0 = w.x * p.chunk; let r1 = min(p.N, r0 + p.chunk);
  for (var cb = 0u; cb < p.C; cb = cb + p.cp) {
    let c = cb + t % p.cp; var acc = 0.0;
    if (c < p.C) {
      let m = select(0.0, mean[c], p.pass_ == 1u);
      for (var r = r0 + lane; r < r1; r = r + p.lanes) { let x = A[r * p.C + c]; if (p.pass_ == 1u) { acc = acc + (x - m) * (x - m); } else { acc = acc + x; } }
    }
    sh[t] = acc; workgroupBarrier();
    if (lane == 0u && c < p.C) { var s = 0.0; for (var q = 0u; q < p.lanes; q = q + 1u) { s = s + sh[q * p.cp + t]; } part[w.x * p.C + c] = s; }
    workgroupBarrier();
  }
}`;

const STATS_END = `
struct P { C: u32, nwg: u32, pass_: u32, goff: u32, boff: u32, N: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> part: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> mean: array<f32>;
@group(0) @binding(4) var<storage, read_write> aff: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gi: vec3<u32>) {
  let c = gi.x; if (c >= p.C) { return; }
  var s = 0.0; for (var q = 0u; q < p.nwg; q = q + 1u) { s = s + part[q * p.C + c]; }
  if (p.pass_ == 0u) { mean[c] = s / f32(p.N); return; }
  let sc = W[p.goff + c] / sqrt(s / f32(p.N) + 1e-5);
  aff[c] = sc; aff[p.C + c] = W[p.boff + c] - mean[c] * sc;
}`;

const ADD = `
struct P { total: u32, nx: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let g = (w.y * p.nx + w.x) * 256u + l.x; if (g >= p.total) { return; }
  O[g] = A[g] + B[g];
}`;

const SCORES = `
struct P { N: u32, C: u32, d: u32, chp: u32, nx: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> pe: array<f32>;
@group(0) @binding(3) var<storage, read> u: array<f32>;
@group(0) @binding(4) var<storage, read_write> S: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let v = (w.y * p.nx + w.x) * 256u + l.x; if (v >= p.N) { return; }
  let pos = vec3<u32>(v / (p.d * p.d), (v / p.d) % p.d, v % p.d);
  var s = vec4<f32>(0.0);
  for (var c = 0u; c < p.C; c = c + 1u) {
    let ax = c / p.chp; let j = c % p.chp;
    let t = A[v * p.C + c] + pe[(ax * p.d + pos[ax]) * p.chp + j];
    s = s + t * vec4<f32>(u[c], u[p.C + c], u[2u * p.C + c], u[3u * p.C + c]);
  }
  S[v * 4u] = s.x; S[v * 4u + 1u] = s.y; S[v * 4u + 2u] = s.z; S[v * 4u + 3u] = s.w;
}`;

const MAXP = `
struct P { N: u32, chunk: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read_write> pm: array<f32>;
var<workgroup> sh: array<vec4<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let r0 = w.x * p.chunk; let r1 = min(p.N, r0 + p.chunk); var m = vec4<f32>(-3.0e38);
  for (var r = r0 + l.x; r < r1; r = r + 256u) { m = max(m, vec4<f32>(S[r * 4u], S[r * 4u + 1u], S[r * 4u + 2u], S[r * 4u + 3u])); }
  sh[l.x] = m; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) { if (l.x < s) { sh[l.x] = max(sh[l.x], sh[l.x + s]); } workgroupBarrier(); }
  if (l.x == 0u) { for (var h = 0u; h < 4u; h = h + 1u) { pm[w.x * 4u + h] = sh[0][h]; } }
}`;

const MAXE = `
struct P { nwg: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pm: array<f32>;
@group(0) @binding(2) var<storage, read_write> mx: array<f32>;
@compute @workgroup_size(4)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  var m = -3.0e38; for (var q = 0u; q < p.nwg; q = q + 1u) { m = max(m, pm[q * 4u + l.x]); } mx[l.x] = m;
}`;

const SUMS = `
struct P { N: u32, C: u32, chunk: u32, lanes: u32, cp: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read> mx: array<f32>;
@group(0) @binding(3) var<storage, read> A: array<f32>;
@group(0) @binding(4) var<storage, read_write> ps: array<f32>;
var<workgroup> sh: array<vec4<f32>, 256>;
var<workgroup> se: array<vec4<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let t = l.x; let lane = t / p.cp; let r0 = w.x * p.chunk; let r1 = min(p.N, r0 + p.chunk);
  let M = vec4<f32>(mx[0], mx[1], mx[2], mx[3]);
  for (var cb = 0u; cb < p.C; cb = cb + p.cp) {
    let c = cb + t % p.cp; var acc = vec4<f32>(0.0); var es = vec4<f32>(0.0);
    if (c < p.C) {
      for (var r = r0 + lane; r < r1; r = r + p.lanes) {
        let e = exp(vec4<f32>(S[r * 4u], S[r * 4u + 1u], S[r * 4u + 2u], S[r * 4u + 3u]) - M);
        acc = acc + e * A[r * p.C + c]; es = es + e;
      }
    }
    sh[t] = acc; se[t] = es; workgroupBarrier();
    if (lane == 0u && c < p.C) {
      var s = vec4<f32>(0.0); var q4 = vec4<f32>(0.0);
      for (var q = 0u; q < p.lanes; q = q + 1u) { s = s + sh[q * p.cp + t]; q4 = q4 + se[q * p.cp + t]; }
      for (var h = 0u; h < 4u; h = h + 1u) { ps[(w.x * 4u + h) * (p.C + 1u) + c] = s[h]; if (c == 0u) { ps[(w.x * 4u + h) * (p.C + 1u) + p.C] = q4[h]; } }
    }
    workgroupBarrier();
  }
}`;

const TOKENS = `
struct P { N: u32, C: u32, g3: u32, b3: u32, g2: u32, b2: u32, nx: u32, _a: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> cv: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let v = (w.y * p.nx + w.x) * 256u + l.x; if (v >= p.N) { return; }
  let o = v * p.C; let n = f32(p.C);
  var m = 0.0; for (var c = 0u; c < p.C; c = c + 1u) { m = m + A[o + c]; } m = m / n;
  var q = 0.0; for (var c = 0u; c < p.C; c = c + 1u) { let d = A[o + c] - m; q = q + d * d; }
  let s3 = 1.0 / sqrt(q / n + 1e-5);
  var m2 = 0.0; for (var c = 0u; c < p.C; c = c + 1u) { m2 = m2 + (A[o + c] - m) * s3 * W[p.g3 + c] + W[p.b3 + c] + cv[c]; } m2 = m2 / n;
  var q2 = 0.0; for (var c = 0u; c < p.C; c = c + 1u) { let d = (A[o + c] - m) * s3 * W[p.g3 + c] + W[p.b3 + c] + cv[c] - m2; q2 = q2 + d * d; }
  let s2 = 1.0 / sqrt(q2 / n + 1e-5);
  for (var c = 0u; c < p.C; c = c + 1u) {
    let y = (A[o + c] - m) * s3 * W[p.g3 + c] + W[p.b3 + c] + cv[c];
    O[o + c] = (y - m2) * s2 * W[p.g2 + c] + W[p.b2 + c];
  }
}`;

const LABEL = `
struct P { N: u32, nx: u32, pts: u32, _a: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let q = (w.y * p.nx + w.x) * 256u + l.x; if (q * 4u >= p.N) { return; }
  var word = 0u;
  for (var i = 0u; i < 4u; i = i + 1u) {
    let v = q * 4u + i; if (v >= p.N) { break; }
    let m = 1.0 / (1.0 + exp(-Y[v * 2u])); let s = 1.0 / (1.0 + exp(-Y[v * 2u + 1u]));
    var lab = 0u; if (m > 0.5) { lab = u32(clamp(ceil(s * f32(p.pts)), 0.0, f32(p.pts))); }
    word = word | (lab << (8u * i));
  }
  O[q] = word;
}`;

export function bundleParcGpu(device: GPUDevice, model: BundleParcModel, opts: { keep?: boolean } = {}): GpuBundleParc {
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC, U = GPUBufferUsage.UNIFORM;
  const own: GPUBuffer[] = [];
  const buf = (bytes: number, usage = S | CD | CS) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage }); own.push(b); return b; };

  // ── the weights in one buffer; each tensor's offset in floats; the doubling residuals' transposed 2×2×2 weights
  //    [cin][cout][a][b][c] rearranged into eight [cout][cin] matrices, one per corner ──
  const off = new Map<string, number>(); const parts: Float32Array[] = []; let at = 0;
  const put = (k: string, a: Float32Array) => { off.set(k, at); parts.push(a); at += a.length; };
  for (const [k, t] of model.tensors) put(k, t.data);
  for (let j = 0; j < 4; j++) {
    const t = tensor(model, `decoder.layers.${j}.upsample.resconv.weight`), [cin, cout] = t.shape;
    for (let k = 0; k < 8; k++) {
      const m = new Float32Array(cout * cin);
      for (let ci = 0; ci < cin; ci++) for (let co = 0; co < cout; co++) m[co * cin + ci] = t.data[(ci * cout + co) * 8 + k];
      put(`decoder.layers.${j}.upsample.resconv.corner${k}`, m);
    }
  }
  const W = new Float32Array(at); { let o = 0; for (const p of parts) { W.set(p, o); o += p.length; } }
  const wbuf = buf(W.byteLength); device.queue.writeBuffer(wbuf, 0, W);
  const o_ = (k: string) => { const v = off.get(k); if (v === undefined) throw new Error(`BundleParc: no weights ${k}`); return v; };

  const pipe = (code: string) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const P = { gemm: pipe(GEMM), dw3: pipe(DW3), dwt2: pipe(DWT2), stats: pipe(STATS), statsEnd: pipe(STATS_END), add: pipe(ADD), scores: pipe(SCORES), maxp: pipe(MAXP), maxe: pipe(MAXE), sums: pipe(SUMS), tokens: pipe(TOKENS), label: pipe(LABEL) };

  // ── buffers, sized on the first encode for its cube ──
  let D = 0;
  let IN!: GPUBuffer, DW!: GPUBuffer, EXP!: GPUBuffer, B1!: GPUBuffer, B2!: GPUBuffer, B3!: GPUBuffer, ZB!: GPUBuffer, HEAD!: GPUBuffer, LAB!: GPUBuffer, SC!: GPUBuffer;
  const F: GPUBuffer[] = [];
  const NWG = 1024;
  const PART = buf(NWG * 512 * 4), MEAN = buf(512 * 4), AFF = buf(2 * 2048 * 4), DUMMY = buf(16), PM = buf(NWG * 4 * 4), MX = buf(16), PS = buf(NWG * 4 * 257 * 4), UV = buf(4 * 512 * 4), CV = buf(512 * 4), PE = buf(3 * 176 * 90 * 4);
  const PSREAD = buf(NWG * 4 * 257 * 4, GPUBufferUsage.MAP_READ | CD);
  const kept = new Map<string, { b: GPUBuffer; floats: number }>();

  let enc: GPUCommandEncoder | undefined; const temp: GPUBuffer[] = [];
  const encoder = () => (enc ??= device.createCommandEncoder());
  const flush = async () => { if (enc) { device.queue.submit([enc.finish()]); enc = undefined; } await device.queue.onSubmittedWorkDone(); for (const b of temp.splice(0)) b.destroy(); };
  const uni = (vals: number[]) => { const b = device.createBuffer({ size: Math.ceil(vals.length / 4) * 16, usage: U | CD }); device.queue.writeBuffer(b, 0, new Uint32Array([...vals, ...new Array((4 - vals.length % 4) % 4).fill(0)])); temp.push(b); return b; };
  const run = (pl: GPUComputePipeline, bufs: GPUBuffer[], groups: [number, number?, number?]) => {
    const bg = device.createBindGroup({ layout: pl.getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const pass = encoder().beginComputePass(); pass.setPipeline(pl); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(groups[0], groups[1] ?? 1, groups[2] ?? 1); pass.end();
  };
  const split = (n: number): [number, number] => { const nx = Math.min(n, 65535); return [nx, Math.ceil(n / nx)]; };
  const keep = (name: string, b: GPUBuffer, floats: number) => {
    if (!opts.keep) return;
    const c = device.createBuffer({ size: floats * 4, usage: CS | CD }); own.push(c);
    encoder().copyBufferToBuffer(b, 0, c, 0, floats * 4); kept.set(name, { b: c, floats });
  };

  // ── the steps ──
  /** out (N × M) = act(in'·Wᵀ + b) (+ res). */
  const gemm = (inp: GPUBuffer, out: GPUBuffer, N: number, K: number, M: number, w: string, b: string, o: { act?: 0 | 1 | 2; aff?: boolean; res?: GPUBuffer; inMode?: 0 | 1; outMode?: 0 | 1; d?: number; od?: number; corner?: number } = {}) => {
    const tiles = Math.ceil(N / 64), [nx, ny] = split(tiles), k = o.corner ?? 0;
    run(P.gemm, [uni([N, K, M, o.act ?? 0, o_(w), o_(b), o.aff ? 1 : 0, o.res ? 1 : 0, o.inMode ?? 0, o.outMode ?? 0, o.d ?? 0, o.od ?? 0, (k >> 2) & 1, (k >> 1) & 1, k & 1, nx]),
      inp, wbuf, AFF, o.res ?? DUMMY, out], [nx, ny, Math.ceil(M / 64)]);
  };
  const dw3 = (inp: GPUBuffer, out: GPUBuffer, id: number, C: number, stride: 1 | 2, pre: string) => {
    const d = stride === 1 ? id : Math.floor((id - 1) / 2) + 1, total = d * d * d * C, [nx, ny] = split(Math.ceil(total / 256));
    run(P.dw3, [uni([d, id, C, stride, o_(`${pre}.weight`), o_(`${pre}.bias`), nx, total]), inp, wbuf, out], [nx, ny]);
    return d;
  };
  const dwt2 = (inp: GPUBuffer, out: GPUBuffer, id: number, C: number, pre: string) => {
    const d = 2 * id, total = d * d * d * C, [nx, ny] = split(Math.ceil(total / 256));
    run(P.dwt2, [uni([d, id, C, 0, o_(`${pre}.weight`), o_(`${pre}.bias`), nx, total]), inp, wbuf, out], [nx, ny]);
  };
  /** The group norm's scale and shift for `inp` (N × C) into AFF. */
  const groupNorm = (inp: GPUBuffer, N: number, C: number, pre: string) => {
    const nwg = Math.min(NWG, Math.ceil(N / 256)), chunk = Math.ceil(N / nwg), cp = Math.min(C, 256), lanes = 256 / cp;
    for (const pass of [0, 1]) {
      run(P.stats, [uni([N, C, chunk, pass, lanes, cp]), inp, MEAN, PART], [nwg]);
      run(P.statsEnd, [uni([C, nwg, pass, o_(`${pre}.weight`), o_(`${pre}.bias`), N]), PART, wbuf, MEAN, AFF], [Math.ceil(C / 64)]);
    }
  };
  /** ConvNeXt block: out = in + W3·GELU(W2·GN(dw3(in))). */
  const convNext = (inp: GPUBuffer, out: GPUBuffer, d: number, C: number, r: number, pre: string, name?: string) => {
    const N = d * d * d;
    dw3(inp, DW, d, C, 1, `${pre}.conv1`); if (name) keep(`${name}.dw`, DW, N * C);
    groupNorm(DW, N, C, `${pre}.gn1`);
    gemm(DW, EXP, N, C, C * r, `${pre}.conv2.weight`, `${pre}.conv2.bias`, { act: 1, aff: true }); if (name) keep(`${name}.expand`, EXP, N * C * r);
    gemm(EXP, out, N, C * r, C, `${pre}.conv3.weight`, `${pre}.conv3.bias`, { res: inp }); if (name) keep(`${name}.out`, out, N * C);
  };
  /** The halving block: out (half grid, cout) = resconv(stride 2) + W3·GELU(W2·GN(dw3 stride 2(in))). */
  const down = (inp: GPUBuffer, res: GPUBuffer, out: GPUBuffer, d: number, cin: number, cout: number, r: number, pre: string) => {
    const h = Math.floor((d - 1) / 2) + 1, N = h * h * h;
    gemm(inp, res, N, cin, cout, `${pre}.resconv.weight`, `${pre}.resconv.bias`, { inMode: 1, d: h, od: d });
    dw3(inp, DW, d, cin, 2, `${pre}.conv1`);
    groupNorm(DW, N, cin, `${pre}.gn1`);
    gemm(DW, EXP, N, cin, cin * r, `${pre}.conv2.weight`, `${pre}.conv2.bias`, { act: 1, aff: true });
    gemm(EXP, out, N, cin * r, cout, `${pre}.conv3.weight`, `${pre}.conv3.bias`, { res });
    return h;
  };

  const ensure = (side: number) => {
    if (side === D) return;
    if (D) throw new Error("BundleParc on the card: one cube size per instance");
    D = side; const N0 = D ** 3;
    if (D % 16) throw new Error(`BundleParc on the card: the cube's side (${D}) must divide by 16`);
    IN = buf(N0 * N_COEFS * 4); DW = buf(N0 * 64 * 4); EXP = buf(N0 * 128 * 4);
    B1 = buf(N0 * 32 * 4); B2 = buf(N0 * 32 * 4); B3 = buf(N0 * 32 * 4);
    for (let l = 0; l < 4; l++) F.push(buf((D >> l) ** 3 * CHANNELS[l] * 4));
    ZB = buf((D >> 4) ** 3 * 512 * 4); HEAD = buf(N0 * 2 * 4); LAB = buf(Math.ceil(N0 / 4) * 4); SC = buf(N0 * 4 * 4);
  };

  const prompts = new Map<string, Float64Array>();
  const g: GpuBundleParc = {
    prompts,
    async encode(cube, side) {
      ensure(side);
      device.queue.writeBuffer(IN, 0, cube);
      const N0 = D ** 3;
      gemm(IN, B1, N0, N_COEFS, 32, "stem.weight", "stem.bias"); keep("stem", B1, N0 * 32);
      let d = D, x = B1;
      for (let l = 0; l < 4; l++) {
        const c = CHANNELS[l], r = ENC_RATIO[l], N = d ** 3, pre = `encoder.layers.${l}`;
        const t = x === B1 ? B2 : B1;
        convNext(x, t, d, c, r, `${pre}.conv1`, l === 0 ? "enc0.conv1" : undefined); keep(`enc${l}.conv1`, t, N * c);
        convNext(t, F[l], d, c, r, `${pre}.conv2`); keep(`enc${l}.res`, F[l], N * c);
        const h = down(F[l], B3, x, d, c, CHANNELS[l + 1], r, `${pre}.down`); keep(`enc${l}.down`, x, h ** 3 * CHANNELS[l + 1]);
        d = h;
      }
      convNext(x, ZB, d, 512, 4, "bottleneck"); keep("bottleneck", ZB, d ** 3 * 512);
      await flush();
    },
    async decode(tract) {
      if (!D) throw new Error("BundleParc on the card: encode first");
      const t0 = performance.now(), name = model.tracts[tract];
      let p = promptEmbedding(model, tract); prompts.set(`${name}.prompt`, p);
      let zin = ZB, d = D >> 4;
      for (let j = 0; j < 4; j++) {
        const cin = CHANNELS[4 - j], cout = CHANNELS[3 - j], r = DEC_RATIO[j], L = `decoder.layers.${j}`, od = 2 * d, N = od ** 3, Nin = d ** 3;
        const [b1, b2, b3] = zin === B1 ? [B2, B3, B1] : zin === B2 ? [B3, B1, B2] : [B1, B2, B3];   // b3 may be zin
        // The doubling block: resconv (8 corners) into b1; the depthwise path; + resconv.
        for (let k = 0; k < 8; k++) gemm(zin, b1, Nin, cin, cout, `${L}.upsample.resconv.corner${k}`, `${L}.upsample.resconv.bias`, { outMode: 1, d, od, corner: k });
        dwt2(zin, DW, d, cin, `${L}.upsample.conv1`);
        groupNorm(DW, N, cin, `${L}.upsample.gn1`);
        gemm(DW, EXP, N, cin, cin * r, `${L}.upsample.conv2.weight`, `${L}.upsample.conv2.bias`, { act: 1, aff: true });
        gemm(EXP, b2, N, cin * r, cout, `${L}.upsample.conv3.weight`, `${L}.upsample.conv3.bias`, { res: b1 }); keep(`${name}.dec${j}.up`, b2, N * cout);
        // + the encoder's output at this level, then a ConvNeXt block.
        { const total = N * cout, [nx, ny] = split(Math.ceil(total / 256)); run(P.add, [uni([total, nx]), b2, F[3 - j], b1], [nx, ny]); }
        convNext(b1, b2, od, cout, r, `${L}.conv1`); keep(`${name}.dec${j}.conv1`, b2, N * cout);
        // The prompt's attention: scores, the maximum, the partial sums -- read back once.
        p = promptIn(model, j, p); prompts.set(`${name}.dec${j}.promptIn`, p);
        const { u } = attnScores(model, j, p), pe = positionTables(cout, [od, od, od]);
        const peAll = new Float32Array(3 * od * pe.chp); pe.tables.forEach((t, a) => peAll.set(t, a * od * pe.chp));
        device.queue.writeBuffer(UV, 0, u); device.queue.writeBuffer(PE, 0, peAll);
        { const [nx, ny] = split(Math.ceil(N / 256)); run(P.scores, [uni([N, cout, od, pe.chp, nx]), b2, PE, UV, SC], [nx, ny]); }
        const nwg = Math.min(NWG, Math.ceil(N / 256)), chunk = Math.ceil(N / nwg), cp = Math.min(cout, 256), lanes = 256 / cp;
        run(P.maxp, [uni([N, chunk]), SC, PM], [nwg]);
        run(P.maxe, [uni([nwg]), PM, MX], [1]);
        run(P.sums, [uni([N, cout, chunk, lanes, cp]), SC, MX, b2, PS], [nwg]);
        const psFloats = nwg * 4 * (cout + 1);
        encoder().copyBufferToBuffer(PS, 0, PSREAD, 0, psFloats * 4);
        await flush();
        await PSREAD.mapAsync(GPUMapMode.READ, 0, psFloats * 4);
        const ps = new Float32Array(PSREAD.getMappedRange(0, psFloats * 4).slice(0)); PSREAD.unmap();
        const avg: Float64Array[] = [];
        for (let h = 0; h < HEADS; h++) {
          const a = new Float64Array(cout); let e = 0;
          for (let q = 0; q < nwg; q++) { const o = (q * 4 + h) * (cout + 1); for (let c = 0; c < cout; c++) a[c] += ps[o + c]; e += ps[o + cout]; }
          for (let c = 0; c < cout; c++) a[c] /= e; avg.push(a);
        }
        const upd = attnUpdate(model, j, p, avg); p = upd.prompt; prompts.set(`${name}.dec${j}.promptOut`, p);
        device.queue.writeBuffer(CV, 0, upd.toVoxels);
        // The voxels' MLP (ReLU) with its residual, then the two layer norms around the vector every voxel gets.
        gemm(b2, EXP, N, cout, cout * r, `${L}.tkn2img.mlp.lin1.weight`, `${L}.tkn2img.mlp.lin1.bias`, { act: 2 });
        gemm(EXP, b1, N, cout * r, cout, `${L}.tkn2img.mlp.lin2.weight`, `${L}.tkn2img.mlp.lin2.bias`, { res: b2 });
        { const [nx, ny] = split(Math.ceil(N / 256)); run(P.tokens, [uni([N, cout, o_(`${L}.tkn2img.norm3.weight`), o_(`${L}.tkn2img.norm3.bias`), o_(`${L}.tkn2img.norm2.weight`), o_(`${L}.tkn2img.norm2.bias`), nx]), b1, wbuf, CV, b3], [nx, ny]); }
        keep(`${name}.dec${j}.attn`, b3, N * cout);
        convNext(b3, b1, od, cout, r, `${L}.conv2`); keep(`${name}.dec${j}.out`, b1, N * cout);
        zin = b1; d = od;
      }
      const N0 = D ** 3;
      gemm(zin, HEAD, N0, 32, 2, "head.conv.weight", "head.conv.bias"); keep(`${name}.head`, HEAD, N0 * 2);
      { const q = Math.ceil(N0 / 4), [nx, ny] = split(Math.ceil(q / 256)); run(P.label, [uni([N0, nx, 10]), HEAD, LAB], [nx, ny]); }
      const bytes = Math.ceil(N0 / 4) * 4, rd = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | CD });
      encoder().copyBufferToBuffer(LAB, 0, rd, 0, bytes);
      await flush();
      await rd.mapAsync(GPUMapMode.READ);
      const labels = new Uint8Array(rd.getMappedRange().slice(0)).subarray(0, N0); rd.unmap(); rd.destroy();
      return { labels, seconds: (performance.now() - t0) / 1000 };
    },
    async read(name) {
      const k = kept.get(name); if (!k) throw new Error(`BundleParc on the card: nothing kept as ${name}`);
      const rd = device.createBuffer({ size: k.floats * 4, usage: GPUBufferUsage.MAP_READ | CD });
      const e = device.createCommandEncoder(); e.copyBufferToBuffer(k.b, 0, rd, 0, k.floats * 4); device.queue.submit([e.finish()]);
      await rd.mapAsync(GPUMapMode.READ); const out = new Float32Array(rd.getMappedRange().slice(0)); rd.unmap(); rd.destroy();
      return out;
    },
    destroy() { for (const b of own) b.destroy(); for (const k of kept.values()) k.b.destroy(); },
  };
  return g;
}
