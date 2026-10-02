// TRACTCLOUD ON THE GRAPHICS CARD -- the same network as tractcloud.ts (the processor version, itself checked against
// the original in PyTorch), as a chain of small compute steps over a batch of streamlines. Single precision; checked
// against the processor version in tractcloud.test.ts.
//
// The steps, per batch (B streamlines, P = 15 points):
//   conv1     one thread per (streamline, feature, point): the 100 context streamlines, maxed.   -> cat[:, 0:64]
//   knn       one thread per (streamline, point): its 5 nearest points in the current features.
//   edge A/B  one thread per (streamline, feature, point): the two halves of W [x_q - x_p, x_p] = Wa x_q + (Wb-Wa) x_p.
//   edge max  one thread per (streamline, feature, point): the layer over the 5 neighbors, maxed.  -> cat[:, 64:128] …
//   conv5     one thread per (streamline, feature): 512 in, 15 points, then max and mean over the points.
//   dense ×3  one thread per (streamline, output).
//   argmax    one thread per streamline.
// Batch norms are folded into scale and shift, as in the processor version. At most 5 storage buffers per step (the
// standard allows 8).
import type { TractCloudModel } from "./tractcloud.ts";

const WG = 64;

export interface GpuTractCloud {
  /** Clusters (0..1599) of all streamlines. ctx (N, 100): for each streamline, the streamline index of each context
   *  (its 20 local neighbors, then the 80 global ones). */
  classify(feat: Float32Array, ctx: Int32Array): Promise<Int32Array>;
  /**
   * For the last classify, per streamline: how sure the name is -- the probability of its tract (softmax over the 1,600
   * clusters, summed per tract) minus that of the best other tract. Mike Halle keeps this margin in his rank field: on an
   * HCP subject it predicts which streamlines change name across TractCloud's random draws (AUROC 0.875; his mail,
   * 2026-10-01). The name itself is unchanged: the tract of the most likely cluster, as upstream.
   */
  margins(): Float32Array;
  destroy(): void;
}

export function tractCloudGpu(device: GPUDevice, model: TractCloudModel, batch = 1024): GpuTractCloud {
  const P = model.P, { leak, k, kGlobal, kPoint } = model.json.settings, KC = k + kGlobal;
  const own: GPUBuffer[] = [];
  const buf = (size: number, usage: number) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage }); own.push(b); return b; };
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const upload = (a: Float32Array | Int32Array) => { const b = buf(a.byteLength, S | CD); device.queue.writeBuffer(b, 0, a); return b; };

  // ── weights, folded ──
  const f32 = (a: ArrayLike<number>) => Float32Array.from(a);
  /** An edge layer's weights as [WA (out×C) | WD (out×C) | s (out) | t (out)]. */
  const edgeWeights = (L: TractCloudModel["conv2"], C: number) => {
    const w = new Float32Array(2 * L.out * C + 2 * L.out);
    for (let o = 0; o < L.out; o++) for (let c = 0; c < C; c++) {
      const wa = L.W[o * L.inp + c], wb = L.W[o * L.inp + C + c];
      w[o * C + c] = wa; w[L.out * C + o * C + c] = wb - wa;
    }
    w.set(f32(L.s), 2 * L.out * C); w.set(f32(L.t), 2 * L.out * C + L.out);
    return upload(w);
  };
  const denseWeights = (L: TractCloudModel["lin1"]) => {
    const w = new Float32Array(L.out * L.inp + 2 * L.out);
    w.set(f32(L.W), 0); w.set(f32(L.s), L.out * L.inp); w.set(f32(L.t), L.out * L.inp + L.out);
    return upload(w);
  };
  const w1 = edgeWeights(model.conv1, 3), w2 = edgeWeights(model.conv2, 64), w3 = edgeWeights(model.conv3, 64), w4 = edgeWeights(model.conv4, 128);
  const w5 = denseWeights(model.conv5), l1 = denseWeights(model.lin1), l2 = denseWeights(model.lin2), l3 = denseWeights(model.lin3);

  // ── activations for one batch ──
  const CAT = 512;
  const cat = buf(batch * CAT * P * 4, S);                  // x1 | x2 | x3 | x4, channel-major per streamline
  const A = buf(batch * 256 * P * 4, S), Bh = buf(batch * 256 * P * 4, S);
  const nb = buf(batch * P * kPoint * 4, S);
  const pooled = buf(batch * 2048 * 4, S), h1 = buf(batch * 512 * 4, S), h2 = buf(batch * 256 * 4, S), logits = buf(batch * 1600 * 4, S);
  const out = buf(batch * 4, S | CS), read = buf(batch * 4, GPUBufferUsage.MAP_READ | CD);
  const marginOut = buf(batch * 4, S | CS), readMargin = buf(batch * 4, GPUBufferUsage.MAP_READ | CD);
  const T = model.json.tracts.length;
  const lut = buf(1600 * 4, S | CD); device.queue.writeBuffer(lut, 0, Int32Array.from(model.json.clusterToTract.slice(0, 1600)));
  let lastMargins = new Float32Array(0);
  const params = buf(16, GPUBufferUsage.UNIFORM | CD);      // count in this batch, first streamline

  const lr = `fn lr(x: f32) -> f32 { return select(${leak} * x, x, x > 0.0); }`;
  const head = `struct Prm { count: u32, first: u32, _a: u32, _b: u32 };\n@group(0) @binding(0) var<uniform> prm: Prm;\n${lr}\n`;
  const pipe = (code: string) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: head + code }), entryPoint: "main" } });

  const conv1 = pipe(`
@group(0) @binding(1) var<storage, read> feat: array<f32>;
@group(0) @binding(2) var<storage, read> ctx: array<i32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> cat: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= prm.count * ${64 * P}u) { return; }
  let b = i / ${64 * P}u; let o = (i / ${P}u) % 64u; let p = i % ${P}u;
  let f = prm.first + b;
  let x = vec3f(feat[f * ${3 * P}u + p * 3u], feat[f * ${3 * P}u + p * 3u + 1u], feat[f * ${3 * P}u + p * 3u + 2u]);
  let wa = vec3f(w[o * 3u], w[o * 3u + 1u], w[o * 3u + 2u]);
  let wd = vec3f(w[192u + o * 3u], w[192u + o * 3u + 1u], w[192u + o * 3u + 2u]);
  let s = w[384u + o]; let t = w[448u + o];
  let base = dot(wd, x);
  var m = -3.4e38;
  for (var j = 0u; j < ${KC}u; j++) {
    let c = u32(ctx[f * ${KC}u + j]) * ${3 * P}u + p * 3u;
    m = max(m, lr(s * (dot(wa, vec3f(feat[c], feat[c + 1u], feat[c + 2u])) + base) + t));
  }
  cat[b * ${CAT * P}u + o * ${P}u + p] = m;
}`);

  const knn = (C: number, off: number) => pipe(`
@group(0) @binding(1) var<storage, read> cat: array<f32>;
@group(0) @binding(2) var<storage, read_write> nb: array<u32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= prm.count * ${P}u) { return; }
  let b = i / ${P}u; let p = i % ${P}u; let base = b * ${CAT * P}u + ${off * P}u;
  var d: array<f32, ${P}>;
  for (var q = 0u; q < ${P}u; q++) {
    var s = 0.0;
    for (var c = 0u; c < ${C}u; c++) { let e = cat[base + c * ${P}u + p] - cat[base + c * ${P}u + q]; s += e * e; }
    d[q] = s;
  }
  var taken = 0u;
  for (var j = 0u; j < ${kPoint}u; j++) {
    var best = 0u; var bd = 3.4e38;
    for (var q = 0u; q < ${P}u; q++) { if (((taken >> q) & 1u) == 0u && d[q] < bd) { bd = d[q]; best = q; } }
    taken |= 1u << best; nb[i * ${kPoint}u + j] = best;
  }
}`);

  const edgeAB = (C: number, out: number, off: number) => pipe(`
@group(0) @binding(1) var<storage, read> cat: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
@group(0) @binding(4) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= prm.count * ${out * P}u) { return; }
  let b = i / ${out * P}u; let o = (i / ${P}u) % ${out}u; let p = i % ${P}u;
  let base = b * ${CAT * P}u + ${off * P}u + p;
  var a = 0.0; var d = 0.0;
  for (var c = 0u; c < ${C}u; c++) { let v = cat[base + c * ${P}u]; a += w[o * ${C}u + c] * v; d += w[${out * C}u + o * ${C}u + c] * v; }
  A[i] = a; B[i] = d;
}`);

  const edgeMax = (C: number, out: number, dst: number) => pipe(`
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read> nb: array<u32>;
@group(0) @binding(4) var<storage, read> w: array<f32>;
@group(0) @binding(5) var<storage, read_write> cat: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= prm.count * ${out * P}u) { return; }
  let b = i / ${out * P}u; let o = (i / ${P}u) % ${out}u; let p = i % ${P}u;
  let s = w[${2 * out * C}u + o]; let t = w[${2 * out * C + out}u + o];
  let row = b * ${out * P}u + o * ${P}u;
  var m = -3.4e38;
  for (var j = 0u; j < ${kPoint}u; j++) { let q = nb[(b * ${P}u + p) * ${kPoint}u + j]; m = max(m, lr(s * (A[row + q] + B[row + p]) + t)); }
  cat[b * ${CAT * P}u + ${dst * P}u + o * ${P}u + p] = m;
}`);

  const conv5 = pipe(`
@group(0) @binding(1) var<storage, read> cat: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> pooled: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= prm.count * 1024u) { return; }
  let b = i / 1024u; let o = i % 1024u;
  var acc: array<f32, ${P}>;
  let base = b * ${CAT * P}u;
  for (var c = 0u; c < ${CAT}u; c++) {
    let wv = w[o * ${CAT}u + c];
    for (var p = 0u; p < ${P}u; p++) { acc[p] += wv * cat[base + c * ${P}u + p]; }
  }
  let s = w[${1024 * CAT}u + o]; let t = w[${1024 * CAT + 1024}u + o];
  var mx = -3.4e38; var sm = 0.0;
  for (var p = 0u; p < ${P}u; p++) { let v = lr(s * acc[p] + t); mx = max(mx, v); sm += v; }
  pooled[b * 2048u + o] = mx; pooled[b * 2048u + 1024u + o] = sm / ${P}.0;
}`);

  const dense = (inp: number, out: number, act: boolean) => pipe(`
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= prm.count * ${out}u) { return; }
  let b = i / ${out}u; let o = i % ${out}u;
  var s = 0.0;
  for (var c = 0u; c < ${inp}u; c++) { s += w[o * ${inp}u + c] * x[b * ${inp}u + c]; }
  let v = w[${out * inp}u + o] * s + w[${out * inp + out}u + o];
  y[i] = ${act ? "lr(v)" : "v"};
}`);

  const argmax = pipe(`
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<i32>;
@group(0) @binding(3) var<storage, read> lut: array<i32>;
@group(0) @binding(4) var<storage, read_write> margin: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let b = g.x; if (b >= prm.count) { return; }
  var best = 0u; var bv = x[b * 1600u];
  for (var c = 1u; c < 1600u; c++) { let v = x[b * 1600u + c]; if (v > bv) { bv = v; best = c; } }
  y[b] = i32(best);
  // The margin: softmax over the clusters (shifted by the largest logit), summed per tract.
  var p: array<f32, ${T}>;
  var total = 0.0;
  for (var c = 0u; c < 1600u; c++) {
    let e = exp(x[b * 1600u + c] - bv);
    total += e;
    let t = lut[c];
    if (t >= 0 && t < ${T}) { p[t] += e; }
  }
  let mine = lut[best];
  var other = 0.0;
  for (var t = 0; t < ${T}; t++) { if (t != mine) { other = max(other, p[t]); } }
  margin[b] = (p[mine] - other) / total;
}`);

  const K = { knn64a: knn(64, 0), knn64b: knn(64, 64), knn128: knn(128, 128),
    ab2: edgeAB(64, 64, 0), ab3: edgeAB(64, 128, 64), ab4: edgeAB(128, 256, 128),
    mx2: edgeMax(64, 64, 64), mx3: edgeMax(64, 128, 128), mx4: edgeMax(128, 256, 256),
    d1: dense(2048, 512, true), d2: dense(512, 256, true), d3: dense(256, 1600, false) };

  const bind = (p: GPUComputePipeline, bufs: GPUBuffer[]) => device.createBindGroup({ layout: p.getBindGroupLayout(0),
    entries: [params, ...bufs].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const groups = (n: number) => Math.ceil(n / WG);

  let featBuf: GPUBuffer | undefined, ctxBuf: GPUBuffer | undefined;
  return {
    async classify(feat, ctx) {
      const N = feat.length / (3 * P);
      featBuf?.destroy(); ctxBuf?.destroy();
      featBuf = device.createBuffer({ size: feat.byteLength, usage: S | CD }); device.queue.writeBuffer(featBuf, 0, feat);
      ctxBuf = device.createBuffer({ size: ctx.byteLength, usage: S | CD }); device.queue.writeBuffer(ctxBuf, 0, ctx);
      const steps: [GPUComputePipeline, GPUBuffer[], (n: number) => number][] = [
        [conv1, [featBuf, ctxBuf, w1, cat], (n) => n * 64 * P],
        [K.knn64a, [cat, nb], (n) => n * P], [K.ab2, [cat, w2, A, Bh], (n) => n * 64 * P], [K.mx2, [A, Bh, nb, w2, cat], (n) => n * 64 * P],
        [K.knn64b, [cat, nb], (n) => n * P], [K.ab3, [cat, w3, A, Bh], (n) => n * 128 * P], [K.mx3, [A, Bh, nb, w3, cat], (n) => n * 128 * P],
        [K.knn128, [cat, nb], (n) => n * P], [K.ab4, [cat, w4, A, Bh], (n) => n * 256 * P], [K.mx4, [A, Bh, nb, w4, cat], (n) => n * 256 * P],
        [conv5, [cat, w5, pooled], (n) => n * 1024],
        [K.d1, [pooled, l1, h1], (n) => n * 512], [K.d2, [h1, l2, h2], (n) => n * 256], [K.d3, [h2, l3, logits], (n) => n * 1600],
        [argmax, [logits, out, lut, marginOut], (n) => n],
      ];
      const bound = steps.map(([p, b, n]) => [p, bind(p, b), n] as const);
      const result = new Int32Array(N), margins = new Float32Array(N);
      for (let first = 0; first < N; first += batch) {
        const count = Math.min(batch, N - first);
        device.queue.writeBuffer(params, 0, new Uint32Array([count, first, 0, 0]));
        const enc = device.createCommandEncoder();
        for (const [p, bg, n] of bound) { const pass = enc.beginComputePass(); pass.setPipeline(p); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(groups(n(count))); pass.end(); }
        enc.copyBufferToBuffer(out, 0, read, 0, count * 4);
        enc.copyBufferToBuffer(marginOut, 0, readMargin, 0, count * 4);
        device.queue.submit([enc.finish()]);                // one batch a submit: short command buffers (the macOS watchdog)
        await Promise.all([read.mapAsync(GPUMapMode.READ, 0, count * 4), readMargin.mapAsync(GPUMapMode.READ, 0, count * 4)]);
        result.set(new Int32Array(read.getMappedRange(0, count * 4)).slice(), first);
        margins.set(new Float32Array(readMargin.getMappedRange(0, count * 4)).slice(), first);
        read.unmap(); readMargin.unmap();
      }
      lastMargins = margins;
      return result;
    },
    margins: () => lastMargins,
    destroy() { for (const b of own) b.destroy(); featBuf?.destroy(); ctxBuf?.destroy(); },
  };
}

/** The contexts of every streamline as streamline indices: its k local neighbors (topk indexes ds), then the globals. */
export function contexts(topk: Int32Array, ds: Int32Array, glob: Int32Array, k: number): Int32Array {
  const N = topk.length / k, KC = k + glob.length, out = new Int32Array(N * KC);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < k; j++) out[i * KC + j] = ds[topk[i * k + j]];
    out.set(glob, i * KC + k);
  }
  return out;
}
