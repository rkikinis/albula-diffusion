// RAPIDPARC ON THE GRAPHICS CARD -- the same network as rapidparc.ts (the processor version, itself checked against
// RapidParc in PyTorch), as a chain of compute steps over a batch of groups of 2,000 streamlines. Single precision, as the
// original; checked against the processor version in rapidparc-gpu.test.ts.
//
// The steps, per batch of G groups (R = G × 2,000 rows):
//   mm         C = act(scale · A·B + bias): one general multiply, 16 × 16 tiles, strided so it serves every linear layer
//              (B = Wᵀ read from the weights), the attention scores (Q·Kᵀ per group, z = group) and the weighted sum
//              (softmax · V per group).
//   softmax    one workgroup per row of scores (2,000 wide), in place.
//   addnorm    out = LayerNorm(a + b), one workgroup per row (128 wide).
//   name       one thread per row: the cluster of the largest logit, and the margin (softmax over the 1,600, summed per
//              tract: this tract's share minus the best other's), as tractcloud-gpu.ts.
// The group's scaling to [-1, 1] and the 45 coordinates are done on the processor (a few hundred thousand numbers).
import { CONTEXT, D, FF, HIDDEN, CLASSES, normalizeCube, type Linear, type RapidParcModel } from "./rapidparc.ts";

export interface GpuRapidParc {
  /** One draw: rows (n × 45, the whole set already scaled; rapidparc.ts groupRows of them in the draw's order, padded to
   *  `groups` × 2,000). Returns, per padded row, the cluster (0..1599) and the margin. */
  classify(rows: Float32Array, groups: number): Promise<{ cluster: Int32Array; margin: Float32Array }>;
  destroy(): void;
}

export function rapidParcGpu(device: GPUDevice, model: RapidParcModel, lut: Int32Array, batchGroups = 4): GpuRapidParc {
  const own: GPUBuffer[] = [];
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const buf = (bytes: number, usage: number) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage }); own.push(b); return b; };

  // ── the weights, in one buffer; each tensor's offset (in floats) ──
  const parts: Float32Array[] = []; let at = 0;
  const put = (a: Float32Array) => { const o = at; parts.push(a); at += a.length; return o; };
  const lin = (L: Linear) => ({ L, W: put(L.W), b: put(L.b) });
  const emb = lin(model.emb), cls1 = lin(model.cls1), cls2 = lin(model.cls2);
  const layers = model.layers.map((L) => ({ qkv: lin(L.qkv), o: lin(L.o), ff1: lin(L.ff1), ff2: lin(L.ff2), n1g: put(L.n1g), n1b: put(L.n1b), n2g: put(L.n2g), n2b: put(L.n2b) }));
  const W = new Float32Array(at); { let o = 0; for (const p of parts) { W.set(p, o); o += p.length; } }
  const wbuf = buf(W.byteLength, S | CD); device.queue.writeBuffer(wbuf, 0, W);
  const lutBuf = buf(CLASSES * 4, S | CD); device.queue.writeBuffer(lutBuf, 0, lut.subarray(0, CLASSES));
  const T = Math.max(...Array.from(lut.subarray(0, CLASSES))) + 1;

  // ── activations for one batch ──
  const R = batchGroups * CONTEXT;
  const in45 = buf(R * 45 * 4, S | CD), x = buf(R * D * 4, S | CD), h = buf(R * D * 4, S), y = buf(R * D * 4, S), att = buf(R * D * 4, S);
  const qkv = buf(R * 3 * D * 4, S), ff = buf(R * FF * 4, S), hid = buf(R * HIDDEN * 4, S), logits = buf(R * CLASSES * 4, S);
  const scores = buf(batchGroups * CONTEXT * CONTEXT * 4, S);
  const outC = buf(R * 4, S | CS), outM = buf(R * 4, S | CS);
  const readC = buf(R * 4, GPUBufferUsage.MAP_READ | CD), readM = buf(R * 4, GPUBufferUsage.MAP_READ | CD);

  const mm = device.createComputePipeline({ layout: "auto", compute: { entryPoint: "main", module: device.createShaderModule({ code: `
struct P { M: u32, N: u32, K: u32, act: u32,
  aoff: u32, ars: u32, acs: u32, abs_: u32,
  boff: u32, brs: u32, bcs: u32, bbs: u32,
  coff: u32, crs: u32, cbs: u32, bias: u32,
  scale: f32, _a: f32, _b: f32, _c: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
var<workgroup> ta: array<array<f32, 16>, 16>;
var<workgroup> tb: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let row = w.y * 16u + l.y; let col = w.x * 16u + l.x; let z = w.z;
  let a0 = p.aoff + z * p.abs_; let b0 = p.boff + z * p.bbs;
  var acc = 0.0;
  for (var t = 0u; t < p.K; t = t + 16u) {
    let ak = t + l.x; let bk = t + l.y;
    ta[l.y][l.x] = select(0.0, A[a0 + row * p.ars + ak * p.acs], row < p.M && ak < p.K);
    tb[l.y][l.x] = select(0.0, B[b0 + bk * p.brs + col * p.bcs], bk < p.K && col < p.N);
    workgroupBarrier();
    for (var i = 0u; i < 16u; i = i + 1u) { acc = acc + ta[l.y][i] * tb[i][l.x]; }
    workgroupBarrier();
  }
  if (row < p.M && col < p.N) {
    var v = acc * p.scale;
    if (p.bias != 0xffffffffu) { v = v + B[p.bias + col]; }
    if (p.act == 1u) { v = max(v, 0.0); } else if (p.act == 2u) { v = select(0.01 * v, v, v > 0.0); }
    C[p.coff + z * p.cbs + row * p.crs + col] = v;
  }
}` }) } });

  const softmax = device.createComputePipeline({ layout: "auto", compute: { entryPoint: "main", module: device.createShaderModule({ code: `
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let n = ${CONTEXT}u; let o = (w.x + w.y * 65535u) * n;
  var m = -3.4e38;
  for (var i = l.x; i < n; i = i + 256u) { m = max(m, S[o + i]); }
  red[l.x] = m; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) { if (l.x < s) { red[l.x] = max(red[l.x], red[l.x + s]); } workgroupBarrier(); }
  let mx = red[0]; workgroupBarrier();
  var sum = 0.0;
  for (var i = l.x; i < n; i = i + 256u) { let e = exp(S[o + i] - mx); S[o + i] = e; sum = sum + e; }
  red[l.x] = sum; workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) { if (l.x < s) { red[l.x] = red[l.x] + red[l.x + s]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  for (var i = l.x; i < n; i = i + 256u) { S[o + i] = S[o + i] * inv; }
}` }) } });

  const addnorm = device.createComputePipeline({ layout: "auto", compute: { entryPoint: "main", module: device.createShaderModule({ code: `
struct P { g: u32, b: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> Y: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<storage, read> Wt: array<f32>;
var<workgroup> red: array<f32, ${D}>;
@compute @workgroup_size(${D})
fn main(@builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
  let i = (w.x + w.y * 65535u) * ${D}u + l.x;
  let v = X[i] + Y[i];
  red[l.x] = v; workgroupBarrier();
  for (var s = ${D / 2}u; s > 0u; s = s >> 1u) { if (l.x < s) { red[l.x] = red[l.x] + red[l.x + s]; } workgroupBarrier(); }
  let mean = red[0] / ${D}.0; workgroupBarrier();
  let d = v - mean; red[l.x] = d * d; workgroupBarrier();
  for (var s = ${D / 2}u; s > 0u; s = s >> 1u) { if (l.x < s) { red[l.x] = red[l.x] + red[l.x + s]; } workgroupBarrier(); }
  let inv = 1.0 / sqrt(red[0] / ${D}.0 + 1e-5);
  O[i] = d * inv * Wt[p.g + l.x] + Wt[p.b + l.x];
}` }) } });

  const name = device.createComputePipeline({ layout: "auto", compute: { entryPoint: "main", module: device.createShaderModule({ code: `
struct P { rows: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> L: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<i32>;
@group(0) @binding(3) var<storage, read_write> cluster: array<i32>;
@group(0) @binding(4) var<storage, read_write> margin: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = g.x; if (r >= p.rows) { return; }
  let o = r * ${CLASSES}u;
  var best = 0u; var mx = L[o];
  for (var c = 1u; c < ${CLASSES}u; c = c + 1u) { if (L[o + c] > mx) { mx = L[o + c]; best = c; } }
  cluster[r] = i32(best);
  var pt: array<f32, ${T}>;
  var total = 0.0;
  for (var c = 0u; c < ${CLASSES}u; c = c + 1u) { let e = exp(L[o + c] - mx); pt[lut[c]] = pt[lut[c]] + e; total = total + e; }
  let mine = lut[best]; var other = 0.0;
  for (var t = 0; t < ${T}; t = t + 1) { if (t != mine) { other = max(other, pt[t]); } }
  margin[r] = (pt[mine] - other) / total;
}` }) } });

  // ── the steps of one batch, with their parameters fixed once (G groups) ──
  const NONE = 0xffffffff;
  type Mm = { M: number; N: number; K: number; act?: number; A: GPUBuffer; aoff?: number; ars: number; acs: number; abs?: number;
    B: GPUBuffer; boff?: number; brs: number; bcs: number; bbs?: number; C: GPUBuffer; coff?: number; crs: number; cbs?: number; bias?: number; scale?: number; z?: number };
  const steps: ((pass: GPUComputePassEncoder) => void)[] = [];
  const addMm = (s: Mm) => {
    const u = new ArrayBuffer(80), U = new Uint32Array(u), F = new Float32Array(u);
    U.set([s.M, s.N, s.K, s.act ?? 0, s.aoff ?? 0, s.ars, s.acs, s.abs ?? 0, s.boff ?? 0, s.brs, s.bcs, s.bbs ?? 0, s.coff ?? 0, s.crs, s.cbs ?? 0, s.bias ?? NONE]);
    F[16] = s.scale ?? 1;
    const ub = buf(80, GPUBufferUsage.UNIFORM | CD); device.queue.writeBuffer(ub, 0, u);
    const bg = device.createBindGroup({ layout: mm.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: s.A } }, { binding: 2, resource: { buffer: s.B } }, { binding: 3, resource: { buffer: s.C } }] });
    steps.push((pass) => { pass.setPipeline(mm); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(s.N / 16), Math.ceil(s.M / 16), s.z ?? 1); });
  };
  /** y = act(x·Wᵀ + b) over all R rows; x has xc columns, y yc columns (written from column yo). */
  const linear = (l: { L: Linear; W: number; b: number }, X: GPUBuffer, xc: number, Y: GPUBuffer, yc: number, act = 0, yo = 0) =>
    addMm({ M: R, N: l.L.out, K: l.L.inp, act, A: X, ars: xc, acs: 1, B: wbuf, boff: l.W, brs: 1, bcs: l.L.inp, C: Y, coff: yo, crs: yc, bias: l.b });
  const rowsXY = (n: number) => [Math.min(n, 65535), Math.ceil(n / 65535)] as const;
  const addAddNorm = (X: GPUBuffer, Y: GPUBuffer, O: GPUBuffer, g: number, b: number) => {
    const ub = buf(16, GPUBufferUsage.UNIFORM | CD); device.queue.writeBuffer(ub, 0, Uint32Array.from([g, b, 0, 0]));
    const bg = device.createBindGroup({ layout: addnorm.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: X } }, { binding: 2, resource: { buffer: Y } }, { binding: 3, resource: { buffer: O } }, { binding: 4, resource: { buffer: wbuf } }] });
    steps.push((pass) => { pass.setPipeline(addnorm); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(...rowsXY(R)); });
  };
  const smBg = device.createBindGroup({ layout: softmax.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: scores } }] });
  const G = batchGroups, C3 = 3 * D;

  linear(emb, in45, 45, x, D, 2, 45);                                     // the 83 learned features beside the 45 coordinates
  for (const L of layers) {
    linear(L.qkv, x, D, qkv, C3);
    // scores[g] = Q·Kᵀ / √128 for each group g (z): Q rows of the group, K the same rows' second third.
    addMm({ M: CONTEXT, N: CONTEXT, K: D, A: qkv, ars: C3, acs: 1, abs: CONTEXT * C3, B: qkv, boff: D, brs: 1, bcs: C3, bbs: CONTEXT * C3,
      C: scores, crs: CONTEXT, cbs: CONTEXT * CONTEXT, scale: 1 / Math.sqrt(D), z: G });
    steps.push((pass) => { pass.setPipeline(softmax); pass.setBindGroup(0, smBg); pass.dispatchWorkgroups(...rowsXY(G * CONTEXT)); });
    // att[g] = softmax · V
    addMm({ M: CONTEXT, N: D, K: CONTEXT, A: scores, ars: CONTEXT, acs: 1, abs: CONTEXT * CONTEXT, B: qkv, boff: 2 * D, brs: C3, bcs: 1, bbs: CONTEXT * C3,
      C: att, crs: D, cbs: CONTEXT * D, z: G });
    linear(L.o, att, D, y, D);
    addAddNorm(x, y, h, L.n1g, L.n1b);
    linear(L.ff1, h, D, ff, FF, 1);
    linear(L.ff2, ff, FF, y, D);
    addAddNorm(h, y, x, L.n2g, L.n2b);
  }
  linear(cls1, x, D, hid, HIDDEN, 1);
  linear(cls2, hid, HIDDEN, logits, CLASSES);
  { const ub = buf(16, GPUBufferUsage.UNIFORM | CD); device.queue.writeBuffer(ub, 0, Uint32Array.from([R, 0, 0, 0]));
    const bg = device.createBindGroup({ layout: name.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ub } }, { binding: 1, resource: { buffer: logits } }, { binding: 2, resource: { buffer: lutBuf } }, { binding: 3, resource: { buffer: outC } }, { binding: 4, resource: { buffer: outM } }] });
    steps.push((pass) => { pass.setPipeline(name); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(R / 64)); }); }

  return {
    async classify(rows, groups) {
      const cluster = new Int32Array(groups * CONTEXT), margin = new Float32Array(groups * CONTEXT);
      for (let g0 = 0; g0 < groups; g0 += G) {
        // The groups of this batch, each scaled to [-1, 1] on its own (as the network does); a short last batch is
        // padded with copies of its first group, whose results are dropped.
        const ins = new Float32Array(R * 45), xs = new Float32Array(R * D);
        for (let g = 0; g < G; g++) {
          const src = Math.min(g0 + g, groups - 1);
          const n45 = normalizeCube(rows.subarray(src * CONTEXT * 45, (src + 1) * CONTEXT * 45));
          ins.set(n45, g * CONTEXT * 45);
          for (let r = 0; r < CONTEXT; r++) xs.set(n45.subarray(r * 45, r * 45 + 45), (g * CONTEXT + r) * D);
        }
        device.queue.writeBuffer(in45, 0, ins); device.queue.writeBuffer(x, 0, xs);
        const enc = device.createCommandEncoder();
        // One pass a step: each step reads what the one before it wrote.
        for (const st of steps) { const pass = enc.beginComputePass(); st(pass); pass.end(); }
        const real = Math.min(G, groups - g0) * CONTEXT;
        enc.copyBufferToBuffer(outC, 0, readC, 0, real * 4); enc.copyBufferToBuffer(outM, 0, readM, 0, real * 4);
        device.queue.submit([enc.finish()]);
        await Promise.all([readC.mapAsync(GPUMapMode.READ, 0, real * 4), readM.mapAsync(GPUMapMode.READ, 0, real * 4)]);
        cluster.set(new Int32Array(readC.getMappedRange(0, real * 4)).slice(), g0 * CONTEXT);
        margin.set(new Float32Array(readM.getMappedRange(0, real * 4)).slice(), g0 * CONTEXT);
        readC.unmap(); readM.unmap();
      }
      return { cluster, margin };
    },
    destroy() { for (const b of own) b.destroy(); },
  };
}
