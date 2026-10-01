// TRACTCLOUD, WRITTEN FROM FIRST PRINCIPLES -- the network that names tracts (Xue, Zhang, O'Donnell et al., "TractCloud:
// registration-free tractography parcellation with a novel local-global streamline point cloud representation",
// MICCAI 2023). Ron, 2026-09-30: "TractCloud first"; the port is plain code, no machine-learning runtime ("why not c?").
// The trained weights are SlicerDMRI's release v1.0.0, converted by model/make-model.py (README there).
//
// WHAT IT DOES. Every streamline is resampled to 15 points at equal arc length and the whole set is moved so its mean
// lands on the atlas's (HCP) mean. Each streamline is then shown to the network with context: its 20 nearest streamlines
// (searched in a random 10% of the brain) and 80 streamlines drawn at random from the whole brain. The network (a
// DGCNN: point-wise layers over the streamline's points and their 5 nearest points in feature space) answers one of
// 1600 classes: 800 fiber clusters of the O'Donnell-Zhang atlas and their 800 "outlier" twins; a table turns the
// cluster into one of 42 tracts or Other. So THE NAME NEEDS THE WHOLE BRAIN AROUND IT -- track the whole brain first.
//
// This file is the PROCESSOR version, in double precision: the reference the graphics-card version is checked against,
// itself checked layer by layer against the original running in PyTorch (tractcloud.test.ts;
// Contents/tools/tractcloud-reference.py in the workspace).
//
// THE RANDOM DRAWS. The original draws its sample and its 80 global streamlines anew every run, so names change from
// run to run (measured on PAT16: 79% of streamlines keep their tract between draws). Here they come from a seeded
// generator, so the same streamlines always get the same names.

export interface TractInfo { abbr: string; name: string; category: string }
export interface ModelJson {
  settings: { points: number; k: number; kGlobal: number; kPoint: number; kSampleRate: number; embDims: number; classes: number; bnEps: number; leak: number };
  massCenter: number[][];
  tracts: TractInfo[];
  clusterToTract: number[];
  tensors: Record<string, { offset: number; shape: number[] }>;
  source: Record<string, string>;
}

/** A point-wise layer with its batch norm folded in: y = s * (W x) + t, then the leaky ReLU. W is [out][in]. */
interface Layer { out: number; inp: number; W: Float64Array; s: Float64Array; t: Float64Array; bias?: Float64Array }

export interface TractCloudModel {
  json: ModelJson;
  P: number;
  conv1: Layer; conv2: Layer; conv3: Layer; conv4: Layer; conv5: Layer;
  lin1: Layer; lin2: Layer; lin3: Layer;
}

export function loadModel(weights: ArrayBuffer, json: ModelJson): TractCloudModel {
  const all = new Float32Array(weights);
  const get = (name: string) => {
    const e = json.tensors[name];
    if (!e) throw new Error(`TractCloud model: no tensor ${name}`);
    const n = e.shape.reduce((a, b) => a * b, 1);
    return Float64Array.from(all.subarray(e.offset, e.offset + n));
  };
  const eps = json.settings.bnEps;
  const layer = (w: string, bn: string | null, bias: string | null): Layer => {
    const W = get(w), shape = json.tensors[w].shape, out = shape[0], inp = shape[1];
    const s = new Float64Array(out).fill(1), t = new Float64Array(out);
    const b = bias ? get(bias) : undefined;
    if (b) t.set(b);
    if (bn) {
      const g = get(`${bn}.weight`), beta = get(`${bn}.bias`), mean = get(`${bn}.running_mean`), vr = get(`${bn}.running_var`);
      for (let o = 0; o < out; o++) { const k = g[o] / Math.sqrt(vr[o] + eps); s[o] = k; t[o] = k * (t[o] - mean[o]) + beta[o]; }
    }
    return { out, inp, W, s, t, bias: b };
  };
  return {
    json, P: json.settings.points,
    conv1: layer("conv1.0.weight", "bn1", null), conv2: layer("conv2.0.weight", "bn2", null),
    conv3: layer("conv3.0.weight", "bn3", null), conv4: layer("conv4.0.weight", "bn4", null),
    conv5: layer("conv5.0.weight", "bn5", null),
    lin1: layer("linear1.weight", "bn6", null), lin2: layer("linear2.weight", "bn7", "linear2.bias"),
    lin3: layer("linear3.weight", null, "linear3.bias"),
  };
}

/** One streamline (x,y,z,… RAS mm) at `n` points of equal arc length along it -- the original's extract_ras_features. */
export function resample(pts: Float32Array | number[], n = 15): Float64Array {
  const m = pts.length / 3, out = new Float64Array(3 * n);
  if (m < 2) { for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[3 * i + c] = pts[c] ?? 0; return out; }
  const cum = new Float64Array(m);
  for (let i = 1; i < m; i++) cum[i] = cum[i - 1] + Math.hypot(pts[3 * i] - pts[3 * i - 3], pts[3 * i + 1] - pts[3 * i - 2], pts[3 * i + 2] - pts[3 * i - 1]);
  const total = cum[m - 1];
  if (total < 1e-12) { for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[3 * i + c] = pts[c]; return out; }
  let seg = 0;
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * total;
    while (seg < m - 2 && cum[seg + 1] <= t) seg++;             // searchsorted(side="right") - 1, clipped to [0, m-2]
    const len = cum[seg + 1] - cum[seg], f = (t - cum[seg]) / (len < 1e-12 ? 1 : len);
    for (let c = 0; c < 3; c++) out[3 * i + c] = pts[3 * seg + c] + f * (pts[3 * seg + 3 + c] - pts[3 * seg + c]);
  }
  return out;
}

/** All streamlines resampled and moved so their mean (per point) is the atlas's: feat, (N, P, 3) row-major. */
export function prepare(streamlines: (Float32Array | number[])[], model: TractCloudModel): Float64Array {
  const P = model.P, N = streamlines.length, feat = new Float64Array(N * P * 3);
  streamlines.forEach((s, i) => feat.set(resample(s, P), i * P * 3));
  const mean = new Float64Array(P * 3);
  for (let i = 0; i < N; i++) for (let j = 0; j < P * 3; j++) mean[j] += feat[i * P * 3 + j] / N;
  const mc = model.json.massCenter;
  for (let i = 0; i < N; i++) for (let p = 0; p < P; p++) for (let c = 0; c < 3; c++) feat[(i * P + p) * 3 + c] += mc[p][c] - mean[p * 3 + c];
  return feat;
}

/** A small seeded generator (mulberry32), so the same streamlines always get the same draws. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** The random choices: the sample searched for local neighbors (sorted) and the global streamlines (with repeats). */
export function draw(N: number, model: TractCloudModel, seed = 20260930): { ds: Int32Array; glob: Int32Array } {
  const r = rng(seed), { kSampleRate, kGlobal, k } = model.json.settings;
  const nds = Math.max(k, Math.min(N, Math.floor(N * kSampleRate)));
  const idx = Int32Array.from({ length: N }, (_, i) => i);
  for (let i = 0; i < nds; i++) { const j = i + Math.floor(r() * (N - i)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
  const ds = idx.slice(0, nds).sort();
  const glob = Int32Array.from({ length: kGlobal }, () => Math.floor(r() * N));
  return { ds, glob };
}

/** Each streamline's k nearest streamlines among the sample (indices into ds), by the original's distance: the
 *  Euclidean norm of the flattened 45 numbers (divided by 15, which does not change the order). */
export function localNeighbors(feat: Float64Array, ds: Int32Array, model: TractCloudModel): Int32Array {
  const P3 = model.P * 3, k = model.json.settings.k, N = feat.length / P3, out = new Int32Array(N * k);
  const bestD = new Float64Array(k), bestI = new Int32Array(k);
  for (let i = 0; i < N; i++) {
    bestD.fill(Infinity); bestI.fill(-1);
    for (let j = 0; j < ds.length; j++) {
      let d = 0; const a = i * P3, b = ds[j] * P3;
      for (let q = 0; q < P3; q++) { const e = feat[a + q] - feat[b + q]; d += e * e; }
      if (d >= bestD[k - 1]) continue;
      let p = k - 1;
      while (p > 0 && bestD[p - 1] > d) { bestD[p] = bestD[p - 1]; bestI[p] = bestI[p - 1]; p--; }
      bestD[p] = d; bestI[p] = j;
    }
    out.set(bestI, i * k);
  }
  return out;
}

export interface Recorded { conv1: Float64Array; conv2: Float64Array; conv3: Float64Array; conv4: Float64Array; conv5: Float64Array; lin1: Float64Array; lin2: Float64Array; lin3: Float64Array }

const lrelu = (x: number, a: number) => (x > 0 ? x : a * x);

/**
 * AN EDGE LAYER over P points: for point p and each of its neighbors q, the input is [x_q - x_p, x_p]; the output is
 * the layer applied to it, maxed over the neighbors. Since W [x_q - x_p, x_p] = Wa x_q + (Wb - Wa) x_p, the two halves
 * are computed once per point, not once per edge. x is (C, P); nb[p] lists the neighbors of p.
 */
function edgeLayer(L: Layer, x: Float64Array, C: number, P: number, nb: Int32Array, K: number, leak: number): Float64Array {
  const A = new Float64Array(L.out * P), B = new Float64Array(L.out * P);
  for (let o = 0; o < L.out; o++) {
    const w = o * L.inp;
    for (let p = 0; p < P; p++) {
      let a = 0, b = 0;
      for (let c = 0; c < C; c++) { const wa = L.W[w + c], wb = L.W[w + C + c], v = x[c * P + p]; a += wa * v; b += (wb - wa) * v; }
      A[o * P + p] = a; B[o * P + p] = b;
    }
  }
  const y = new Float64Array(L.out * P);
  for (let o = 0; o < L.out; o++) for (let p = 0; p < P; p++) {
    let m = -Infinity;
    for (let j = 0; j < K; j++) { const q = nb[p * K + j]; const v = lrelu(L.s[o] * (A[o * P + q] + B[o * P + p]) + L.t[o], leak); if (v > m) m = v; }
    y[o * P + p] = m;
  }
  return y;
}

/** The kPoint nearest points of each point in feature space, itself included (the original's tract_knn). */
function pointNeighbors(x: Float64Array, C: number, P: number, K: number): Int32Array {
  const out = new Int32Array(P * K), d = new Float64Array(P), order = Array.from({ length: P }, (_, i) => i);
  for (let p = 0; p < P; p++) {
    for (let q = 0; q < P; q++) { let s = 0; for (let c = 0; c < C; c++) { const e = x[c * P + p] - x[c * P + q]; s += e * e; } d[q] = s; }
    order.sort((a, b) => d[a] - d[b] || a - b);
    for (let j = 0; j < K; j++) out[p * K + j] = order[j];
  }
  return out;
}

function dense(L: Layer, x: Float64Array, leak: number | null): Float64Array {
  const y = new Float64Array(L.out);
  for (let o = 0; o < L.out; o++) {
    let s = 0; const w = o * L.inp;
    for (let i = 0; i < L.inp; i++) s += L.W[w + i] * x[i];
    const v = L.s[o] * s + L.t[o];
    y[o] = leak === null ? v : lrelu(v, leak);
  }
  return y;
}

/**
 * THE NETWORK on streamlines [from, to): the cluster (0..1599) of each. topk (N, k) indexes ds. With `record`, the
 * layer outputs of the first streamline of the range are returned too, in the reference's shapes (conv: (C, P);
 * lin1/lin2 BEFORE their batch norm; lin3 = the logits).
 */
export function classifyCpu(model: TractCloudModel, feat: Float64Array, ds: Int32Array, glob: Int32Array, topk: Int32Array,
  from = 0, to = feat.length / (model.P * 3), record?: Recorded[]): Int32Array {
  const { P } = model, { k, kGlobal, kPoint, leak } = model.json.settings, Kc = k + kGlobal, P3 = P * 3;
  const out = new Int32Array(to - from);
  const L1 = model.conv1;
  for (let f = from; f < to; f++) {
    const x = feat.subarray(f * P3, f * P3 + P3);
    // conv1: context j at point p gives [ctx - x, x]; W1 [d, x] = W1a ctx + (W1b - W1a) x.
    const base = new Float64Array(64 * P);
    for (let o = 0; o < L1.out; o++) for (let p = 0; p < P; p++) {
      let b = 0; for (let c = 0; c < 3; c++) b += (L1.W[o * 6 + 3 + c] - L1.W[o * 6 + c]) * x[p * 3 + c];
      base[o * P + p] = b;
    }
    const x1 = new Float64Array(L1.out * P).fill(-Infinity);
    for (let j = 0; j < Kc; j++) {
      const src = j < k ? ds[topk[f * k + j]] : glob[j - k];
      const cx = feat.subarray(src * P3, src * P3 + P3);
      for (let o = 0; o < L1.out; o++) {
        const w0 = L1.W[o * 6], w1 = L1.W[o * 6 + 1], w2 = L1.W[o * 6 + 2];
        for (let p = 0; p < P; p++) {
          const v = lrelu(L1.s[o] * (w0 * cx[p * 3] + w1 * cx[p * 3 + 1] + w2 * cx[p * 3 + 2] + base[o * P + p]) + L1.t[o], leak);
          if (v > x1[o * P + p]) x1[o * P + p] = v;
        }
      }
    }
    const x2 = edgeLayer(model.conv2, x1, 64, P, pointNeighbors(x1, 64, P, kPoint), kPoint, leak);
    const x3 = edgeLayer(model.conv3, x2, 64, P, pointNeighbors(x2, 64, P, kPoint), kPoint, leak);
    const x4 = edgeLayer(model.conv4, x3, 128, P, pointNeighbors(x3, 128, P, kPoint), kPoint, leak);
    const cat = new Float64Array(512 * P);
    cat.set(x1, 0); cat.set(x2, 64 * P); cat.set(x3, 128 * P); cat.set(x4, 256 * P);
    const L5 = model.conv5, x5 = new Float64Array(L5.out * P), pooled = new Float64Array(2 * L5.out);
    for (let o = 0; o < L5.out; o++) {
      let mx = -Infinity, sum = 0;
      for (let p = 0; p < P; p++) {
        let s = 0; const w = o * L5.inp;
        for (let i = 0; i < L5.inp; i++) s += L5.W[w + i] * cat[i * P + p];
        const v = lrelu(L5.s[o] * s + L5.t[o], leak);
        x5[o * P + p] = v; if (v > mx) mx = v; sum += v;
      }
      pooled[o] = mx; pooled[L5.out + o] = sum / P;
    }
    const h1 = dense(model.lin1, pooled, leak), h2 = dense(model.lin2, h1, leak), logits = dense(model.lin3, h2, null);
    let best = 0; for (let c = 1; c < logits.length; c++) if (logits[c] > logits[best]) best = c;
    out[f - from] = best;
    if (record) {
      // lin1/lin2 before their batch norm, as the reference records them (the Linear's own output, bias included).
      const pre = (L: Layer, inp: Float64Array) => Float64Array.from({ length: L.out }, (_, o) => {
        let s = L.bias?.[o] ?? 0; for (let i = 0; i < L.inp; i++) s += L.W[o * L.inp + i] * inp[i]; return s;
      });
      record.push({ conv1: x1, conv2: x2, conv3: x3, conv4: x4, conv5: x5, lin1: pre(model.lin1, pooled), lin2: pre(model.lin2, h1), lin3: logits });
    }
  }
  return out;
}

/** The tract (index into model.json.tracts) of each cluster. */
export function tractsOf(model: TractCloudModel, clusters: Int32Array): Int32Array {
  const lut = model.json.clusterToTract;
  return clusters.map((c) => lut[Math.max(0, Math.min(lut.length - 1, c))]);
}
