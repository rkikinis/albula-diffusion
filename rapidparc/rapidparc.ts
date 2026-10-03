// RAPIDPARC, THE NETWORK THAT NAMES TRACTS (von Bornhaupt, Bisten, …, Schultz, "RapidParc: A Global-Context Transformer
// for Parallel, Accurate, and Lesion-Robust Tractogram Parcellation", Imaging Neuroscience 2026; github.com/MedVisBonn/
// RapidParc, BSD-3-Clause). Ron, 2026-10-03: Albula switches to it, as Mike Halle's tractline did on 2026-10-02 — on
// TractCloud's own test split 94.5 % of tracts right against TractCloud's 92.0 %, draws agreeing on 98 % of tracts, about
// 25 times faster (tractline's docs/labelers.md). Written from RapidParc's source (utils/model.py, utils/transforms3D.py,
// run.py at commit 4d03d3c), with Mike's tractline/labelers/rapidparc.py (Apache-2.0) as the second reading; its released
// weights (model/, BSD-3, README there).
//
// WHAT IT DOES. Every streamline of 40 mm or more becomes 15 points chosen by index (not arc length:
// np.round(linspace(0, n - 1, 15))); the whole set is scaled into [-1, 1] per axis; it is shuffled and cut into groups of
// 2,000; each group is scaled into [-1, 1] again and goes through the network as one context: the 45 coordinates plus a
// linear layer's 83 (leaky ReLU) make 128 features a streamline; eight transformer encoder layers (one attention head
// across all 2,000 streamlines of the group, feed-forward 256, normalization after each step, as PyTorch's default);
// a classifier 128 → 256 → 1,600. The 1,600 are the O'Donnell-Zhang atlas's 800 clusters and their 800 outlier twins —
// the same classes, the same cluster → tract table and the same 43 tract names as TractCloud's (checked identical,
// 2026-10-03), so tractcloud/model/model.json's table serves both.
//
// This file is the PROCESSOR version, the reference the graphics-card version is checked against; itself checked layer
// by layer against RapidParc's network in PyTorch (rapidparc.test.ts; Contents/tools/rapidparc-reference.py).

export const POINTS = 15, CONTEXT = 2000, MIN_LENGTH_MM = 40, D = 128, FF = 256, HIDDEN = 256, CLASSES = 1600, LAYERS = 8;

export interface Linear { out: number; inp: number; W: Float32Array; b: Float32Array }   // W is [out][inp], as PyTorch
export interface EncoderLayer { qkv: Linear; o: Linear; ff1: Linear; ff2: Linear; n1g: Float32Array; n1b: Float32Array; n2g: Float32Array; n2b: Float32Array }
export interface RapidParcModel { name: string; emb: Linear; layers: EncoderLayer[]; cls1: Linear; cls2: Linear }

/** A .safetensors file: an 8-byte little-endian header length, a JSON header, the raw tensors. */
export function readSafetensors(buf: ArrayBuffer): Map<string, { shape: number[]; data: Float32Array }> {
  const dv = new DataView(buf), n = Number(dv.getBigUint64(0, true));
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n))) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const out = new Map<string, { shape: number[]; data: Float32Array }>();
  for (const [k, v] of Object.entries(head)) {
    if (k === "__metadata__") continue;
    if (v.dtype !== "F32") throw new Error(`RapidParc weights: ${k} is ${v.dtype}, not F32`);
    const [a, b] = v.data_offsets;
    out.set(k, { shape: v.shape, data: new Float32Array(buf.slice(8 + n + a, 8 + n + b)) });
  }
  return out;
}

export function loadRapidParc(buf: ArrayBuffer, name = "rapidparc"): RapidParcModel {
  const t = readSafetensors(buf);
  const get = (k: string) => { const e = t.get(k); if (!e) throw new Error(`RapidParc weights: no tensor ${k}`); return e; };
  const lin = (k: string): Linear => { const w = get(`${k}.weight`); return { out: w.shape[0], inp: w.shape[1], W: w.data, b: get(`${k}.bias`).data }; };
  const layers: EncoderLayer[] = [];
  for (let l = 0; l < LAYERS; l++) {
    const p = `embeddingTransformer.layers.${l}`;
    const w = get(`${p}.self_attn.in_proj_weight`);
    layers.push({
      qkv: { out: w.shape[0], inp: w.shape[1], W: w.data, b: get(`${p}.self_attn.in_proj_bias`).data },
      o: lin(`${p}.self_attn.out_proj`), ff1: lin(`${p}.linear1`), ff2: lin(`${p}.linear2`),
      n1g: get(`${p}.norm1.weight`).data, n1b: get(`${p}.norm1.bias`).data, n2g: get(`${p}.norm2.weight`).data, n2b: get(`${p}.norm2.bias`).data,
    });
  }
  return { name, emb: lin("embedding_layer.linear"), layers, cls1: lin("classifier.1"), cls2: lin("classifier.3") };
}

/** np.round: halves go to the even neighbor. */
const roundHalfEven = (x: number) => { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };
/** 15 points of a streamline (xyz, RAS mm) chosen by index, RapidParc's rule. */
export function resampleByIndex(s: Float32Array, n = POINTS): Float32Array {
  // numpy's linspace exactly: position = index × step, the step rounded first, the last position the stop itself — 7 × (3/14)
  // is 1.4999…, not 1.5, and the rounding that follows tells them apart.
  const m = s.length / 3, out = new Float32Array(n * 3), step = n > 1 ? (m - 1) / (n - 1) : 0;
  for (let p = 0; p < n; p++) {
    const i = roundHalfEven(p === n - 1 ? m - 1 : p * step);
    out[p * 3] = s[i * 3]; out[p * 3 + 1] = s[i * 3 + 1]; out[p * 3 + 2] = s[i * 3 + 2];
  }
  return out;
}

export const lengthMm = (s: Float32Array) => { let L = 0; for (let i = 3; i < s.length; i += 3) L += Math.hypot(s[i] - s[i - 3], s[i + 1] - s[i - 2], s[i + 2] - s[i - 1]); return L; };

/** Rows of 45 (15 points × xyz) scaled into [-1, 1] per axis over all rows (RapidParc's normalize_to_identity_cube). An
 *  axis with no extent divides by 1 (RapidParc's would divide by zero; no real tractogram has one). In float32, as
 *  PyTorch computes it. */
export function normalizeCube(x: Float32Array): Float32Array {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < x.length; i++) { const a = i % 3; if (x[i] < lo[a]) lo[a] = x[i]; if (x[i] > hi[a]) hi[a] = x[i]; }
  const out = new Float32Array(x.length), f = Math.fround;
  for (let i = 0; i < x.length; i++) { const a = i % 3, e = f(hi[a] - lo[a]) || 1; out[i] = f(f(2 * f(x[i] - lo[a])) / e) - 1; }
  return out;
}

/** A seeded shuffle of 0..n-1 (the draws: each seed its own grouping, the same seed the same names). */
export function shuffle(n: number, seed: number): Int32Array {
  let s = seed >>> 0;
  const rnd = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const p = new Int32Array(n); for (let i = 0; i < n; i++) p[i] = i;
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; }
  return p;
}

/** The groups of one draw: rows of `x` (n × 45, already scaled over the whole set) in the order of `perm`, padded with the
 *  first shuffled rows to a multiple of CONTEXT (RapidParc pads once with the first rows; repeating covers n < CONTEXT/2
 *  too, as tractline does). Returns the padded rows and how many are real. */
export function groupRows(x: Float32Array, perm: Int32Array): { rows: Float32Array; groups: number } {
  const n = perm.length, groups = Math.max(1, Math.ceil(n / CONTEXT)), rows = new Float32Array(groups * CONTEXT * 45);
  for (let r = 0; r < groups * CONTEXT; r++) rows.set(x.subarray(perm[r % n] * 45, perm[r % n] * 45 + 45), r * 45);
  return { rows, groups };
}

// ── the network, on the processor ──
function linear(L: Linear, x: Float32Array, rows: number, act?: "relu" | "leaky"): Float32Array {
  const y = new Float32Array(rows * L.out);
  for (let r = 0; r < rows; r++) {
    const xo = r * L.inp, yo = r * L.out;
    for (let o = 0; o < L.out; o++) {
      let s = L.b[o]; const wo = o * L.inp;
      for (let i = 0; i < L.inp; i++) s += L.W[wo + i] * x[xo + i];
      y[yo + o] = act === "relu" ? (s > 0 ? s : 0) : act === "leaky" ? (s > 0 ? s : 0.01 * s) : s;
    }
  }
  return y;
}
function addNorm(x: Float32Array, y: Float32Array, g: Float32Array, b: Float32Array, rows: number): Float32Array {
  const out = new Float32Array(rows * D);
  for (let r = 0; r < rows; r++) {
    let mean = 0; for (let i = 0; i < D; i++) mean += x[r * D + i] + y[r * D + i]; mean /= D;
    let v = 0; for (let i = 0; i < D; i++) { const d = x[r * D + i] + y[r * D + i] - mean; v += d * d; } v /= D;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < D; i++) out[r * D + i] = (x[r * D + i] + y[r * D + i] - mean) * inv * g[i] + b[i];
  }
  return out;
}
/** One head of self-attention over all `rows` streamlines of a group. */
function attention(qkv: Float32Array, rows: number): Float32Array {
  const out = new Float32Array(rows * D), sc = new Float64Array(rows), k = 1 / Math.sqrt(D);
  for (let r = 0; r < rows; r++) {
    const q = r * 3 * D; let mx = -Infinity;
    for (let c = 0; c < rows; c++) { let s = 0; const kk = c * 3 * D + D; for (let i = 0; i < D; i++) s += qkv[q + i] * qkv[kk + i]; sc[c] = s * k; if (sc[c] > mx) mx = sc[c]; }
    let sum = 0; for (let c = 0; c < rows; c++) { sc[c] = Math.exp(sc[c] - mx); sum += sc[c]; }
    for (let c = 0; c < rows; c++) { const w = sc[c] / sum, v = c * 3 * D + 2 * D; for (let i = 0; i < D; i++) out[r * D + i] += w * qkv[v + i]; }
  }
  return out;
}

export interface Recorded { emb: Float32Array; layer0: Float32Array; enc: Float32Array; hidden: Float32Array }

/** One group (rows × 45, `rows` = CONTEXT in use) through the network: its logits (rows × 1600). */
export function forwardGroupCpu(m: RapidParcModel, group: Float32Array, rec?: Partial<Recorded>): Float32Array {
  const rows = group.length / 45, x45 = normalizeCube(group);
  const e = linear(m.emb, x45, rows, "leaky");
  let x = new Float32Array(rows * D);
  for (let r = 0; r < rows; r++) { x.set(x45.subarray(r * 45, r * 45 + 45), r * D); x.set(e.subarray(r * 83, r * 83 + 83), r * D + 45); }
  if (rec) rec.emb = x;
  m.layers.forEach((L, l) => {
    const a = linear(L.o, attention(linear(L.qkv, x, rows), rows), rows);
    const h = addNorm(x, a, L.n1g, L.n1b, rows);
    x = addNorm(h, linear(L.ff2, linear(L.ff1, h, rows, "relu"), rows), L.n2g, L.n2b, rows);
    if (rec && l === 0) rec.layer0 = x;
  });
  if (rec) rec.enc = x;
  const hid = linear(m.cls1, x, rows, "relu");
  if (rec) rec.hidden = hid;
  return linear(m.cls2, hid, rows);
}
