// BRAINSTEMSEG'S NETWORK, THE PROCESSOR'S SHARE (Ron, 2026-10-22: "just taking their model and implement a gpu version").
// BrainstemSeg (Rudravaram, Landman, Schilling et al., Vanderbilt; preprint bioRxiv 10.64898/2026.06.09.731210) finds 97
// brainstem and deep white-matter pathways -- the medial lemniscus (ML) and its thalamocortical part (MLc), the
// spinothalamic tract (STT, STTc), the corticospinal tract (CST), the frontopontine and parietopontine tracts and the rest
// (model/tracts.json) -- as a voxel mask each, from white-matter fiber orientation distributions (FODs), with BundleParc's
// network (Théberge et al., SCIL-VITAL; MIT). Their weights: CC BY 4.0, model/README.md.
//
// The network (vendor/bundleparc/bundleparcnet.py in the workspace, read line by line):
//   stem      1×1×1 convolution, 45 → 32
//   encoder   four levels (32→64, 64→128, 128→256, 256→512 channels; expansion 2, 3, 4, 4), each two ConvNeXt blocks
//             and a block that halves the grid; the second block's output kept for the decoder
//   ConvNeXt  x + W3·GELU(W2·GN(depthwise 3×3×3 (x))): GN is a group norm with one group per channel
//   bottleneck one ConvNeXt block at 512 channels
//   prompt    GELU(Linear 97 → 512) of the tract's one-hot vector
//   decoder   four levels (512→256, 256→128, 128→64, 64→32; expansion 4, 4, 3, 2): a block that doubles the grid (a
//             depthwise 2×2×2 transposed convolution path plus a full one as the residual), + the encoder's output, a
//             ConvNeXt block, the prompt's attention step (below), a ConvNeXt block
//   head      1×1×1 convolution 32 → 2: the mask's logit and the position along the tract
// The attention step at a level (TwoWayAttentionBlock3D of bundleparcnet.py, SAM-Med3D's, 4 heads): the one prompt
// token attends to every voxel (its key: the voxel + a sine position code; its value: the voxel), is added to and layer-
// normed; the voxels go through an MLP (ReLU) with a residual and a layer norm; then each voxel attends to the one updated
// prompt token -- with one key the weight is exactly 1, so every voxel gets the same vector added -- and a layer norm.
// Because the query side is ONE token, the scores are a linear function of each voxel and the weighted values a weighted
// average of the voxels: the port computes those two directly and never builds the per-voxel keys and values (the same
// numbers up to rounding; bundleparc-gpu.test.ts measures it).
//
// The processor does: the weights, the input as their pipeline prepares it (fit, pad, one z-score), the position code's
// tables, and the prompt side of each attention step (a few hundred numbers, in double precision).
import { readSafetensors } from "../rapidparc/rapidparc.ts";

export const N_COEFS = 45, VOLUME = 176, NB_PTS = 10, HEADS = 4, EPS = 1e-5;
export const CHANNELS = [32, 64, 128, 256, 512];
export const ENC_RATIO = [2, 3, 4, 4], DEC_RATIO = [4, 4, 3, 2];

export interface Tensor { shape: number[]; data: Float32Array }
export interface BundleParcModel { name: string; tensors: Map<string, Tensor>; tracts: string[] }

/** The converted weights (model/bundleparc-brainstemseg.safetensors) and the tract list (model/tracts.json). */
export function loadBundleParc(weights: ArrayBuffer, tractsJson: { tracts: string[] }, name = "brainstemseg"): BundleParcModel {
  const tensors = readSafetensors(weights) as Map<string, Tensor>;
  for (const k of ["stem.weight", "head.conv.weight", "prompt_embedding.0.weight", "bottleneck.conv2.weight", "decoder.layers.3.tkn2img.norm2.weight"]) {
    if (!tensors.has(k)) throw new Error(`BundleParc weights: ${k} is missing`);
  }
  const n = tensors.get("prompt_embedding.0.weight")!.shape[1];
  if (n !== tractsJson.tracts.length) throw new Error(`BundleParc weights: ${n} prompts, ${tractsJson.tracts.length} tract names`);
  return { name, tensors, tracts: tractsJson.tracts };
}
export const tensor = (m: BundleParcModel, k: string): Tensor => {
  const t = m.tensors.get(k); if (!t) throw new Error(`BundleParc weights: ${k} is missing`); return t;
};

// ── small dense algebra for the prompt side, double precision ──

/** y = W·x + b, W [out][in] row-major (as PyTorch's Linear). */
export function linear(W: Tensor, b: Tensor | undefined, x: ArrayLike<number>): Float64Array {
  const [out, inp] = W.shape, y = new Float64Array(out);
  for (let o = 0; o < out; o++) { let s = b ? b.data[o] : 0; for (let i = 0; i < inp; i++) s += W.data[o * inp + i] * x[i]; y[o] = s; }
  return y;
}
/** The error function (W. J. Cody's rational approximations): within 1.5e-12 of Python's math.erf on [-10, 10]
 *  (1,281 points, checked 2026-10-22) -- far below single precision, which the network runs in. */
export function erf(x: number): number {
  const ax = Math.abs(x);
  if (ax < 0.5) {
    const t = x * x;
    const p = (((0.185777706184603153 * t + 3.16112374387056560) * t + 113.864154151050156) * t + 377.485237685302021) * t + 3209.37758913846947;
    const q = (((t + 23.6012909523441209) * t + 244.024637934444173) * t + 1282.61652607737228) * t + 2844.23683343917062;
    return x * p / q;
  }
  return Math.sign(x) * (1 - erfc(ax));
}
function erfc(x: number): number {
  if (x < 4) {
    const p = (((((((5.64188496988670089e-1 * x + 8.88314979438837594) * x + 66.1191906371416295) * x + 298.635138197400131) * x + 881.952221241769090) * x + 1712.04761263407058) * x + 2051.07837782607147) * x + 1230.33935479799725);
    const q = (((((((x + 15.7449261107098347) * x + 117.693950891312499) * x + 537.181101862009858) * x + 1621.38957456669019) * x + 3290.79923573345963) * x + 4362.61909014324716) * x + 3439.36767414372164) * x + 1230.33935480374942;
    return Math.exp(-x * x) * p / q;
  }
  const z = 1 / (x * x);
  const p = ((((1.63153871373020978e-2 * z + 3.05326634961232344e-1) * z + 3.60344899949804439e-1) * z + 1.25781726111229246e-1) * z + 1.60837851487422766e-2) * z + 6.58749161529837803e-4;
  const q = ((((z + 2.56852019228982242) * z + 1.87295284992346725) * z + 5.27905102951428412e-1) * z + 6.05183413124413191e-2) * z + 2.33520497626869185e-3;
  return Math.exp(-x * x) / x * (0.564189583547756287 - z * p / q);
}
/** PyTorch's GELU (exact: x · Φ(x)). */
export const gelu = (x: number) => 0.5 * x * (1 + erf(x / Math.SQRT2));
export function layerNorm(x: ArrayLike<number>, g: Tensor, b: Tensor): Float64Array {
  const n = x.length; let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
  let v = 0; for (let i = 0; i < n; i++) v += (x[i] - m) ** 2; v /= n;
  const s = 1 / Math.sqrt(v + EPS), y = new Float64Array(n);
  for (let i = 0; i < n; i++) y[i] = (x[i] - m) * s * g.data[i] + b.data[i];
  return y;
}

// ── the position code (bundleparcnet.py PositionalEncoding3D, from tatp22/multidim-positional-encoding) ──

/** Per axis, a table [position][j] of the code's channels for that axis: chp = ceil(C/6)·2 channels each (sin, cos
 *  interleaved at frequencies 1/10000^(2f/chp)); channel k of the code is axis floor(k/chp)'s entry k mod chp, the
 *  first C of the 3·chp kept. Double precision, rounded to single at the end. */
export function positionTables(C: number, dims: number[]): { chp: number; tables: Float32Array[]; invFreq: Float64Array } {
  let chp = Math.ceil(C / 6) * 2; if (chp % 2) chp++;
  const invFreq = new Float64Array(chp / 2);
  for (let f = 0; f < chp / 2; f++) invFreq[f] = 1 / Math.pow(10000, (2 * f) / chp);
  const tables = dims.map((n) => {
    const t = new Float32Array(n * chp);
    for (let p = 0; p < n; p++) for (let f = 0; f < chp / 2; f++) { t[p * chp + 2 * f] = Math.sin(p * invFreq[f]); t[p * chp + 2 * f + 1] = Math.cos(p * invFreq[f]); }
    return t;
  });
  return { chp, tables, invFreq };
}

// ── the input, as their bundleparc_predict_fitpad.py prepares it ──

export interface Prepared {
  /** The 176-voxel cube, CHANNEL LAST (voxel (x, y, z) at ((x·S + y)·S + z)·45), z-scored. */
  cube: Float32Array;
  /** The fitted grid's size, where it sits in the cube, and the scale per axis (old voxels per new one). */
  fit: [number, number, number]; offset: [number, number, number]; step: [number, number, number];
}
/** `fod`: the FODs on their 1 mm grid, CHANNEL LAST ((x·Y + y)·Z + z)·C with C >= 45 coefficients (MRtrix3's basis,
 *  as dwi2fod writes them); only the first 45 are used. Fit: scaled uniformly so the longest side is 176 -- scipy's
 *  affine_transform as dipy's reslice calls it (output voxel o reads input o·step, linear, zero outside); pad: centered
 *  in the cube; one mean and one standard deviation over the whole cube, padding included. */
export function prepareInput(fod: Float32Array, dims: [number, number, number], C: number): Prepared {
  const [X, Y, Z] = dims, S = VOLUME, m = Math.max(X, Y, Z);
  const fit = dims.map((d) => Math.max(1, Math.min(S, Math.round(d * S / m)))) as [number, number, number];
  const step = dims.map((d, i) => d / fit[i]) as [number, number, number];
  const offset = fit.map((f) => Math.floor((S - f) / 2)) as [number, number, number];
  const cube = new Float32Array(S * S * S * N_COEFS);
  const at = (x: number, y: number, z: number, c: number) => fod[((x * Y + y) * Z + z) * C + c];
  for (let x = 0; x < fit[0]; x++) {
    const sx = x * step[0], x0 = Math.floor(sx), fx = sx - x0;
    for (let y = 0; y < fit[1]; y++) {
      const sy = y * step[1], y0 = Math.floor(sy), fy = sy - y0;
      for (let z = 0; z < fit[2]; z++) {
        const sz = z * step[2], z0 = Math.floor(sz), fz = sz - z0;
        // scipy's order-1 spline: the 8 neighbors, a neighbor beyond the last voxel contributing 0 ("constant" mode
        // reaches it only for a point beyond the edge, which the fit never asks for: (fit - 1)·step < size - 1).
        const o = (((x + offset[0]) * S + (y + offset[1])) * S + (z + offset[2])) * N_COEFS;
        for (let c = 0; c < N_COEFS; c++) {
          let v = 0;
          for (let a = 0; a < 2; a++) { const xi = x0 + a, wx = a ? fx : 1 - fx; if (xi >= X || wx === 0) continue;
            for (let b = 0; b < 2; b++) { const yi = y0 + b, wy = b ? fy : 1 - fy; if (yi >= Y || wy === 0) continue;
              for (let d = 0; d < 2; d++) { const zi = z0 + d, wz = d ? fz : 1 - fz; if (zi >= Z || wz === 0) continue;
                v += wx * wy * wz * at(xi, yi, zi, c); } } }
          cube[o + c] = v;
        }
      }
    }
  }
  let s = 0; for (let i = 0; i < cube.length; i++) s += cube[i];
  const mean = s / cube.length; let q = 0; for (let i = 0; i < cube.length; i++) q += (cube[i] - mean) ** 2;
  const sd = Math.sqrt(q / cube.length);
  for (let i = 0; i < cube.length; i++) cube[i] = (cube[i] - mean) / sd;
  return { cube, fit, offset, step };
}

// ── the prompt side of one attention step ──

export interface AttnPrompt {
  /** Per head h, the score of voxel token t is (t + pe)·u[h] (+ a constant the softmax ignores), already / √dh. */
  u: Float32Array;
}
/** Before the voxels are read: the prompt (after this level's prompt encoding) gives the per-head score vectors. */
export function attnScores(m: BundleParcModel, level: number, prompt: Float64Array): AttnPrompt {
  const p = `decoder.layers.${level}.tkn2img.cross_attn_image_to_token`;
  const C = prompt.length, dh = C / HEADS, Q = linear(tensor(m, `${p}.q_proj.weight`), tensor(m, `${p}.q_proj.bias`), prompt);
  const Wk = tensor(m, `${p}.k_proj.weight`).data, u = new Float32Array(HEADS * C), sc = 1 / Math.sqrt(dh);
  for (let h = 0; h < HEADS; h++) for (let c = 0; c < C; c++) {
    let s = 0; for (let j = h * dh; j < (h + 1) * dh; j++) s += Q[j] * Wk[j * C + c];
    u[h * C + c] = s * sc;
  }
  return { u };
}
/** After the voxels were summed: `avg[h]` is the softmax-weighted average of the voxel tokens (without the position
 *  code) under head h. Returns the updated prompt and the vector every voxel gets (the second attention). */
export function attnUpdate(m: BundleParcModel, level: number, prompt: Float64Array, avg: Float64Array[]): { prompt: Float64Array; toVoxels: Float32Array } {
  const L = `decoder.layers.${level}.tkn2img`, a = `${L}.cross_attn_image_to_token`, C = prompt.length, dh = C / HEADS;
  const Wv = tensor(m, `${a}.v_proj.weight`).data, bv = tensor(m, `${a}.v_proj.bias`).data;
  const heads = new Float64Array(C);
  for (let h = 0; h < HEADS; h++) for (let j = h * dh; j < (h + 1) * dh; j++) {
    let s = bv[j]; for (let c = 0; c < C; c++) s += Wv[j * C + c] * avg[h][c]; heads[j] = s;
  }
  const out = linear(tensor(m, `${a}.out_proj.weight`), tensor(m, `${a}.out_proj.bias`), heads);
  const keys = layerNorm(prompt.map((x, i) => x + out[i]), tensor(m, `${L}.norm4.weight`), tensor(m, `${L}.norm4.bias`));
  const b = `${L}.cross_attn_token_to_image`;
  const v = linear(tensor(m, `${b}.v_proj.weight`), tensor(m, `${b}.v_proj.bias`), keys);
  const toVoxels = Float32Array.from(linear(tensor(m, `${b}.out_proj.weight`), tensor(m, `${b}.out_proj.bias`), v));
  return { prompt: keys, toVoxels };
}
/** The prompt entering decoder level `level`: GELU(Linear) of the previous level's (the first: of the embedding). */
export function promptIn(m: BundleParcModel, level: number, prev: ArrayLike<number>): Float64Array {
  return linear(tensor(m, `decoder.layers.${level}.prompt_encoding.0.weight`), tensor(m, `decoder.layers.${level}.prompt_encoding.0.bias`), prev).map(gelu);
}
/** The tract's prompt embedding: GELU(W·onehot + b) = GELU(column i of W + b). */
export function promptEmbedding(m: BundleParcModel, tract: number): Float64Array {
  const W = tensor(m, "prompt_embedding.0.weight"), b = tensor(m, "prompt_embedding.0.bias"), [out, n] = W.shape, y = new Float64Array(out);
  for (let o = 0; o < out; o++) y[o] = gelu(W.data[o * n + tract] + b.data[o]);
  return y;
}

// ── the output, as their bundleparc_predict_fitpad.py returns it ──

/** One tract's labels on the cube (0 outside, 1..10 along the tract) back onto the FODs' grid (`dims`, channel order as
 *  `prepareInput` read it): unpadded, then nearest neighbor -- nibabel's resample_from_to with order 0: target voxel i
 *  reads the fitted grid at i / step, rounded; a point past the fitted grid's last voxel reads 0, as scipy's "constant"
 *  mode does. */
export function toFodGrid(labels: Uint8Array, prep: Prepared, dims: [number, number, number]): Uint8Array {
  const [X, Y, Z] = dims, S = VOLUME, out = new Uint8Array(X * Y * Z);
  const near = (i: number, a: number) => { const f = i / prep.step[a]; return f > prep.fit[a] - 1 ? -1 : Math.round(f); };
  const xs = Int32Array.from({ length: X }, (_, i) => near(i, 0)), ys = Int32Array.from({ length: Y }, (_, i) => near(i, 1)), zs = Int32Array.from({ length: Z }, (_, i) => near(i, 2));
  for (let x = 0; x < X; x++) { if (xs[x] < 0) continue;
    for (let y = 0; y < Y; y++) { if (ys[y] < 0) continue;
      const row = ((xs[x] + prep.offset[0]) * S + (ys[y] + prep.offset[1])) * S + prep.offset[2], o = (x * Y + y) * Z;
      for (let z = 0; z < Z; z++) if (zs[z] >= 0) out[o + z] = labels[row + zs[z]];
    }
  }
  return out;
}
