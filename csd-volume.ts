// CSD OVER A WHOLE SCAN, on the processor in parallel workers (csd-worker.ts): the voxels inside the brain mask split into
// chunks, each worker fitting its chunk with the exact solver (csd.ts csdFitFast, warm-started voxel to voxel). A job, not
// a wait: a whole brain takes minutes (Contents/docs/csd-ptt-plan-2026-10-01.md in the workspace).
import type { DiffusionSeries } from "./dwi.ts";
import { csdModel, type Kernel } from "./csd.ts";

export interface FodVolume {
  dims: [number, number, number]; ijkToRAS: number[];
  /** Per voxel: [fluid, gray matter, 45 white-matter FOD coefficients] (nx numbers; zero outside the mask). */
  coeffs: Float32Array; nx: number; lmax: number;
  /** The gradients' frame: the FOD's directions are in it (patient RAS for a DiffusionSeries, CONVENTION 1). */
  frame: "RAS";
  seconds: number; voxels: number;
}

/** Directions where the FOD must not be negative: n points spread over a hemisphere (Fibonacci). */
export function hemisphere(n = 300): Float64Array {
  const out = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const z = 1 - (i + 0.5) / n, r = Math.sqrt(1 - z * z), ph = i * Math.PI * (3 - Math.sqrt(5));
    out.set([r * Math.cos(ph), r * Math.sin(ph), z], 3 * i);
  }
  return out;
}

export async function csdVolume(dwi: DiffusionSeries, mask: Uint8Array, kernel: Kernel, opts: { workers?: number; constraint?: ArrayLike<number>; onProgress?: (f: number) => void; workerUrl?: URL } = {}): Promise<FodVolume> {
  const t0 = performance.now();
  const g = new Float64Array(dwi.gradients.length * 3); dwi.gradients.forEach((v, i) => g.set(v, 3 * i));
  const constraint = opts.constraint ?? hemisphere(300);
  const model = csdModel(g, dwi.bValues, kernel, constraint);       // checked here, once, before any worker starts
  const [nx, ny, nz] = dwi.volumes[0].dims, n = nx * ny * nz, rows = dwi.bValues.length;
  const vox: number[] = []; for (let v = 0; v < n; v++) if (mask[v]) vox.push(v);
  const coeffs = new Float32Array(n * model.nx);
  const W = Math.max(1, opts.workers ?? Math.min(8, (navigator.hardwareConcurrency ?? 4) - 1));
  const CHUNK = 2000, chunks: number[][] = [];
  for (let s = 0; s < vox.length; s += CHUNK) chunks.push(vox.slice(s, s + CHUNK));
  let done = 0;
  const url = opts.workerUrl ?? new URL("./csd-worker.ts", import.meta.url);
  await Promise.all(Array.from({ length: Math.min(W, chunks.length) }, async () => {
    const w = new Worker(url, { type: "module" });
    try {
      w.postMessage({ init: { gradients: g, bValues: Float64Array.from(dwi.bValues), kernel, constraint: Float64Array.from(constraint) } });
      for (let c = chunks.shift(); c; c = chunks.shift()) {
        const sig = new Float32Array(c.length * rows);
        c.forEach((v, i) => { for (let r = 0; r < rows; r++) sig[i * rows + r] = Number(dwi.volumes[r].data[v]); });
        const out: Float32Array = await new Promise((res, rej) => { w.onmessage = (e) => res(e.data as Float32Array); w.onerror = (e) => rej(new Error(e.message)); w.postMessage({ signals: sig, count: c.length }, [sig.buffer]); });
        c.forEach((v, i) => coeffs.set(out.subarray(i * model.nx, (i + 1) * model.nx), v * model.nx));
        done += c.length; opts.onProgress?.(done / vox.length);
      }
    } finally { w.terminate(); }
  }));
  return { dims: [nx, ny, nz], ijkToRAS: dwi.volumes[0].ijkToRAS, coeffs, nx: model.nx, lmax: kernel.lmax, frame: "RAS", seconds: (performance.now() - t0) / 1000, voxels: vox.length };
}
