// THE CARD'S TENSOR FIT MUST AGREE WITH THE PROCESSOR REFERENCE (extensions/diffusion/tensor.ts, itself checked against
// DIPY): a small synthetic scan here, and OpenNeuro ds001226 PAT16 when it is on disk. Skipped without a graphics card.
//   deno test -A --no-check --unstable-webgpu extensions/diffusion/tensor-gpu.test.ts
import { assert } from "jsr:@std/assert@1";
import { fitTensorsGpu } from "./tensor-gpu.ts";
import { fitTensors } from "./tensor.ts";
import { DWI_CONVENTION, type DiffusionSeries, fromFsl } from "./dwi.ts";
import { parseNiftiVolumes } from "albula";
import { ABSENT, testData } from "albula/testing";

const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
const adapter = gpu ? await gpu.requestAdapter().catch(() => null) : null;
const device = adapter ? await adapter.requestDevice().catch(() => null) : null;

function compare(a: { fa: Float32Array; v1: Float32Array; md: Float32Array }, b: { fa: Float32Array; v1: Float32Array; md: Float32Array }, mask: Uint8Array) {
  let faMax = 0, angMax = 0, mdMax = 0, n = 0;
  for (let v = 0; v < mask.length; v++) {
    if (!mask[v]) continue;
    n++;
    faMax = Math.max(faMax, Math.abs(a.fa[v] - b.fa[v]));
    mdMax = Math.max(mdMax, Math.abs(a.md[v] - b.md[v]) / Math.max(Math.abs(b.md[v]), 1e-6));
    if (b.fa[v] > 0.3) {
      const c = Math.abs(a.v1[3 * v] * b.v1[3 * v] + a.v1[3 * v + 1] * b.v1[3 * v + 1] + a.v1[3 * v + 2] * b.v1[3 * v + 2]);
      angMax = Math.max(angMax, (Math.acos(Math.min(1, c)) * 180) / Math.PI);
    }
  }
  return { faMax, angMax, mdMax, n };
}

Deno.test({
  name: "card = processor on a synthetic scan (random tensors, two shells, one b=0)",
  ignore: !device,
  fn: async () => {
    const nv = 500, dims: [number, number, number] = [10, 10, 5];
    const dirs: [number, number, number][] = [];
    for (let i = 0; i < 30; i++) { const z = 1 - (2 * (i + 0.5)) / 30, r = Math.sqrt(1 - z * z), t = i * 2.39996; dirs.push([r * Math.cos(t), r * Math.sin(t), z]); }
    const bs = [0, ...Array(15).fill(800), ...Array(15).fill(1400)];
    const gs = [[0, 0, 0] as [number, number, number], ...dirs];
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
    const tensors = Array.from({ length: nv }, () => {
      const a = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5]; const l = Math.hypot(...a); const u = a.map((x) => x / l);
      const l1 = 0.8e-3 + 1.2e-3 * rnd(), l2 = 0.2e-3 + 0.4e-3 * rnd();
      return (g: number[]) => l2 + (l1 - l2) * (g[0] * u[0] + g[1] * u[1] + g[2] * u[2]) ** 2;
    });
    const vols = bs.map((b, i) => ({ dims, ijkToRAS: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1], dtype: "<f4",
      data: new Float32Array(nv).map((_, v) => Math.round(1000 * Math.exp(-b * tensors[v](gs[i])) * (1 + 0.02 * (rnd() - 0.5)))) }));
    const dwi: DiffusionSeries = { volumes: vols, bValues: bs, gradients: gs, ijkToRAS: vols[0].ijkToRAS, source: "synthetic", convention: DWI_CONVENTION };
    const mask = new Uint8Array(nv).fill(1);
    const cpu = fitTensors(dwi, { mask });
    const g = await fitTensorsGpu(device!, dwi, { mask, chunkVoxels: 128 });       // several chunks on purpose
    const r = compare(g, cpu, mask);
    assert(r.faMax < 1e-3 && r.angMax < 0.5 && r.mdMax < 1e-3, JSON.stringify(r));
  },
});

const D = testData("openneuro-ds001226", "sub-PAT16/ses-preop/dwi") ?? ABSENT;
const HAVE = (() => { try { Deno.statSync(`${D}sub-PAT16_ses-preop_acq-AP_dwi.nii.gz`); return true; } catch { return false; } })();
Deno.test({
  name: "card = processor on PAT16 (the whole brain mask)",
  ignore: !device || !HAVE,
  fn: async () => {
    const I = `${D}sub-PAT16_ses-preop_acq-AP_dwi`;
    const dwi = fromFsl(await parseNiftiVolumes(Deno.readFileSync(`${I}.nii.gz`)), Deno.readTextFileSync(`${I}.bval`), Deno.readTextFileSync(`${I}.bvec`));
    const t0 = performance.now();
    const cpu = fitTensors(dwi);
    const cpuMs = performance.now() - t0;
    const g = await fitTensorsGpu(device!, dwi, { mask: cpu.mask });
    const r = compare(g, cpu, cpu.mask);
    console.log(`PAT16: processor ${(cpuMs / 1000).toFixed(2)} s; card ${(g.ms.total / 1000).toFixed(2)} s (prepare ${(g.ms.prepare / 1000).toFixed(2)}, card incl. upload/readback ${(g.ms.gpu / 1000).toFixed(2)}); FA max diff ${r.faMax.toExponential(1)}, direction ${r.angMax.toFixed(2)}°, MD ${r.mdMax.toExponential(1)}`);
    assert(r.faMax < 2e-3, `FA differs by ${r.faMax}`);
    assert(r.angMax < 1, `direction differs by ${r.angMax}°`);
  },
});
