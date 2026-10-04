// THE CARD'S UKF AGAINST THE PROCESSOR'S (critic, 2026-09-30, qa/2026-09-30-ukf-gpu-numerics.md, finding 6: nothing tested
// them against each other, and a units error in the card's projection went unseen). Same seeds on PAT16, both run to the
// end; the fibers must cover the same white matter and end in the same places. Measured when written: density maps
// correlate 0.972, 90% of fiber ends within 0.7 mm (was 0.876 and 17 mm before the fix).
//   deno test -A --no-check --unstable-webgpu extensions/diffusion/ukf-gpu.test.ts
import { assert } from "jsr:@std/assert@1";
import { fromFsl } from "./dwi.ts";
import { parseNiftiVolumes } from "albula";
import { fitTensors } from "./tensor.ts";
import { prepareUkfData, trackUkf } from "./ukf.ts";
import { trackUkfGpu } from "./ukf-gpu.ts";
import { ABSENT, testData } from "albula/testing";

const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
const adapter = gpu ? await gpu.requestAdapter().catch(() => null) : null;
const L = adapter?.limits;
const device = adapter && L ? await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } }).catch(() => null) : null;
const D = testData("openneuro-ds001226", "sub-PAT16/ses-preop/dwi") ?? ABSENT;
const HAVE = (() => { try { Deno.statSync(`${D}sub-PAT16_ses-preop_acq-AP_dwi.nii.gz`); return true; } catch { return false; } })();

async function compare(freeWater: boolean) {
    const I = `${D}sub-PAT16_ses-preop_acq-AP_dwi`;
    const dwi = fromFsl(await parseNiftiVolumes(Deno.readFileSync(`${I}.nii.gz`)), Deno.readTextFileSync(`${I}.bval`), Deno.readTextFileSync(`${I}.bvec`));
    const fit = fitTensors(dwi);
    const [nx, ny, nz] = fit.dims, seeds: number[][] = [];
    for (let k = 0; k < nz; k += 2) for (let j = 0; j < ny; j += 2) for (let i = 0; i < nx; i += 2) {
      const v = (k * ny + j) * nx + i;
      if (fit.mask[v] && fit.fa[v] > 0.25 && Math.abs(i - nx / 2) < 15 && Math.abs(j - ny / 2) < 15 && Math.abs(k - nz / 2) < 10) seeds.push([i, j, k]);
    }
    const S = seeds.slice(0, 300), data = prepareUkfData(dwi, fit.mask);
    // The plain model with tracking rule 2's thresholds (tracking-rules.ts; critic, 2026-10-03, finding 18).
    const o = freeWater ? { freeWater } : { freeWater, seedingThreshold: 0.1, stoppingFA: 0.08, stoppingThreshold: 0.06, recordLength: 1.8 };
    const cpu = trackUkf(data, S, o), card = await trackUkfGpu(device!, data, S, o);
    // Same seeds accepted.
    assert(Math.abs(cpu.fibers.length - card.fibers.length) <= 1, `fibers: processor ${cpu.fibers.length}, card ${card.fibers.length}`);
    // Where the fibers end (per seed, either orientation of the fiber).
    const C = new Map(cpu.fibers.map((f) => [f.seed, f.points])), ends: number[] = [];
    for (const g of card.fibers) {
      const c = C.get(g.seed); if (!c) continue;
      const a0 = [c[0], c[1], c[2]], a1 = [c.at(-3)!, c.at(-2)!, c.at(-1)!], b0 = [g.points[0], g.points[1], g.points[2]], b1 = [g.points.at(-3)!, g.points.at(-2)!, g.points.at(-1)!];
      const d = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
      ends.push(Math.min(d(a0, b0) + d(a1, b1), d(a0, b1) + d(a1, b0)) / 2);
    }
    ends.sort((p, q) => p - q);
    const end90 = ends[Math.floor(ends.length * 0.9)];
    // Tract density on the diffusion grid.
    const M = fit.ijkToRAS, [a, b, c, d, e, f, g2, h, i2] = [M[0], M[1], M[2], M[4], M[5], M[6], M[8], M[9], M[10]];
    const A_ = e * i2 - f * h, B_ = -(d * i2 - f * g2), C_ = d * h - e * g2, det = a * A_ + b * B_ + c * C_;
    const R = [A_, -(b * i2 - c * h), b * f - c * e, B_, a * i2 - c * g2, -(a * f - c * d), C_, -(a * h - b * g2), a * e - b * d].map((x) => x / det);
    const dens = (fs: { points: Float32Array }[]) => { const o = new Float32Array(nx * ny * nz); for (const fb of fs) { const seen = new Set<number>(); const p = fb.points; for (let q = 0; q < p.length; q += 3) { const u = [p[q] - M[3], p[q + 1] - M[7], p[q + 2] - M[11]]; const ii = Math.round(R[0] * u[0] + R[1] * u[1] + R[2] * u[2]), jj = Math.round(R[3] * u[0] + R[4] * u[1] + R[5] * u[2]), kk = Math.round(R[6] * u[0] + R[7] * u[1] + R[8] * u[2]); if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) continue; const v = (kk * ny + jj) * nx + ii; if (!seen.has(v)) { seen.add(v); o[v]++; } } } return o; };
    const x = dens(cpu.fibers), y = dens(card.fibers);
    let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let v = 0; v < x.length; v++) { if (!x[v] && !y[v]) continue; n++; sx += x[v]; sy += y[v]; sxx += x[v] ** 2; syy += y[v] ** 2; sxy += x[v] * y[v]; }
    const r = (n * sxy - sx * sy) / Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    console.log(`  ${freeWater ? "free water" : "plain two-tensor"}: ${card.fibers.length} fibers; 90% of ends within ${end90.toFixed(2)} mm; density correlation ${r.toFixed(3)}; card ${(card.ms.total / 1000).toFixed(1)} s, processor ${(cpu.ms / 1000).toFixed(1)} s`);
    // Limits near what is measured (2026-10-03: free water 0.06 mm / r 0.973; plain with rule 2's thresholds 0.10 mm /
    // r 0.970 -- longer fibers part more), so a real regression fails instead of passing under a limit a hundred times wider.
    assert(end90 < (freeWater ? 0.5 : 0.3), `90% of fiber ends within ${end90.toFixed(2)} mm`);
    assert(r > (freeWater ? 0.96 : 0.955), `density correlation ${r.toFixed(3)}`);
}

Deno.test({ name: "UKF: the graphics card's fibers agree with the processor's (PAT16, 300 seeds)", ignore: !device || !HAVE, fn: () => compare(true) });
// THE PLAIN TWO-TENSOR MODEL (2026-10-03; the ORG atlas behind the tract names was tracked with it): the same check.
Deno.test({ name: "UKF, plain two-tensor: the graphics card's fibers agree with the processor's (PAT16, 300 seeds)", ignore: !device || !HAVE, fn: () => compare(false) });

Deno.test({ name: "every variant of the card tracker's shader compiles (a shader that does not compile runs nothing, silently)", ignore: !navigator.gpu, fn: async () => {
  const { wgsl } = await import("./ukf-gpu.ts");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return;
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: { maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage } });
  const bad: string[] = [];
  for (let code = 0; code < 128; code++) {
    const b = [0, 1, 2, 3, 4, 5, 6].map((k) => ((code >> k) & 1) === 1) as [boolean, boolean, boolean, boolean, boolean, boolean, boolean];
    if (b[5] && !b[3]) continue;                          // parInverse needs wgInverse
    if (!b[6] && b[4]) continue;                          // onePass is the free-water model's only (the 7th: free water)
    const info = await device.createShaderModule({ code: wgsl(...b) }).getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === "error");
    if (errs.length) bad.push(`${b.map(Number).join("")}: ${errs[0].lineNum}:${errs[0].linePos} ${errs[0].message.split("\n")[0]}`);
  }
  device.destroy();
  if (bad.length) throw new Error(`shader variants that do not compile:\n${bad.join("\n")}`);
}});
