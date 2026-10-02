// The agreement measures (agreement.ts): tract mix r, center shift, label agreement; the scan's noise level and added noise.
import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { addScanNoise, agreement, noiseSigma } from "./agreement.ts";
import type { Named } from "./tractcloud/name-tracts.ts";
import type { DiffusionSeries } from "./dwi.ts";

const line = (x: number, y: number) => Float32Array.from([x, y, 0, x, y, 10, x, y, 20]);
const named = (tracts: number[]): Named => ({ tract: Int32Array.from(tracts), side: new Int8Array(tracts.length), draws: 1, seconds: 0 });

Deno.test("identical runs agree fully; a moved tract shows its shift; relabeling shows in the label agreement", () => {
  const sl = [...Array.from({ length: 20 }, () => line(0, 0)), ...Array.from({ length: 30 }, () => line(10, 0))];
  const n = named([...Array(20).fill(1), ...Array(30).fill(2)]);
  const same = agreement(sl, n, sl, n);
  assertAlmostEquals(same.mixR, 1, 1e-12);
  assertEquals([same.centerMedianMm, same.tractsCompared, same.labelAgreement], [0, 2, 1]);
  const moved = [...Array.from({ length: 20 }, () => line(3, 4)), ...sl.slice(20)];
  const m = agreement(sl, n, moved, n);
  assertEquals(m.tractsCompared, 2);
  assertAlmostEquals(m.center95Mm, 5, 1e-6);                     // tract 1 moved by (3, 4, 0)
  assertEquals(m.labelAgreement, undefined, "different streamlines: no label agreement");
  const relabeled = named([...Array(15).fill(1), ...Array(5).fill(2), ...Array(30).fill(2)]);
  const r = agreement(sl, n, sl, relabeled);
  assertAlmostEquals(r.labelAgreement!, 45 / 50, 1e-12);
  assert(r.mixR < 1);
});

Deno.test("the noise level is found from the background, and added noise has that size", () => {
  const dims: [number, number, number] = [40, 40, 10], n = 40 * 40 * 10, sigma = 7;
  const mask = new Uint8Array(n); for (let v = 0; v < n; v++) { const i = v % 40, j = Math.floor(v / 40) % 40; mask[v] = Math.hypot(i - 20, j - 20) < 12 ? 1 : 0; }
  const clean: DiffusionSeries = { volumes: [{ dims, ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], data: Float32Array.from(mask, (m) => m ? 500 : 0), dtype: "<f4" }],
    bValues: [0], gradients: [[0, 0, 0]], ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], source: "test", convention: 1 as never };
  const noisy = addScanNoise(clean, sigma, 1);
  // Outside: a Rayleigh background, from which σ is found within 5%.
  const est = noiseSigma(noisy, mask);
  assertAlmostEquals(est, sigma, sigma * 0.05);
  // Inside, at signal 500, the added noise is nearly Gaussian with standard deviation σ.
  let s = 0, s2 = 0, k = 0; for (let v = 0; v < n; v++) if (mask[v]) { const x = noisy.volumes[0].data[v] as number; s += x; s2 += x * x; k++; }
  const sd = Math.sqrt(s2 / k - (s / k) ** 2);
  assertAlmostEquals(sd, sigma, 0.5);
  assertEquals(addScanNoise(clean, sigma, 1).volumes[0].data, noisy.volumes[0].data, "seeded: the same noise again");
});
