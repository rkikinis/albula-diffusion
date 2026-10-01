// writeNrrdDwi round-trips through the reader (dwi.ts fromNrrdDwi): the same grid, b-values, gradient directions in
// patient RAS and voxels -- so the file handed to the original UKFTractography says what our own series says.
//   deno test -A --no-check dwi-nrrd-write.test.ts   (with core's config)
import { assert, assertEquals } from "jsr:@std/assert";
import { fromNrrdDwi, type DiffusionSeries } from "./dwi.ts";
import { writeNrrdDwi } from "./dwi-nrrd-write.ts";

Deno.test("a diffusion series written as NRRD DWI reads back the same (an oblique grid, three shells, a b=0)", async () => {
  const dims: [number, number, number] = [4, 3, 2], n = 24;
  const c = Math.cos(0.3), s = Math.sin(0.3);
  const ijkToRAS = [-2 * c, 2 * s, 0, 10, 2 * s, 2 * c, 0, -20, 0, 0, 2.5, 30, 0, 0, 0, 1];   // oblique, a flip, anisotropic
  const g: [number, number, number][] = [[0, 0, 0], [1, 0, 0], [0, Math.SQRT1_2, Math.SQRT1_2], [0.6, 0.8, 0]];
  const b = [0, 700, 1200, 2800];
  const series = { name: "t", bValues: b, gradients: g, ijkToRAS, source: "test", convention: 1,
    volumes: b.map((_, q) => ({ dims, ijkToRAS, data: Float32Array.from({ length: n }, (_, i) => 100 * q + i), dtype: "<f4" })) } as unknown as DiffusionSeries;
  const back = await fromNrrdDwi(writeNrrdDwi(series));
  assertEquals(back.volumes[0].dims, dims);
  back.ijkToRAS.forEach((v, i) => assert(Math.abs(v - ijkToRAS[i]) < 1e-6, `ijkToRAS[${i}]`));
  back.bValues.forEach((v, i) => assert(Math.abs(v - b[i]) < 1e-3, `b[${i}] ${v}`));
  back.gradients.forEach((v, i) => v.forEach((x, k) => assert(Math.abs(x - g[i][k]) < 1e-6, `g[${i}][${k}] ${x} vs ${g[i][k]}`)));
  back.volumes.forEach((v, q) => { for (let i = 0; i < n; i++) assertEquals(Number(v.data[i]), 100 * q + i); });
});

Deno.test("written in LPS (Slicer's convention) it reads back to the same RAS series", async () => {
  const dims: [number, number, number] = [2, 2, 2];
  const ijkToRAS = [-2, 0, 0, 10, 0, 2, 0, -20, 0, 0, 2.5, 30, 0, 0, 0, 1];
  const g: [number, number, number][] = [[0, 0, 0], [0.6, 0.8, 0]];
  const series = { bValues: [0, 1000], gradients: g, ijkToRAS, source: "t", convention: 1,
    volumes: [0, 1].map((q) => ({ dims, ijkToRAS, data: Float32Array.from({ length: 8 }, (_, i) => q * 10 + i), dtype: "<f4" })) } as unknown as DiffusionSeries;
  const back = await fromNrrdDwi(writeNrrdDwi(series, { space: "LPS" }));
  back.ijkToRAS.forEach((v, i) => assert(Math.abs(v - ijkToRAS[i]) < 1e-6, `ijkToRAS[${i}]`));
  back.gradients[1].forEach((x, k) => assert(Math.abs(x - g[1][k]) < 1e-6, `g[${k}]`));
});
