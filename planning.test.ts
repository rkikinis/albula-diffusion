import { assert } from "jsr:@std/assert@1";
import { wholeBrainSeeds } from "./planning.ts";
import type { TensorFit } from "./tensor.ts";

Deno.test("whole-brain seeds do not depend on the order the slices are stored in", () => {
  const [nx, ny, nz] = [7, 6, 5], n = nx * ny * nz;
  const fa = Float32Array.from({ length: n }, (_, v) => 0.1 + ((v * 37) % 11) / 11);
  const up = { dims: [nx, ny, nz], ijkToRAS: [2, 0, 0, -5, 0, 2, 0, -6, 0, 0, 2, 1, 0, 0, 0, 1], mask: new Uint8Array(n).fill(1), fa } as unknown as TensorFit;
  // The same grid, slices stored the other way round.
  const fa2 = new Float32Array(n);
  for (let k = 0; k < nz; k++) for (let v = 0; v < nx * ny; v++) fa2[(nz - 1 - k) * nx * ny + v] = fa[k * nx * ny + v];
  const down = { ...up, fa: fa2, ijkToRAS: [2, 0, 0, -5, 0, 2, 0, -6, 0, 0, -2, 1 + 2 * (nz - 1), 0, 0, 0, 1] } as unknown as TensorFit;
  const a = wholeBrainSeeds(up), b = wholeBrainSeeds(down);
  assert(a.length === b.length && a.length > 0);
  for (let i = 0; i < a.length; i++) for (let c = 0; c < 3; c++) assert(Math.abs(a[i][c] - b[i][c]) < 1e-9, `seed ${i} differs`);
});
