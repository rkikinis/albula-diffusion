// compareFibers on sets whose answer is known: identical sets; one set shifted by 1 mm; a set with a fiber missing.
//   deno test -A --no-check fiber-compare.test.ts   (with core's config)
import { assert, assertEquals } from "jsr:@std/assert";
import { compareFibers } from "./fiber-compare.ts";

const line = (x0: number, y: number, z: number, n: number) => Float32Array.from({ length: n * 3 }, (_, i) => (i % 3 === 0 ? x0 + Math.floor(i / 3) : i % 3 === 1 ? y : z));
const A = [line(0, 0, 0, 30), line(0, 10, 0, 20), line(0, 20, 5, 40)];
const starts = [[10, 0, 0], [10, 10, 0], [10, 20, 5]];

Deno.test("identical sets: every start matched, ends 0 mm apart, lengths equal, maps identical", () => {
  const c = compareFibers(A, A, starts);
  assertEquals([c.inA, c.inB, c.inBoth], [3, 3, 3]);
  assertEquals(c.endMedianMm, 0);
  assertEquals(c.lengthRatioMedian, 1);
  assert(Math.abs(c.densityCorrelation - 1) < 1e-12);
});

Deno.test("one set reversed and shifted 0.5 mm: ends 0.5 mm apart (reversal does not matter)", () => {
  const B = A.map((f) => { const g = new Float32Array(f.length); for (let i = 0; i < f.length; i += 3) { const j = f.length - 3 - i; g[i] = f[j]; g[i + 1] = f[j + 1] + 0.5; g[i + 2] = f[j + 2]; } return g; });
  const c = compareFibers(A, B, starts);
  assertEquals(c.inBoth, 3);
  assert(Math.abs(c.endMedianMm - 0.5) < 1e-5, String(c.endMedianMm));
});

Deno.test("a fiber missing in one set is counted, not matched to another", () => {
  const c = compareFibers(A, [A[0], A[2]], starts);
  assertEquals([c.inA, c.inB, c.inBoth], [3, 2, 2]);
});
