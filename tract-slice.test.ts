import { assert, assertEquals } from "jsr:@std/assert@1";
import { sliceCrossings } from "./tract-slice.ts";

Deno.test("a straight streamline crosses an axial plane once, at the right place, with its direction", () => {
  const f = Float32Array.from([0, 0, -2, 1, 0, -1, 2, 0, 0.5, 3, 0, 2]);
  const c = sliceCrossings([[f]], { origin: [0, 0, 0], normal: [0, 0, 1] });
  assertEquals(c.length, 1);
  assert(Math.abs(c[0].p[0] - 1 - 1 / 1.5) < 1e-6 && Math.abs(c[0].p[2]) < 1e-6, `at ${c[0].p}`);
  assert(c[0].dir[2] > 0.8);
  assertEquals(c[0].set, 0);
});

Deno.test("a U-shaped streamline crosses twice; one lying off the plane not at all; an oblique plane; several sets", () => {
  const u = Float32Array.from([0, 0, 5, 0, 0, -5, 3, 0, -5, 3, 0, 5]);
  const off = Float32Array.from([0, 0, 1, 5, 0, 1]);
  const c = sliceCrossings([[off], [u]], { origin: [0, 0, 0], normal: [0, 0, 1] });
  assertEquals(c.map((x) => x.set), [1, 1]);
  // Oblique plane x + z = 0 through a streamline along x at z = 1: crosses at x = -1.
  const line = Float32Array.from([-5, 2, 1, 5, 2, 1]);
  const o = sliceCrossings([[line]], { origin: [0, 0, 0], normal: [1, 0, 1] });
  assertEquals(o.length, 1);
  assert(Math.abs(o[0].p[0] + 1) < 1e-6);
});

Deno.test("a point exactly on the plane is counted once, not twice", () => {
  const f = Float32Array.from([0, 0, -1, 0, 0, 0, 0, 0, 1]);
  assertEquals(sliceCrossings([[f]], { origin: [0, 0, 0], normal: [0, 0, 1] }).length, 1);
});
