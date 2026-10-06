import { assert, assertEquals } from "jsr:@std/assert@1";
import { crossingOutlines, sliceCrossings, trimEnds } from "./tract-slice.ts";

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

Deno.test("trimming the ends: the length less twice the trim, the cut points on the line; too short gives null; 0 leaves it", () => {
  const f = Float32Array.from([0, 0, 0, 4, 0, 0, 10, 0, 0, 10, 6, 0]);      // 16 mm, a corner at (10,0,0)
  const t = trimEnds(f, 2.5)!;
  let L = 0; for (let i = 3; i < t.length; i += 3) L += Math.hypot(t[i] - t[i - 3], t[i + 1] - t[i - 2], t[i + 2] - t[i - 1]);
  assert(Math.abs(L - 11) < 1e-5, `length ${L}`);
  assertEquals(Array.from(t.slice(0, 3)), [2.5, 0, 0]);
  assertEquals(Array.from(t.slice(-3)).map((x) => +x.toFixed(5)), [10, 3.5, 0]);
  assertEquals(trimEnds(f, 8), null);
  assertEquals(trimEnds(f, 0), f);
});

Deno.test("the outline of the crossings: one loop around the bundle, its own small loop around a stray fiber", () => {
  const z = 7, bundle: [number, number, number][] = [];
  for (let k = 0; k < 40; k++) bundle.push([10 + 1.5 * Math.cos(k), -5 + 1.5 * Math.sin(k * 1.3), z]);
  const stray: [number, number, number] = [22, -5, z];
  const loops = crossingOutlines([...bundle, stray], [0, 0, z], [1, 0, 0], [0, 1, 0], 1, 0.5);
  assertEquals(loops.length, 2, "the bundle and the stray fiber");
  const inside = (loop: [number, number, number][], x: number, y: number) => {
    let w = false;
    for (let i = 0, j = loop.length - 1; i < loop.length; j = i++) {
      const [xi, yi] = loop[i], [xj, yj] = loop[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) w = !w;
    }
    return w;
  };
  const around = loops.find((l) => inside(l, 10, -5))!, small = loops.find((l) => inside(l, 22, -5))!;
  assert(around && small && around !== small, "each loop encloses its own crossings");
  for (const p of small) assert(Math.abs(Math.hypot(p[0] - 22, p[1] + 5) - 1) < 0.6, "the stray fiber's loop lies about 1 mm from it");
  for (const l of loops) for (const p of l) assertEquals(p[2], z, "on the slice");
  assertEquals(crossingOutlines([], [0, 0, 0], [1, 0, 0], [0, 1, 0]), []);
});
