// The corticospinal tract's anatomical gates (cst-gates.ts) on synthetic slices.
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc cst-gates.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { crusErrant, crusOf, posteriorLimbOf, type PlaneImage } from "./cst-gates.ts";

/** A 40 x 40 mm slice at 0.5 mm, colored by a function of the in-plane position (x = the head's left, y = its front). */
function slice(color: (x: number, y: number) => [number, number, number]): PlaneImage {
  const nu = 80, nv = 80, step = 0.5, u0 = -20, v0 = -20, rgb = new Float32Array(3 * nu * nv);
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) { const c = color(u0 + i * step, v0 + j * step); rgb.set(c, 3 * (j * nu + i)); }
  return { nu, nv, step, u0, v0, rgb };
}

Deno.test("the crus: the pink region on its side, cut off by the green band; behind the band is errant, in the crus is not", () => {
  // On the head's left (x > 0): a pink crus (y 2..10), a green band (y 0..2), blue tegmentum (y -10..0).
  const img = slice((x, y) => x < 3 || x > 15 ? [0, 0, 0] : y > 2 && y < 10 ? [0.5, 0.05, 0.45] : y >= 0 && y <= 2 ? [0.05, 0.5, 0.1] : y > -10 && y < 0 ? [0.1, 0.05, 0.6] : [0, 0, 0]);
  const { crus, green } = crusOf(img, 1);
  assert(crus.some(Boolean), "the crus is found on the left");
  assert(!crusOf(img, -1).crus.some(Boolean), "nothing on the right");
  assertEquals(crusErrant(img, crus, green, [9, 6]), false, "in the crus");
  assertEquals(crusErrant(img, crus, green, [9, -3]), true, "behind the green band, in the tegmentum");
  assertEquals(crusErrant(img, crus, green, [9, 2.5]), false, "at the crus's edge (1 mm of slack)");
  assertEquals(crusErrant(img, crus, green, [-9, 6]), true, "far from it");
});

Deno.test("the posterior limb: the blue band 10-35 mm from the midline on its side", () => {
  const img = slice((x, y) => x > 12 && x < 18 && y > -15 && y < 5 ? [0.1, 0.1, 0.7] : x > 2 && x < 6 ? [0.1, 0.1, 0.7] : [0.05, 0.4, 0.05]);
  const limb = posteriorLimbOf(img, 1);
  const at = (x: number, y: number) => limb[Math.round((y + 20) / 0.5) * 80 + Math.round((x + 20) / 0.5)];
  assertEquals(at(15, -5), 1, "the limb's blue");
  assertEquals(at(4, -5), 0, "a blue strip near the midline is not the limb");
});

Deno.test("the crus border from the FA ridge: behind the ridge, where FA has fallen half-way to the level behind", async () => {
  const { crusBorderFromRidge, dorsalOfBorder } = await import("./cst-gates.ts");
  // Left side (x > 0): the surface at y = 8 (outside FA 0), a pink crus y 3..8 (FA 0.7, red 0.45), behind it y < 3 tegmentum
  // (FA 0.4, blue): the border should come out near y = 3.
  const img = slice((x, y) => x < 3 || x > 20 ? [0, 0, 0] : y >= 8 ? [0, 0, 0] : y >= 3 ? [0.45, 0.05, 0.53] : [0.05, 0.05, 0.4]);
  const b = crusBorderFromRidge(img, 1);
  assert(b.length > 10, `columns found: ${b.length}`);
  for (const [, y] of b) assert(y > 1 && y < 4, `border at y = ${y}, not near 3`);
  assertEquals(dorsalOfBorder(b, [10, 0]), true, "behind it");
  assertEquals(dorsalOfBorder(b, [10, 6]), false, "in the crus");
  assertEquals(crusBorderFromRidge(img, -1).length, 0, "nothing on the right");
});
