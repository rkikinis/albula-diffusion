// The Tract review's own arithmetic (review.ts): which streamlines are judged, on which side, at which levels.
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc review.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { CST, cstOf, levelsOf, midlineX, sideToJudge } from "./review.ts";
import type { TractSetData } from "./tracts-dicom.ts";

/** A synthetic corticospinal tract as the real ones are (the test cases, 2026-10-06): 60 fibers from z = -40 (brainstem)
 *  to z = +60 (cortex) around x = cx, a narrow trunk 3-4 mm across up to z = +20, then a fan widening to 30 mm. */
function tract(cx: number): Float32Array[] {
  const width = (z: number) => z < 20 ? 3.5 - 0.01 * z : 3.3 + (z - 20) * 0.7;
  return Array.from({ length: 60 }, (_, k) => {
    const a = (2 * Math.PI * k) / 60, r = 0.4 + 0.6 * ((k * 7) % 10) / 10, pts: number[] = [];
    for (let z = -40; z <= 60; z += 1) pts.push(cx + Math.cos(a) * r * width(z), 10 + Math.sin(a) * r * width(z) * 0.6, z);
    return new Float32Array(pts);
  });
}

Deno.test("the corticospinal tract of each side, the midline between them, and the side away from the tumor", () => {
  const left = tract(-20), right = tract(24);
  const sets: TractSetData[] = [
    { label: CST, side: -1, color: [1, 0, 0], streamlines: left }, { label: CST, side: 1, color: [1, 0, 0], streamlines: right },
    { label: "arcuate fasciculus", side: -1, color: [0, 1, 0], streamlines: tract(-40) },
  ];
  const c = cstOf(sets);
  assertEquals([c.left.length, c.right.length], [60, 60]);
  const mid = midlineX(c.left, c.right);
  assert(Math.abs(mid - 2) < 0.5, `midline at ${mid}, not 2`);
  assertEquals(sideToJudge(30, mid), -1, "a tumor on the right: the left is judged");
  assertEquals(sideToJudge(-30, mid), 1, "a tumor on the left: the right is judged");
  assertEquals(sideToJudge(undefined, mid), -1, "no tumor (a control): the left");
  assertEquals(sideToJudge(1.5, mid), 1, "the midline is the tracts', not the scanner's x = 0");
  // One side missing: the midline from the other, 25 mm across.
  assertEquals(midlineX(c.left, []), midlineX(c.left, c.left) + 25);
});

Deno.test("the levels: the internal capsule 6 mm below where the fan begins, the peduncle 22 mm below it, the coronal through the tract", () => {
  const l = levelsOf(tract(-20));
  assert(l, "no levels");
  // The spread passes 1.4 times the trunk's a few millimeters above z = 20 (smoothed): the internal capsule near 20.
  assert(Math.abs(l!.ic - 20) <= 4, `internal capsule at ${l!.ic}, not near 20`);
  assertEquals(l!.crus, l!.ic - 22);
  assert(Math.abs(l!.coronal - 10) <= 1, `coronal at ${l!.coronal}, not through the tract (y = 10)`);
  assertEquals(levelsOf([]), undefined);
});
