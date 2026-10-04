// The tracking rules (tracking-rules.ts): the shell a rule tracks, and the every-voxel seeds.
//   deno test -A --no-check tracking-rules.test.ts
import { assertEquals } from "jsr:@std/assert@1";
import { everyBrainVoxel, shellNearest, TRACKING_RULE, TRACKING_RULES } from "./tracking-rules.ts";
import type { TensorFit } from "./tensor.ts";

Deno.test("the shell nearest b = 3000: ds001226's 2800 among 700/1200/2800; a single shell is that shell; ties go to the higher", () => {
  assertEquals(shellNearest([0, 700, 700, 1200, 2800, 2805, 0], 3000), 2800);
  assertEquals(shellNearest([0, 1000, 1000], 3000), 1000);
  assertEquals(shellNearest([0, 2000, 4000], 3000), 4000);
  assertEquals(shellNearest([0, 5, 10], 3000), undefined);
});

Deno.test("rule 2 is the default and is Mike Halle's tractline's (the ORG atlas's) settings", () => {
  assertEquals(TRACKING_RULE, 2);
  assertEquals(TRACKING_RULES[2].ukf, { freeWater: false, seedingThreshold: 0.1, stoppingFA: 0.08, stoppingThreshold: 0.06, recordLength: 1.8 });
});

Deno.test("every brain voxel a seed, at its center; with a draw, jittered inside its own voxel and repeatable", () => {
  const dims: [number, number, number] = [3, 2, 2], n = 12, mask = new Uint8Array(n).fill(1), seedMask = new Uint8Array(n); seedMask[1] = seedMask[7] = 1;
  const fit = { dims, ijkToRAS: [2, 0, 0, 10, 0, 2, 0, 20, 0, 0, 2, 30, 0, 0, 0, 1], mask, seedMask } as unknown as TensorFit;
  assertEquals(everyBrainVoxel(fit), [[12, 20, 30], [12, 20, 32]]);
  const a = everyBrainVoxel(fit, 5), b = everyBrainVoxel(fit, 5);
  assertEquals(a, b);
  for (const [p, c] of a.map((p, i) => [p, everyBrainVoxel(fit)[i]] as const)) for (let k = 0; k < 3; k++) assertEquals(Math.abs(p[k] - c[k]) <= 1, true);
});
