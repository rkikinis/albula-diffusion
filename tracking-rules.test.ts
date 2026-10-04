// The tracking rules (tracking-rules.ts): the shell a rule tracks, and the every-voxel seeds.
//   deno test -A --no-check tracking-rules.test.ts
import { assertEquals } from "jsr:@std/assert@1";
import { brainFromT1, everyBrainVoxel, shellNearest, TRACKING_RULE, TRACKING_RULES } from "./tracking-rules.ts";
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

Deno.test("rule 3 is rule 2 with the brain taken from the T1 (SynthStrip); rule 2 stays the default until the module has it", () => {
  const { brain, id, ...r3 } = TRACKING_RULES[3], { brain: b2, id: i2, ...r2 } = TRACKING_RULES[2];
  assertEquals([id, brain, i2, b2], [3, "t1-synthstrip", 2, "median-otsu"]);
  assertEquals(r3, r2);
});

Deno.test("a T1-space mask put on the fit's grid by voxel centers, through a flipped and shifted affine", () => {
  // The mask: 1 mm voxels, the x axis flipped (as ds001226's mask files are), a block inside x in [-3, 2], y in [0, 3], z in [0, 1].
  const mdims = [10, 6, 4], mA = [-1, 0, 0, 5, 0, 1, 0, -1, 0, 0, 1, -1, 0, 0, 0, 1], data = new Uint8Array(240);
  for (let k = 0; k < 4; k++) for (let j = 0; j < 6; j++) for (let i = 0; i < 10; i++) {
    const x = 5 - i, y = j - 1, z = k - 1;
    if (x >= -3 && x <= 2 && y >= 0 && y <= 3 && z >= 0 && z <= 1) data[(k * 6 + j) * 10 + i] = 1;
  }
  // The fit's grid: 2 mm voxels from (-4, 0, 0).
  const out = brainFromT1({ dims: [4, 3, 2], ijkToRAS: [2, 0, 0, -4, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1] }, { dims: mdims, ijkToRAS: mA, data });
  // Centers x = -4, -2, 0, 2; y = 0, 2, 4; z = 0, 2: inside are x in {-2, 0, 2}, y in {0, 2}, z = 0.
  const want = new Uint8Array(24); for (const i of [1, 2, 3]) for (const j of [0, 1]) want[j * 4 + i] = 1;
  assertEquals(out, want);
});
