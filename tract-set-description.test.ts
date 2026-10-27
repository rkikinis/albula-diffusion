import { assert, assertEquals } from "jsr:@std/assert@1";
import { describeTractSet } from "./tract-set-description.ts";

// The Prisma scan's eight sets of 2026-10-27, by the fields their records carry.
Deno.test("each kind of tract set gets the title the approved mockup gives it", () => {
  const std = { rules: { tracking: 3, labeler: "rapidparc 6a15; clusters ORG-800FC-100HCP" }, brain: "SynthStrip's brain mask of the T1" };
  assertEquals(describeTractSet(std).title, "Fiber tracts, standard");
  assertEquals(describeTractSet({ ...std, rules: { ...std.rules, tracking: 7 } }).title, "Fiber tracts stopping at the cortex");
  assertEquals(describeTractSet({ ...std, rules: { ...std.rules, tracking: 7 }, cstGate: { rule: 2 } }).title, "Fiber tracts stopping at the cortex, motor pathway checked");
  assertEquals(describeTractSet({ ...std, rules: { ...std.rules, labeler: "tractcloud 5a84 (kGlobal 500)" } }).title, "Fiber tracts, named by a second network");
  const tl = { madeWith: { pipeline: "tractline v0.5.2 (ef860a1d), Michael Halle" } };
  assertEquals(describeTractSet(tl).title, "Fiber tracts, brain outline from the diffusion scan");
  assertEquals(describeTractSet({ madeWith: { pipeline: "tractline v0.5.2 (ef860a1d), Michael Halle; brain mask replaced" } }).title, "Fiber tracts, brain outline from the T1");
  assertEquals(describeTractSet({ ...std, cstByAnatomy: { what: "x" } }).title, "Fiber tracts, motor pathway by anatomy");
});

Deno.test("the line names the method with its reference and the motor pathway's counts; titles fit DICOM's 64", () => {
  const d = describeTractSet({ rules: { tracking: 3, labeler: "rapidparc" }, brain: "SynthStrip's brain mask of the T1" }, { left: 569, right: 878 });
  assertEquals(d.line, "Two-tensor tractography (Malcolm 2010), brain from the T1 (SynthStrip, Hoopes 2022), named against an atlas (Zhang 2018) — motor pathway 569 left, 878 right");
  const tl = describeTractSet({ madeWith: { pipeline: "tractline v0.5.2" } }, { left: 2, right: 32 });
  assert(tl.line.startsWith("Two-tensor tractography (tractline, Halle 2026), brain found on the diffusion scan"));
  assert(/second research program, tractline \(M\. Halle, 2026\)/.test(tl.tip) && !/Mike/.test(tl.tip + tl.line + tl.title));
  for (const r of [{}, { cstGate: {}, rules: { tracking: 7 } }]) assert(describeTractSet(r).title.length <= 64);
});
