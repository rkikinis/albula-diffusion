// @full-tier -- the FIXED COHORT, whole brain on the graphics card (42-69 s a case with tracking rule 3 and the T1 alignment, 2026-10-04; twelve cases): the rebuild runs it
// only in the full tier (Contents/tools/Rebuild SlicerAlbula App.command).
//
// THE CASE LIBRARY DOES NOT DRIFT (Ron, 2026-10-01: "Are there tests that you can add/improve now?"; Mike: an algorithm
// is not implemented if it isn't tested). Every case with a stored result (test/cases/<id>-ukf.json) is rerun through
// case-run.ts -- the code the batch tool uses -- and compared with it: the same number of streamlines (within 0.5%), the
// same named tracts near the tumor (at least 5 streamlines within 6 mm, the list the app shows; since 2026-10-04) and
// within 8 mm (where its gray band ends), each tract's count within 10%, and the same streamlines outside the brain.
// The cases: PAT16 since 2026-10-01; since 2026-10-03 the twelve of Mike Halle's tractline cohort (validation step 5:
// "a fixed cohort that every change is checked against"), written by Contents/tools/dmri-cohort.ts. A change that moves
// them on purpose reruns that tool, rewriting the reference files in the same commit, and says why.
//
//   deno test -A --no-check --unstable-webgpu --config ../../src/SlicerLive/deno.jsonc cases.regression.test.ts
import { assert, assertEquals } from "jsr:@std/assert";
import { ABSENT, testData } from "albula/testing";
import { runCase, synthstripMaskPath } from "./case-run.ts";
import { TRACKING_RULE, TRACKING_RULES } from "./tracking-rules.ts";
import { GRAY_BAND_MM, NEAR_MM } from "./planning.ts";
import { OUTSIDE_RULE } from "./outside-brain.ts";
import { MOTION_RULE } from "./motion.ts";
import { loadModel, type ModelJson } from "./tractcloud/tractcloud.ts";

const DS = (testData("openneuro-ds001226", "") ?? ABSENT).replace(/\/$/, "");
const exists = (f: string) => { try { return Deno.statSync(f).isFile; } catch { return false; } };
// Tracking rule 3 also needs the case's SynthStrip mask (Contents/tools/synthstrip-masks.ts); without it the case is skipped.
const have = (id: string) => exists(`${DS}/sub-${id}/ses-preop/dwi/sub-${id}_ses-preop_acq-AP_dwi.nii.gz`) && (TRACKING_RULES[TRACKING_RULE].brain !== "t1-synthstrip" || exists(synthstripMaskPath(DS, id)));
const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
const CASES = [...Deno.readDirSync(new URL("./test/cases/", import.meta.url))].map((e) => e.name.match(/^(\w+)-ukf\.json$/)?.[1]).filter((x): x is string => !!x).sort();

for (const id of CASES) Deno.test({ name: `${id}, two-tensor: the same streamlines and the same named tracts near the tumor as the stored result`, ignore: !have(id) || !adapter, fn: async () => {
  const L = adapter!.limits;
  const device = await adapter!.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });
  try {
    const M = new URL("./tractcloud/model/", import.meta.url);
    const model = loadModel(Deno.readFileSync(new URL("weights.f32", M)).buffer, JSON.parse(Deno.readTextFileSync(new URL("model.json", M))) as ModelJson);
    const ref = JSON.parse(Deno.readTextFileSync(new URL(`./test/cases/${id}-ukf.json`, import.meta.url))) as { streamlines: number; near6: { tract: string; within6: number }[]; near8: { tract: string; within8: number }[]; trackingRule?: number; outsideBrain?: number; outsideRule?: number };
    // The stored lists are at 6 and 8 mm: the app's distance and the end of its gray band. A change of either constant
    // must come with new references (critic, 2026-10-04, finding 4).
    assertEquals([NEAR_MM, NEAR_MM + GRAY_BAND_MM], [6, 8], "planning.ts NEAR_MM or GRAY_BAND_MM changed: the references are at 6 and 8 mm");
    assertEquals(ref.outsideRule ?? 0, OUTSIDE_RULE.on ? OUTSIDE_RULE.id : 0, "the stored result was made with the outside-the-brain test in another state: rerun Contents/tools/dmri-cohort.ts");
    // A reference made under another rule is a stale file, not a regression (critic, 2026-10-04, finding 10).
    assertEquals(ref.trackingRule, TRACKING_RULE, `the stored result was made under tracking rule ${ref.trackingRule}, the default is ${TRACKING_RULE}: rerun Contents/tools/dmri-cohort.ts`);
    assertEquals(ref.motionRule ?? 0, MOTION_RULE, `the stored result was made under head-movement rule ${ref.motionRule ?? 0}, the default is ${MOTION_RULE}: rerun Contents/tools/dmri-cohort.ts`);
    const r = await runCase(DS, id, device, model, "ukf");
    console.log(`${id}: ${r.streamlines} streamlines (stored ${ref.streamlines}), ${r.seconds.total} s`);
    assert(Math.abs(r.streamlines - ref.streamlines) <= ref.streamlines * 0.005, `streamlines ${r.streamlines} against ${ref.streamlines}`);
    // The list the app shows: at least 5 streamlines within 6 mm (planning.ts NEAR_MM, since 2026-10-04; 8 before).
    const near = new Map(r.tracts.filter((t) => t.within6 >= 5).map((t) => [t.tract, t.within6]));
    const want = new Map(ref.near6.map((t) => [t.tract, t.within6]));
    assertEquals(r.outsideBrain, ref.outsideBrain, "the streamlines outside the brain (outside-brain.ts) changed");
    const gained = [...near.keys()].filter((k) => !want.has(k)), lost = [...want.keys()].filter((k) => !near.has(k));
    assertEquals({ gained, lost }, { gained: [], lost: [] }, "the named tracts near the tumor changed");
    for (const [k, n] of want) assert(Math.abs(near.get(k)! - n) <= Math.max(2, n * 0.1), `${k}: ${near.get(k)} streamlines within 6 mm, stored ${n}`);
    // Within 8 mm: what the gray band lists.
    const near8 = new Map(r.tracts.filter((t) => t.within8 >= 5).map((t) => [t.tract, t.within8])), want8 = new Map(ref.near8.map((t) => [t.tract, t.within8]));
    assertEquals({ gained: [...near8.keys()].filter((k) => !want8.has(k)), lost: [...want8.keys()].filter((k) => !near8.has(k)) }, { gained: [], lost: [] }, "the named tracts within 8 mm changed");
  } finally { device.destroy(); }
}});
