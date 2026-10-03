// @full-tier -- the FIXED COHORT, whole brain on the graphics card (about 40 s a case, twelve cases): the rebuild runs it
// only in the full tier (Contents/tools/Rebuild SlicerAlbula App.command).
//
// THE CASE LIBRARY DOES NOT DRIFT (Ron, 2026-10-01: "Are there tests that you can add/improve now?"; Mike: an algorithm
// is not implemented if it isn't tested). Every case with a stored result (test/cases/<id>-ukf.json) is rerun through
// case-run.ts -- the code the batch tool uses -- and compared with it: the same number of streamlines (within 0.5%), the
// same named tracts near the tumor (at least 5 streamlines within 8 mm), and each tract's count near the tumor within 10%.
// The cases: PAT16 since 2026-10-01; since 2026-10-03 the twelve of Mike Halle's tractline cohort (validation step 5:
// "a fixed cohort that every change is checked against"), written by Contents/tools/dmri-cohort.ts. A change that moves
// them on purpose reruns that tool, rewriting the reference files in the same commit, and says why.
//
//   deno test -A --no-check --unstable-webgpu --config ../../src/SlicerLive/deno.jsonc cases.regression.test.ts
import { assert, assertEquals } from "jsr:@std/assert";
import { ABSENT, testData } from "albula/testing";
import { runCase } from "./case-run.ts";
import { loadModel, type ModelJson } from "./tractcloud/tractcloud.ts";

const DS = (testData("openneuro-ds001226", "") ?? ABSENT).replace(/\/$/, "");
const have = (id: string) => { try { return Deno.statSync(`${DS}/sub-${id}/ses-preop/dwi/sub-${id}_ses-preop_acq-AP_dwi.nii.gz`).isFile; } catch { return false; } };
const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
const CASES = [...Deno.readDirSync(new URL("./test/cases/", import.meta.url))].map((e) => e.name.match(/^(\w+)-ukf\.json$/)?.[1]).filter((x): x is string => !!x).sort();

for (const id of CASES) Deno.test({ name: `${id}, two-tensor: the same streamlines and the same named tracts near the tumor as the stored result`, ignore: !have(id) || !adapter, fn: async () => {
  const L = adapter!.limits;
  const device = await adapter!.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });
  try {
    const M = new URL("./tractcloud/model/", import.meta.url);
    const model = loadModel(Deno.readFileSync(new URL("weights.f32", M)).buffer, JSON.parse(Deno.readTextFileSync(new URL("model.json", M))) as ModelJson);
    const ref = JSON.parse(Deno.readTextFileSync(new URL(`./test/cases/${id}-ukf.json`, import.meta.url))) as { streamlines: number; near8: { tract: string; within8: number }[] };
    const r = await runCase(DS, id, device, model, "ukf");
    console.log(`${id}: ${r.streamlines} streamlines (stored ${ref.streamlines}), ${r.seconds.total} s`);
    assert(Math.abs(r.streamlines - ref.streamlines) <= ref.streamlines * 0.005, `streamlines ${r.streamlines} against ${ref.streamlines}`);
    const near = new Map(r.tracts.filter((t) => t.within8 >= 5).map((t) => [t.tract, t.within8]));
    const want = new Map(ref.near8.map((t) => [t.tract, t.within8]));
    const gained = [...near.keys()].filter((k) => !want.has(k)), lost = [...want.keys()].filter((k) => !near.has(k));
    assertEquals({ gained, lost }, { gained: [], lost: [] }, "the named tracts near the tumor changed");
    for (const [k, n] of want) assert(Math.abs(near.get(k)! - n) <= Math.max(2, n * 0.1), `${k}: ${near.get(k)} streamlines within 8 mm, stored ${n}`);
  } finally { device.destroy(); }
}});
