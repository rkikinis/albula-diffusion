// @full-tier -- one case of the library, whole brain on the graphics card (about a minute): the rebuild runs it only in the
// full tier (Contents/tools/Rebuild SlicerAlbula App.command).
//
// THE CASE LIBRARY DOES NOT DRIFT (Ron, 2026-10-01: "Are there tests that you can add/improve now?"; Mike: an algorithm
// is not implemented if it isn't tested). PAT16 is rerun through case-run.ts -- the code the batch tool uses -- and
// compared with its stored result (test/cases/PAT16-ukf.json): the same number of streamlines (within 0.5%), the same
// named tracts near the tumor (at least 5 streamlines within 8 mm), and each tract's count near the tumor within 10%.
// A change that moves them on purpose rewrites the reference file in the same commit and says why.
//
//   deno test -A --no-check --unstable-webgpu --config ../../src/SlicerLive/deno.jsonc cases.regression.test.ts
import { assert, assertEquals } from "jsr:@std/assert";
import { ABSENT, testData } from "albula/testing";
import { runCase } from "./case-run.ts";
import { loadModel, type ModelJson } from "./tractcloud/tractcloud.ts";

const DS = (testData("openneuro-ds001226", "") ?? ABSENT).replace(/\/$/, "");
const HAVE = (() => { try { return Deno.statSync(`${DS}/sub-PAT16/ses-preop/dwi/sub-PAT16_ses-preop_acq-AP_dwi.nii.gz`).isFile; } catch { return false; } })();
const adapter = await navigator.gpu?.requestAdapter().catch(() => null);

Deno.test({ name: "PAT16, two-tensor: the same streamlines and the same named tracts near the tumor as the stored result", ignore: !HAVE || !adapter, fn: async () => {
  const L = adapter!.limits;
  const device = await adapter!.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });
  try {
    const M = new URL("./tractcloud/model/", import.meta.url);
    const model = loadModel(Deno.readFileSync(new URL("weights.f32", M)).buffer, JSON.parse(Deno.readTextFileSync(new URL("model.json", M))) as ModelJson);
    const ref = JSON.parse(Deno.readTextFileSync(new URL("./test/cases/PAT16-ukf.json", import.meta.url))) as { streamlines: number; near8: { tract: string; within8: number }[] };
    const r = await runCase(DS, "PAT16", device, model, "ukf");
    console.log(`PAT16: ${r.streamlines} streamlines (stored ${ref.streamlines}), ${r.seconds.total} s`);
    assert(Math.abs(r.streamlines - ref.streamlines) <= ref.streamlines * 0.005, `streamlines ${r.streamlines} against ${ref.streamlines}`);
    const near = new Map(r.tracts.filter((t) => t.within8 >= 5).map((t) => [t.tract, t.within8]));
    const want = new Map(ref.near8.map((t) => [t.tract, t.within8]));
    const gained = [...near.keys()].filter((k) => !want.has(k)), lost = [...want.keys()].filter((k) => !near.has(k));
    assertEquals({ gained, lost }, { gained: [], lost: [] }, "the named tracts near the tumor changed");
    for (const [k, n] of want) assert(Math.abs(near.get(k)! - n) <= Math.max(2, n * 0.1), `${k}: ${near.get(k)} streamlines within 8 mm, stored ${n}`);
  } finally { device.destroy(); }
}});
