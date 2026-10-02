// ONE CASE OF THE CASE LIBRARY, from its BIDS files to the named tracts near its tumor -- the pipeline the batch tool
// (Contents/tools/dmri-cases.ts) runs over the library and the regression test (cases.regression.test.ts) runs on one
// case, so the test checks exactly what produced the stored results. Moved here from the tool on 2026-10-01 (Ron: "Are
// there tests that you can add/improve now?").
//
// The steps, with the app's defaults: read the diffusion scan and its reversed-phase pair (NIfTI + FSL), correct the
// distortion, fit the tensor (b up to 1500), seed the whole brain, follow tracts (UKF on the graphics card, or CSD + PTT
// on the processor's workers), name them (TractCloud), and measure each tract's streamlines against the dataset's tumor
// mask (inside at 0.5 and above).
import { parseNiftiVolumes } from "albula";
import { fromFsl } from "./dwi.ts";
import { fitTensors } from "./tensor.ts";
import { prepareUkfData } from "./ukf.ts";
import { correctWithReversed, sortByDistance, streamlineDistances, tractName, trackUkfSeeds, wholeBrainSeeds } from "./planning.ts";
import type { TractCloudModel } from "./tractcloud/tractcloud.ts";
import { nameTracts, SHORT } from "./tractcloud/name-tracts.ts";
import { estimateResponses, kernelFromResponses } from "./responses.ts";
import { csdVolume } from "./csd-volume.ts";
import { trackPttParallel } from "./ptt.ts";

export const CASE_DEFAULTS = { stopFA: 0.15, maxB: 1500, margins: [0, 5, 8] } as const;

export interface CaseTract { tract: string; abbr: string; category: string; total: number; closest: number; within0: number; within5: number; within8: number }
export interface CaseResult {
  id: string; corrected: string; streamlines: number; short: number; other: number; named: number; tumorVoxels: number;
  method: "ukf" | "ptt";
  seconds: { read_correct_fit: number; csd: number; track: number; name: number; total: number };
  /** Every named tract with at least one streamline within the largest margin, with its counts. */
  tracts: CaseTract[];
}

/** Run case `id` of the BIDS dataset at `ds` (ds001226's layout: ses-preop, acq-AP / acq-PA, derivatives/tumor_masks). */
export async function runCase(ds: string, id: string, device: GPUDevice, model: TractCloudModel, method: "ukf" | "ptt" = "ukf", opts: { pttWorkerUrl?: URL; csdWorkerUrl?: URL; /** The card tracker's options (checking variants, e.g. onePass). */ ukf?: Record<string, unknown> } = {}): Promise<CaseResult> {
  const t0 = performance.now(), p = `${ds}/sub-${id}/ses-preop`;
  const rd = (f: string) => Deno.readFileSync(`${p}/${f}`), tx = (f: string) => Deno.readTextFileSync(`${p}/${f}`);
  const dwi = fromFsl(await parseNiftiVolumes(rd(`dwi/sub-${id}_ses-preop_acq-AP_dwi.nii.gz`)), tx(`dwi/sub-${id}_ses-preop_acq-AP_dwi.bval`), tx(`dwi/sub-${id}_ses-preop_acq-AP_dwi.bvec`));
  const rev = fromFsl(await parseNiftiVolumes(rd(`dwi/sub-${id}_ses-preop_acq-PA_dwi.nii.gz`)), tx(`dwi/sub-${id}_ses-preop_acq-PA_dwi.bval`), tx(`dwi/sub-${id}_ses-preop_acq-PA_dwi.bvec`));
  const corrected = await correctWithReversed(dwi, rev.volumes.filter((_, i) => rev.bValues[i] < 50).map((v) => v.data), "PA");
  const fit = fitTensors(dwi, { maxB: CASE_DEFAULTS.maxB });
  const t1 = performance.now();
  let sl: Float32Array[], csdSeconds = 0;
  if (method === "ptt") {
    const k = kernelFromResponses(estimateResponses(dwi));
    const fod = await csdVolume(dwi, fit.mask, k, opts.csdWorkerUrl ? { workerUrl: opts.csdWorkerUrl } : {});
    csdSeconds = fod.seconds;
    sl = await trackPttParallel(fod, wholeBrainSeeds(fit), opts.pttWorkerUrl ? { workerUrl: opts.pttWorkerUrl } : {});
  } else sl = await trackUkfSeeds(device, prepareUkfData(dwi, fit.mask), wholeBrainSeeds(fit), CASE_DEFAULTS.stopFA, undefined, undefined, opts.ukf ?? {});
  const t2 = performance.now();
  const named = await nameTracts(device, model, sl);
  const OTHER = model.json.tracts.length - 1;
  let short = 0, other = 0;
  for (const t of named.tract) { if (t === SHORT) short++; else if (t === OTHER) other++; }
  let tracts: CaseTract[] = [], tumorVoxels = 0;
  const maskFile = `${ds}/derivatives/tumor_masks/sub-${id}/anat/sub-${id}_space_T1_label-tumor.nii`;
  try {
    const m = (await parseNiftiVolumes(Deno.readFileSync(maskFile)))[0];
    const data = m.data as ArrayLike<number>;
    for (let v = 0; v < data.length; v++) if (data[v] >= 0.5) tumorVoxels++;
    const margins = CASE_DEFAULTS.margins, widest = Math.max(...margins);
    const dist = await streamlineDistances({ dims: m.dims, ijkToRAS: m.ijkToRAS, inside: (v) => data[v] >= 0.5 }, sl, widest + 2);
    const sorted = sortByDistance(model, named, dist, widest, 1);   // every tract, with its counts
    tracts = sorted.near.map((e) => ({ tract: tractName(model, e.tract, e.side), abbr: model.json.tracts[e.tract].abbr, category: model.json.tracts[e.tract].category,
      total: e.idx.length, closest: +e.d.toFixed(2), within0: e.idx.filter((i) => dist[i] <= 0).length, within5: e.idx.filter((i) => dist[i] <= 5).length, within8: e.idx.filter((i) => dist[i] <= 8).length }));
  } catch (e) { if (!id.startsWith("CON")) throw e; }   // healthy volunteers have no tumor mask
  const t3 = performance.now();
  return { id, corrected, streamlines: sl.length, short, other, named: sl.length - short - other, tumorVoxels, method,
    seconds: { read_correct_fit: +((t1 - t0) / 1000).toFixed(1), csd: +csdSeconds.toFixed(1), track: +((t2 - t1 - csdSeconds * 1000) / 1000).toFixed(1), name: +named.seconds.toFixed(1), total: +((t3 - t0) / 1000).toFixed(1) }, tracts };
}
