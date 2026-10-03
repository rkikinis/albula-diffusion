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
import { correctWithReversed, sortByDistance, stageText, streamlineDistances, tractName, trackUkfSeeds, wholeBrainSeeds, type StageTimes, type TrackTiming } from "./planning.ts";
import type { TractCloudModel } from "./tractcloud/tractcloud.ts";
import { nameTracts, SHORT, type Named } from "./tractcloud/name-tracts.ts";
import { loadRapidParc, type RapidParcModel } from "./rapidparc/rapidparc.ts";
/** RapidParc's standard weights from the file beside this module (case runs are Deno's), read once. */
let rp: RapidParcModel | undefined;
const defaultRapidParc = () => rp ??= loadRapidParc(Deno.readFileSync(new URL("./rapidparc/model/rapidparc.safetensors", import.meta.url)).buffer);
import { addScanNoise, noiseSigma } from "./agreement.ts";
import { estimateResponses, kernelFromResponses } from "./responses.ts";
import { csdVolume } from "./csd-volume.ts";
import { trackPttParallel } from "./ptt.ts";

export const CASE_DEFAULTS = { stopFA: 0.15, maxB: 1500, margins: [0, 5, 8] } as const;

export interface CaseTract { tract: string; abbr: string; category: string; total: number; closest: number; within0: number; within5: number; within8: number }
export interface CaseResult {
  id: string; corrected: string; streamlines: number; short: number; other: number; named: number; tumorVoxels: number;
  method: "ukf" | "ptt";
  seconds: { read_correct_fit: number; csd: number; track: number; name: number; total: number };
  /** Step by step (planning.ts StageTimes, milliseconds), and the same in words. */
  stages: StageTimes;
  stagesText: string;
  /** With opts.keep: the streamlines and their names, for agreement measures (agreement.ts). */
  kept?: { sl: Float32Array[]; named: Named };
  /** With opts.noiseSeed: the noise level that was added (σ, the scan's own, measured from its background). */
  noiseSigma?: number;
  /** Every named tract with at least one streamline within the largest margin, with its counts. */
  tracts: CaseTract[];
}

/** Run case `id` of the BIDS dataset at `ds` (ds001226's layout: ses-preop, acq-AP / acq-PA, derivatives/tumor_masks). */
export async function runCase(ds: string, id: string, device: GPUDevice, model: TractCloudModel, method: "ukf" | "ptt" = "ukf", opts: { pttWorkerUrl?: URL; csdWorkerUrl?: URL; /** The card tracker's options (checking variants, e.g. onePass). */ ukf?: Record<string, unknown>; /** The naming draw's seed (label noise floor). */ nameSeed?: number; /** Who names: RapidParc (the default since 2026-10-03; its weights) or TractCloud ("tractcloud"). */ labeler?: RapidParcModel | "tractcloud"; /** Add the scan's own noise again, seeded (scan noise floor). */ noiseSeed?: number; /** Return the streamlines and names. */ keep?: boolean; /** How the brain mask is made (tensor.ts). */ maskMethod?: "head" | "median-otsu"; /** Whole-brain starting points (planning.ts WHOLE_BRAIN_SEEDS). */ seeds?: number; /** Their draw's seed (seed noise floor). */ seedDraw?: number; /** false: seeds anywhere in the head mask, as before 2026-10-02. */ seedMask?: boolean } = {}): Promise<CaseResult> {
  const t0 = performance.now(), p = `${ds}/sub-${id}/ses-preop`;
  const rd = (f: string) => Deno.readFileSync(`${p}/${f}`), tx = (f: string) => Deno.readTextFileSync(`${p}/${f}`);
  const stages: StageTimes = {};
  const dwi = fromFsl(await parseNiftiVolumes(rd(`dwi/sub-${id}_ses-preop_acq-AP_dwi.nii.gz`)), tx(`dwi/sub-${id}_ses-preop_acq-AP_dwi.bval`), tx(`dwi/sub-${id}_ses-preop_acq-AP_dwi.bvec`));
  const rev = fromFsl(await parseNiftiVolumes(rd(`dwi/sub-${id}_ses-preop_acq-PA_dwi.nii.gz`)), tx(`dwi/sub-${id}_ses-preop_acq-PA_dwi.bval`), tx(`dwi/sub-${id}_ses-preop_acq-PA_dwi.bvec`));
  stages.read = performance.now() - t0;
  // THE PARTNER ON ITS OWN GRID: 11 of the 29 have a PA slab placed 2.3 mm and 0.8° away; until 2026-10-02 it was paired
  // voxel by voxel here (Mike Halle's pipeline aligns it by the scanner's coordinates; so does this now).
  const corrected = await correctWithReversed(dwi, rev.volumes.filter((_, i) => rev.bValues[i] < 50).map((v) => v.data), "PA", undefined,
    { partnerGrid: { dims: rev.volumes[0].dims, ijkToRAS: rev.volumes[0].ijkToRAS }, times: stages });
  let added: number | undefined;
  if (opts.noiseSeed !== undefined) {
    // THE SCAN'S OWN NOISE AGAIN (agreement.ts): measured from the background outside the brain, added to every volume.
    const m = fitTensors(dwi, { maxB: CASE_DEFAULTS.maxB, maskMethod: opts.maskMethod, seedMask: false }).mask;
    added = noiseSigma(dwi, m);
    const noisy = addScanNoise(dwi, added, opts.noiseSeed);
    dwi.volumes = noisy.volumes;
  }
  const tFit = performance.now();
  const fit = fitTensors(dwi, { maxB: CASE_DEFAULTS.maxB, maskMethod: opts.maskMethod, seedMask: opts.seedMask });
  const t1 = performance.now();
  stages.fit = t1 - tFit;
  const detail: TrackTiming = { prepare: 0, gpu: 0, assemble: 0, between: 0 };
  let sl: Float32Array[], csdSeconds = 0;
  if (method === "ptt") {
    const k = kernelFromResponses(estimateResponses(dwi));
    const fod = await csdVolume(dwi, fit.mask, k, opts.csdWorkerUrl ? { workerUrl: opts.csdWorkerUrl } : {});
    csdSeconds = fod.seconds;
    sl = await trackPttParallel(fod, wholeBrainSeeds(fit, opts.seeds, opts.seedDraw), opts.pttWorkerUrl ? { workerUrl: opts.pttWorkerUrl } : {});
  } else {
    const tData = performance.now(), data = prepareUkfData(dwi, fit.mask), ts = performance.now(), seeds = wholeBrainSeeds(fit, opts.seeds, opts.seedDraw);
    stages.seeds = performance.now() - ts;
    sl = await trackUkfSeeds(device, data, seeds, CASE_DEFAULTS.stopFA, undefined, detail, opts.ukf ?? {});
    stages.trackDetail = { ...detail, data: ts - tData };
  }
  const t2 = performance.now();
  stages.track = t2 - t1 - csdSeconds * 1000 - (stages.seeds ?? 0);
  const rapidParc = opts.labeler === "tractcloud" ? undefined : opts.labeler ?? defaultRapidParc();
  const named = await nameTracts(device, model, sl, { ...(opts.nameSeed !== undefined ? { seed: opts.nameSeed } : {}), ...(rapidParc ? { rapidParc } : {}) });
  stages.name = named.seconds * 1000;
  const tDist = performance.now();
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
  stages.distances = t3 - tDist;
  stages.total = t3 - t0;
  return { id, corrected, stages, stagesText: stageText(stages), ...(opts.keep ? { kept: { sl, named } } : {}), ...(added !== undefined ? { noiseSigma: +added.toFixed(2) } : {}), streamlines: sl.length, short, other, named: sl.length - short - other, tumorVoxels, method,
    seconds: { read_correct_fit: +((t1 - t0) / 1000).toFixed(1), csd: +csdSeconds.toFixed(1), track: +((t2 - t1 - csdSeconds * 1000) / 1000).toFixed(1), name: +named.seconds.toFixed(1), total: +((t3 - t0) / 1000).toFixed(1) }, tracts };
}
