// DIFFUSION SCANS IN A BIDS SESSION -> one Enhanced MR Image object each, the b-value and direction on every frame
// (export-dicom-dwi.ts). A kind of the BIDS import (bids-kinds.ts). TEMPORARILY in core (step 1 of
// Contents/docs/EXTENSIONS.md); step 2 moves it into the diffusion extension, which registers it itself.
import type { BuiltObject } from "albula";
import type { BidsKind, BidsKindContext } from "albula";
import { parseNiftiVolumes } from "albula";
import { fromFsl } from "./dwi.ts";
import { diffusionToEnhancedMR } from "./export-dicom-dwi.ts";

/** Diffusion 1 -> 2, 2026-09-29 (bids.ts keeps the history of the rules): ISOTROPIC for a direction-less b > 0, UUID UIDs. */
export const DWI_RULE = 2;
const DWI_NAME = /_dwi\.nii(\.gz)?$/;
const AGENCIES = ["IEC", "FDA", "MHW"];

export const bidsDiffusion: BidsKind = {
  name: "diffusion",
  folders: ["dwi"],
  async check(ctx: BidsKindContext) {
    // A diffusion scan needs the safety standard, stated (critic 2026-09-28, finding 8).
    const any = (await ctx.files("dwi")).some((e) => DWI_NAME.test(e));
    if (any && !AGENCIES.includes(String(ctx.options.safetyStandardAgency ?? ""))) {
      throw new Error("this session has a diffusion scan: say which safety standard applies to the scanner (IEC, FDA or MHW) and why -- DICOM requires it for a diffusion object, and BIDS does not record it");
    }
  },
  async build(ctx: BidsKindContext): Promise<BuiltObject[]> {
    const out: BuiltObject[] = [], st = ctx.study, agency = ctx.options.safetyStandardAgency as "IEC" | "FDA" | "MHW";
    const dwiFiles = (await ctx.files("dwi")).filter((e) => DWI_NAME.test(e));
    for (const e of dwiFiles) {
      const m = /^(.*_dwi)\.nii(\.gz)?$/.exec(e)!;
      const bval = await ctx.readText(`${ctx.dir}/dwi/${m[1]}.bval`), bvec = await ctx.readText(`${ctx.dir}/dwi/${m[1]}.bvec`);
      if (!bval || !bvec) { ctx.skip(e, "no .bval/.bvec beside it"); continue; }
      ctx.say(`reading ${e}`);
      try {
        const sidecarText = (await ctx.readText(`${ctx.dir}/dwi/${m[1]}.json`)) ?? "";
        const side = sidecarText ? JSON.parse(sidecarText) as Record<string, unknown> : {};
        const kept = Object.fromEntries(Object.entries(side).filter(([k]) => !ctx.sidecarLeftOut.includes(k)));
        const series = fromFsl(await parseNiftiVolumes(await Deno.readFile(`${ctx.dir}/dwi/${e}`), e), bval, bvec, e);
        const acq = /_(?:acq|dir)-([A-Za-z0-9]+)/.exec(e)?.[1];
        const desc = `${String(side.SeriesDescription ?? "DWI")}${acq ? ` (${acq})` : ""}`.slice(0, 64);
        const number = ctx.nextSeriesNumber();
        const exp = await diffusionToEnhancedMR(series, { patientName: st.patientName, patientID: st.patientID, studyInstanceUID: st.studyInstanceUID, frameOfReferenceUID: st.frameOfReferenceUID, studyDescription: st.studyDescription, comments: st.comments, extra: st.extra, studyDate: st.studyDate, studyTime: st.studyTime }, {
          seriesDescription: desc, seriesNumber: number, sourceDescription: sidecarText ? JSON.stringify(kept, null, 2) : undefined,
          safetyStandardAgency: agency,
          uids: { series: await ctx.uid(m[1], `dwi rule ${DWI_RULE}`, "series"), sop: await ctx.uid(m[1], `dwi rule ${DWI_RULE}`, "instance") },
        });
        const shells = [...new Set(series.bValues.map((b) => Math.round(b)))].sort((a, b) => a - b);
        const noDir = series.bValues.filter((b, i) => b > 0 && Math.hypot(...series.gradients[i]) < 0.5).length;
        const [nx, ny] = series.volumes[0].dims;
        out.push({
          role: `dwi${acq ? `-${acq}` : ""}`, description: `diffusion: ${e}, ${series.volumes.length} volumes, b = ${shells.join("/")}`,
          seriesInstanceUID: exp.seriesInstanceUID,
          files: [{
            name: `${exp.sopInstanceUID}.dcm`, bytes: exp.bytes,
            index: { sopInstanceUID: exp.sopInstanceUID, seriesInstanceUID: exp.seriesInstanceUID, studyInstanceUID: st.studyInstanceUID, modality: "MR", seriesNumber: number,
              seriesDescription: desc, frameOfReferenceUID: st.frameOfReferenceUID, displayedSize: `${nx}x${ny}`, numberOfFrames: exp.frames, newStudy: st.newStudy } as BuiltObject["files"][number]["index"],
          }],
          notes: [series.source, `safety standard ${agency}: ${String(ctx.options.safetyReason ?? "(no reason given)")}`,
            ...(noDir ? [`${noDir} volume(s) with b > 0 and no direction: written as ISOTROPIC (trace) images`] : []),
            `sidecar kept in the private block without: ${ctx.sidecarLeftOut.filter((k) => k in side).join(", ") || "(none of the left-out fields present)"}`],
        });
      } catch (err) { ctx.skip(e, (err as Error).message); }
    }
    return out;
  },
};
