// DIFFUSION SCANS IN A BIDS SESSION -> one Enhanced MR Image object each, the b-value and direction on every frame
// (export-dicom-dwi.ts). A kind of the BIDS import, registered by hooks.ts through the SDK (registerBidsKind): core
// builds the anatomy and the masks of a session, this extension its dwi/ folder -- and, since 2026-10-22, the b = 0 field
// maps meant for a diffusion scan in fmap/ (`…_epi` with IntendedFor naming a dwi/ file: the reversed phase-encoding pair
// OpenNeuro ds005123 stores there; ds001226 stored its reversed scan in dwi/). Each is written as a diffusion object whose
// every frame is b = 0, with its phase-encoding direction from its sidecar, so the import job's choice of the reversed
// scan (import-job.ts partnerOf) takes it as it takes one stored in dwi/. Other field maps (for functional MRI: magnitude,
// phase difference) are skipped and said.
import type { BuiltObject } from "albula";
import type { BidsKind, BidsKindContext } from "albula";
import { parseNiftiVolumes } from "albula";
import { fromFsl, type DiffusionSeries } from "./dwi.ts";
import { diffusionToEnhancedMR } from "./export-dicom-dwi.ts";

/** Diffusion 1 -> 2, 2026-09-29 (bids.ts keeps the history of the rules): ISOTROPIC for a direction-less b > 0, UUID UIDs. */
export const DWI_RULE = 2;
/** The b = 0 field maps for diffusion (fmap/…_epi), rule 1, 2026-10-22. Its own rule: the diffusion scans' UIDs are unchanged. */
export const FMAP_RULE = 1;
const DWI_NAME = /_dwi\.nii(\.gz)?$/;
const EPI_NAME = /_epi\.nii(\.gz)?$/;
const AGENCIES = ["IEC", "FDA", "MHW"];

const listOf = (v: unknown): string[] => (Array.isArray(v) ? v : v === undefined || v === null ? [] : [v]).map(String);
/** Which of THIS session's diffusion files a field map is meant for: named in its IntendedFor (BIDS: a path relative to
 *  the subject, or a "bids::" URI; one or a list), or -- BIDS 1.7 and later -- its B0FieldIdentifier named in that scan's
 *  B0FieldSource. A path to another session's scan, or to none, is not this session's (critic 2026-10-22, finding 8). */
function meantFor(side: Record<string, unknown>, dwis: { file: string; side: Record<string, unknown> }[]): string | undefined {
  const named = listOf(side.IntendedFor).map((p) => p.replace(/^bids::/, ""));
  for (const d of dwis) if (named.some((p) => /(^|\/)dwi\//.test(p) && p.split("/").pop() === d.file)) return d.file;
  const ids = listOf(side.B0FieldIdentifier);
  for (const d of dwis) if (ids.length && listOf(d.side.B0FieldSource).some((s) => ids.includes(s))) return d.file;
  return undefined;
}

/** One object: the series as an Enhanced MR diffusion object, with its sidecar and its record. */
async function writeObject(ctx: BidsKindContext, e: string, series: DiffusionSeries, sidecarText: string, role: string, uidRule: string, kind: string): Promise<BuiltObject> {
  const st = ctx.study, agency = ctx.options.safetyStandardAgency as "IEC" | "FDA" | "MHW";
  const side = sidecarText ? JSON.parse(sidecarText) as Record<string, unknown> : {};
  const kept = Object.fromEntries(Object.entries(side).filter(([k]) => !ctx.sidecarLeftOut.includes(k)));
  const base = e.replace(/\.nii(\.gz)?$/, "");
  // A field map is named by its direction (dir-PA), a diffusion scan as before (its first acq- or dir- entity).
  const acq = role === "dwi" ? /_(?:acq|dir)-([A-Za-z0-9]+)/.exec(e)?.[1] : /_dir-([A-Za-z0-9]+)/.exec(e)?.[1] ?? /_acq-([A-Za-z0-9]+)/.exec(e.replace(/_acq-dwi/, ""))?.[1];
  const desc = `${String(side.SeriesDescription ?? (role === "dwi" ? "DWI" : "DWI b0 field map"))}${acq ? ` (${acq})` : ""}`.slice(0, 64);
  const number = ctx.nextSeriesNumber();
  const exp = await diffusionToEnhancedMR(series, { patientName: st.patientName, patientID: st.patientID, studyInstanceUID: st.studyInstanceUID, frameOfReferenceUID: st.frameOfReferenceUID, studyDescription: st.studyDescription, comments: st.comments, extra: st.extra, studyDate: st.studyDate, studyTime: st.studyTime }, {
    seriesDescription: desc, seriesNumber: number, sourceDescription: sidecarText ? JSON.stringify(kept, null, 2) : undefined,
    // The sidecar's phase-encoding axis as the standard attribute: i along the rows (ROW), j along the columns (COL);
    // the writer keeps the voxel axes as the image's (export-dicom-dwi.ts). Its sign stays in the private block.
    ...(typeof side.PhaseEncodingDirection === "string" && /^[ij]-?$/.test(side.PhaseEncodingDirection) ? { phaseEncoding: (side.PhaseEncodingDirection[0] === "i" ? "ROW" : "COLUMN") as "ROW" | "COLUMN" } : {}),
    safetyStandardAgency: agency,
    uids: { series: await ctx.uid(base, uidRule, "series"), sop: await ctx.uid(base, uidRule, "instance") },
  });
  const shells = [...new Set(series.bValues.map((b) => Math.round(b)))].sort((a, b) => a - b);
  const noDir = series.bValues.filter((b, i) => b > 0 && Math.hypot(...series.gradients[i]) < 0.5).length;
  const [nx, ny] = series.volumes[0].dims;
  return {
    role: `${role}${acq ? `-${acq}` : ""}`, description: `${kind}: ${e}, ${series.volumes.length} volumes, b = ${shells.join("/")}`,
    seriesInstanceUID: exp.seriesInstanceUID,
    files: [{
      name: `${exp.sopInstanceUID}.dcm`, bytes: exp.bytes,
      index: { sopInstanceUID: exp.sopInstanceUID, seriesInstanceUID: exp.seriesInstanceUID, studyInstanceUID: st.studyInstanceUID, modality: "MR", seriesNumber: number,
        seriesDescription: desc, frameOfReferenceUID: st.frameOfReferenceUID, displayedSize: `${nx}x${ny}`, numberOfFrames: exp.frames, newStudy: st.newStudy } as BuiltObject["files"][number]["index"],
    }],
    notes: [series.source, `safety standard ${agency}: ${String(ctx.options.safetyReason ?? "(no reason given)")}`,
      ...(noDir ? [`${noDir} volume(s) with b > 0 and no direction: written as ISOTROPIC (trace) images`] : []),
      `sidecar kept in the private block without: ${ctx.sidecarLeftOut.filter((k) => k in side).join(", ") || "(none of the left-out fields present)"}`],
  };
}

export const bidsDiffusion: BidsKind = {
  name: "diffusion",
  folders: ["dwi", "fmap"],
  async check(ctx: BidsKindContext) {
    // A diffusion scan needs the safety standard, stated (critic 2026-09-28, finding 8).
    const any = (await ctx.files("dwi")).some((e) => DWI_NAME.test(e));
    if (any && !AGENCIES.includes(String(ctx.options.safetyStandardAgency ?? ""))) {
      throw new Error("this session has a diffusion scan: say which safety standard applies to the scanner (IEC, FDA or MHW) and why -- DICOM requires it for a diffusion object, and BIDS does not record it");
    }
  },
  async build(ctx: BidsKindContext): Promise<BuiltObject[]> {
    const out: BuiltObject[] = [];
    const dwiFiles = (await ctx.files("dwi")).filter((e) => DWI_NAME.test(e));
    for (const e of dwiFiles) {
      const m = /^(.*_dwi)\.nii(\.gz)?$/.exec(e)!;
      const bval = await ctx.readText(`${ctx.dir}/dwi/${m[1]}.bval`), bvec = await ctx.readText(`${ctx.dir}/dwi/${m[1]}.bvec`);
      if (!bval || !bvec) { ctx.skip(e, "no .bval/.bvec beside it"); continue; }
      ctx.say(`reading ${e}`);
      try {
        const sidecarText = (await ctx.readText(`${ctx.dir}/dwi/${m[1]}.json`)) ?? "";
        const series = fromFsl(await parseNiftiVolumes(await Deno.readFile(`${ctx.dir}/dwi/${e}`), e), bval, bvec, e);
        out.push(await writeObject(ctx, e, series, sidecarText, "dwi", `dwi rule ${DWI_RULE}`, "diffusion"));
      } catch (err) { ctx.skip(e, (err as Error).message); }
    }
    // THE b = 0 FIELD MAPS FOR DIFFUSION (fmap/…_epi, meant for one of this session's dwi/ scans): every volume b = 0.
    const dwis: { file: string; side: Record<string, unknown> }[] = [];
    for (const e of dwiFiles) { const t = await ctx.readText(`${ctx.dir}/dwi/${e.replace(/\.nii(\.gz)?$/, "")}.json`); try { dwis.push({ file: e, side: t ? JSON.parse(t) : {} }); } catch { dwis.push({ file: e, side: {} }); } }
    for (const e of (await ctx.files("fmap")).filter((f) => /\.nii(\.gz)?$/.test(f))) {
      if (!EPI_NAME.test(e)) { ctx.skip(`fmap/${e}`, "a field map for another kind of scan (only the b = 0 pairs, …_epi, meant for a diffusion scan are read)"); continue; }
      const m = /^(.*_epi)\.nii(\.gz)?$/.exec(e)!;
      const sidecarText = (await ctx.readText(`${ctx.dir}/fmap/${m[1]}.json`)) ?? "";
      let side: Record<string, unknown> = {};
      try { side = sidecarText ? JSON.parse(sidecarText) : {}; } catch { ctx.skip(`fmap/${e}`, "its sidecar is not valid JSON"); continue; }
      const target = meantFor(side, dwis);
      if (!target) { ctx.skip(`fmap/${e}`, "meant for another scan (neither its IntendedFor nor its B0FieldIdentifier names a diffusion scan of this session)"); continue; }
      ctx.say(`reading ${e}`);
      try {
        const vols = await parseNiftiVolumes(await Deno.readFile(`${ctx.dir}/fmap/${e}`), e), n = vols.length, zeros = new Array(n).fill(0).join(" ");
        const series = fromFsl(vols, `${zeros}\n`, `${zeros}\n${zeros}\n${zeros}\n`, e);
        series.source = `${series.source}; a b = 0 field map (fmap/${e}) for dwi/${target}: every volume b = 0`;
        out.push(await writeObject(ctx, e, series, sidecarText, "dwi-b0-fieldmap", `fmap epi rule ${FMAP_RULE}`, "diffusion field map"));
      } catch (err) { ctx.skip(`fmap/${e}`, (err as Error).message); }
    }
    return out;
  },
};
