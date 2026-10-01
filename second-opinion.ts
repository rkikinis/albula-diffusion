// THE SECOND OPINION on a diffusion scan's b-values and directions: dcm2niix (vendor/dcm2niix, WebAssembly, in its own
// worker) reads the same DICOM files Albula's reader read, and the two are compared volume by volume. Ron, 2026-09-30
// ("2 yes", Lauren O'Donnell's suggestion): dcm2niix is actively maintained and learns new scanner tags first, so a new
// scanner generation shows up as a DISAGREEMENT the day it appears, not as a silent wrong direction. Albula's reader
// stays the reader (it keeps the DICOM record); this only checks it.
//
// dcm2niix's output (NIfTI + .bval/.bvec, FSL's convention) is read by Albula's own FSL reader (logic/diffusion/dwi.ts
// fromFsl, itself checked against dcm2niix and nibabel), so both sides are in one convention (DWI_CONVENTION 1: unit
// gradients in patient RAS). Directions are compared without their sign (an axis has none).
import { workerUrl } from "albula";
import { fromFsl, isotropicVolumes, type DiffusionSeries } from "./dwi.ts";
import { parseNiftiVolumes } from "albula";

export const DCM2NIIX_VERSION = "v1.0.20260724 (@niivue/dcm2niix 1.3.20260724)";

export interface SecondOpinion {
  /** Agreement within the tolerances: b within 1 s/mm² (or 0.5%), directions within 0.5°. */
  agree: boolean;
  /** In words, for the panel. */
  said: string;
  volumes: number;
  worstDeg: number;
  worstB: number;
  ms: number;
}

type Converted = { name: string; arrayBuffer(): Promise<ArrayBuffer> };
interface Dcm2niixClass { new(): { init(): Promise<void>; input(files: unknown): { run(): Promise<Converted[]> } } }
let lib: Promise<Dcm2niixClass> | undefined;
async function dcm2niix(): Promise<Dcm2niixClass> {
  // The package's own loader, served beside the app; it starts its worker beside itself.
  lib ??= import(workerUrl("./vendor/dcm2niix/index.js").href).then((m) => m.Dcm2niix as Dcm2niixClass);
  return await lib;
}

/** Run dcm2niix on these DICOM files and compare its diffusion reading with `ours`. */
export async function secondOpinion(files: ArrayBuffer[], ours: DiffusionSeries): Promise<SecondOpinion> {
  const t0 = performance.now();
  const Cls = await dcm2niix();
  const d = new Cls();
  await d.init();
  const list = files.map((b, i) => { const f = new File([b], `f${i}.dcm`); (f as unknown as { _webkitRelativePath: string })._webkitRelativePath = `in/f${i}.dcm`; return f; });
  const out = await d.input(list).run();
  const pick = (ext: string) => out.filter((f) => f.name.endsWith(ext));
  const niis = pick(".nii"), bvals = pick(".bval"), bvecs = pick(".bvec");
  if (!niis.length || !bvals.length || !bvecs.length) throw new Error(`dcm2niix found no diffusion series in these files (it wrote ${out.map((f) => f.name.split(".").pop()).join(", ") || "nothing"})`);
  // The series of the same size, when dcm2niix split the files into several.
  const series = await Promise.all(niis.map(async (n) => {
    const stem = n.name.replace(/\.nii$/, "");
    const bv = bvals.find((f) => f.name === `${stem}.bval`), bc = bvecs.find((f) => f.name === `${stem}.bvec`);
    if (!bv || !bc) return undefined;
    return fromFsl(await parseNiftiVolumes(new Uint8Array(await n.arrayBuffer())), new TextDecoder().decode(await bv.arrayBuffer()), new TextDecoder().decode(await bc.arrayBuffer()));
  }));
  const ref = series.filter((s): s is DiffusionSeries => !!s).sort((a, b) => Math.abs(a.bValues.length - ours.bValues.length) - Math.abs(b.bValues.length - ours.bValues.length))[0];
  if (!ref) throw new Error("dcm2niix wrote no .bval/.bvec beside its image");
  // THE TRACE IMAGE: dcm2niix sets the scanner's trace volume aside (it writes it as _ADC); Albula keeps it.
  let idx = ours.bValues.map((_, i) => i);
  if (ref.bValues.length !== idx.length) {
    const trace = new Set(isotropicVolumes(ours));
    const kept = idx.filter((i) => !trace.has(i));
    if (kept.length === ref.bValues.length) idx = kept;
    else return { agree: false, said: `dcm2niix finds ${ref.bValues.length} volumes where Albula finds ${ours.bValues.length}: check this scan before making tracts.`, volumes: ours.bValues.length, worstDeg: NaN, worstB: NaN, ms: performance.now() - t0 };
  }
  let worstDeg = 0, worstB = 0, missing = 0;
  idx.forEach((i, k) => {
    const b = ours.bValues[i], rb = ref.bValues[k];
    const diff = Math.abs(b - rb);
    if (diff > Math.max(1, 0.005 * Math.abs(rb))) worstB = Math.max(worstB, diff);   // 1 s/mm² or 0.5% (GE rounds shells to 5)
    if (rb < 50) return;
    const g = ours.gradients[i], r = ref.gradients[k], lg = Math.hypot(...g), lr = Math.hypot(...r);
    if (lr < 0.5) return;                                   // dcm2niix gives no direction: nothing to compare
    if (lg < 0.5) { missing++; return; }
    const c = Math.min(1, Math.abs(g[0] * r[0] + g[1] * r[1] + g[2] * r[2]) / (lg * lr));
    worstDeg = Math.max(worstDeg, Math.acos(c) * 180 / Math.PI);
  });
  const agree = worstB === 0 && worstDeg <= 0.5 && missing === 0;
  const said = agree
    ? `dcm2niix agrees: ${idx.length} volumes, b-values equal, directions within ${worstDeg.toFixed(2)}°.`
    : `dcm2niix reads this scan differently${worstB ? `: b-values differ by up to ${worstB.toFixed(0)} s/mm²` : ""}${worstDeg > 0.5 ? `${worstB ? ";" : ":"} directions differ by up to ${worstDeg.toFixed(1)}°` : ""}${missing ? `; ${missing} volumes have no direction here` : ""}. Check before making tracts.`;
  return { agree, said, volumes: idx.length, worstDeg, worstB, ms: performance.now() - t0 };
}
