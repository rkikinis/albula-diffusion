// DIFFUSION B-VALUES AND DIRECTIONS FROM SINGLE-FRAME DICOM FILES, vendor by vendor. Ron, 2026-09-29: "Siemens and GE
// are the dominant players in diffusion. It's worth the effort to figure them out." Plan and test data:
// Contents/docs/dmri-review-2026-09-29-vendor-note.md (workspace), Contents/tools/fetch-dwi-vendors.sh.
//
// Multi-frame (enhanced) files carry the standard MR Diffusion macro and are read in dicom-series.ts. A single-frame
// file may carry, in this order of trust:
//   1. the standard attributes at the top level: DiffusionBValue (0018,9087) and DiffusionGradientOrientation
//      (0018,9089), the direction in patient coordinates (LPS);
//   2. Siemens: the CSA image header (0029,1010) -- "B_value" and "DiffusionGradientDirection" (LPS) -- and the older
//      private elements (0019,100C) b-value and (0019,100E) direction;
//   3. GE: (0043,1039) whose first value is the b-value (sometimes with 1e9 added as a flag), and (0019,10BB/BC/BD)
//      the direction's three components;
//   4. Philips (older, single-frame): (2001,1003) b-value, (2005,10B0/B1/B2) the direction's components.
// The coordinate frame of each vendor's private direction is not documented by the vendors; it is SET BY MEASUREMENT
// against dcm2niix on the public example sets (see VENDOR_RULE and the test), not assumed.
//
// Each rule is versioned (VENDOR_RULE); the reader names which rule gave a series its directions.

// Version 2 (2026-09-29, after the critic): private values read by their dictionary type (implicit VR); a GE/Philips
// b = 0 is weak evidence; the Siemens syngo 2004 b = 0 marker; short Siemens directions kept when the header says
// DIRECTIONAL; a mosaic's empty SliceNormalVector falls back to the image's own normal.
export const VENDOR_RULE = 2;

import { privateAt as at, bytesOf, dicomNumber as num, parseCsa, privateNumbers, type DicomJsonRaw as Raw } from "albula";
import type { VolumeInterpreter, VolumeKey } from "albula";

export interface DiffusionInfo {
  bValue: number;
  /** Unit direction in patient LPS, absent for b = 0 or a trace image. */
  direction?: [number, number, number];
  /** Which rule produced it: "standard", "siemens-csa", "siemens-private", "ge", "philips". */
  /**
   * WEAK: a b-value of 0 read from a vendor field that sits on EVERY MR image, diffusion or not (GE (0043,1039), Philips
   * (2001,1003)). On its own it says nothing; it counts only when another image of the same series has b > 0
   * (weakDiffusionDropped in dicom-series.ts). Critic, 2026-09-29, finding 3: every GE T1, fMRI and field map had become
   * "diffusion b = 0".
   */
  weak?: boolean;
  source: string;
}

const unit = (g: number[] | undefined): [number, number, number] | undefined => {
  if (!g || g.length < 3 || g.some((x) => !Number.isFinite(x))) return undefined;
  const l = Math.hypot(g[0], g[1], g[2]);
  return l > 0.5 ? [g[0] / l, g[1] / l, g[2] / l] : undefined;
};

/**
 * The diffusion information of one single-frame image, or undefined when it has none. `ds` is the naturalized dataset,
 * `raw` the DICOM JSON form (private elements are read from it by tag). Directions are in patient LPS for every rule
 * whose frame is established; see `vendorFrames` for the vendors' private frames.
 */
export function diffusionOf(ds: Record<string, unknown>, raw: Raw): DiffusionInfo | undefined {
  const maker = String(ds.Manufacturer ?? "").toUpperCase();
  // 1. The standard attributes -- complete when the b-value is 0 or the direction is there too. A b-value WITHOUT its
  // direction (GE MR29 writes (0018,9087) and not (0018,9089)) falls through to the vendor's private direction.
  const bStd = num(ds.DiffusionBValue);
  const gStd = unit((ds.DiffusionGradientOrientation as number[] | undefined)?.map(Number));
  if (bStd !== undefined && (bStd === 0 || gStd)) return { bValue: bStd, ...(bStd > 0 && gStd ? { direction: gStd } : {}), source: "standard" };
  // 2. Siemens.
  if (maker.includes("SIEMENS")) {
    const csa = bytesOf(raw, at(raw, "0029", "SIEMENS CSA HEADER", "10"));
    if (csa) {
      const t = parseCsa(csa);
      const b = num(t.get("B_value")?.[0]);
      if (b !== undefined) {
        const g = t.get("DiffusionGradientDirection")?.map(Number);
        // THE syngo 2004 b = 0 MARKER: every image says B_value 1000, and the b = 0 images carry the direction
        // (-1.0001, -1.0001, -1.0001). Measured on both SiemensTrio-Syngo2004A sets against dcm2niix (b = 0 there).
        if (g && g.length >= 3 && g.slice(0, 3).every((x) => Math.abs(x + 1.0001) < 1e-4)) return { bValue: 0, source: "siemens-csa (syngo b = 0 marker)" };
        // A SHORT direction under DIRECTIONAL is still a direction (SiemensTrioTim1 stores two at length 0.045-0.07,
        // which dcm2niix normalizes); anywhere else a vector under 0.5 is not trusted (unit()).
        const directional = String(t.get("DiffusionDirectionality")?.[0] ?? "").toUpperCase() === "DIRECTIONAL";
        const len = g && g.length >= 3 ? Math.hypot(g[0], g[1], g[2]) : 0;
        const d = directional && len > 1e-3 ? [g![0] / len, g![1] / len, g![2] / len] as [number, number, number] : unit(g);
        return { bValue: b, ...(b > 0 && d ? { direction: d } : {}), source: "siemens-csa" };
      }
    }
    const b = privateNumbers(raw, at(raw, "0019", "SIEMENS MR HEADER", "0C"), "IS")[0];
    if (b !== undefined) {
      const g = privateNumbers(raw, at(raw, "0019", "SIEMENS MR HEADER", "0E"), "FD");
      return { bValue: b, ...(b > 0 && unit(g) ? { direction: unit(g) } : {}), source: "siemens-private" };
    }
  }
  // 3. GE: the direction is in the IMAGE's frame (vendorFrames.ge).
  if (maker.includes("GE")) {
    let b = bStd ?? privateNumbers(raw, at(raw, "0043", "GEMS_PARM_01", "39"), "IS")[0];
    if (b !== undefined) {
      if (b >= 1e9) b = b % 1e9;                       // the flag GE adds on some software
      const g = ["BB", "BC", "BD"].map((e) => privateNumbers(raw, at(raw, "0019", "GEMS_ACQU_01", e), "DS")[0] ?? NaN);
      // A LOWER SHELL is written as a shortened vector under the series' one maximum b: b scales by the length squared,
      // rounded to 5 and at least 5 (dcm2niix's geCorrectBvecs, BSD-2, issues 163 and 245). Critic 2026-09-29, finding 6.
      const len = g.every(Number.isFinite) ? Math.hypot(g[0], g[1], g[2]) : NaN;
      if (b > 0 && len > 0.03 && len < 0.97) { const s = b * len * len; b = s > 0 && s < 5 ? 5 : Math.floor((s + 2.5) / 5) * 5; }
      const pe = String(ds.InPlanePhaseEncodingDirection ?? "").toUpperCase();
      const d = b > 0 && len > 0.03 ? vendorFrames.ge(g.map((x) => x / len), ds) : undefined;
      const why = !(len > 0.03) ? "no gradient vector" : pe !== "COL" ? `phase encoding "${pe || "not stated"}", not validated` : "no image orientation";
      return { bValue: b, ...(d ? { direction: d } : {}), source: b > 0 && !d ? `ge (direction not determined: ${why})` : "ge", ...(b === 0 && bStd === undefined ? { weak: true } : {}) };
    }
  }
  // 4. Philips (older single-frame).
  if (maker.includes("PHILIPS")) {
    const b = privateNumbers(raw, at(raw, "2001", "Philips Imaging DD 001", "03"), "FL")[0];
    if (b !== undefined) {
      const g = ["B0", "B1", "B2"].map((e) => privateNumbers(raw, at(raw, "2005", "Philips MR Imaging DD 001", e), "FL")[0] ?? NaN);
      const d = vendorFrames.philips(g, ds);
      return { bValue: b, ...(b > 0 && d ? { direction: d } : {}), source: "philips", ...(b === 0 ? { weak: true } : {}) };
    }
  }
  // 5. Canon (formerly Toshiba): the direction only as text in ImageComments (0020,4000), "b=1500(-0.445,-0.895,0.000)".
  if (maker.includes("CANON") || maker.includes("TOSHIBA")) {
    const m = /b=\s*([\d.]+)\s*\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)/.exec(String(ds.ImageComments ?? ""));
    if (m) {
      const b = bStd ?? Number(m[1]);
      const d = b > 0 ? vendorFrames.canon([Number(m[2]), Number(m[3]), Number(m[4])], ds) : undefined;
      return { bValue: b, ...(d ? { direction: d } : {}), source: "canon-comments" };
    }
  }
  // A standard b-value whose direction no rule found: the b-value still stands.
  if (bStd !== undefined) return { bValue: bStd, source: "standard (no direction)" };
  return undefined;
}

/**
 * The vendors' private direction frames, turned into patient LPS.
 *
 * GE: (0019,10BB/10BC/10BD) are components along the image's own axes -- the row direction (ImageOrientationPatient's
 * first vector), the column direction (its second) and the slice normal (their cross product) -- as dcm2niix's notes
 * describe ("relative to freq, phase, slice"; its geCorrectBvecs, BSD-2). With column phase-encoding (0018,1312 COL):
 *     LPS = −a·row − b·column + c·normal.
 * MEASURED 2026-09-29 against dcm2niix: GE release 14.0 (Signa HDx, axial) and MR29.1 (dcm_qa_ge, OBLIQUE, four series)
 * -- the tilted set is what separates the image frame from the patient frame. ROW phase-encoding swaps the in-plane
 * axes; no example has been validated, so no direction is given for it (said in `source`), rather than a guess.
 *
 * Canon / Toshiba (the vector in ImageComments): also the image's frame, NOT the scanner's (the dcm_qa_canon README
 * says bore; the numbers say image). MEASURED 2026-09-29 against dcm2niix on all 11 dcm_qa_canon series -- axial,
 * sagittal, coronal, tilted in one and two axes, both phase-encoding directions:
 *     column phase:  LPS = −a·column − b·row + c·normal;     row phase:  LPS = a·row − b·column − c·normal.
 * Matching dcm2niix is not ground truth here: that README warns these scanners do not flip the vector for an AP versus a
 * PA acquisition, so an AP series' y sign can be wrong in the file itself; nothing in the file says which it is.
 *
 * Philips (older single-frame (2005,10B0/B1/B2)): version 1 takes them as LPS; no single-frame Philips file with these
 * and without the standard attributes has been checked yet.
 */
export const vendorFrames = {
  ge: (g: number[], ds: Record<string, unknown>) => {
    if (String(ds.InPlanePhaseEncodingDirection ?? "").toUpperCase() !== "COL") return undefined;
    const f = imageFrame(ds);
    if (!f || g.some((x) => !Number.isFinite(x))) return undefined;
    const { r, c, n } = f;
    return unit([0, 1, 2].map((k) => -g[0] * r[k] - g[1] * c[k] + g[2] * n[k]));
  },
  philips: (g: number[], _ds: Record<string, unknown>) => unit(g),
  canon: (g: number[], ds: Record<string, unknown>) => {
    const f = imageFrame(ds);
    if (!f || g.some((x) => !Number.isFinite(x))) return undefined;
    const pe = String(ds.InPlanePhaseEncodingDirection ?? "").toUpperCase();
    if (pe !== "COL" && pe !== "ROW") return undefined;
    const neg = (v: number[]) => v.map((x) => -x);
    const [a, b, c] = pe === "COL" ? [neg(f.c), neg(f.r), f.n] : [f.r, neg(f.c), neg(f.n)];
    return unit([0, 1, 2].map((k) => g[0] * a[k] + g[1] * b[k] + g[2] * c[k]));
  },
};

/** The image's row direction, column direction and slice normal (row × column), from ImageOrientationPatient. */
function imageFrame(ds: Record<string, unknown>) {
  const iop = (ds.ImageOrientationPatient as number[] | undefined)?.map(Number);
  if (!iop || iop.length !== 6 || iop.some((x) => !Number.isFinite(x))) return undefined;
  const r = iop.slice(0, 3), c = iop.slice(3, 6);
  return { r, c, n: [r[1] * c[2] - r[2] * c[1], r[2] * c[0] - r[0] * c[2], r[0] * c[1] - r[1] * c[0]] };
}


// ── the interpreter: diffusion as the DICOM reader's volume key (volume-interpreters.ts) ─────────────────────────────

const unitDir = (g: unknown): [number, number, number] | undefined => {
  const v = Array.isArray(g) ? g.map(Number) : undefined;
  return v && v.length === 3 && Math.hypot(...v) > 0.5 ? v as [number, number, number] : undefined;
};
function diffusionKey(b: number, dir: [number, number, number] | undefined, source?: string, weak?: boolean): VolumeKey {
  return {
    key: `${b}|${dir?.map((v) => v.toFixed(4)).join(",") ?? ""}`,
    label: `b ${b}${dir ? ` · ${dir.map((v) => v.toFixed(2)).join(", ")}` : ""}`,
    // The rule and its version travel with the values (critic, 2026-09-29, finding 11): a person or a later step can
    // see where a direction came from, or why there is none.
    meta: { bValue: b, ...(dir ? { gradient: dir } : {}), ...(source ? { source, vendorRule: VENDOR_RULE } : {}) },
    ...(weak ? { weak: true } : {}),
  };
}

/**
 * DIFFUSION, as the DICOM reader's volume key: a single-frame file by the vendor rules above; a frame of an enhanced
 * file by the standard MR Diffusion macro. A WEAK b = 0 (a vendor field every MR image carries) stands only when
 * another image of the same series has b > 0; otherwise the series is not diffusion and its images lose the label
 * (critic, 2026-09-29, finding 3). Volumes carry meta.diffusion = { bValue, gradient (LPS), source, vendorRule }.
 */
export const diffusionInterpreter: VolumeInterpreter = {
  name: "diffusion",
  instance(ds, raw) {
    const d = diffusionOf(ds, raw as Raw);
    return d ? diffusionKey(d.bValue, d.direction, d.source, d.weak) : undefined;
  },
  frame(group) {
    const diff = group("MRDiffusionSequence");
    const b = diff?.DiffusionBValue != null ? num(diff.DiffusionBValue) : undefined;
    if (b === undefined) return undefined;
    const seq = diff?.DiffusionGradientDirectionSequence;
    const item = (Array.isArray(seq) ? seq[0] : seq) as Record<string, unknown> | undefined;
    return diffusionKey(b, unitDir(item?.DiffusionGradientOrientation));
  },
  finish(images) {
    const series = new Set(images.filter((i) => ((i.volumeKeys?.diffusion?.meta.bValue as number | undefined) ?? 0) > 0).map((i) => i.seriesInstanceUID));
    for (const i of images) {
      const k = i.volumeKeys?.diffusion;
      if (!k?.weak || series.has(i.seriesInstanceUID)) continue;
      delete i.volumeKeys!.diffusion;
      if (!Object.keys(i.volumeKeys!).length) delete i.volumeKeys;
    }
  },
};
