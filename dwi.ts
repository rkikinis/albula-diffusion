// ONE DIFFUSION SERIES, IN ONE CONVENTION -- whatever file it came from.
//
// The diffusion MRI work (Contents/docs/dmri-review-2026-09-28.md in the workspace; Ron, 2026-09-28: milestone 1 is
// reading, maps on the card, tracking from seeds in the view). Every source says where the diffusion gradients point in
// its own way, and a sign lost on the way is the classic silent error of the field: the tracts still draw, just wrong.
// So every reader below converts ONCE, here, to the one convention the rest of the code uses:
//
//   CONVENTION 1 -- each volume's gradient is a unit vector in PATIENT space, RAS (x to the patient's right, y to
//   anterior, z to superior), the same space as the volume's ijkToRAS; a b=0 volume's gradient is [0, 0, 0]; b-values
//   in s/mm² as the source gives them. A volume with b > 0 and NO direction (a scanner's trace, or "isotropic", image)
//   also has [0, 0, 0]: it is not a direction, and consumers must not use it as one (`isotropicVolumes` finds them;
//   the tensor fit leaves them out, the DICOM writer marks them ISOTROPIC; critic 2026-09-28, finding 9).
//
// The sources and their rules:
//  - NIfTI with FSL .bval/.bvec: the vectors are along the IMAGE AXES (i, j, k), and FSL flips the first one when the
//    image is stored with a positive determinant ("neurological"), so x is negated in that case before the axes are
//    turned into patient space. (FSL's convention, as dcm2niix writes it and MRtrix reads it.)
//  - NRRD, Slicer's DWI form: `DWMRI_gradient_NNNN` in the MEASUREMENT FRAME (each parenthesized vector a column),
//    which maps into the file's `space` (LPS, RAS, LAS); each b-value is DWMRI_b-value · |g|² (the longest gradient
//    carries the reference b). The list axis may come first (Slicer writes it so) or last (DWIConvert does).
//  - DICOM: the per-frame gradient read in logic/readers/dicom-series.ts (`meta.diffusion`, LPS).
// Checked 2026-09-28 against Slicer's DWIConvert on OpenNeuro ds001226 sub-PAT16 (logic/diffusion/dwi.test.ts).
import type { Volume } from "albula";
import { nrrdDecode as decode, nrrdGeometry as geometry, nrrdVectors as parseVectors, nrrdSampleReader as sampleReader, nrrdSpaceFlip as spaceFlip, nrrdSplitHeader as splitHeader, NRRD_TYPE_BYTES as TYPE_BYTES } from "albula";

export const DWI_CONVENTION = 1;

export interface DiffusionSeries {
  name?: string;
  /** One 3D volume per gradient, all on the same grid (`ijkToRAS`). */
  volumes: Volume[];
  /** s/mm², one per volume. */
  bValues: number[];
  /** Unit vectors in patient RAS, one per volume; [0, 0, 0] where b is 0 (CONVENTION 1). */
  gradients: [number, number, number][];
  ijkToRAS: number[];
  /** Which reader and which of its rules made the gradients, for the record and for bug reports. */
  source: string;
  convention: typeof DWI_CONVENTION;
}

/** The volumes with b > 0 and no gradient direction (trace / isotropic images), by index. */
export function isotropicVolumes(s: Pick<DiffusionSeries, "bValues" | "gradients">): number[] {
  return s.bValues.map((b, i) => (b > 0 && Math.hypot(...s.gradients[i]) < 0.5 ? i : -1)).filter((i) => i >= 0);
}
const noteIsotropic = (s: DiffusionSeries): DiffusionSeries => {
  const n = isotropicVolumes(s).length;
  return n ? { ...s, source: `${s.source}; ${n} volume(s) with b > 0 and no direction (trace images)` } : s;
};

const unit = (v: number[]): [number, number, number] => {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 1e-12 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
};
/** The 3x3 part of a row-major 4x4, its columns scaled to unit length (the axes' directions, spacing removed). */
function axes(m: number[]): number[][] {
  const cols = [0, 1, 2].map((c) => unit([m[c], m[4 + c], m[8 + c]]));
  return [0, 1, 2].map((r) => cols.map((col) => col[r]));   // rows of the direction matrix
}
const det3 = (m: number[]) =>
  m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) + m[2] * (m[4] * m[9] - m[5] * m[8]);
const mul = (R: number[][], v: number[]): number[] => R.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);

/** Numbers from a .bval/.bvec text: whitespace separated, rows kept. */
function table(text: string): number[][] {
  return text.trim().split(/\r?\n/).map((l) => l.trim()).filter(Boolean).map((l) => l.split(/[\s,]+/).map(Number));
}

/** A NIfTI's volumes with their FSL .bval/.bvec. */
export function fromFsl(volumes: Volume[], bvalText: string, bvecText: string, name?: string): DiffusionSeries {
  const n = volumes.length;
  const bvals = table(bvalText).flat();
  let bv = table(bvecText);
  // One row per volume is accepted as a courtesy; with exactly three volumes the two layouts look the same, and FSL's
  // own (three rows, one per axis) is the one read (critic finding 14: a three-volume file written one row per volume
  // is misread -- rare; a b > 0 volume that comes out without a direction is then named in `source`).
  if (bv.length === n && bv[0].length === 3 && n !== 3) bv = [0, 1, 2].map((r) => bv.map((row) => row[r]));
  if (bvals.length !== n || bv.length !== 3 || bv.some((row) => row.length !== n)) {
    throw new Error(`the gradient files do not match the scan: ${n} volumes, ${bvals.length} b-values, bvec ${bv.length}x${bv[0]?.length ?? 0}`);
  }
  const M = volumes[0].ijkToRAS;
  const flipX = det3(M) > 0;      // FSL: a positive determinant means its first voxel axis is mirrored
  const R = axes(M);
  const gradients = Array.from({ length: n }, (_, i) => {
    const g = [flipX ? -bv[0][i] : bv[0][i], bv[1][i], bv[2][i]];
    return bvals[i] > 0 ? unit(mul(R, g)) : [0, 0, 0] as [number, number, number];
  });
  return noteIsotropic({ name, volumes, bValues: bvals, gradients, ijkToRAS: M, source: `NIfTI + FSL bval/bvec${flipX ? " (x flipped: positive determinant)" : ""}`, convention: DWI_CONVENTION });
}

/**
 * Slicer's NRRD DWI: the header's fields and the (decoded or raw) data bytes. `header` is the text of a detached .nhdr or
 * the start of an attached .nrrd; `dataBytes` is the data file for a detached header (omit it for an attached one).
 */
export async function fromNrrdDwi(header: Uint8Array, dataBytes?: Uint8Array, name?: string): Promise<DiffusionSeries> {
  // A detached header (.nhdr) may end with one newline; the splitter looks for the blank line that ends a header.
  const { f, body } = splitHeader(dataBytes ? new Uint8Array([...header, 0x0a, 0x0a]) : header);
  if ((f["modality"] ?? "").toUpperCase() !== "DWMRI") throw new Error("NRRD: not a diffusion file (no modality:=DWMRI)");
  const sizes = (f["sizes"] ?? "").split(/\s+/).filter(Boolean).map(Number);
  const kinds = (f["kinds"] ?? "").split(/\s+/).filter(Boolean).map((k) => k.toLowerCase());
  if (sizes.length !== 4) throw new Error(`NRRD DWI: 4 dimensions expected, got ${sizes.length}`);
  const listAxis = kinds.findIndex((k) => k === "list" || k === "vector" || k === "covariant-vector");
  if (listAxis !== 0 && listAxis !== 3) throw new Error("NRRD DWI: the gradient axis must be first or last");
  const spatial = sizes.filter((_, i) => i !== listAxis) as [number, number, number];
  const nvol = sizes[listAxis];
  const type = (f["type"] ?? "").toLowerCase();
  const bpp = TYPE_BYTES[type];
  if (!bpp) throw new Error(`NRRD DWI: unsupported type "${type}"`);
  const n3 = spatial[0] * spatial[1] * spatial[2];
  const raw = await decode(f, dataBytes ?? body, n3 * nvol * bpp);
  const little = (f["endian"] ?? "little").toLowerCase() !== "big";
  const read = sampleReader(type, raw, new DataView(raw.buffer, raw.byteOffset, raw.byteLength), little);
  const geom = geometry(f);
  const volumes: Volume[] = Array.from({ length: nvol }, (_, t) => {
    const data = new Float32Array(n3);
    for (let v = 0; v < n3; v++) data[v] = read((listAxis === 0 ? v * nvol + t : t * n3 + v) * bpp);
    return { dims: spatial, ijkToRAS: geom.ijkToRAS, data, dtype: "<f4", name: `${name ?? "dwi"} [${t}]` };
  });
  // Gradients: in the measurement frame; frame -> the file's space -> RAS.
  const bRef = Number(f["dwmri_b-value"]);
  if (!(bRef >= 0)) throw new Error("NRRD DWI: no DWMRI_b-value");
  const cols = parseVectors(f["measurement frame"]);
  const MF = cols.length === 3 ? [0, 1, 2].map((r) => cols.map((c) => c[r])) : [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  // The space's signs from the one place NRRD spaces are decided (render/nrrd.ts); a space with no patient meaning
  // (scanner-xyz, 3D-right-handed) cannot give patient-space directions, and says so.
  const space = (f["space"] ?? "").toLowerCase();
  const toRAS = spaceFlip(space);
  if (!toRAS || !geom.anatomical) throw new Error(`NRRD DWI: the space "${space || "(none)"}" has no patient orientation, so the gradient directions cannot be put in patient space`);
  const keys = Object.keys(f).filter((k) => /^dwmri_gradient_\d+$/.test(k)).sort((a, b) => Number(a.slice(15)) - Number(b.slice(15)));
  if (keys.length !== nvol) throw new Error(`NRRD DWI: ${keys.length} gradients for ${nvol} volumes`);
  const raws = keys.map((k) => f[k].trim().split(/\s+/).map(Number));
  const maxLen = Math.max(...raws.map((g) => Math.hypot(g[0], g[1], g[2])));
  const bValues = raws.map((g) => (maxLen > 0 ? bRef * (Math.hypot(g[0], g[1], g[2]) / maxLen) ** 2 : 0));
  const gradients = raws.map((g, i) => {
    if (!(bValues[i] > 0)) return [0, 0, 0] as [number, number, number];
    const s = mul(MF, g);
    return unit([s[0] * toRAS[0], s[1] * toRAS[1], s[2] * toRAS[2]]);
  });
  return noteIsotropic({ name, volumes, bValues, gradients, ijkToRAS: geom.ijkToRAS, source: `NRRD DWI (space ${space}, measurement frame)`, convention: DWI_CONVENTION });
}

/** Volumes from the DICOM reader, each with `meta.diffusion = { bValue, gradient }` (gradient in LPS). */
export function fromDicomVolumes(volumes: Volume[], name?: string): DiffusionSeries {
  const bValues: number[] = [], gradients: [number, number, number][] = [];
  for (const v of volumes) {
    const d = (v.meta?.diffusion ?? {}) as { bValue?: number; gradient?: number[] };
    const b = Number(d.bValue ?? 0);
    bValues.push(b);
    gradients.push(b > 0 && d.gradient ? unit([-d.gradient[0], -d.gradient[1], d.gradient[2]]) : [0, 0, 0]);
  }
  return noteIsotropic({ name, volumes, bValues, gradients, ijkToRAS: volumes[0].ijkToRAS, source: "DICOM (MR diffusion attributes, LPS)", convention: DWI_CONVENTION });
}
