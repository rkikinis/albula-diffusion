// THE DIFFUSION KIND OF THE BIDS IMPORT (bids-dwi.ts) through core's import, on a small dataset made here (core's test
// data builder, copied from logic/import/bids.test.ts): a diffusion scan with .bval/.bvec beside a T1 and a mask.
// Checked: the same UIDs on a second run, the gradients read back through the app's DICOM reader, the safety standard
// required, the sidecar's who-and-where left out, and dciodvfy (when installed).
import { assert, assertEquals } from "jsr:@std/assert@1";
import { dicomIO, parseInstances, volumesOfSeries } from "albula";
import { buildBidsSubject, dcmjs, dciodvfy, HAS_DCIODVFY, readBidsDataset, setDicomLibrary } from "albula/testing";
import { phaseEncodingOf } from "./diffusion-vendors.ts";
import { fromDicomVolumes } from "./dwi.ts";
import "./hooks.ts";                                         // the diffusion kind and interpreter, as the app registers them

setDicomLibrary(dcmjs);

/** A NIfTI-1 file (.nii), `dims` as the header's dim[] ([ndim, nx, ny, nz, nt]): int16 (datatype 4) or float32 (16), sform from a row-major 4x4. */
function nifti(dims: number[], affine: number[], data: Int16Array | Float32Array): Uint8Array {
  const buf = new ArrayBuffer(352 + data.byteLength), v = new DataView(buf);
  v.setInt32(0, 348, true);
  dims.forEach((d, i) => v.setInt16(40 + 2 * i, d, true));   // dim[0] = the number of dimensions, then the sizes
  const f32 = data instanceof Float32Array;
  v.setInt16(70, f32 ? 16 : 4, true); v.setInt16(72, f32 ? 32 : 16, true);
  v.setFloat32(76, 1, true);
  for (let c = 0; c < 3; c++) v.setFloat32(80 + 4 * c, Math.hypot(affine[c], affine[4 + c], affine[8 + c]), true);
  v.setFloat32(108, 352, true);             // vox_offset
  v.setFloat32(112, 1, true);               // scl_slope
  v.setInt16(254, 1, true);                 // sform_code: scanner
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) v.setFloat32(280 + 16 * r + 4 * c, affine[4 * r + c], true);
  new Uint8Array(buf, 344, 4).set([0x6e, 0x2b, 0x31, 0]);   // "n+1\0"
  new Uint8Array(buf, 352).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  return new Uint8Array(buf);
}

async function makeDataset(): Promise<string> {
  const root = await Deno.makeTempDir({ prefix: "bids-test-" });
  const w = (p: string, b: Uint8Array | string) => { Deno.mkdirSync(p.slice(0, p.lastIndexOf("/")), { recursive: true }); typeof b === "string" ? Deno.writeTextFileSync(p, b) : Deno.writeFileSync(p, b); };
  w(`${root}/dataset_description.json`, JSON.stringify({ Name: "TestSet", License: "CC0", Authors: ["A", "B"], DatasetDOI: "doi:10.18112/openneuro.ds999999.v1.0.0" }));
  w(`${root}/participants.tsv`, "participant_id\t sex\t age\ttumor type & grade\theight (cm)\n sub-01\tF\t47\tGlioma II\t165\n");
  // T1: 8x6x5 at 1 mm, RAS, a slight tilt and a 1e-10 term (float noise, as real NIfTI affines have).
  const T1A = [0.99881786, 0.04542019, 0.0174408, -4, -0.04541337, 0.99896795, -0.00079298, -3, -0.01745886, 3.5163339e-10, 0.99984759, -2, 0, 0, 0, 1];
  const t1 = new Int16Array(8 * 6 * 5).map((_, i) => i * 7 % 1000);
  w(`${root}/sub-01/ses-pre/anat/sub-01_ses-pre_T1w.nii`, nifti([3, 8, 6, 5, 1], T1A, t1));
  w(`${root}/sub-01/ses-pre/anat/sub-01_ses-pre_T1w.json`, JSON.stringify({ ScanningSequence: "GR_IR", SequenceVariant: "SP_MP", EchoTime: 0.00418, RepetitionTime: 1.75, InversionTime: 0.9, FlipAngle: 9, SeriesDescription: "T1_mprage", ImageType: ["ORIGINAL", "PRIMARY", "M", "ND"] }));
  // DWI: 4x4x3 at 2.5 mm, LAS, four volumes (b0 + three directions).
  const DA = [-2.5, 0, 0, 5, 0, 2.5, 0, -5, 0, 0, 2.5, -3, 0, 0, 0, 1];
  const dwi = new Int16Array(4 * 4 * 3 * 4).map((_, i) => 100 + i);
  w(`${root}/sub-01/ses-pre/dwi/sub-01_ses-pre_acq-AP_dwi.nii`, nifti([4, 4, 4, 3, 4], DA, dwi));
  w(`${root}/sub-01/ses-pre/dwi/sub-01_ses-pre_acq-AP_dwi.bval`, "0 1000 1000 2000\n");
  w(`${root}/sub-01/ses-pre/dwi/sub-01_ses-pre_acq-AP_dwi.bvec`, "0 1 0 0.6\n0 0 1 0.8\n0 0 0 0\n");
  w(`${root}/sub-01/ses-pre/dwi/sub-01_ses-pre_acq-AP_dwi.json`, JSON.stringify({ SeriesDescription: "DTI_test", PhaseEncodingDirection: "j-", InstitutionName: "Somewhere", DeviceSerialNumber: "123" }));
  // Mask on the T1's grid but stored LAS (i reversed): the same points, the other index order. Fractional values.
  const MA = [-T1A[0], T1A[1], T1A[2], T1A[3] + 7 * T1A[0], -T1A[4], T1A[5], T1A[6], T1A[7] + 7 * T1A[4], -T1A[8], T1A[9], T1A[10], T1A[11] + 7 * T1A[8], 0, 0, 0, 1];
  const mask = new Float32Array(8 * 6 * 5);
  // In T1 index terms: voxels (i=2..3, j=1..2, k=1) at 0.8, (i=5, j=4, k=3) at 0.3 (below the cut).
  const setT1 = (i: number, j: number, k: number, x: number) => { mask[(k * 6 + j) * 8 + (7 - i)] = x; };
  for (const i of [2, 3]) for (const j of [1, 2]) setT1(i, j, 1, 0.8);
  setT1(5, 4, 3, 0.3);
  w(`${root}/derivatives/tumor_masks/sub-01/anat/sub-01_space_T1_label-tumor.nii`, nifti([3, 8, 6, 5, 1], MA, mask));
  return root;
}

const opts = { masks: ["tumor_masks"], maskAlgorithm: "manual delineation", safetyStandardAgency: "IEC" as const, safetyReason: "test" };
const ab = (u: Uint8Array) => u.slice().buffer as ArrayBuffer;

Deno.test("BIDS import with the diffusion kind: T1, diffusion and mask, stable UIDs, the gradients read back", async () => {
  const root = await makeDataset();
  try {
    const ds = await readBidsDataset(root);
    assertEquals(ds.id, "ds999999");
    const { objects: a, skipped } = await buildBidsSubject(ds, "01", "pre", opts);
    const { objects: b } = await buildBidsSubject(ds, "sub-01", "ses-pre", opts);
    assertEquals(skipped, []);
    assertEquals(a.map((o) => o.role), ["T1w", "dwi-AP", "seg-tumor"]);
    assertEquals(a.map((o) => o.seriesInstanceUID), b.map((o) => o.seriesInstanceUID), "a second run gives the same series UIDs");
    assertEquals(a.map((o) => o.files.map((f) => f.index.sopInstanceUID)), b.map((o) => o.files.map((f) => f.index.sopInstanceUID)));
    const [t1, dwi, seg] = a;
    assert(t1.files.every((f) => f.index.newStudy?.patientID === "ds999999-sub-01"), "the T1 rows create the patient");
    assertEquals(new Set(a.flatMap((o) => o.files.map((f) => f.index.studyInstanceUID))).size, 1, "one study");
    assertEquals(seg.files[0].index.derivedFrom?.parentSeriesUID, t1.seriesInstanceUID, "the SEG hangs under the T1");

    // The T1 through our own reader: same values at the same positions (the grid may be re-ordered).
    const t1Back = volumesOfSeries(await parseInstances(t1.files.map((f) => ab(f.bytes)))).frames;
    assertEquals(t1Back.length, 1);
    const M = t1Back[0].ijkToRAS;
    const T1A = [0.99881786, 0.04542019, 0.0174408, -4, -0.04541337, 0.99896795, -0.00079298, -3, -0.01745886, 3.5163339e-10, 0.99984759, -2];
    for (let i = 0; i < 12; i++) assert(Math.abs(M[i] - T1A[i]) < 1e-5, `T1 geometry [${i}]: ${M[i]} against ${T1A[i]} (a lost exponent shows up here)`);
    assertEquals([...t1Back[0].data].slice(0, 20), [...new Int16Array(8 * 6 * 5).map((_, i) => i * 7 % 1000)].slice(0, 20));
    const meta = dicomIO;
    const io = await meta();
    const first = io.naturalize(io.readFile(ab(t1.files[0].bytes)).dict) as Record<string, unknown>;
    assertEquals([first.PatientSex, first.PatientAge, first.RepetitionTime, first.EchoTime, first.AdmittingDiagnosesDescription], ["F", "047Y", 1750, 4.18, "Glioma II"]);
    assertEquals(first.ImageType, ["DERIVED", "SECONDARY", "M", "ND"]);

    // The diffusion object: four volumes, same voxels, same gradients in patient space.
    const dBack = fromDicomVolumes(volumesOfSeries(await parseInstances([ab(dwi.files[0].bytes)])).frames);
    assertEquals(dBack.bValues, [0, 1000, 1000, 2000]);
    // FSL on an LAS grid (negative determinant): no flip; +x along i, which points to patient left (RAS -x).
    const want = [[0, 0, 0], [-1, 0, 0], [0, 1, 0], [-0.6, 0.8, 0]];
    dBack.gradients.forEach((g, i) => g.forEach((x, c) => assert(Math.abs(x - want[i][c]) < 1e-6, `gradient ${i}: ${g} against ${want[i]}`)));

    // The SEG: the four voxels at 0.8, not the one at 0.3.
    const sd = io.naturalize(io.readFile(ab(seg.files[0].bytes)).dict) as Record<string, unknown>;
    assertEquals(sd.SOPClassUID === "1.2.840.10008.5.1.4.1.1.66.7" || sd.SOPClassUID === "1.2.840.10008.5.1.4.1.1.66.4", true);
    assert(seg.description.includes("4 voxels"), seg.description);
    const segs = sd.SegmentSequence as Record<string, unknown>[];
    const tumor = segs.find((s) => s.SegmentLabel === "Tumor")!;
    assertEquals([tumor.SegmentAlgorithmType, tumor.SegmentAlgorithmName], ["SEMIAUTOMATIC", "manual delineation"]);

    if (HAS_DCIODVFY) {
      // Accepted, each a dicom3tools simplification (Contents/docs/upstream-issues-dicom3tools.md): Patient Orientation
      // (item 2), and the MR FOV/Geometry macro's encoding attributes, which it requires always where the standard
      // requires them for ORIGINAL frames only (item 3; module/mr.tpl lines 75-79: Condition="Always").
      const known = /PatientOrientation\(0020,0020\)> - Missing attribute for Type 2C Conditional - Module=<GeneralImage>|MRFOVGeometrySequence\(0018,9125\)\[1\]\/(MRAcquisitionFrequencyEncodingSteps|MRAcquisitionPhaseEncodingStepsInPlane|PercentSampling|PercentPhaseFieldOfView)\(0018,[0-9a-f]{4}\)> - Missing attribute for Type 1C Conditional - Module=<MRFOVGeometryMacro>/;
      for (const o of a) assertEquals(dciodvfy(o.files[0].bytes)!.errors.filter((e) => !known.test(e)), [], `${o.role}: dciodvfy`);
    }
  } finally { await Deno.remove(root, { recursive: true }); }
});

Deno.test("BIDS: no safety standard with a diffusion scan -> refused before anything is built", async () => {
  const root = await makeDataset();
  try {
    let err = "";
    try { await skippedOf(root, { masks: [] }); } catch (e) { err = String(e); }
    assert(/safety standard/.test(err), err || "no error");
  } finally { await Deno.remove(root, { recursive: true }); }
});

Deno.test("BIDS diffusion: no invented dates, the sidecar's who-and-where left out", async () => {
  const root = await makeDataset();
  try {
    const { objects } = await skippedOf(root);
    const io = await dicomIO();
    for (const o of objects) {
      const d = io.naturalize(io.readFile(ab(o.files[0].bytes)).dict) as Record<string, unknown>;
      assertEquals([o.role, d.StudyDate ?? "", d.StudyTime ?? "", d.AcquisitionDateTime], [o.role, "", "", undefined]);
    }
    const dwi = objects.find((o) => o.role.startsWith("dwi"))!;
    const text = new TextDecoder().decode(dwi.files[0].bytes);
    assert(text.includes("PhaseEncodingDirection") && !text.includes("Somewhere") && !text.includes("DeviceSerialNumber"), "sidecar filtered");
    // THE PHASE-ENCODING DIRECTION (2026-10-03): the axis in the standard attribute (j: along the columns), the sign kept
    // in the private block, and both read back as the scanner's record ("j-") by the diffusion interpreter.
    const file = io.readFile(ab(dwi.files[0].bytes)), nd = io.naturalize(file.dict) as Record<string, unknown>;
    const fov = ((nd.SharedFunctionalGroupsSequence as Record<string, unknown>[])[0].MRFOVGeometrySequence as Record<string, unknown>[])[0];
    assertEquals(fov.InPlanePhaseEncodingDirection, "COLUMN");
    assertEquals(phaseEncodingOf(fov.InPlanePhaseEncodingDirection, file.dict as never), "j-");
    assertEquals(phaseEncodingOf(fov.InPlanePhaseEncodingDirection), "COL");
    assertEquals([objects[0].files[0].index.newStudy?.patientSex, objects[0].files[0].index.newStudy?.patientAge], ["F", "047Y"]);
  } finally { await Deno.remove(root, { recursive: true }); }
});

const skippedOf = async (root: string, o: Record<string, unknown> = opts) => (await buildBidsSubject(await readBidsDataset(root), "01", "pre", o as typeof opts));

Deno.test("BIDS diffusion: a T1 the writers cannot take is skipped and named, and the diffusion scan is still imported", async () => {
  const root = await makeDataset();
  try {
    const p = `${root}/sub-01/ses-pre/anat/sub-01_ses-pre_T1w.nii`;
    const b = Deno.readFileSync(p), v = new DataView(b.buffer);
    // Since 2026-10-07 a scaled or fractional T1 IS written (its own scale, or rounded under one Rescale Slope); what the
    // writers still cannot take is a value that is not a number -- a scale of 1e38 overflows float32 to infinity.
    v.setFloat32(112, 1e38, true);
    Deno.writeFileSync(p, b);
    const r = await skippedOf(root);
    assertEquals(r.objects.map((o) => o.role), ["dwi-AP"]);
    assert(r.skipped.some((k) => /T1w\.nii/.test(k.file) && /not numbers/.test(k.why)), JSON.stringify(r.skipped));
    assert(r.skipped.some((k) => /no T1 series/.test(k.why)), "the mask says why it was not used");
  } finally { await Deno.remove(root, { recursive: true }); }
});
