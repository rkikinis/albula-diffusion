// The diffusion writer (logic/export-dicom-dwi.ts) on a tiny series: each volume's directionality as the standard names
// it, and the safety standard required. The round trip through our reader and dcm2niix on real data is in
// logic/import/bids.test.ts and the record (dmri-review-2026-09-28.md).
//   deno test -A --no-check logic/export-dicom-dwi.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { dcmjs } from "albula/testing";
import { dicomIO } from "albula";
import { setDicomLibrary } from "albula/testing";
import { diffusionToEnhancedMR } from "./export-dicom-dwi.ts";
import { DWI_CONVENTION, type DiffusionSeries } from "./dwi.ts";

setDicomLibrary(dcmjs);
const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const series: DiffusionSeries = {
  volumes: [0, 1, 2].map(() => ({ dims: [2, 2, 1] as [number, number, number], ijkToRAS: I, data: new Int16Array([1, 2, 3, 4]), dtype: "<i2" })),
  bValues: [0, 1000, 1000], gradients: [[0, 0, 0], [1, 0, 0], [0, 0, 0]], ijkToRAS: I, source: "test", convention: DWI_CONVENTION,
};
const subject = { patientName: "T", patientID: "T", studyInstanceUID: "1.2.3", frameOfReferenceUID: "1.2.4", studyDate: "", studyTime: "" };

Deno.test("directionality per volume: NONE at b = 0, DIRECTIONAL with a direction, ISOTROPIC at b > 0 without one", async () => {
  const out = await diffusionToEnhancedMR(series, subject, { safetyStandardAgency: "IEC" });
  const io = await dicomIO();
  const d = io.naturalize(io.readFile(out.bytes.slice().buffer as ArrayBuffer).dict) as Record<string, unknown>;
  const frames = d.PerFrameFunctionalGroupsSequence as { MRDiffusionSequence: { DiffusionDirectionality: string } | { DiffusionDirectionality: string }[] }[];
  const dir = frames.map((f) => (Array.isArray(f.MRDiffusionSequence) ? f.MRDiffusionSequence[0] : f.MRDiffusionSequence).DiffusionDirectionality);
  assertEquals(dir, ["NONE", "DIRECTIONAL", "ISOTROPIC"]);
  assertEquals([d.StudyDate ?? "", d.AcquisitionDateTime, d.DimensionOrganizationType], ["", undefined, undefined]);
});

Deno.test("no safety standard -> refused in words", async () => {
  let err = "";
  try { await diffusionToEnhancedMR(series, subject, { safetyStandardAgency: undefined as unknown as "IEC" }); } catch (e) { err = String(e); }
  assert(/Safety Standard Agency is required/.test(err), err || "accepted");
});

Deno.test("a NIfTI value scale: the stored whole numbers written with the scale as Rescale Slope; a value off the scale refused", async () => {
  const slope = Math.fround(1.816361427307129), meta = { sclSlope: slope, sclInter: 0 };
  const scaled: DiffusionSeries = { ...series, volumes: series.volumes.map((v) => ({ ...v, meta, data: Float32Array.from(v.data as ArrayLike<number>, (x) => x * slope) })) };
  const out = await diffusionToEnhancedMR(scaled, subject, { safetyStandardAgency: "IEC" });
  const io = await dicomIO();
  const d = io.naturalize(io.readFile(out.bytes.slice().buffer as ArrayBuffer).dict) as Record<string, unknown>;
  const shared = d.SharedFunctionalGroupsSequence as Record<string, unknown> | Record<string, unknown>[];
  const pvt = (Array.isArray(shared) ? shared[0] : shared).PixelValueTransformationSequence as Record<string, unknown> | Record<string, unknown>[];
  const r = Array.isArray(pvt) ? pvt[0] : pvt;
  assertEquals([Number(r.RescaleSlope), Number(r.RescaleIntercept)], [1.816361427, 0]);
  const px = new Uint16Array((d.PixelData as ArrayBuffer[])[0]);
  assertEquals(Array.from(px.slice(0, 4)), [1, 2, 3, 4]);
  // Every value read back as stored * slope equals the scan's own to float32 precision.
  for (let i = 0; i < 4; i++) assert(Math.abs(px[i] * 1.816361427 - (scaled.volumes[0].data as Float32Array)[i]) < 1e-5);
  // A value the scale cannot reach: refused, not rounded.
  const off: DiffusionSeries = { ...scaled, volumes: scaled.volumes.map((v, t) => t ? v : { ...v, data: Float32Array.from([1.5, 2, 3, 4]) }) };
  let err = "";
  try { await diffusionToEnhancedMR(off, subject, { safetyStandardAgency: "IEC" }); } catch (e) { err = String(e); }
  assert(/does not turn back into a whole number/.test(err), err || "accepted");
  // Fractions without any scale: refused as before.
  const bare: DiffusionSeries = { ...series, volumes: series.volumes.map((v) => ({ ...v, data: Float32Array.from([0.5, 1, 2, 3]) })) };
  err = "";
  try { await diffusionToEnhancedMR(bare, subject, { safetyStandardAgency: "IEC" }); } catch (e) { err = String(e); }
  assert(/no value scale/.test(err), err || "accepted");
});
