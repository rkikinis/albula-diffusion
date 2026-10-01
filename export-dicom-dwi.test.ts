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
