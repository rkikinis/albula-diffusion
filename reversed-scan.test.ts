// THE REVERSED SCAN OF A CONVERTED CASE IS FOUND (critic 2026-10-22, reversed-scan regression, findings 1 and 6). A scan
// and its reversed scan written by Albula's own diffusion writer -- as the BIDS import converts the public tumor cases --
// read back through the job's own reader (seriesFromInstances: the facts from the real headers, nothing set by hand),
// and the job's own choice (planCases) takes the reversed scan. A Siemens TENSOR_B0 header is never taken.
// From 2026-10-21 to 2026-10-22 the writer's Manufacturer "SlicerAlbula" made every converted scan "derived", the
// reversed scan of 37 cases was refused, and their tracts were made without the correction. No test read a real header.
//
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc reversed-scan.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { dcmjs, setDicomLibrary } from "albula/testing";
import { parseInstances } from "albula";
import { diffusionToEnhancedMR } from "./export-dicom-dwi.ts";
import { DWI_CONVENTION, type DiffusionSeries } from "./dwi.ts";
import { partnerRefusals, planCases, seriesFromInstances } from "./import-job.ts";
import "./hooks.ts";   // the vendors' readers, as the job imports them (the phase-encoding sign comes through them)

setDicomLibrary(dcmjs);
const M = [-2, 0, 0, 100, 0, -2, 0, 120, 0, 0, 2, -40, 0, 0, 0, 1];
const vol = () => ({ dims: [4, 4, 2] as [number, number, number], ijkToRAS: M, data: new Int16Array(32).fill(100), dtype: "<i2" as const });
const subject = { patientName: "Test^Converted", patientID: "T1", studyInstanceUID: "2.25.1001", frameOfReferenceUID: "2.25.1002", studyDate: "", studyTime: "" };
async function written(description: string, bValues: number[], gradients: number[][], pe: "j" | "j-") {
  const s: DiffusionSeries = { volumes: bValues.map(vol), bValues, gradients, ijkToRAS: M, source: "test", convention: DWI_CONVENTION };
  const out = await diffusionToEnhancedMR(s, subject, { safetyStandardAgency: "IEC", seriesDescription: description, phaseEncoding: "COLUMN",
    sourceDescription: JSON.stringify({ PhaseEncodingDirection: pe }) });
  const instances = await parseInstances([out.bytes.slice().buffer as ArrayBuffer], { headers: true });
  return { instances, uid: out.seriesInstanceUID as string, description };
}
const row = (w: { uid: string; description: string }) => ({ seriesUID: w.uid, studyUID: subject.studyInstanceUID, patientUID: "p1", description: w.description });

Deno.test("a converted scan's reversed scan is chosen; a TENSOR_B0 map never is", async () => {
  const dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.7, 0.7, 0], [0.7, 0, 0.7], [0, 0.7, 0.7]];
  const ap = await written("DTI (AP)", [0, ...dirs.map(() => 1000)], [[0, 0, 0], ...dirs], "j-");
  const pa = await written("PA b0 (PA)", [0, 0], [[0, 0, 0], [0, 0, 0]], "j");
  const apS = seriesFromInstances(row(ap), ap.instances), paS = seriesFromInstances(row(pa), pa.instances);
  // The writer's own stamp is there -- and does not make the scan a computed map.
  assertEquals(String(paS.header["00080070"]?.Value?.[0]), "SlicerAlbula");
  assertEquals([apS.facts.derived, paS.facts.derived], [false, false]);
  const plans = planCases([apS.facts, paS.facts]);
  assertEquals(plans.length, 1);
  assertEquals(plans[0].dwi.uid, ap.uid);
  assertEquals(plans[0].partner?.uid, pa.uid, "the reversed scan is taken");

  // The same reversed scan marked as Siemens' computed TENSOR_B0 map (its image type, in the real header): refused, and
  // the reason said.
  const t = await written("TENSOR_B0 (AP)", [0, 0], [[0, 0, 0], [0, 0, 0]], "j");
  const h = t.instances[0].header!.json as Record<string, { vr?: string; Value?: unknown[] }>;
  h["00080008"] = { vr: "CS", Value: ["DERIVED", "PRIMARY", "TENSOR_B0", "NONE"] };
  const tS = seriesFromInstances(row(t), t.instances);
  assertEquals(tS.facts.derived, true);
  const only = planCases([apS.facts, tS.facts]);
  assertEquals(only[0].partner, undefined, "a computed map is never the reversed scan");
  assertEquals(only[0].partnerRefusals?.find((r) => r.uid === t.uid)?.why, "a computed map (its image type)");
  assert(partnerRefusals(apS.facts, [apS.facts, paS.facts]).every((r) => r.why === ""), "the true reversed scan has no reason to be refused");
});

// THE FULL-LENGTH TWIN THROUGH REAL HEADERS (2026-10-23): the HCP protocols scan every diffusion run in both directions at
// full length. Both written by the diffusion writer and read back: the PA run is the AP run's partner (the record's
// phase-encoding sign comes through the reader), not a case of its own.
Deno.test("a full-length reversed twin read from real headers is the scan's partner", async () => {
  const dirs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.7, 0.7, 0], [0.7, 0, 0.7], [0, 0.7, 0.7]];
  const ap = await written("dMRI_dir99_AP", [0, ...dirs.map(() => 1000)], [[0, 0, 0], ...dirs], "j-");
  const pa = await written("dMRI_dir99_PA", [0, ...dirs.map(() => 1000)], [[0, 0, 0], ...dirs.map((d) => d.map((x) => -x))], "j");
  const apS = seriesFromInstances(row(ap), ap.instances), paS = seriesFromInstances(row(pa), pa.instances);
  assertEquals(apS.facts.volumes, paS.facts.volumes);
  const plans = planCases([paS.facts, apS.facts]);
  assertEquals(plans.map((p) => [p.dwi.uid, p.partner?.uid]), [[ap.uid, pa.uid]]);
});
