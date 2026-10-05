// The tracts as a DICOM Tractography Results object (tracts-dicom.ts): written and read back unchanged (points, sides,
// colors, labels, provenance), valid by dciodvfy (when installed), and a whole brain's worth written in reasonable time.
//   deno test -A --no-check tracts-dicom.test.ts
import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { dcmjs, dciodvfy, HAS_DCIODVFY, setDicomLibrary } from "albula/testing";
import { dicomToTracts, tractsToDicom, UNNAMED, type TractSetData, type TractsRun, type TractsSource } from "./tracts-dicom.ts";

setDicomLibrary(dcmjs);
const source: TractsSource = {
  patientStudy: { PatientName: "Test^Tracts", PatientID: "T1", PatientBirthDate: "", PatientSex: "O", StudyInstanceUID: "2.25.1001", StudyDate: "", StudyTime: "", StudyID: "", AccessionNumber: "", ReferringPhysicianName: "" },
  frameOfReferenceUID: "2.25.1002", seriesInstanceUID: "2.25.1003",
  instances: [{ sopClassUID: "1.2.840.10008.5.1.4.1.1.4.1", sopInstanceUID: "2.25.1004" }],
};
const run: TractsRun = { algorithmName: "UKF two-tensor (Albula GPU port)", algorithmVersion: "tracking rule 3", algorithmParameters: "seedingThreshold=0.1 stoppingFA=0.08", model: "multi", provenance: { trackingRule: 3, date: "2026-10-05" } };
const line = (x0: number, n: number) => Float32Array.from(Array.from({ length: n }, (_, i) => [x0 + i, -2 * i, 0.5 * i]).flat());

Deno.test("tracts round-trip through a Tractography Results object: points, sides, colors, labels, provenance", async () => {
  const sets: TractSetData[] = [
    { label: "arcuate fasciculus", side: -1, color: [0.9, 0.2, 0.1], streamlines: [line(10, 5), line(11, 7)] },
    { label: "corpus callosum 3", side: 0, color: [0.1, 0.5, 0.9], streamlines: [line(-3, 4)] },
    { label: UNNAMED, side: 0, color: [0.6, 0.6, 0.6], streamlines: [line(0, 3)] },
    { label: "empty one", side: 1, color: [0, 1, 0], streamlines: [] },
  ];
  const w = await tractsToDicom(sets, source, run);
  assertEquals([w.trackSets, w.tracks], [3, 4]);
  const r = await dicomToTracts(w.bytes);
  assertEquals(r.sets.map((s) => [s.label, s.side, s.streamlines.length]), [["arcuate fasciculus", -1, 2], ["corpus callosum 3", 0, 1], [UNNAMED, 0, 1]]);
  assertEquals([...r.sets[0].streamlines[1]], [...line(11, 7)]);            // RAS in, LPS in the file, RAS out
  for (let c = 0; c < 3; c++) assertAlmostEquals(r.sets[0].color[c], sets[0].color[c], 0.01);
  assertEquals(r.provenance, run.provenance);
  assertEquals(r.referencedSeries, source.seriesInstanceUID);
  if (HAS_DCIODVFY) {
    const v = dciodvfy(w.bytes)!;
    assertEquals(v.errors, [], `dciodvfy errors: ${v.errors.join(" | ")}`);
  }
});

Deno.test("a whole brain's worth (50,000 streamlines of 40 points) is written and read in seconds", async () => {
  const many: Float32Array[] = Array.from({ length: 50000 }, (_, i) => line(i % 100, 40));
  const sets: TractSetData[] = Array.from({ length: 70 }, (_, t) => ({ label: `tract ${t}`, side: (t % 3 - 1) as -1 | 0 | 1, color: [0.5, 0.5, 0.5] as [number, number, number], streamlines: many.slice(t * 700, (t + 1) * 700) }));
  sets.push({ label: UNNAMED, side: 0, color: [0.6, 0.6, 0.6], streamlines: many.slice(49000) });
  const t0 = performance.now();
  const w = await tractsToDicom(sets, source, run);
  const t1 = performance.now();
  const r = await dicomToTracts(w.bytes);
  const t2 = performance.now();
  console.log(`  ${w.tracks} tracks in ${w.trackSets} sets: ${(w.bytes.length / 1e6).toFixed(1)} MB, written in ${((t1 - t0) / 1000).toFixed(1)} s, read in ${((t2 - t1) / 1000).toFixed(1)} s`);
  assertEquals(r.sets.reduce((s, x) => s + x.streamlines.length, 0), 50000);
  assert(t2 - t0 < 30000, "writing and reading took more than 30 s");
});
