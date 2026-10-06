// The import-time job's choices (import-job.ts), on made-up series: which scan is tracked, which series is its reversed
// partner, which image is its MRI of the anatomy, when stored tracts are current, and how streamlines become track sets
// (critic, 2026-10-05, qa/2026-10-05-dmri-import-job.md, findings 6, 7, 11, 14).
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc import-job.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { anatomyOf, colorFaPath, isCurrent, isDiffusionScan, jobRules, partnerOf, planCases, trackSets, writeColorFA, type SeriesFacts } from "./import-job.ts";
import { packRGB24 } from "albula";
import type { TensorFit } from "./tensor.ts";
import { SHORT } from "./tractcloud/name-tracts.ts";
import { UNNAMED } from "./tracts-dicom.ts";

const AX = [-2.5, 0, 0, 120, 0, 2.5, 0, -100, 0, 0, 2.5, -60, 0, 0, 0, 1];
/** A series: `b` its b-values (NaN for an image that is not diffusion), `dirs` its distinct directions. */
const S = (uid: string, description: string, b: number[], o: Partial<SeriesFacts> = {}): SeriesFacts => ({
  uid, studyUID: "st1", patientUID: "p1", description, volumes: b.length, bValues: b, directions: o.directions ?? (b.filter((x) => x > 50).length), ijkToRAS: AX, derived: false, ...o });
const dwi = (uid: string, n: number, o: Partial<SeriesFacts> = {}) => S(uid, `DTI ${n}`, [0, ...Array(n - 1).fill(1000)], o);
const t1 = (uid: string, description = "T1_mprage", o: Partial<SeriesFacts> = {}) => S(uid, description, [NaN], o);

Deno.test("a diffusion scan needs seven volumes and six directions", () => {
  assert(isDiffusionScan(dwi("a", 7)));
  assert(!isDiffusionScan(dwi("a", 6)));
  assert(!isDiffusionScan(S("t", "trace", [0, 1000, 1000, 1000, 1000, 1000, 1000, 1000], { directions: 1 })));
});

Deno.test("the reversed partner: the record's reversed pair first, then b = 0 only; a pair the record refuses is skipped; elsewhere placed or bigger never", () => {
  const scan = dwi("scan", 102, { phaseEncoding: "j-" });
  const b0 = S("b0", "PA b0", [0, 0]);
  const recorded = S("rec", "PA", [0, 0, 0], { phaseEncoding: "j" });
  const wrongAxis = S("lr", "LR", [0, 0], { phaseEncoding: "i-" });
  assertEquals(partnerOf(scan, [scan, b0, recorded])?.uid, "rec");
  assertEquals(partnerOf(scan, [scan, b0])?.uid, "b0");
  assertEquals(partnerOf(scan, [scan, wrongAxis]), undefined);
  // A short diffusion scan with its own b = 0 (PA, 7 volumes) is a partner, not a scan of its own (finding 7).
  const shortPA = dwi("pa7", 7, { phaseEncoding: "j" });
  const cases = planCases([scan, shortPA, t1("t1")]);
  assertEquals(cases.map((c) => [c.dwi.uid, c.partner?.uid]), [["scan", "pa7"]]);
  // Placed elsewhere (another slab 40 mm away) or another study: not a partner.
  const far = S("far", "PA", [0, 0], { ijkToRAS: AX.map((v, i) => (i === 11 ? v + 40 : v)) });
  assertEquals(partnerOf(scan, [scan, far, S("other", "PA", [0, 0], { studyUID: "st2" })]), undefined);
});

Deno.test("the MRI of the anatomy: named as a T1, one volume, not computed; same study first, then the same patient; none named so, none chosen", () => {
  const scan = dwi("scan", 30);
  assertEquals(anatomyOf(scan, [scan, S("flair", "FLAIR", [NaN]), t1("t1", "T1_mprage")])?.uid, "t1");
  assertEquals(anatomyOf(scan, [scan, S("flair", "FLAIR", [NaN]), S("adc", "ADC", [NaN], { derived: true })]), undefined);   // quality first: wait
  assertEquals(anatomyOf(scan, [scan, t1("t10", "T10 something")]), undefined);                                            // "T10" is not a T1
  assertEquals(anatomyOf(scan, [scan, t1("mp", "MPRAGE sag")])?.uid, "mp");
  assertEquals(anatomyOf(scan, [scan, t1("ours", "T1 resampled", { derived: true })]), undefined);
  assertEquals(anatomyOf(scan, [scan, t1("other", "T1", { studyUID: "st2" }), t1("own", "T1 post", { studyUID: "st1" })])?.uid, "own");
  assertEquals(anatomyOf(scan, [scan, t1("other", "T1", { studyUID: "st2" })])?.uid, "other");                             // same patient, other study
  assertEquals(anatomyOf(scan, [scan, t1("stranger", "T1", { studyUID: "st3", patientUID: "p2" })]), undefined);
});

Deno.test("stored tracts are current only for the same code, rules and inputs", () => {
  const v = { code: "abc", labeler: "rapidparc x", synthstrip: "haversack 0.15.0" };
  const rules = jobRules(v), inputs = { diffusion: "d", partner: "p", t1: "t" };
  assert(isCurrent({ rules, inputs }, rules, inputs));
  assert(!isCurrent({ rules: jobRules({ ...v, code: "abd" }), inputs }, rules, inputs), "a code change must make them stale");
  assert(!isCurrent({ rules, inputs: { ...inputs, partner: null } }, rules, inputs), "a reversed scan that arrived later must make them stale");
  assert(!isCurrent({ rules, inputs: { ...inputs, t1: "other" } }, rules, inputs));
  assert(!isCurrent({ rules: { ...rules, extra: 1 }, inputs }, rules, inputs));
  assert(!isCurrent(undefined, rules, inputs));
});

Deno.test("track sets: one per named tract and side, the rest in Unnamed, last", () => {
  const names = [{ name: "arcuate" }, { name: "corpus callosum" }, { name: "Other" }];
  const sl = [0, 1, 2, 3, 4].map((i) => new Float32Array([i, 0, 0, i + 1, 0, 0]));
  const sets = trackSets(sl, Int32Array.from([0, 0, 1, 2, SHORT]), Int8Array.from([-1, 1, 0, 0, 0]), new Uint8Array(5), names);
  assertEquals(sets.map((s) => [s.label, s.side, s.streamlines.length]), [["arcuate", -1, 1], ["arcuate", 1, 1], ["corpus callosum", 0, 1], [UNNAMED, 0, 2]]);
});

Deno.test("the direction-colored map is stored beside the tracts: the Color FA, packed as the slice views draw it", async () => {
  const dir = await Deno.makeTempDir();
  try {
    // Two voxels: FA 0.5 along left-right (red), FA 1 along up-down (blue).
    const fit = { dims: [2, 1, 1], ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], fa: new Float32Array([0.5, 1]), v1: new Float32Array([1, 0, 0, 0, 0, 1]) } as unknown as TensorFit;
    await writeColorFA(dir, "1.2.3", fit);
    const bytes = await Deno.readFile(`${dir}/${colorFaPath("1.2.3")}`);
    const text = new TextDecoder().decode(bytes.subarray(0, 400)), at = text.indexOf("\n\n") + 2;
    assert(/type: float/.test(text) && /encoding: gzip/.test(text) && /sizes: 2 1 1/.test(text), text);
    const raw = new Uint8Array(await new Response(new Blob([bytes.subarray(at)]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
    const v = new Float32Array(raw.buffer);
    assertEquals([...v], [packRGB24(128, 0, 0), packRGB24(0, 0, 255)]);
  } finally { await Deno.remove(dir, { recursive: true }); }
});
