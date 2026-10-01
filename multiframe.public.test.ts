// DIFFUSION in public multi-frame files from three vendors (the fMRI files are core's test, dicom-multiframe.public.test.ts) (Philips, Canon, Siemens XA30), checked
// against dcm2niix's own outputs for the same series (b-values, gradient directions, repetition time). The files are
// fetched by Contents/data/multiframe/fetch-multiframe.py (BSD-2, pinned commits) into Contents/data/dicom/multiframe;
// without them these tests are ignored, not passed.
//
// Checked by hand on 2026-09-25, beyond what is asserted here: the Canon and both Siemens series equal pydicom's own
// reading voxel for voxel; the two Philips series differed by up to 1 in every slice because their rescale slope is
// not a whole number (0.70, 1.07) and the reader stored whole numbers -- float32 since 2026-09-25, exact.
import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { parseInstances, volumesOfSeries } from "albula";
import { ABSENT, dcmjs, setDicomLibrary, testData } from "albula/testing";
import "./hooks.ts";                                         // the diffusion interpreter, as the app registers it
setDicomLibrary(dcmjs);

const BASE = testData("multiframe") ?? ABSENT;
const have = (p: string) => { try { Deno.statSync(p); return true; } catch { return false; } };

async function read(name: string) {
  const dir = BASE + name;
  const files = [...Deno.readDirSync(dir)].map((e) => e.name).filter((n) => !n.endsWith(".gz")).sort();
  const bufs = files.map((f) => { const u = Deno.readFileSync(`${dir}/${f}`); return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer; });
  return volumesOfSeries(await parseInstances(bufs));
}
const ref = (name: string, ext: string) => {
  const dir = BASE + name + "_ref";
  const f = [...Deno.readDirSync(dir)].find((e) => e.name.endsWith(ext))!.name;
  return Deno.readTextFileSync(`${dir}/${f}`);
};
const bvals = (name: string) => ref(name, ".bval").trim().split(/\s+/).map(Number);

for (const c of [
  { name: "philips_dti", volumes: 17, dims: [128, 128, 82] },
  { name: "canon_dti", volumes: 13, dims: [80, 80, 40], sameOrder: true },
  { name: "xa30_dwi", volumes: 21, dims: [72, 72, 39], sameOrder: true },
]) {
  Deno.test({
    name: `public multi-frame ${c.name}: ${c.volumes} volumes of ${c.dims.join("x")}, as dcm2niix finds them`,
    ignore: !have(BASE + c.name),
    fn: async () => {
      const r = await read(c.name);
      assertEquals(r.frames.length, c.volumes);
      for (const v of r.frames) assertEquals(v.dims, c.dims);
      assertEquals(r.leftOut, []);
      const ours = r.frames.map((v) => (v.meta?.diffusion as { bValue: number } | undefined)?.bValue);
      assert(ours.every((b) => b !== undefined), "every diffusion volume carries its b-value");
      // THE SAME ORDER, checked, not sorted (critic, 2026-09-25, finding 16): Canon and Siemens as dcm2niix lists them;
      // Philips with its b=0 first, where dcm2niix moves it last, and the directional volumes in the same order.
      const refB = bvals(c.name);
      const refIndex = (j: number) => c.sameOrder ? j : (j === 0 ? refB.length - 1 : j - 1);
      assertEquals(ours, ours.map((_, j) => refB[refIndex(j)]));
      // THE GRADIENT DIRECTIONS against dcm2niix's .bvec, every directional volume (only b-values were checked before).
      // .bvec is in FSL's image convention, whose row axis runs the other way from DICOM's, so in our image axes the
      // direction is (x, -y, z) of the .bvec -- on all three vendors, to three decimals.
      const vec = ref(c.name, ".bvec").trim().split("\n").map((l) => l.trim().split(/\s+/).map(Number));
      const m = r.frames[0].ijkToRAS;
      const axes = [0, 1, 2].map((k) => { const a = [m[k], m[4 + k], m[8 + k]]; const n = Math.hypot(...a); return a.map((x) => x / n); });
      let checked = 0;
      r.frames.forEach((v, j) => {
        const g = (v.meta?.diffusion as { gradient?: number[] } | undefined)?.gradient;
        if (!g) return;
        const ras = [-g[0], -g[1], g[2]];                       // the scanner's LPS, in RAS
        const img = axes.map((a) => a[0] * ras[0] + a[1] * ras[1] + a[2] * ras[2]);
        const want = [vec[0][refIndex(j)], -vec[1][refIndex(j)], vec[2][refIndex(j)]];
        for (let k = 0; k < 3; k++) assertAlmostEquals(img[k], want[k], 0.002, `volume ${j}, axis ${k}`);
        checked++;
      });
      assertEquals(checked, ours.filter((b) => b! > 0).length, "every directional volume has a direction");
    },
  });
}

// Critic, 2026-09-25, finding 1, on real headers: the Siemens XA30 diffusion series with a second copy of its b=0 file
// (a new SOPInstanceUID and InstanceNumber) had become ONE volume of 72x72x858 at 0.133 mm. It is 22 volumes.
Deno.test({
  name: "public xa30_dwi with a second b=0 file: 22 volumes of 72x72x39, not one merged stack",
  ignore: !have(BASE + "xa30_dwi"),
  fn: async () => {
    const dir = BASE + "xa30_dwi";
    const files = [...Deno.readDirSync(dir)].map((e) => e.name).filter((n) => n.endsWith(".dcm")).sort();
    const bufs = files.map((f) => { const u = Deno.readFileSync(`${dir}/${f}`); return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer; });
    const D = dcmjs.data as unknown as { DicomMessage: { readFile(b: ArrayBuffer): { dict: Record<string, { Value: unknown[] }>; meta: Record<string, { Value: unknown[] }> } }; DicomDict: new (m: unknown) => { dict: unknown; write(): ArrayBuffer } };
    const p = D.DicomMessage.readFile(bufs[0]);
    p.dict["00080018"].Value = ["2.25.99999999999999999999"]; p.meta["00020003"].Value = ["2.25.99999999999999999999"];
    p.dict["00200013"].Value = [22];
    const extra = new D.DicomDict(p.meta); extra.dict = p.dict;
    const r = volumesOfSeries(await parseInstances([...bufs, extra.write()]));
    assertEquals(r.frames.length, 22);
    for (const v of r.frames) assertEquals(v.dims, [72, 72, 39]);
  },
});
