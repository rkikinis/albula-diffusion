// DIFFUSION IN ONE ENHANCED MR FILE (synthetic, built with dcmjs): one volume per b-value and direction, read through
// the app's DICOM reader with this extension's interpreter registered. The builder is core's
// (logic/readers/dicom-multiframe.test.ts), copied.
import { assertEquals } from "jsr:@std/assert@1";
import { parseInstances, volumesOfSeries } from "albula";
import { dcmjs, setDicomLibrary } from "albula/testing";
import "./hooks.ts";                                         // the diffusion interpreter, as the app registers it
setDicomLibrary(dcmjs);

const ENHANCED_MR = "1.2.840.10008.5.1.4.1.1.4.1";
const SEG = "1.2.840.10008.5.1.4.1.1.66.4";
const uid = () => `2.25.${Math.floor(Math.random() * 1e15)}${Math.floor(Math.random() * 1e15)}`;

interface Frame { z: number; value: number; t?: number; te?: number; b?: number; dir?: [number, number, number] }

/** One Enhanced MR file: nx x ny frames, each filled with its `value`, at (0, 0, z) mm, axial, 0.5 x 0.8 mm pixels. */
function enhancedMr(frames: Frame[], opts: { nx?: number; ny?: number; sopClass?: string; slope?: number } = {}): ArrayBuffer {
  const nx = opts.nx ?? 4, ny = opts.ny ?? 3;
  const px = new Uint16Array(nx * ny * frames.length);
  frames.forEach((f, i) => px.fill(f.value, i * nx * ny, (i + 1) * nx * ny));
  const sop = uid(), sopClass = opts.sopClass ?? ENHANCED_MR;
  const ds: Record<string, unknown> = {
    SOPClassUID: sopClass, SOPInstanceUID: sop, StudyInstanceUID: uid(), SeriesInstanceUID: uid(), FrameOfReferenceUID: uid(),
    Modality: "MR", PatientName: "TEST^MULTIFRAME", PatientID: "TEST-MF", SeriesDescription: "synthetic", InstanceNumber: 1,
    Rows: ny, Columns: nx, NumberOfFrames: frames.length, BitsAllocated: 16, BitsStored: 12, HighBit: 11, PixelRepresentation: 0,
    SamplesPerPixel: 1, PhotometricInterpretation: "MONOCHROME2",
    SharedFunctionalGroupsSequence: [{
      PlaneOrientationSequence: [{ ImageOrientationPatient: [1, 0, 0, 0, 1, 0] }],
      PixelMeasuresSequence: [{ PixelSpacing: [0.8, 0.5], SliceThickness: 2 }],
      ...(opts.slope ? { PixelValueTransformationSequence: [{ RescaleSlope: opts.slope, RescaleIntercept: 0, RescaleType: "US" }] } : {}),
    }],
    PerFrameFunctionalGroupsSequence: frames.map((f) => ({
      PlanePositionSequence: [{ ImagePositionPatient: [0, 0, f.z] }],
      FrameContentSequence: [{ ...(f.t !== undefined ? { TemporalPositionIndex: f.t } : {}), FrameAcquisitionDateTime: `20260925${String(120000 + (f.t ?? 0) * 2).padStart(6, "0")}.000000` }],
      ...(f.te !== undefined ? { MREchoSequence: [{ EffectiveEchoTime: f.te }] } : {}),
      ...(f.b !== undefined ? { MRDiffusionSequence: [{
        DiffusionBValue: f.b, DiffusionDirectionality: f.dir ? "DIRECTIONAL" : "NONE",
        ...(f.dir ? { DiffusionGradientDirectionSequence: [{ DiffusionGradientOrientation: f.dir }] } : {}),
      }] } : {}),
    })),
    PixelData: [px.buffer],
    _meta: {
      MediaStorageSOPClassUID: { Value: [sopClass], vr: "UI" },
      MediaStorageSOPInstanceUID: { Value: [sop], vr: "UI" },
      TransferSyntaxUID: { Value: ["1.2.840.10008.1.2.1"], vr: "UI" },
    },
  };
  return (dcmjs.data as unknown as { datasetToDict(d: unknown): { write(): ArrayBuffer } }).datasetToDict(ds).write();
}

Deno.test("diffusion in one file: one volume per b-value and direction, which the volume keeps", async () => {
  const kinds: { b: number; dir?: [number, number, number] }[] = [{ b: 0 }, { b: 1000, dir: [1, 0, 0] }, { b: 1000, dir: [0, 0.6, 0.8] }];
  const frames: Frame[] = [];
  kinds.forEach((kd, i) => { for (const z of [0, 2]) frames.push({ z, value: 10 * (i + 1), ...kd }); });
  const inst = await parseInstances([enhancedMr(frames)]);
  const { frames: vols, timing } = volumesOfSeries(inst);
  assertEquals(vols.length, 3);
  assertEquals(vols.map((v) => v.data[0]), [10, 20, 30]);
  // in the order the scanner stored them, not sorted
  assertEquals(timing.map((t) => t.label), ["b 0", "b 1000 · 1.00, 0.00, 0.00", "b 1000 · 0.00, 0.60, 0.80"]);
  assertEquals(vols[1].meta?.diffusion, { bValue: 1000, gradient: [1, 0, 0] });
  assertEquals(timing[2].keys?.diffusion?.gradient, [0, 0.6, 0.8]);
});
