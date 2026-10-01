// A DIFFUSION SERIES AS ONE ENHANCED MR IMAGE OBJECT (PS3.3 A.36.2), the standard's own form for diffusion: every
// frame carries its position and, in the MR Diffusion macro (C.8.13.5.9), its b-value and gradient direction in
// patient coordinates. Ron, 2026-09-28: "convert to proper dicom" -- the OpenNeuro diffusion cases go into the DICOM
// database like everything else (Contents/docs/dmri-review-2026-09-28.md in the workspace).
//
// What it writes, and why:
//  - one frame per slice per volume, volume-major (all slices of volume 0, then volume 1, ...), 16-bit as stored;
//  - shared: pixel measures, plane orientation, frame anatomy, pixel value transformation, MR frame type;
//  - per frame: frame content (dimension indices), plane position, MR diffusion;
//  - dimensions: the stack's in-stack position, and the diffusion volume index (the frame content's temporal position
//    index is NOT used: diffusion volumes are not time points);
//  - Image Type DERIVED\PRIMARY\DIFFUSION\NONE: these frames were made from a NIfTI file, not taken off the scanner,
//    so the MR acquisition macros the standard requires for ORIGINAL frames (timing, coils, echoes) are not claimed.
// The gradient directions come from logic/diffusion/dwi.ts (convention 1, RAS) and are written in LPS, as the
// standard's patient coordinates are.
import type { DiffusionSeries } from "./dwi.ts";
import { dicomIO } from "albula";

/** Albula's private block (PS3.5 §7.8), shared with the SEG writer's creator: here, the source's own description. */
const PRIVATE = { creatorTag: "00770010", creator: "SlicerAlbula provenance 1", sidecarTag: "00771002" } as const;

export interface DwiExportSubject {
  patientName: string;
  patientID: string;
  studyInstanceUID: string;
  frameOfReferenceUID: string;
  studyDescription?: string;
  comments?: string;
  /** Further patient/study attributes as the source states them (sex, age, attribution). */
  extra?: Record<string, unknown>;
  /** The study's date and time, the SAME for every series of the study (critic 2026-09-28, finding 3: each writer read
   *  its own clock). "" is allowed (Type 2: unknown, as for a BIDS dataset, which keeps no dates). Omitted: now. */
  studyDate?: string;
  studyTime?: string;
}

const ENHANCED_MR = "1.2.840.10008.5.1.4.1.1.4.1";
const EXPLICIT_VR_LE = "1.2.840.10008.1.2.1";
const lps = (v: number[]) => [-v[0], -v[1], v[2]];
const unit = (v: number[]) => { const l = Math.hypot(v[0], v[1], v[2]); return v.map((x) => x / l); };
const ds10 = (x: number) => Number(x.toPrecision(10));   // DS holds 16 characters

/** The Part-10 bytes of one Enhanced MR Image object holding every volume of `dwi`. */
export async function diffusionToEnhancedMR(
  dwi: DiffusionSeries,
  subject: DwiExportSubject,
  opts: {
    seriesDescription?: string; seriesNumber?: number;
    /** The source's own description of the acquisition (a BIDS sidecar's JSON text), kept whole in the private block. */
    sourceDescription?: string;
    /** Applicable Safety Standard Agency (0018,9174): IEC, FDA or MHW -- required in every Enhanced MR object. It describes
     *  the acquisition, so the caller states it and why (the writer does not guess). */
    safetyStandardAgency: "IEC" | "FDA" | "MHW";
    /** Stable UIDs, for an import that must give the object the same identity on every run (logic/import/bids.ts). */
    uids?: { series: string; sop: string };
  },
): Promise<{ bytes: Uint8Array; sopInstanceUID: string; seriesInstanceUID: string; frames: number }> {
  // Required in every Enhanced MR object, and a property of the acquisition the caller must state (critic 2026-09-28,
  // finding 8: a missing value used to vanish silently from the file).
  if (!["IEC", "FDA", "MHW"].includes(opts.safetyStandardAgency as string)) {
    throw new Error("the Applicable Safety Standard Agency is required for a diffusion object (IEC, FDA or MHW): say which, and why");
  }
  const dcm = await dicomIO();
  const [nx, ny, nz] = dwi.volumes[0].dims;
  const nv = dwi.volumes.length;
  const M = dwi.ijkToRAS;
  // Columns of the grid: i (along a row), j (down the rows), k (slice to slice).
  const col = (c: number) => [M[c], M[4 + c], M[8 + c]];
  const si = Math.hypot(...col(0)), sj = Math.hypot(...col(1)), sk = Math.hypot(...col(2));
  const iop = [...lps(unit(col(0))), ...lps(unit(col(1)))].map(ds10);
  const pos = (k: number) => lps([M[3] + k * M[2], M[7] + k * M[6], M[11] + k * M[10]]).map(ds10);

  // Pixel data: stored integers only (a diffusion scan is 16-bit as acquired); refuse anything else rather than round.
  // Signed when any value is negative; unsigned when any is above 32767 and none negative; both at once cannot be stored.
  let lo = Infinity, hi = -Infinity;
  for (const v of dwi.volumes) for (let i = 0; i < v.data.length; i++) {
    const x = v.data[i];
    if (!Number.isInteger(x)) throw new Error(`a diffusion volume holds a fractional value (${x}); DICOM pixels are whole numbers`);
    if (x < lo) lo = x; if (x > hi) hi = x;
  }
  const signed = lo < 0;
  if (signed ? (lo < -32768 || hi > 32767) : hi > 65535) throw new Error(`the values run from ${lo} to ${hi}, past what a 16-bit DICOM pixel holds`);
  const n3 = nx * ny * nz;
  const px = signed ? new Int16Array(n3 * nv) : new Uint16Array(n3 * nv);
  dwi.volumes.forEach((v, t) => px.set(v.data as ArrayLike<number>, t * n3));

  const now = new Date();
  const p2 = (n: number) => String(n).padStart(2, "0");
  const date = `${now.getFullYear()}${p2(now.getMonth() + 1)}${p2(now.getDate())}`;
  const time = `${p2(now.getHours())}${p2(now.getMinutes())}${p2(now.getSeconds())}`;
  const sop = opts.uids?.sop ?? dcm.newUid(), series = opts.uids?.series ?? dcm.newUid(), dimOrg = dcm.newUid();

  const perFrame: Record<string, unknown>[] = [];
  for (let t = 0; t < nv; t++) {
    const b = dwi.bValues[t];
    const g = dwi.gradients[t];
    const directional = b > 0 && Math.hypot(g[0], g[1], g[2]) > 0.5;
    // NONE only for b = 0; a b > 0 volume without a direction is a trace (ISOTROPIC) image (critic, finding 9).
    const diffusion: Record<string, unknown> = {
      DiffusionBValue: ds10(b),
      DiffusionDirectionality: directional ? "DIRECTIONAL" : b > 0 ? "ISOTROPIC" : "NONE",
      ...(directional ? { DiffusionGradientDirectionSequence: [{ DiffusionGradientOrientation: lps(g).map(ds10) }] } : {}),
    };
    for (let k = 0; k < nz; k++) {
      perFrame.push({
        FrameContentSequence: [{ DimensionIndexValues: [k + 1, t + 1], StackID: "1", InStackPositionNumber: k + 1 }],
        PlanePositionSequence: [{ ImagePositionPatient: pos(k) }],
        MRDiffusionSequence: [diffusion],
      });
    }
  }

  const s = subject;
  const ds: Record<string, unknown> = {
    // SOP Common, Patient, General Study, Patient Study, General Series, MR Series, Frame of Reference.
    SpecificCharacterSet: "ISO_IR 192",
    SOPClassUID: ENHANCED_MR, SOPInstanceUID: sop,
    PatientName: s.patientName, PatientID: s.patientID, PatientBirthDate: "", PatientSex: "",
    ...(s.comments ? { PatientComments: s.comments } : {}),
    StudyInstanceUID: s.studyInstanceUID, StudyDate: s.studyDate ?? date, StudyTime: s.studyTime ?? time, ReferringPhysicianName: "", StudyID: "1",
    AccessionNumber: "", StudyDescription: s.studyDescription ?? "",
    Modality: "MR", SeriesInstanceUID: series, SeriesNumber: opts.seriesNumber ?? 2,
    SeriesDescription: opts.seriesDescription ?? "Diffusion", SeriesDate: date, SeriesTime: time,
    FrameOfReferenceUID: s.frameOfReferenceUID, PositionReferenceIndicator: "",
    // Enhanced General Equipment: this application wrote the object.
    Manufacturer: "SlicerAlbula", ManufacturerModelName: "SlicerAlbula", DeviceSerialNumber: "none", SoftwareVersions: "diffusion writer 1",
    // Enhanced MR Image, Image Pixel.
    ImageType: ["DERIVED", "PRIMARY", "DIFFUSION", "NONE"],
    // Content date/time: when these pixels were made (now). No AcquisitionDateTime or AcquisitionDuration: they are
    // required only for ORIGINAL frames, and the import's clock is not the scan's (critic, finding 3).
    ContentDate: date, ContentTime: time,
    InstanceNumber: 1, BurnedInAnnotation: "NO", LossyImageCompression: "00", PresentationLUTShape: "IDENTITY",
    ContentQualification: "RESEARCH", ResonantNucleus: "1H", KSpaceFiltering: "NONE", ImageComments: dwi.source,
    // The Enhanced MR Image module's image-level frame description (Type 1), the same as every frame's (dciodvfy).
    PixelPresentation: "MONOCHROME", VolumetricProperties: "VOLUME", VolumeBasedCalculationTechnique: "NONE",
    ComplexImageComponent: "MAGNITUDE", AcquisitionContrast: "DIFFUSION",
    PatientPosition: "", ApplicableSafetyStandardAgency: opts.safetyStandardAgency,
    SamplesPerPixel: 1, PhotometricInterpretation: "MONOCHROME2", Rows: ny, Columns: nx,
    BitsAllocated: 16, BitsStored: 16, HighBit: 15, PixelRepresentation: signed ? 1 : 0,
    NumberOfFrames: nz * nv,
    // Multi-frame Dimension: in-stack position, then the diffusion volume (its index is the second value).
    DimensionOrganizationSequence: [{ DimensionOrganizationUID: dimOrg }],
    // No DimensionOrganizationType: "3D" means one spatial volume, and this object holds many at the same positions
    // (critic, finding 10). It is Type 3.
    DimensionIndexSequence: [
      { DimensionOrganizationUID: dimOrg, DimensionIndexPointer: 0x00209057, FunctionalGroupPointer: 0x00209111, DimensionDescriptionLabel: "slice" },
      { DimensionOrganizationUID: dimOrg, DimensionIndexPointer: 0x00189087, FunctionalGroupPointer: 0x00189117, DimensionDescriptionLabel: "diffusion volume" },
    ],
    AcquisitionContextSequence: [],
    SharedFunctionalGroupsSequence: [{
      PixelMeasuresSequence: [{ PixelSpacing: [ds10(sj), ds10(si)], SliceThickness: ds10(sk), SpacingBetweenSlices: ds10(sk) }],
      PlaneOrientationSequence: [{ ImageOrientationPatient: iop }],
      FrameAnatomySequence: [{
        AnatomicRegionSequence: [{ CodeValue: "12738006", CodingSchemeDesignator: "SCT", CodeMeaning: "Brain" }],
        FrameLaterality: "U",
      }],
      PixelValueTransformationSequence: [{ RescaleIntercept: 0, RescaleSlope: 1, RescaleType: "US" }],
      MRImageFrameTypeSequence: [{
        FrameType: ["DERIVED", "PRIMARY", "DIFFUSION", "NONE"], PixelPresentation: "MONOCHROME",
        VolumetricProperties: "VOLUME", VolumeBasedCalculationTechnique: "NONE", ComplexImageComponent: "MAGNITUDE",
        AcquisitionContrast: "DIFFUSION",
      }],
    }],
    PerFrameFunctionalGroupsSequence: perFrame,
    ...(s.extra ?? {}),
    PixelData: [px.buffer],
    _vrMap: { PixelData: "OW" },
    _meta: {
      MediaStorageSOPClassUID: { Value: [ENHANCED_MR], vr: "UI" },
      MediaStorageSOPInstanceUID: { Value: [sop], vr: "UI" },
      TransferSyntaxUID: { Value: [EXPLICIT_VR_LE], vr: "UI" },
    },
  };
  const file = dcm.toFile(ds);
  if (opts.sourceDescription) {
    file.dict[PRIVATE.creatorTag] = { vr: "LO", Value: [PRIVATE.creator] };
    file.dict[PRIVATE.sidecarTag] = { vr: "UT", Value: [opts.sourceDescription] };
  }
  const bytes = new Uint8Array(file.write());
  return { bytes, sopInstanceUID: sop, seriesInstanceUID: series, frames: nz * nv };
}
