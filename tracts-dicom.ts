// A WHOLE-BRAIN TRACKING AS ONE DICOM TRACTOGRAPHY RESULTS OBJECT (PS3.3 A.90, Supplement 181; SOP class
// 1.2.840.10008.5.1.4.1.1.66.6), and back. The first piece of tracking at import time (Contents/docs/DMRI-AT-IMPORT.md
// in the workspace; Ron, 2026-10-05: "automatic import", "remake automatically, versioned"): the named tracts of a
// diffusion scan are kept in the DICOM database beside it, so the resident's button only measures them against the tumor.
//
// The layout, decided 2026-10-01 (Ron; dicom-tracts-and-scenes-review-2026-09-30.md): ONE object per run, a Track Set per
// named tract (its name as the label, its side as a laterality modifier, its color as the set's display color), and one
// Track Set "Unnamed" for the streamlines no name fits. Points are millimeters in the patient coordinate system (LPS):
// Albula's streamlines are RAS, so x and y change sign on the way in and out.
//
// Codes, kept deliberately generic until the anatomy codes are settled with Mike Halle (the semantics line, CLAUDE.md):
// every set's anatomical type is Brain (SCT 12738006), with Left (SCT 7771000) or Right (SCT 24028007) as a modifier; the
// tract itself is named by the label. The diffusion model is Multi Tensor (DCM 113232, CID 7261) for two-tensor UKF,
// Single Tensor (113231) otherwise; the algorithm family Deterministic (DCM 113211, CID 7262).
//
// What made the tracts goes into Albula's private block (creator "SlicerAlbula provenance 1", element 0077,1002, as JSON),
// as the caller gives it: the import job writes every rule's version, a fingerprint of the extension's code, the naming
// network's and SynthStrip's versions, the series it was made from and the date (import-job.ts jobRules). That is what
// tells, later, whether the stored tracts are still current.
import { dicomIO, rgbToDicomLab } from "albula";

export const TRACTOGRAPHY_RESULTS = "1.2.840.10008.5.1.4.1.1.66.6";
const EXPLICIT_VR_LE = "1.2.840.10008.1.2.1";
const PRIVATE = { creatorTag: "00770010", creator: "SlicerAlbula provenance 1", provenanceTag: "00771002" } as const;
const BRAIN = { CodeValue: "12738006", CodingSchemeDesignator: "SCT", CodeMeaning: "Brain" };
const SIDE: Record<string, { CodeValue: string; CodingSchemeDesignator: string; CodeMeaning: string }> = {
  "-1": { CodeValue: "7771000", CodingSchemeDesignator: "SCT", CodeMeaning: "Left" },
  "1": { CodeValue: "24028007", CodingSchemeDesignator: "SCT", CodeMeaning: "Right" },
};
const MODEL = {
  multi: { CodeValue: "113232", CodingSchemeDesignator: "DCM", CodeMeaning: "Multi Tensor" },
  single: { CodeValue: "113231", CodingSchemeDesignator: "DCM", CodeMeaning: "Single Tensor" },
};
const DETERMINISTIC = { CodeValue: "113211", CodingSchemeDesignator: "DCM", CodeMeaning: "Deterministic" };
/** Where the direction-colored map stored with a tracts object lives, relative to the database's folder (the import job
 *  writes it, the Tract review reads it; Contents/docs/TRACT-REVIEW.md). */
export const colorFaPath = (tractsSeriesUID: string) => `SlicerAlbula-Cache/colorfa-${tractsSeriesUID}.nrrd`;
/** The b = 0 image beside them (Ron, 2026-10-06: "substantia nigra should be visible on the B0 images, which are heavily
 *  T2 weighted"), on the same grid -- the fit's S0, after every correction. */
export const b0Path = (tractsSeriesUID: string) => `SlicerAlbula-Cache/b0-${tractsSeriesUID}.nrrd`;
/** The Track Set holding the streamlines no tract name fits (Ron, 2026-10-01). */
export const UNNAMED = "Unnamed";

/** One named tract on one side (or the unnamed streamlines): its streamlines in RAS mm, x y z per point. */
export interface TractSetData { label: string; side: -1 | 0 | 1; color: [number, number, number]; streamlines: Float32Array[] }

/** The diffusion series the tracts were made from, as the database knows it: what the new object is filed under. */
export interface TractsSource {
  /** Patient and study attributes copied as they are (PatientName, PatientID, PatientBirthDate, PatientSex, StudyInstanceUID,
   *  StudyDate, StudyTime, StudyID, AccessionNumber, ReferringPhysicianName). */
  patientStudy: Record<string, unknown>;
  frameOfReferenceUID: string;
  seriesInstanceUID: string;
  /** The diffusion series' instances (SOP class and instance UIDs): the images the tracking used. */
  instances: { sopClassUID: string; sopInstanceUID: string }[];
  /** Other series the tracts depend on -- the T1 they were aligned to (their points are on it), the reversed scan --
   *  listed after the diffusion series (critic, 2026-10-05, finding 12). */
  alsoReferenced?: { seriesInstanceUID: string; instances: { sopClassUID: string; sopInstanceUID: string }[] }[];
}

export interface TractsRun {
  /** "UKF two-tensor (Albula's GPU port of UKFTractography)" and the like; and its version (the tracking rule). */
  algorithmName: string; algorithmVersion: string;
  /** The tracker's settings, in words ("seedingThreshold=0.1 stoppingFA=0.08 ..."). */
  algorithmParameters: string;
  model: "multi" | "single";
  /** Everything that made these tracts and decides whether they are still current: rule versions, haversack, commit, date. */
  provenance: Record<string, unknown>;
  seriesDescription?: string; seriesNumber?: number;
  /** Stable UIDs, when the same run must keep its identity. */
  uids?: { series: string; sop: string };
}

const lpsPoints = (p: Float32Array) => { const o = new Float32Array(p.length); for (let k = 0; k < p.length; k += 3) { o[k] = -p[k]; o[k + 1] = -p[k + 1]; o[k + 2] = p[k + 2]; } return o; };
const da = () => { const d = new Date(), z = (n: number, w = 2) => String(n).padStart(w, "0"); return { date: `${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}`, time: `${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}` }; };

/** The tracts as a Part 10 file. Sets without streamlines are left out (a Track Set needs at least one track). */
export async function tractsToDicom(sets: TractSetData[], source: TractsSource, run: TractsRun): Promise<{ bytes: Uint8Array; sopInstanceUID: string; seriesInstanceUID: string; trackSets: number; tracks: number }> {
  const dcm = await dicomIO();
  const sop = run.uids?.sop ?? dcm.newUid(), series = run.uids?.series ?? dcm.newUid(), now = da();
  const used = sets.filter((s) => s.streamlines.some((p) => p.length >= 6));
  if (!used.length) throw new Error("no streamlines to write");
  let tracks = 0;
  const TrackSetSequence = used.map((s, i) => {
    const items = s.streamlines.filter((p) => p.length >= 6).map((p) => ({ PointCoordinatesData: lpsPoints(p).buffer }));
    tracks += items.length;
    const side = SIDE[String(s.side)];
    return {
      TrackSetNumber: i + 1,
      TrackSetLabel: s.label.slice(0, 64),
      TrackSetAnatomicalTypeCodeSequence: [{ ...BRAIN, ...(side ? { ModifierCodeSequence: [side] } : {}) }],
      TrackSequence: items,
      RecommendedDisplayCIELabValue: rgbToDicomLab(s.color),
      DiffusionModelCodeSequence: [MODEL[run.model]],
      TrackingAlgorithmIdentificationSequence: [{
        AlgorithmFamilyCodeSequence: [DETERMINISTIC], AlgorithmName: run.algorithmName.slice(0, 64), AlgorithmVersion: run.algorithmVersion.slice(0, 64),
        AlgorithmParameters: run.algorithmParameters,
      }],
    };
  });
  const refsOf = (list: TractsSource["instances"]) => list.map((r) => ({ ReferencedSOPClassUID: r.sopClassUID, ReferencedSOPInstanceUID: r.sopInstanceUID }));
  const refs = refsOf(source.instances), more = source.alsoReferenced ?? [];
  const ds: Record<string, unknown> = {
    ...source.patientStudy,
    // UTF-8 (ISO_IR 192): the provenance's text and copied names may hold any character (critic, 2026-10-05, finding 4:
    // "°" and "·" under no declared character set failed dciodvfy and read back garbled).
    SpecificCharacterSet: "ISO_IR 192",
    SOPClassUID: TRACTOGRAPHY_RESULTS, SOPInstanceUID: sop,
    // BodyPartExamined: the brain is not a paired structure, so General Series' Laterality (Type 2C) is not required.
    Modality: "MR", BodyPartExamined: "BRAIN", SeriesInstanceUID: series, SeriesNumber: run.seriesNumber ?? 900,
    SeriesDescription: (run.seriesDescription ?? "Fiber tracts (whole brain)").slice(0, 64),
    SeriesDate: now.date, SeriesTime: now.time,
    FrameOfReferenceUID: source.frameOfReferenceUID, PositionReferenceIndicator: "",
    Manufacturer: "SlicerAlbula", ManufacturerModelName: "SlicerAlbula", DeviceSerialNumber: "none",
    SoftwareVersions: `diffusion tracts writer 1; ${run.algorithmName}`.slice(0, 64),
    InstanceNumber: 1, ContentLabel: "TRACTS", ContentDescription: "Whole-brain tractography, named", ContentCreatorName: "SlicerAlbula",
    ContentDate: now.date, ContentTime: now.time, InstanceCreationDate: now.date, InstanceCreationTime: now.time,
    TrackSetSequence,
    // The images the tracking used: in this study, so Referenced Series Sequence (Common Instance Reference module).
    ReferencedSeriesSequence: [{ SeriesInstanceUID: source.seriesInstanceUID, ReferencedInstanceSequence: refs },
      ...more.map((m) => ({ SeriesInstanceUID: m.seriesInstanceUID, ReferencedInstanceSequence: refsOf(m.instances) }))],
    ReferencedInstanceSequence: [...refs, ...more.flatMap((m) => refsOf(m.instances))],
    _meta: {
      MediaStorageSOPClassUID: { Value: [TRACTOGRAPHY_RESULTS], vr: "UI" },
      MediaStorageSOPInstanceUID: { Value: [sop], vr: "UI" },
      TransferSyntaxUID: { Value: [EXPLICIT_VR_LE], vr: "UI" },
    },
  };
  const file = dcm.toFile(ds as never);
  file.dict[PRIVATE.creatorTag] = { vr: "LO", Value: [PRIVATE.creator] };
  file.dict[PRIVATE.provenanceTag] = { vr: "UT", Value: [JSON.stringify(run.provenance)] };
  return { bytes: new Uint8Array(file.write()), sopInstanceUID: sop, seriesInstanceUID: series, trackSets: used.length, tracks };
}

/** A Tractography Results object back into tract sets (RAS mm), with Albula's provenance when it wrote the object. */
export async function dicomToTracts(bytes: Uint8Array): Promise<{ sets: TractSetData[]; provenance?: Record<string, unknown>; sopInstanceUID: string; referencedSeries?: string }> {
  const dcm = await dicomIO();
  const f = dcm.readFile(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
  const ds = dcm.naturalize(f.dict) as Record<string, unknown>;
  if (ds.SOPClassUID !== TRACTOGRAPHY_RESULTS) throw new Error(`not a Tractography Results object (${String(ds.SOPClassUID)})`);
  const raw = (f.dict as Record<string, { Value?: unknown[] }>);
  let provenance: Record<string, unknown> | undefined;
  if (raw[PRIVATE.creatorTag]?.Value?.[0] === PRIVATE.creator) { try { provenance = JSON.parse(String(raw[PRIVATE.provenanceTag]?.Value?.[0] ?? "")); } catch { /* not ours */ } }
  type Item = { TrackSetLabel?: string; TrackSetAnatomicalTypeCodeSequence?: { ModifierCodeSequence?: { CodeValue?: string }[] }[]; TrackSequence?: { PointCoordinatesData?: ArrayBuffer | ArrayBuffer[] }[]; RecommendedDisplayCIELabValue?: number[] };
  const sets = ((ds.TrackSetSequence ?? []) as Item[]).map((s) => {
    const mod = s.TrackSetAnatomicalTypeCodeSequence?.[0]?.ModifierCodeSequence?.[0]?.CodeValue;
    const side: -1 | 0 | 1 = mod === "7771000" ? -1 : mod === "24028007" ? 1 : 0;
    const lab = s.RecommendedDisplayCIELabValue, rgb = lab ? dcm.dicomLabToRgb(lab) : [0.6, 0.6, 0.6];
    const streamlines = (s.TrackSequence ?? []).map((t) => {
      const b = Array.isArray(t.PointCoordinatesData) ? t.PointCoordinatesData[0] : t.PointCoordinatesData;
      return lpsPoints(new Float32Array(b as ArrayBuffer));
    });
    return { label: String(s.TrackSetLabel ?? ""), side, color: [rgb[0], rgb[1], rgb[2]] as [number, number, number], streamlines };
  });
  const rs = (ds.ReferencedSeriesSequence as { SeriesInstanceUID?: string }[] | undefined)?.[0]?.SeriesInstanceUID;
  return { sets, provenance, sopInstanceUID: String(ds.SOPInstanceUID), ...(rs ? { referencedSeries: rs } : {}) };
}
