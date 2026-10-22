// THE TRACT TABLE THAT NAMES A CLUSTER, and the two steps around it that every naming network shares (NAMING.md).
//
// Both naming networks -- RapidParc's (the app's and the import job's) and TractCloud's (validation runs only) -- put a
// streamline into one of the 1,600 classes of the O'Donnell group's atlas ORG-800FC-100HCP (800 fiber clusters and an
// outlier twin of each). TRACTCLOUD'S 42-TRACT TABLE (SlicerDMRI/TractCloud tract_mapping.py; RapidParc's own copy is the
// same, checked 2026-10-03) turns a class into one of 42 tracts or Other: tractcloud-42.json here, taken from TractCloud's
// converted release (../tractcloud/model/model.json; tractcloud-42.test.ts keeps the two identical). The same file holds
// the average streamline of TractCloud's training data (ORG-800FC-100HCP's subjects, 15 points), on which a brain is
// centered before the side of a streamline is read.
//
// The table is loaded alone: TractCloud's weights are read only by a run that asks for TractCloud's network.

export const CLUSTERS_NAME = "ORG-800FC-100HCP";
export const TABLE_NAME = "TractCloud-42";

export interface TractInfo { abbr: string; name: string; category: string }
/** What a table holds (TractCloud's model.json holds the same fields, among its tensors). */
export interface TableJson {
  massCenter: number[][];
  tracts: TractInfo[];
  clusterToTract: number[];
  settings: { points: number };
}
export interface TractTable { json: TableJson; P: number }

/** The table from its file (tractcloud-42.json). */
export function loadTable(json: TableJson): TractTable {
  if (!Array.isArray(json.clusterToTract) || json.clusterToTract.length !== 1600) throw new Error("the tract table is not a table of 1,600 classes");
  return { json, P: json.settings.points };
}

/** One streamline (x,y,z,… RAS mm) at `n` points of equal arc length along it -- the original's extract_ras_features. */
export function resample(pts: Float32Array | number[], n = 15): Float64Array {
  const m = pts.length / 3, out = new Float64Array(3 * n);
  if (m < 2) { for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[3 * i + c] = pts[c] ?? 0; return out; }
  const cum = new Float64Array(m);
  for (let i = 1; i < m; i++) cum[i] = cum[i - 1] + Math.hypot(pts[3 * i] - pts[3 * i - 3], pts[3 * i + 1] - pts[3 * i - 2], pts[3 * i + 2] - pts[3 * i - 1]);
  const total = cum[m - 1];
  if (total < 1e-12) { for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[3 * i + c] = pts[c]; return out; }
  let seg = 0;
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * total;
    while (seg < m - 2 && cum[seg + 1] <= t) seg++;             // searchsorted(side="right") - 1, clipped to [0, m-2]
    const len = cum[seg + 1] - cum[seg], f = (t - cum[seg]) / (len < 1e-12 ? 1 : len);
    for (let c = 0; c < 3; c++) out[3 * i + c] = pts[3 * seg + c] + f * (pts[3 * seg + 3 + c] - pts[3 * seg + c]);
  }
  return out;
}

/** All streamlines resampled and moved so their mean (per point) is the atlas's: feat, (N, P, 3) row-major. */
/** `centerOn`: the brain's center is the mean of the first `centerOn` streamlines only (default all) -- so streamlines
 *  added to a whole-brain run are placed where the whole brain puts them (nameAgainst, name-tracts.ts). */
export function prepare(streamlines: (Float32Array | number[])[], model: TractTable, centerOn = streamlines.length): Float64Array {
  const P = model.P, N = streamlines.length, feat = new Float64Array(N * P * 3);
  streamlines.forEach((s, i) => feat.set(resample(s, P), i * P * 3));
  const mean = new Float64Array(P * 3);
  for (let i = 0; i < centerOn; i++) for (let j = 0; j < P * 3; j++) mean[j] += feat[i * P * 3 + j] / centerOn;
  const mc = model.json.massCenter;
  for (let i = 0; i < N; i++) for (let p = 0; p < P; p++) for (let c = 0; c < 3; c++) feat[(i * P + p) * 3 + c] += mc[p][c] - mean[p * 3 + c];
  return feat;
}

/** The tract (index into model.json.tracts) of each cluster. */
export function tractsOf(model: TractTable, clusters: Int32Array): Int32Array {
  const lut = model.json.clusterToTract;
  return clusters.map((c) => lut[Math.max(0, Math.min(lut.length - 1, c))]);
}
