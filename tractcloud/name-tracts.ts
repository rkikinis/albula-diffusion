// NAMING A WHOLE-BRAIN TRACTOGRAPHY WITH TRACTCLOUD: streamlines in (RAS mm), a tract per streamline out.
// Short streamlines (under 40 mm) are left out, both from naming and as context: the atlas the network learned from
// has none (measured on PAT16: keeping them doubles the share called an outlier). Several draws of the random context
// can vote; each draw is seeded, so the same streamlines always get the same names.
import { draw, localNeighbors, prepare, tractsOf, type TractCloudModel } from "./tractcloud.ts";
import { contexts, tractCloudGpu } from "./tractcloud-gpu.ts";
import { CONTEXT, groupRows, normalizeCube, resampleByIndex, shuffle, type RapidParcModel } from "../rapidparc/rapidparc.ts";
import { rapidParcGpu } from "../rapidparc/rapidparc-gpu.ts";

export const MIN_LENGTH_MM = 40;
/** A streamline's tract: an index into model.json.tracts, or SHORT. */
export const SHORT = -1;

/** side: +1 right, -1 left, 0 none (the commissures and the middle cerebellar peduncle cross the midline). */
export interface Named { tract: Int32Array; side: Int8Array; draws: number; seconds: number;
  /** How sure each name is (the first draw): its tract's probability minus the best other tract's (tractcloud-gpu.ts margins); NaN for a short streamline. */
  margin?: Float32Array }
/** Tracts without a side. */
const NO_SIDE = new Set(["MCP"]);

const lengthOf = (s: Float32Array) => { let L = 0; for (let i = 3; i < s.length; i += 3) L += Math.hypot(s[i] - s[i - 3], s[i + 1] - s[i - 2], s[i + 2] - s[i - 1]); return L; };

/** Who names: RapidParc when `rapidParc` is given (the default since 2026-10-03, Ron: "2 yes"), else TractCloud. Both name
 *  the same 1,600 clusters with the same table (`model`, TractCloud's model.json), so everything after the clusters is
 *  shared: the 40 mm cut, the side, the draws' vote, the margin. */
export interface NameOptions { draws?: number; seed?: number; rapidParc?: RapidParcModel }

export async function nameTracts(device: GPUDevice, model: TractCloudModel, streamlines: Float32Array[], opts: NameOptions = {}): Promise<Named> {
  return await nameAgainst(device, model, streamlines, [], opts).then((r) => r.context);
}

/**
 * STREAMLINES ADDED TO A WHOLE-BRAIN RUN, NAMED AS THAT RUN WOULD NAME THEM (Ron, 2026-10-01: a second, denser run in
 * the tracts the user picks). TractCloud names a streamline from its context -- its nearest streamlines and a random
 * draw from the whole brain -- and centers the brain on the mean of what it is given; a run seeded densely in one
 * region would be both mis-centered and its own context. So the added streamlines take the context run's center and
 * draw their context from the context run only. Names for `context` come back too, and are exactly nameTracts' (the
 * draws do not depend on `added`); an added copy of a context streamline gets that streamline's name (name-tracts.test.ts).
 */
export async function nameAgainst(device: GPUDevice, model: TractCloudModel, context: Float32Array[], added: Float32Array[], opts: NameOptions = {}): Promise<{ context: Named; added: Named }> {
  const t0 = performance.now(), draws = opts.draws ?? 1, seed = opts.seed ?? 20260930;
  const kept = (set: Float32Array[]) => set.map((s, i) => [s, i] as const).filter(([s]) => lengthOf(s) >= MIN_LENGTH_MM);
  const keepC = kept(context), keepA = kept(added), Nc = keepC.length;
  const out = (n: number): Named => ({ tract: new Int32Array(n).fill(SHORT), side: new Int8Array(n), draws, seconds: 0, margin: new Float32Array(n).fill(NaN) });
  const rc = out(context.length), ra = out(added.length);
  const done = () => { rc.seconds = ra.seconds = (performance.now() - t0) / 1000; return { context: rc, added: ra }; };
  if (Nc === 0 || (!opts.rapidParc && Nc < model.json.settings.k + 1)) return done();   // TractCloud needs k + 1; RapidParc names any number
  // Context first, the added after it; every row's index in the result it belongs to.
  const keep = [...keepC.map(([s, i]) => [s, i, rc] as const), ...keepA.map(([s, i]) => [s, i, ra] as const)];
  const feat = prepare(keep.map(([s]) => s), model, Nc), feat32 = Float32Array.from(feat), N = keep.length;
  const T = model.json.tracts.length, votes = new Int32Array(N * T), first = new Int32Array(N);
  const tally = (d: number, clusters: Int32Array, mg: Float32Array) => {
    const t = tractsOf(model, clusters);
    for (let i = 0; i < N; i++) { votes[i * T + t[i]]++; if (d === 0) { first[i] = t[i]; const [, at, r] = keep[i]; r.margin![at] = mg[i]; } }
  };
  if (opts.rapidParc) {
    // RAPIDPARC (rapidparc/): every streamline is named inside a group of 2,000 whose members are its context. The
    // whole-brain run's streamlines are shuffled into groups of their own (so their names are exactly nameTracts'); the
    // added ones go 500 to a group, the other 1,500 drawn from the whole-brain run -- a group of one dense region alone
    // would be scaled to that region and lose the brain around it.
    const pts = new Float32Array(N * 45); keep.forEach(([s], i) => pts.set(resampleByIndex(s), i * 45));
    const x = normalizeCube(pts), Na = N - Nc, PER = 500;
    const gpu = rapidParcGpu(device, opts.rapidParc, Int32Array.from(model.json.clusterToTract));
    try {
      for (let d = 0; d < draws; d++) {
        const clusters = new Int32Array(N), mg = new Float32Array(N);
        const perm = shuffle(Nc, seed + d), g = groupRows(x, perm), res = await gpu.classify(g.rows, g.groups);
        for (let r = 0; r < Nc; r++) { clusters[perm[r]] = res.cluster[r]; mg[perm[r]] = res.margin[r]; }
        if (Na) {
          const fill = shuffle(Nc, seed + d + 7919), groups = Math.ceil(Na / PER), rows = new Float32Array(groups * CONTEXT * 45);
          let f = 0;
          for (let gi = 0; gi < groups; gi++) for (let r = 0; r < CONTEXT; r++) {
            const a = gi * PER + r, src = r < PER && a < Na ? Nc + a : fill[f++ % Nc];
            rows.set(x.subarray(src * 45, src * 45 + 45), (gi * CONTEXT + r) * 45);
          }
          const ra = await gpu.classify(rows, groups);
          for (let a = 0; a < Na; a++) { const row = Math.floor(a / PER) * CONTEXT + (a % PER); clusters[Nc + a] = ra.cluster[row]; mg[Nc + a] = ra.margin[row]; }
        }
        tally(d, clusters, mg);
      }
    } finally { gpu.destroy(); }
  } else {
    const gpu = tractCloudGpu(device, model);
    try {
      for (let d = 0; d < draws; d++) {
        const { ds, glob } = draw(Nc, model, seed + d);
        const clusters = await gpu.classify(feat32, contexts(localNeighbors(feat, ds, model), ds, glob, model.json.settings.k));
        tally(d, clusters, gpu.margins());
      }
    } finally { gpu.destroy(); }
  }
  for (let i = 0; i < N; i++) {
    let best = first[i];                                  // a tie keeps the first draw's answer
    for (let c = 0; c < T; c++) if (votes[i * T + c] > votes[i * T + best]) best = c;
    const [, at, r] = keep[i];
    r.tract[at] = best;
    // THE SIDE, from where the streamline lies once the brain is centered on the atlas (RAS: +x is the patient's right).
    const info = model.json.tracts[best];
    if (info.category !== "Commissural" && !NO_SIDE.has(info.abbr)) {
      let x = 0; for (let p = 0; p < model.P; p++) x += feat[(i * model.P + p) * 3];
      r.side[at] = x > 0 ? 1 : -1;
    }
  }
  return done();
}
