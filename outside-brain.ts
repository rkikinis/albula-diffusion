// STREAMLINES OUTSIDE THE BRAIN (2026-10-04) -- BUILT, TESTED, AND OFF (OUTSIDE_RULE.on, below): it also caught
// corticospinal fibers running along the medulla. What follows describes what it would do. Tracking rule 3 tracks inside SynthStrip's brain, which holds the CSF
// around it; in PAT29 it followed the left trigeminal nerve through its cistern, and the naming network -- which knows
// only the brain's own tracts -- called it uncinate. Ron: "cranial nerve detection is not a bad thing. The only bad
// thing is mislabeling them." Leaving the fluid out of the tracking mask (rules 4 and 5) took the corpus callosum's edge
// and parts of tumors with it ("treatment worse than the problem"), so the mask stays whole and the NAME changes instead:
// a streamline that runs for a while through the fluid at the brain's edge is not given a tract's name; it is listed as
// "outside the brain, possibly a cranial nerve". (Fan Zhang's cranial-nerve atlases, on the list, could name it later.)
//
// The fluid at the edge: brain voxels whose mean diffusivity says fluid (above FLUID_MD, mm²/s) within EDGE_VOXELS of
// the brain's outside -- the cisterns and the sulci, not the ventricles. A streamline is outside when it CROSSES that
// fluid: at least MIN_RUN consecutive points there (1.8 mm apart under rules 2-3: about 3.6 mm) with brain on both sides
// (at least END_POINTS points before and after). A brain tract does not leave the brain and come back; one that merely
// ENDS in the fluid -- the corticospinal tract at the bottom of the brainstem, a fiber in a sulcus -- keeps its name.
// Measured on PAT29 (dmri-review, 2026-10-04): of the left "uncinate" streamlines reaching back to the pons (the
// trigeminal), 9 of 10 have such a run in their middle and none at an end; of the left corticospinal tract's 349, 103
// have a run of 4 or more, 92 of them at an end. Counted as crossings: with 4 points, 6 of the 10 trigeminal streamlines
// and 70 streamlines in all (0.1%); with 3, all 10, and 154 in all (0.3%), the left corticospinal tract losing 6 of 349
// and no other named tract more than 3. 3 it is.
import type { TensorFit } from "./tensor.ts";
import { outerShell } from "./tracking-rules.ts";

/** Versioned (Ron, 2026-09-25: "modular … and also versioned"). 1: 2026-10-04. OFF (`on: false`) since the same night:
 *  on the twelve it also marked corticospinal streamlines whose lower parts run along the medulla's surface, in voxels
 *  partly fluid (PAT14: 54 of them, left and right; PAT25: 13). Neither FA, nor mean diffusivity, nor the run's angle to
 *  the brain's surface told them from PAT29's trigeminal (dmri-review, 2026-10-04 night). Taking corticospinal fibers
 *  out of the list near a tumor is worse than misnaming a nerve, so the test is kept, tested, and not applied. */
export const OUTSIDE_RULE = { id: 1, on: false, fluidMd: 2.5e-3, edgeVoxels: 3, minRun: 3, endPoints: 2 } as const;

/** The fluid at the brain's edge, on the fit's grid (1 inside): the brain mask the tracking used (seedMask), its outer
 *  shell, and mean diffusivity above the rule's cut-off. */
export function edgeFluid(fit: Pick<TensorFit, "dims" | "md" | "seedMask" | "mask">, rule = OUTSIDE_RULE): Uint8Array {
  const brain = fit.seedMask ?? fit.mask, shell = outerShell(brain, fit.dims, rule.edgeVoxels), out = new Uint8Array(brain.length);
  for (let v = 0; v < brain.length; v++) out[v] = brain[v] && shell[v] && fit.md[v] > rule.fluidMd ? 1 : 0;
  return out;
}

/** For each streamline (RAS mm, x y z per point), 1 when it crosses `region`: at least `minRun` consecutive points in it,
 *  with at least `endPoints` points of the streamline before and after that run. */
export function outsideBrain(sl: Float32Array[], grid: Pick<TensorFit, "dims" | "ijkToRAS">, region: Uint8Array, minRun: number = OUTSIDE_RULE.minRun, endPoints: number = OUTSIDE_RULE.endPoints): Uint8Array {
  const [nx, ny, nz] = grid.dims, M = grid.ijkToRAS;
  const a = M[0], b = M[1], c = M[2], d = M[4], e = M[5], f = M[6], g = M[8], h = M[9], i9 = M[10];
  const det = a * (e * i9 - f * h) - b * (d * i9 - f * g) + c * (d * h - e * g);
  const Ri = [(e * i9 - f * h) / det, (c * h - b * i9) / det, (b * f - c * e) / det, (f * g - d * i9) / det, (a * i9 - c * g) / det, (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det];
  const out = new Uint8Array(sl.length);
  for (let s = 0; s < sl.length; s++) {
    const p = sl[s], n = Math.floor(p.length / 3); let run = 0;
    for (let q = 0; q < n; q++) {
      const k = 3 * q;
      const x = p[k] - M[3], y = p[k + 1] - M[7], z = p[k + 2] - M[11];
      const i = Math.round(Ri[0] * x + Ri[1] * y + Ri[2] * z), j = Math.round(Ri[3] * x + Ri[4] * y + Ri[5] * z), l = Math.round(Ri[6] * x + Ri[7] * y + Ri[8] * z);
      const inside = i >= 0 && j >= 0 && l >= 0 && i < nx && j < ny && l < nz && region[(l * ny + j) * nx + i] === 1;
      if (inside) { run++; continue; }
      // The run ended at point q - 1, with brain after it: a crossing when long enough and not at the start.
      if (run >= minRun && q - run >= endPoints && n - q >= endPoints) { out[s] = 1; break; }
      run = 0;
    }
  }
  return out;
}
