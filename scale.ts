// SIZES IN VOXELS OF THE SCAN, NOT IN MM (Ron, 2026-10-20: "did you forget about scales? from astrophysics to electron
// microscope"; MODULE-DESIGN-RULES "Any scale"). The scene is in the data's own units, so a fixed number of mm is right
// at one scale only. Diffusion MRI exists only at clinical and animal scales, but a grid of 1e-6 mm or 3e19 mm voxels
// must still not hang the module or allocate absurdly, and every method constant is stated in voxels of the scan, with
// the published mm value as the translation on the grid it was published for.
//
// The same two kinds of value as core's render/scene-scale.ts (an extension imports core only through the SDK, which
// does not carry them, so the few lines are here):
//   - a size in voxels: n * voxel;
//   - tuned(mm, voxel, lo, hi): a value published or tuned by eye in mm on clinical grids (UKFTractography's 0.3 mm
//     step, SlicerDMRI's 0.5 mm tube), kept EXACTLY as it is while it lies between `lo` and `hi` voxels, and held to
//     that range outside it -- so a clinical scan gives the same result, bit for bit, and a microscope's or a
//     telescope's grid scales.
// Clinical measurements Ron set in mm (the near-tumor distance, the Within list, the gray band) are not sizes of this
// kind and stay in mm.

/** The three voxel sizes of a grid, from its ijk -> RAS matrix (row-major 4x4), or undefined when it has none. */
export function voxelSizes(ijkToRAS: ArrayLike<number> | undefined): [number, number, number] | undefined {
  if (!ijkToRAS || ijkToRAS.length < 12) return undefined;
  const s = [0, 1, 2].map((c) => Math.hypot(ijkToRAS[c], ijkToRAS[4 + c], ijkToRAS[8 + c]));
  return s.every((x) => Number.isFinite(x) && x > 0) ? [s[0], s[1], s[2]] : undefined;
}

/** A grid's finest voxel size, or undefined. */
export function finestVoxel(ijkToRAS: ArrayLike<number> | undefined): number | undefined {
  const s = voxelSizes(ijkToRAS);
  return s ? Math.min(s[0], s[1], s[2]) : undefined;
}

/** A value tuned in mm on clinical grids, unchanged between `lo` and `hi` voxels, held to that range outside it. */
export function tuned(mm: number, voxel: number | undefined, lo: number, hi: number): number {
  if (voxel === undefined || !(voxel > 0) || !Number.isFinite(voxel)) return mm;
  return Math.min(Math.max(mm, lo * voxel), hi * voxel);
}

/** A length for a line of text: "2 mm", "1.3 mm" in the clinical range, "2.0e-6 mm" far outside it. `fixed` keeps
 *  one decimal ("1.0 mm") as the module's distances are written. */
export function lengthText(mm: number, fixed = false): string {
  if (!Number.isFinite(mm)) return `${mm} mm`;
  const a = Math.abs(mm);
  if (a === 0 || (a >= 0.05 && a < 1e5)) return `${fixed ? mm.toFixed(1) : String(+mm.toFixed(1))} mm`;
  return `${mm.toExponential(1)} mm`;
}

// ── The Diffusion module's own sizes (module.ts) ──────────────────────────────────────────────────────────────────────

/** The drawn radii, in mm on clinical grids: SlicerDMRI's tube radius is 0.5 mm; "lines" are hairlines. */
export const RADIUS = { tubes: 0.5, lines: 0.12 };
/**
 * SIZES IN VOXELS OF THE SCAN, NOT IN MM (Ron, 2026-10-20: "did you forget about scales? from astrophysics to electron
 * microscope"). Each was set by eye, or published, in mm on clinical diffusion grids (2 to 2.5 mm) and is kept
 * exactly there; outside the range of voxels named it is held to that range, so a 1e-6 mm or a 3e19 mm grid draws,
 * probes and seeds at the same size relative to its voxels:
 *   - the tubes' and the lines' radius: 0.1 to 0.5 voxel (tubes) and 0.024 to 0.12 (lines) -- unchanged on 1 to 5 mm;
 *   - the slack the 3D probe allows around a tube, and its step along the ray: 0.5 mm, 0.1 to 0.5 voxel (1 to 5 mm);
 *   - the data probe's reach: 2 mm, 0.5 to 2 voxels (1 to 4 mm);
 *   - the sphere "Seed where I click" fills: 5 mm, 1 to 5 voxels (1 to 5 mm).
 */
export const drawnRadius = (k: keyof typeof RADIUS, voxel: number | undefined) => k === "tubes" ? tuned(RADIUS.tubes, voxel, 0.1, 0.5) : tuned(RADIUS.lines, voxel, 0.024, 0.12);
export const raySlack = (voxel: number | undefined) => tuned(0.5, voxel, 0.1, 0.5);
export const probeReach = (voxel: number | undefined) => tuned(2, voxel, 0.5, 2);
export const seedRadius = (voxel: number | undefined) => tuned(5, voxel, 1, 5);
/** The 3D probe's march along the mouse ray takes at most this many steps, however long the ray is in voxels. */
export const MAX_RAY_STEPS = 4096;
