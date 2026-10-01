# albula-diffusion — diffusion MRI for brain tumor planning, an Albula extension

Research software. Not reviewed or approved by the FDA or any other agency; not for clinical use.
Written with Claude (Anthropic), an AI assistant, under Ron Kikinis's direction.

[Albula](https://github.com/rkikinis/SlicerAlbula) is a viewer for medical images built on Steve Pieper's SlicerLive.
This extension adds diffusion MRI, from the scanner's files to the tracts near a tumor:

- **Reading diffusion DICOM**: the standard attributes, Siemens (CSA header, mosaics), GE, Philips and Canon/Toshiba
  private fields, Enhanced MR multi-frame files — checked against dcm2niix on 27 public scanner sets from those four
  vendors (United Imaging, Hitachi/Fujifilm and Bruker are not among them). dcm2niix itself (WebAssembly) runs as a
  second opinion on every diffusion scan loaded from the DICOM database. NIfTI with FSL `.bval`/`.bvec` and NRRD DWI
  are read by library functions (`dwi.ts`): NIfTI+FSL reaches the app through the BIDS import (`bids-dwi.ts`); the
  app's Load Data does not open either directly yet.
- **Writing** a diffusion series as one Enhanced MR DICOM object, and importing BIDS datasets (`dwi/`).
- **Maps**: the tensor fit (processor and graphics card), FA and Color FA on the slices; **distortion correction**
  from a reversed phase-encoding pair, written from the papers (Chang & Fitzpatrick 1992; Ruthotto et al. 2012;
  Macdonald & Ruthotto 2016).
- **Tracking**: UKF two-tensor free-water tractography — a port of UKFTractography (Rathi and colleagues, Brigham and
  Women's Hospital; `LICENSE-UKF.txt`) to TypeScript and to WebGPU. The graphics-card version is checked against the
  processor version on PAT16 (measured: 90% of fiber ends within 0.06 mm of the processor version's, tract-density maps
  correlating 0.97; the test requires 2 mm and 0.95), on a synthetic
  crossing, and for anatomical plausibility on PAT16. **The port has not yet been compared with the original C++
  program's output.** Single-tensor tracking too.
- **Tract names**: TractCloud (Xue, Zhang, O'Donnell et al., MICCAI 2023) written from the paper as a WebGPU network;
  on the reference case (5,120 UKF streamlines of PAT16) every layer agrees with the original running in PyTorch within
  4e-6, and all 5,120 streamlines get the same cluster as the original (the test allows 0.2%).
- **Fiber distributions and parallel transport tracking** (in the code and the case library; not yet behind the
  module's buttons): multi-shell multi-tissue CSD (`csd.ts`; Jeurissen et al. 2014, Tournier et al. 2007), solved
  exactly as the non-negative least squares of its dual (Lawson & Hanson) — on PAT16's 6,770 reference voxels its fit
  equals DIPY's within 0.1% in 6,742, is better in 28 and worse in none; the response functions estimated from the scan
  as DIPY does (`responses.ts`, within 4% of DIPY's); a whole brain in about 105 s on the processor's workers
  (`csd-volume.ts`). Parallel transport tractography (`ptt.ts`; Aydogan & Shi 2021) on the processor's workers, each
  seed with its own seeded generator. Compared with UKF on 15 cases: about as many streamlines inside meningiomas
  (where none belong), a few more named tracts near gliomas, about three times the time.
- **Planning**: tracking through the whole brain, naming, and showing every named tract that comes within a margin of
  a segmented tumor — whole — in its own color, with its closest distance to the tumor; a tract counts as near when at
  least 5 of its streamlines come within the margin (`planning.ts`, MIN_NEAR_STREAMLINES).

## How it joins Albula

An extension reaches Albula's core only through the SDK (`albula`, core's `sdk/albula.ts`); `boundary.test.ts` fails
on any other path into core, and core never imports an extension. `extension.json` says what it brings:

| entry | what it registers |
|---|---|
| `hooks.ts` | what separates one DICOM volume from the next (the b-value and direction: `diffusion-vendors.ts`), and the BIDS `dwi/` kind (`bids-dwi.ts`). Loaded wherever data is read: the app, the copy writer, the command-line tools. |
| `module.ts` | the Diffusion module: Maps (signal, FA, Color FA, distortion, dcm2niix's check), Tracts (near a structure, from a clicked point; two-tensor or single tensor), In the scene (each tract group with its eye, its color, its distance). |
| assets | `tractcloud/model/` (the trained network, 8.9 MB) and `vendor/dcm2niix/`, copied beside the app's bundle. |

## Building and testing

The extension reaches Albula's core only through its SDK (`sdk/albula.ts`), which is part of Albula's application
source: the SlicerLive branch Albula is built from (`rkikinis/SlicerLive`, Albula's branch — not Steve Pieper's
SlicerLive). It builds inside an Albula workspace, where it sits at `Contents/extensions/diffusion/` beside that source
at `Contents/src/SlicerLive/`, and is listed in `Contents/extensions/extensions.json`; the app's rebuild bundles it,
runs its tests and copies its files. Its tests alone:

    deno test -A --no-check --unstable-webgpu --config <workspace>/Contents/src/SlicerLive/deno.jsonc .

Tests that need data are skipped when it is not on disk. Public: OpenNeuro ds001226 and the dcm2niix test sets. Made
from public data by Albula's own tools: the TractCloud reference (`tractcloud-reference.py`, from PAT16's streamlines)
and the CSD reference. Albula's `Contents/data/test-data.json` lists each collection and how to make it.

Some comments name documents of the Albula workspace (`Contents/docs/…`) that are not published; they record the
reasoning behind a choice and are not needed to build or run anything.

## Licenses and credits

Albula's own code here: Apache 2.0 (`LICENSE`). The UKF port keeps UKFTractography's license (`LICENSE-UKF.txt`),
TractCloud's trained network keeps 3D Slicer's (`tractcloud/model/LICENSE.txt`), dcm2niix keeps its own
(`vendor/dcm2niix/LICENSE.txt`) — see `NOTICE`. The papers each part rests on are in `references.ts` and appear in the
module's Help & Acknowledgment.
