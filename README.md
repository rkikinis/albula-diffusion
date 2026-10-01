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
  crossing, and for anatomical plausibility on PAT16. Against the original UKFTractography (v2.1, built from source;
  `Contents/tools/ukf-reference.ts` in the workspace) from the same starting points on PAT16, paired by seed point: all
  838 of the original's fibers pair up, and 815 (97%) end within 0.1 mm of the port's (the rest part by ordinary
  round-off, mostly where the filter's limits are in force); the port makes no fiber the original does not, since it
  computes a seed's FA as the original does for the simple model (the mean of the two minor eigenvalues). Single-tensor
  tracking too.
- **Tract names**: TractCloud (Xue, Zhang, O'Donnell et al., MICCAI 2023) written from the paper as a WebGPU network;
  on the reference case (5,120 UKF streamlines of PAT16) every layer agrees with the original running in PyTorch within
  4e-6, and all 5,120 streamlines get the same cluster as the original (the test allows 0.2%).
- **Fiber distributions and parallel transport tracking** (the module's "Smooth curves" under Advanced; two-tensor
  stays the default): multi-shell multi-tissue CSD (`csd.ts`; Jeurissen et al. 2014, Tournier et al. 2007), solved
  exactly as the non-negative least squares of its dual (Lawson & Hanson) — on PAT16's 6,770 reference voxels its fit
  equals DIPY's within 0.1% in 6,742, is better in 28 and worse in none; the response functions estimated from the scan
  as DIPY does (`responses.ts`, within 4% of DIPY's); a whole brain in about 105 s on the processor's workers
  (`csd-volume.ts`). Parallel transport tractography (`ptt.ts`; Aydogan & Shi 2021) on the processor's workers, each
  seed with its own seeded generator. Compared with UKF on 15 cases: about as many streamlines inside meningiomas
  (where none belong), a few more named tracts near gliomas, about three times the time.
- **Planning**: tracking through the whole brain, naming, and showing every named tract that comes within a margin of
  a segmented tumor — whole — in its own color, with its closest distance to the tumor; a tract counts as near when at
  least 5 of its streamlines come within the margin (`planning.ts`, MIN_NEAR_STREAMLINES). Tracts that come within the
  margin with fewer are listed after them in gray, hidden; every tract with a side shows the other side's count.
  **Add lines** follows many more streamlines in the tracts shown, on both sides: 20 starting points in every voxel they
  pass through (O'Donnell et al. 2017 seeded tumor patients at 20 a voxel), at most 25,000 a press (`denseSeeds`), named
  in the whole-brain run's context (`tractcloud/name-tracts.ts` `nameAgainst`; a copy of a run's streamline gets that
  streamline's name, 932 of 932 on PAT16). PAT16's right uncinate: 2 lines to 5, 23.1 s.
- **The module, for a neurosurgery resident** (`module.ts`): 1 · the patient's case — diffusion MRI, MRI of the anatomy
  and tumor outline, each ticked when it is there; scans come in through Albula's Load / Save and DICOM database; a
  missing outline is grown from a few Tumor / Not tumor strokes (core's grow from seeds) and saved as an AI result is;
  2 · one button, "Show the fiber tracts near the tumor", and the named tracts it finds; everything else under
  Advanced. Measured in a browser on PAT16: 19.1 s from the button to the list (two-tensor); smooth curves took far
  longer in its first run in a page (491.9 s, measured with the page hidden, which slows it; to be measured again).
  The tract list has a search, the atlas's groups (the corpus callosum together) with Show / Hide all, and one plain
  caveat: a tract thin or missing on the tumor's side may be destroyed, or hidden by edema. After a run the MRI of the
  anatomy is behind the slice views, a dot in the tract's color marks where each shown tract crosses a slice
  (`tract-slice.ts`), and the data probe names the tracts within 2 mm of the pointer (`tract-index.ts`). FA and Color FA
  are drawn over the anatomy at half opacity. Tracts are drawn 2 voxels shorter at each end, where fibers fan out
  (Advanced › Shorten ends; drawing only, the tracts stay whole).
- **Tract names on the face** (`tract-info.ts`, version 1): TractCloud's names with the abbreviation ("Arcuate
  fasciculus, right (AF)"); the tooltip gives the term in Terminologia Neuroanatomica (FIPAT 2017) where there is one
  (36 of 42), and how close it is; clicking a name opens the atlas paper that defines the tract. What each tract is
  for is to be written by a clinician; until then the face says nothing rather than an unchecked claim.
- **Workers** (`extension.json` `workers`): the CSD and PTT workers, bundled by Albula's build for the page.

## How it joins Albula

An extension reaches Albula's core only through the SDK (`albula`, core's `sdk/albula.ts`); `boundary.test.ts` fails
on any other path into core, and core never imports an extension. `extension.json` says what it brings:

| entry | what it registers |
|---|---|
| `hooks.ts` | what separates one DICOM volume from the next (the b-value and direction: `diffusion-vendors.ts`), and the BIDS `dwi/` kind (`bids-dwi.ts`). Loaded wherever data is read: the app, the copy writer, the command-line tools. |
| `module.ts` | the Diffusion module: the patient's case and the one button; under Advanced, Maps (signal, FA, Color FA, distortion, dcm2niix's check), Tracts (near a structure or from a clicked point; two-tensor, smooth curves or single tensor), and the tract list (search, groups, Show / Hide all, Add lines), drawn as tubes in 3D and as dots where they cross the slices; data-probe rows for the tracts under the pointer. |
| assets | `tractcloud/model/` (the trained network, 8.9 MB) and `vendor/dcm2niix/`, copied beside the app's bundle. |

## Building and testing

The extension reaches Albula's core only through its SDK (`sdk/albula.ts`), which is part of Albula's application
source: the SlicerLive branch Albula is built from (`rkikinis/SlicerLive`, Albula's branch — not Steve Pieper's
SlicerLive). It builds inside an Albula workspace, where it sits at `Contents/extensions/diffusion/` beside that source
at `Contents/src/SlicerLive/`, and is listed in `Contents/extensions/extensions.json`; the app's rebuild bundles it,
runs its tests and copies its files. Its tests alone:

    deno test -A --no-check --unstable-webgpu --config <workspace>/Contents/src/SlicerLive/deno.jsonc .

From scratch (Deno 2 is the only tool needed):

    mkdir -p albula/Contents/src albula/Contents/extensions && cd albula
    git clone -b albula https://github.com/rkikinis/SlicerLive.git Contents/src/SlicerLive
    git clone https://github.com/rkikinis/albula-diffusion.git Contents/extensions/diffusion
    cd Contents/extensions/diffusion
    deno test -A --no-check --unstable-webgpu --config ../../src/SlicerLive/deno.jsonc .

The UKF code alone (`ukf.ts`, `ukf-gpu.ts`, and `dwi.ts`, `tensor.ts` they read with) needs only `albula` for two
reader functions; its test against the processor version (`ukf-gpu.test.ts`) runs on OpenNeuro ds001226's PAT16.

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
