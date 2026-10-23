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
  Macdonald & Ruthotto 2016). Checked against FSL's topup (FSL 6.0.7.23, a check tool only) on the development
  cases where the head held still between the two scans: the fields correlate 0.93-0.97, differ by 0.30-0.39 mm at the
  median and 2.0-2.9 mm at the 99th percentile -- as Mike Halle's own version does -- in about 1 s against topup's 190 s
  (`distortion-topup.test.ts`). Motion between the two scans is not modeled; topup models it, and on PAT03 (1.8°) the
  two fields part.
- **Tracking**: UKF two-tensor free-water tractography — a port of UKFTractography (Rathi and colleagues, Brigham and
  Women's Hospital; `LICENSE-UKF.txt`) to TypeScript and to WebGPU. The graphics-card version is checked against the
  processor version on PAT16 (measured: 90% of fiber ends within 0.06 mm of the processor version's, tract-density maps
  correlating 0.97; the test requires 2 mm and 0.95), on a synthetic
  crossing, and for anatomical plausibility on PAT16. Against the original UKFTractography (v2.1, built from source;
  `Contents/tools/ukf-reference.ts` in the workspace) from the same starting points on PAT16, paired by seed point: all
  838 of the original's fibers pair up, and 828 (99%) end within 0.1 mm of the port's (802 within 0.001 mm; the worst
  13.3 mm); the port makes no fiber the original does not, since it computes a seed's FA as the original does for the
  simple model (the mean of the two minor eigenvalues). The filter's limits (w in [0, 1], eigenvalues ≥ 0) are kept by
  the EXACT projection in the filter's metric (2026-10-02; it was 815 with an approximate one: the critic,
  `qa/2026-10-01-ukf-port-vs-original.md`, finding 1). The last 10 differ because the original's own solver (QuadProg++,
  LGPL-3, not ported) is not always exact (finding 5). Plain two-tensor mode (no free water) follows the original's
  Simple2T — unconstrained, eigenvalue floor 100 — and matches it: 822 of 822 fibers within 0.1 mm, the worst 0.01 mm.
  The graphics-card version, which the app runs, has the same exact projection and agrees with the original on 741 of
  838 within 0.1 mm (804 within 1 mm; it was 733 and 794): the rest is its 32-bit arithmetic. Single-tensor tracking too.
- **Tract names**: RapidParc's network (Bisten, Schultz et al., Imaging Neuroscience 2026; University of Bonn, BSD
  license) puts each streamline into a fiber cluster of the O'Donnell group's atlas ORG-800FC-100HCP (Zhang, O'Donnell et al.,
  NeuroImage 2018), and TractCloud's 42-tract table names it: 42 tracts, left and right. TractCloud (Xue … O'Donnell, MICCAI 2023),
  a second network trained on the same atlas, is ported too (written from the paper as a WebGPU network; on 5,120 UKF
  streamlines of PAT16 every layer agrees with the original in PyTorch within 4e-6, every cluster the same) and named the
  tracts until 2026-10-03; it now runs only in validation runs that ask for it. Which tool does what, and how to read a
  tract set's record: `NAMING.md`.
- **Fiber distributions and parallel transport tracking** (the module's "Smooth curves" under Advanced; two-tensor
  stays the default): multi-shell multi-tissue CSD (`csd.ts`; Jeurissen et al. 2014, Tournier et al. 2007), solved
  exactly as the non-negative least squares of its dual (Lawson & Hanson) — on PAT16's 6,770 reference voxels its fit
  equals DIPY's within 0.1% in 6,742, is better in 28 and worse in none; the response functions estimated from the scan
  as DIPY does (`responses.ts`, within 4% of DIPY's); a whole brain in about 105 s on the processor's workers
  (`csd-volume.ts`). Parallel transport tractography (`ptt.ts`; Aydogan & Shi 2021) on the processor's workers, each
  seed with its own seeded generator. Compared with UKF on 15 cases: about as many streamlines inside meningiomas
  (where none belong), a few more named tracts near gliomas, about three times the time.
- **How the whole brain is tracked** is a numbered rule (`tracking-rules.ts`). Rule 3, the default since 2026-10-04:
  Mike Halle's tractline settings (plain two-tensor UKF on the shell nearest b = 3000, a seed in every brain voxel, the
  ORG atlas's thresholds) inside the brain as SynthStrip (Hoopes et al. 2022) finds it on the MRI of the anatomy, asked
  of the haversack server through the SDK; the diffusion scan is aligned to that MRI first (`registration.ts`). Rule 2
  is the same inside DIPY's median_otsu mask of the diffusion scan, which leaves out the lower temporal lobes and part
  of the cerebellum; the module falls back to it, and says so, without an MRI of the anatomy or a server. SynthStrip's
  brain holds the fluid around the brain, so rule 3 can follow a cranial nerve through its cistern, and the naming
  network, which knows only the brain's tracts, gives it the nearest tract's name (PAT29's trigeminal nerve, called
  uncinate). Rules 4 and 5 (the fluid left out of the brain) and a test that would take such streamlines out of the
  tracts (`outside-brain.ts`) were tried and are not used: they also took the corpus callosum's edge, parts of tumors,
  or corticospinal fibers running along the medulla.
- **Planning**: tracking through the whole brain, naming, and showing every named tract that comes within a margin of
  a segmented tumor — whole — in its own color, with its closest distance to the tumor; a tract counts as near when at
  least 5 of its streamlines come within the margin (`planning.ts`, MIN_NEAR_STREAMLINES), 6 mm by default since
  2026-10-04 (`NEAR_MM`; 8 before). Listed after them, in gray and hidden: the tracts that come within the next 2 mm
  (`GRAY_BAND_MM`), and those that come within the margin with fewer streamlines; every tract with a side shows the
  other side's count.
  **Add lines** follows many more streamlines in the tracts shown, on both sides: 20 starting points in every voxel they
  pass through (O'Donnell et al. 2017 seeded tumor patients at 20 a voxel), at most 25,000 a press (`denseSeeds`), named
  in the whole-brain run's context (`naming/name-tracts.ts` `nameAgainst`; a copy of a run's streamline gets that
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
- **Tract names on the face** (`tract-info.ts`, version 1): the 42 tract names of TractCloud's table (on the ORG-800FC-100HCP clusters; which tool
  does what: `NAMING.md`) with the abbreviation ("Arcuate
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
| `module.ts` | the Diffusion module: the patient's case and the one button; under Advanced, Maps (b0 corrected, FA, Color FA, distortion, dcm2niix's check), Tracts (near a structure or from a clicked point; two-tensor, smooth curves or single tensor), and the tract list (search, groups, Show / Hide all, Add lines), drawn as tubes in 3D and as dots where they cross the slices; data-probe rows for the tracts under the pointer. |
| assets | `naming/` (TractCloud's 42-tract table), `rapidparc/model/` (13 MB), `template/` (the MNI 2009c head, 13 MB) and `vendor/dcm2niix/`, copied beside the app's bundle (`extension.json` `assets`). Not shipped: `tractcloud/model/` (validation runs only) and `brainstemseg/model/` (not in the app yet). |

Its modules follow the [Albula Module Design Rules](https://github.com/rkikinis/SlicerAlbula/blob/main/Contents/docs/MODULE-DESIGN-RULES.md),
as every module in Albula does; a new extension's modules start from them.

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
and the CSD reference, and the SynthStrip brain masks of ds001226 (`synthstrip-masks.ts`, through haversack). Albula's
`Contents/data/test-data.json` lists each collection and how to make it.

Some comments name documents of the Albula workspace (`Contents/docs/…`) that are not published; they record the
reasoning behind a choice and are not needed to build or run anything.

## Other people's methods in this extension

Each was written here from its source code or its papers, and checked against the original (or, where no program is
the original, against the field's reference implementation) on the same data. "Version" is what the port was written
from and checked against; the weights are pinned by their sha256 in each `model/README.md`. A stored tract set records
the naming network's fingerprint and the fingerprint of all the code that made it (`NAMING.md`, "Reading a tract
set's record"). Measured agreement as of 2026-10-22.

| method (whose) | files | written from, version | license | checked against | measured | test |
|---|---|---|---|---|---|---|
| UKF two-tensor tractography (Rathi and colleagues, BWH) | `ukf.ts`, `ukf-gpu.ts` (card) | UKFTractography commit 2d2b66153970 (v2.1) | UKF agreement (`LICENSE-UKF.txt`) | the original, built from source, same seeds (PAT16) | 828 of 838 fibers end within 0.1 mm (processor); card 741 of 838 | `ukf-reference.test.ts`, `ukf-gpu.test.ts` |
| TractCloud network (O'Donnell group) | `tractcloud/tractcloud.ts`, `tractcloud-gpu.ts` (card) | SlicerDMRI/TractCloud v1.0.0, source 94de627 | 3D Slicer's (`tractcloud/model/LICENSE.txt`) | the original in PyTorch, 5,120 streamlines | every layer within 4e-6, every cluster the same | `tractcloud/tractcloud.test.ts` |
| RapidParc network (Bisten, Schultz et al., Bonn) | `rapidparc/rapidparc.ts`, `rapidparc-gpu.ts` (card) | MedVisBonn/RapidParc source 4d03d3c, weights v1.0.0 | BSD 3-Clause | the original in PyTorch, 4,174 PAT16 streamlines | layer by layer; at most 2 clusters may differ (the test's bound) | `rapidparc/rapidparc.test.ts`, `rapidparc-gpu.test.ts` |
| BrainstemSeg (Vanderbilt) on BundleParc's network (SCIL-VITAL) | `brainstemseg/bundleparc.ts`, `bundleparc-gpu.ts` (card) | MASILab/brainstemseg_singularity 829caaf = scilpy 2810fe2; weights v4.2 | network MIT, weights CC BY 4.0 | the original in PyTorch: double precision block by block; single precision on Prisma's 97 tracts | 1.8e-6 at worst per block; Dice 0.9998 at the lowest | `brainstemseg/bundleparc-gpu.test.ts` |
| Diffusion tensor fit (the standard WLS fit) | `tensor.ts`, `tensor-gpu.ts` (card) | Salvador 2005, Veraart 2013 | ours | DIPY 1.12.1, PAT16's 151,244 brain voxels | FA within 0.0006, direction within 0.02° | `tensor.test.ts`, `tensor-gpu.test.ts` |
| Multi-shell multi-tissue CSD (Jeurissen 2014, Tournier 2007) | `csd.ts`, `responses.ts`, `csd-volume.ts` | the papers; DIPY read as a second description | ours | DIPY, PAT16's 6,770 reference voxels | equal within 0.1% in 6,742, better in 28, worse in none; responses within 4% | `csd.test.ts`, `responses.test.ts` |
| Parallel transport tractography (Aydogan & Shi 2021) | `ptt.ts` | the paper; DIPY's PTTDirectionGetter as a second description | ours | — (compared with UKF on 15 cases) | — | `ptt.test.ts` |
| Distortion correction from a reversed pair (Chang & Fitzpatrick 1992; Ruthotto 2012; Macdonald & Ruthotto 2016) | `distortion.ts` | the papers | ours | FSL topup 6.0.7.23 (a check tool) | fields correlate 0.93-0.97, 0.30-0.39 mm apart at the median | `distortion-topup.test.ts` |
| Head movement between images, after eddy (Andersson & Sotiropoulos 2016) | `motion.ts` | the paper, not eddy's code | ours | FSL eddy (a check tool) | 0.20-0.30 mm | `motion.test.ts` |
| Rigid alignment to the T1 on normalized gradient fields (Haber & Modersitzki 2006), as Mike Halle's tractline | `registration.ts` | the description, not tractline's code | ours | tractline's alignment; BRAINSFit (a check tool), by the workspace's `registration-check.ts` (2026-10-03) | 0.03-0.09 mm from tractline's, 0.18-0.34 mm from BRAINSFit's | `registration.test.ts` (its own properties) |
| median_otsu brain mask (DIPY) | `median-otsu.ts` | DIPY 1.12.1's source (BSD-3) | ours | DIPY 1.12.1 on PAT16 and PAT05 (2026-10-02, the workspace's record) | identical voxel for voxel | `median-otsu.test.ts` (the fast filter against the plain one; the threshold) |

A dash: no number is recorded here; see the file's header and its test. Not ports: SynthStrip (Hoopes et al. 2022)
runs on Mike Halle's haversack server, asked through the SDK; dcm2niix runs as a second opinion (`vendor/dcm2niix/`).

## Licenses and credits

Albula's own code here: Apache 2.0 (`LICENSE`). The UKF port keeps UKFTractography's license (`LICENSE-UKF.txt`),
TractCloud's trained network keeps 3D Slicer's (`tractcloud/model/LICENSE.txt`), RapidParc's weights the BSD 3-Clause
(`rapidparc/model/LICENSE.txt`), BrainstemSeg's weights CC BY 4.0 and BundleParc's network MIT
(`brainstemseg/model/LICENSE-weights.txt`, `LICENSE-network.txt`), the MNI template its own (`template/LICENSE.txt`),
dcm2niix its own (`vendor/dcm2niix/LICENSE.txt`) — see `NOTICE`. The papers each part rests on are in `references.ts` and appear in the
module's Help & Acknowledgment.
