# albula-diffusion — diffusion MRI for brain tumor planning, an Albula extension

Research software. Not reviewed or approved by the FDA or any other agency; not for clinical use.
Written with Claude (Anthropic), an AI assistant, under Ron Kikinis's direction.

[Albula](https://github.com/rkikinis/SlicerAlbula) is a viewer for medical images built on Steve Pieper's SlicerLive.
This extension adds diffusion MRI, from the scanner's files to the tracts near a tumor:

- **Reading diffusion data** from every major scanner: the standard DICOM attributes, Siemens (CSA header, mosaics),
  GE, Philips and Canon private fields, Enhanced MR multi-frame files, NIfTI with FSL `.bval`/`.bvec`, NRRD DWI —
  checked against dcm2niix on 27 public scanner sets. dcm2niix itself (WebAssembly) runs on every scan as a second
  opinion.
- **Writing** a diffusion series as one Enhanced MR DICOM object, and importing BIDS datasets (`dwi/`).
- **Maps**: the tensor fit (processor and graphics card), FA and Color FA on the slices; **distortion correction**
  from a reversed phase-encoding pair, written from the papers (Chang & Fitzpatrick 1992; Ruthotto et al. 2012;
  Macdonald & Ruthotto 2016).
- **Tracking**: UKF two-tensor free-water tractography (a port of UKFTractography) on the graphics card, checked
  against its processor version; single-tensor tracking.
- **Tract names**: TractCloud (Xue, Zhang, O'Donnell et al., MICCAI 2023) written from the paper as a WebGPU network,
  identical to the original's output on all 5,120 streamlines of the reference case.
- **Planning**: tracking through the whole brain, naming, and showing every named tract that comes within a margin of
  a segmented tumor — whole — in its own color, with its closest distance to the tumor.

## How it joins Albula

An extension reaches Albula's core only through the SDK (`albula`, core's `sdk/albula.ts`); `boundary.test.ts` fails
on any other path into core, and core never imports an extension. `extension.json` says what it brings:

| entry | what it registers |
|---|---|
| `hooks.ts` | what separates one DICOM volume from the next (the b-value and direction: `diffusion-vendors.ts`), and the BIDS `dwi/` kind (`bids-dwi.ts`). Loaded wherever data is read: the app, the copy writer, the command-line tools. |
| `module.ts` | the Diffusion module: Maps (signal, FA, Color FA, distortion, dcm2niix's check), Tracts (near a structure, from a clicked point; two-tensor or single tensor), In the scene (each tract group with its eye, its color, its distance). |
| assets | `tractcloud/model/` (the trained network, 8.9 MB) and `vendor/dcm2niix/`, copied beside the app's bundle. |

## Building and testing

The extension builds inside an Albula workspace, where it sits at `Contents/extensions/diffusion/` and is listed in
`Contents/extensions/extensions.json`; the app's rebuild bundles it, runs its tests and copies its assets. Its tests
alone:

    deno test -A --no-check --unstable-webgpu --config <workspace>/Contents/src/SlicerLive/deno.jsonc .

Tests that need public data (OpenNeuro ds001226, the dcm2niix test sets, the TractCloud reference) are skipped when the
data is not on disk; the workspace's `Contents/data/test-data.json` lists each collection and how to fetch it.

## Licenses and credits

Albula's own code here: Apache 2.0 (`LICENSE`). The UKF port keeps UKFTractography's license (`LICENSE-UKF.txt`),
TractCloud's trained network keeps 3D Slicer's (`tractcloud/model/LICENSE.txt`), dcm2niix keeps its own
(`vendor/dcm2niix/LICENSE.txt`) — see `NOTICE`. The papers each part rests on are in `references.ts` and appear in the
module's Help & Acknowledgment.
