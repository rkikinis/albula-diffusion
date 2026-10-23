# BrainstemSeg's trained network, as Albula reads it

- `bundleparc-brainstemseg.safetensors` -- the 338 tensors (11,211,668 numbers, float32) of BundleParc's network as
  BrainstemSeg trained it: 97 brainstem and deep white-matter pathways (`tracts.json`, in the order of the prompts),
  45 FOD coefficients in, a 176-voxel cube. sha256 `d7be651faac125fa9ee903b8744a927e2b67edae861a2bb418f3633be83ae232`.
- Made by `Contents/tools/brainstemseg-weights.py` (workspace) from the checkpoint `/opt/pipeline/checkpoints/model.ckpt`
  (sha256 `3834f42757cb1c5911f09eb1efb0ba3edf5a0528ee3b924bfd553835033110ef`) inside `brainstemseg_v4_2.sif`
  (Zenodo 10.5281/zenodo.19343085, md5 `61ed4d002f4648e50bc71f78168658f7`), opened with PyTorch's safe loader (nothing in
  the file is run); values unchanged. `tracts.json`: the checkpoint's tract list and settings (epoch, Lightning version).
- The network's design: BundleParc (Théberge, El Yamani, Boré, Rheault, Descoteaux, Jodoin; SCIL-VITAL), MIT
  (`LICENSE-network.txt`); `../bundleparc.ts` and `../bundleparc-gpu.ts` are written from it (scilpy 2810fe2 /
  MASILab/brainstemseg_singularity 829caaf, identical).
- Checked (`../bundleparc-gpu.test.ts`, 2026-10-22): every block against PyTorch in double precision on a 32-voxel cube,
  1.8e-6 of each block's largest value at worst; Prisma dMRI_dir99_AP's 97 tracts against PyTorch's run, Dice 0.9998 at
  the lowest.

License of the weights: CC BY 4.0 (`LICENSE-weights.txt`, with the attribution). When BrainstemSeg publishes new
weights: rerun `brainstemseg-weights.py`, update the sha256 values above, rerun `brainstemseg-dump.py`,
`brainstemseg-reference.py` and `../bundleparc-gpu.test.ts`.
