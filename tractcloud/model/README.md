# TractCloud's trained network, as files the app reads

- `weights.f32` — every tensor of `best_tract_f1_model.pth`, float32 little-endian, one after another.
- `model.json` — where each tensor sits in `weights.f32` and its shape; the atlas center (HCP, 15 points × 3); the
  cluster → tract table (1600 entries: clusters 0–799 and their outlier twins 800–1599, the twins all "Other"); the
  42 tracts + Other with their full names and categories; the settings (15 points, 20 local neighbors from a 10% sample,
  80 global streamlines, 5 point neighbors, 1024 features); the source URLs and SHA-256 of what was converted.
- `make-model.py` — the converter. Rerun it when SlicerDMRI releases a new model; the test in `../tractcloud.test.ts`
  then says whether the port still agrees with the original.

Source: SlicerDMRI/TractCloud, release v1.0.0 (`TrainedModel.tar.gz`, `TrainData_800clu800ol.tar.gz`), source at
commit 94de627. License: 3D Slicer's (BSD style), `LICENSE.txt`. Paper: Xue T, Chen Y, Zhang C, Golby AJ, Makris N,
Rathi Y, Cai W, Zhang F, O'Donnell LJ. TractCloud: registration-free tractography parcellation with a novel local-global
streamline point cloud representation. MICCAI 2023 (https://doi.org/10.1007/978-3-031-43993-3_40).
