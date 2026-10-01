# TractCloud's trained network, turned into files the app reads without PyTorch: every tensor as float32 in one file
# (weights.f32) and an index plus everything else the model needs (model.json: the atlas center, the cluster -> tract
# table, the tract names, the settings). Run once when the upstream model changes:
#   uv run --python 3.12 --with torch --with numpy python make-model.py <TractCloud source> <folder with TrainedModel/ and TrainData_800clu800ol/>
# The source's tract_mapping.py is read for the table; nothing from it runs in the app.
import sys, json, hashlib, types, numpy as np, torch
src, data = sys.argv[1], sys.argv[2]
sys.modules["vtk"] = types.ModuleType("vtk"); sys.path.insert(0, src + "/src")
from tractcloud.tract_mapping import TRACT_NAMES, TRACT_FULL_NAMES, TRACT_CATEGORIES, _CLUSTER_TO_TRACT_LUT
pth = data + "/TrainedModel/best_tract_f1_model.pth"; npy = data + "/TrainData_800clu800ol/HCP_mass_center.npy"
sd = torch.load(pth, map_location="cpu", weights_only=True)
index, parts, off = {}, [], 0
for k, v in sd.items():
    if v.dtype != torch.float32: continue                                   # num_batches_tracked (int64) is not used
    a = v.numpy().astype("<f4").ravel(); index[k] = {"offset": off, "shape": list(v.shape)}; parts.append(a); off += a.size
blob = np.concatenate(parts).astype("<f4").tobytes()
open("weights.f32", "wb").write(blob)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({
    "about": "TractCloud (Xue et al., MICCAI 2023), SlicerDMRI/TractCloud release v1.0.0, converted by make-model.py",
    "source": {"model": "https://github.com/SlicerDMRI/TractCloud/releases/download/v1.0.0/TrainedModel.tar.gz",
               "center": "https://github.com/SlicerDMRI/TractCloud/releases/download/v1.0.0/TrainData_800clu800ol.tar.gz",
               "pth_sha256": sha(pth), "center_sha256": sha(npy), "weights_sha256": hashlib.sha256(blob).hexdigest()},
    "settings": {"points": 15, "k": 20, "kGlobal": 80, "kPoint": 5, "kSampleRate": 0.1, "embDims": 1024, "classes": 1600,
                 "bnEps": 1e-5, "leak": 0.2},
    "massCenter": np.load(npy).astype(float).round(6).tolist(),
    "tracts": [{"abbr": n, "name": TRACT_FULL_NAMES[n], "category": next(c for c, l in TRACT_CATEGORIES.items() if n in l)} for n in TRACT_NAMES],
    "clusterToTract": _CLUSTER_TO_TRACT_LUT.tolist(),
    "tensors": index,
}, open("model.json", "w"), separators=(",", ":"))
print(len(index), "tensors,", off, "numbers")
