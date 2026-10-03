# RapidParc's trained network, as the app reads it

- `rapidparc.safetensors` — the standard weights; `hemiaug.safetensors` — the same network trained with one-sided
  ("hemispheric") augmentation, for brains with lesions and surgery. Both from RapidParc's release v1.0.0
  (https://github.com/MedVisBonn/RapidParc/releases/tag/v1.0.0), unchanged, 6,574,128 bytes each:
  - `rapidparc.safetensors` sha256 `6a158dad6a0b6124946da102b6037a6f7f8a16fb24e7caec15eaf95213e7b793`
  - `hemiaug.safetensors` sha256 `b66dfce6135aac56c3f835264e5ae9630b2497859425ad98cdc6d842c6f94435`
  (the same values RapidParc's package and Mike Halle's tractline check). Read by `../rapidparc.ts` (`readSafetensors`).
- The settings (`rapidparc_args.yaml` in the release): 15 points a streamline, groups of 2,000, 8 encoder layers,
  d_model 128, 1 head, feed-forward 256, classifier 256, 1,600 classes.
- The classes, the cluster → tract table and the tract names are TractCloud's (`../../tractcloud/model/model.json`):
  RapidParc's `mapping_from_800_800_to_43` and `int_to_label` were checked identical to it on 2026-10-03.

License: BSD 3-Clause, Copyright (c) 2026, Visualization and Medical Image Analysis Group, University of Bonn
(`LICENSE.txt`). Paper: von Bornhaupt, Bisten, …, Schultz. RapidParc: A Global-Context Transformer for Parallel, Accurate,
and Lesion-Robust Tractogram Parcellation. Imaging Neuroscience (2026).

When RapidParc publishes new weights: replace the file, update the sha256 above, rerun
`Contents/tools/rapidparc-reference.py` (workspace) and `../rapidparc.test.ts`.
