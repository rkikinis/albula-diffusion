# How Albula names fiber tracts: which tool does what

Written 2026-10-22 after Ron found the record unclear ("It's not transparent to me"): a mail to Steve Pieper said our
tracts were named by TractCloud, and the first correction said "TractCloud's table". Every statement below was checked
against the code on that day; the file and line are given.

## In one paragraph

A whole-brain tractography (UKF two-tensor, `ukf.ts`) is **named by RapidParc's network**: it puts every streamline
into one of the 1,600 fiber clusters of the **O'Donnell-group white matter atlas**; the **atlas's own table** turns a
cluster into one of 42 tract names (or "other"); **Albula** then decides the side (left or right). **TractCloud's network is not
run.** TractCloud is a second network trained on the same atlas; Albula ported it first (2026-09-30) and replaced it
with RapidParc on 2026-10-03 (Ron: "2 yes").

## The three sources, and what each contributes

| what | from | used for | where in the code |
|---|---|---|---|
| **RapidParc's network** (Bisten, von Bornhaupt, Grün, Bauer, Rüber, Schultz; Imaging Neuroscience 2026; University of Bonn, BSD license) | `rapidparc/model/rapidparc.safetensors` (sha256 `6a158dad…`) | each streamline's fiber cluster, and how sure it is | `tractcloud/name-tracts.ts` (`nameAgainst`, the `opts.rapidParc` branch, `rapidParcGpu`) |
| **The atlas's table**: 1,600 clusters → 42 tracts and "other", with names and abbreviations (Zhang, Wu, Norton, Rigolo, Rathi, Makris, O'Donnell; NeuroImage 2018) | stored in `tractcloud/model/model.json` (`clusterToTract`, `tracts`); RapidParc's own copy (`mapping_from_800_800_to_43`, `int_to_label`) was checked identical on 2026-10-03 | the tract name of each cluster | `tractcloud/tractcloud.ts` `tractsOf` |
| **The atlas's average position**, used to center a brain on the atlas | `tractcloud/model/model.json` (`massCenter`, from TractCloud's training data) | deciding the SIDE: after centering, a streamline's mean left-right position says left or right | `tractcloud/tractcloud.ts` `prepare`; `name-tracts.ts` after the vote ("THE SIDE") |
| **TractCloud's network** (Xue, Chen, Zhang, Golby, Makris, Rathi, Cai, Zhang, O'Donnell; MICCAI 2023) | `tractcloud/model/weights.f32` | **nothing** in the app or the import job since 2026-10-03; only validation runs that ask for it (`case-run.ts`, `labeler: "tractcloud"`) | `name-tracts.ts`, the `else` branch |

Both networks give a cluster of the same atlas, so everything after the cluster is shared: the 40 mm cut (shorter
streamlines are not named), the table, the side, the vote over draws, the margin.

## Who calls what

- **The import-time job** (tract sets stored in the database): RapidParc, always (`import-job.ts` `makeTracts(...,
  labeler: RapidParcModel)`; `import-job-main.ts` loads `rapidparc.safetensors`).
- **The Diffusion module in the app** (Make tracts, Add lines): RapidParc, always; if its file is missing it stops with
  "the tract-naming model is missing" -- it never falls back to TractCloud (`module.ts`, `rapidParc()`).
- **Validation runs** (`case-run.ts`): RapidParc by default; TractCloud only when a run asks for it.

## Reading a tract set's record

A stored tract set says, in its provenance, `rules.labeler`, for example:
`rapidparc 6a158dad6a0b, table 75b1af321932, tractcloud 5a843c95eb60, template 1f27aabea9f7`

- `rapidparc …` -- the network that named (its weights' fingerprint);
- `table …` -- `model.json`: the atlas's table and average position;
- `tractcloud …` -- TractCloud's weights' fingerprint. They are LOADED with the table (one model object holds both) but
  NOT RUN. Listed because the job loads them -- misleading, and on the list below to change.
- `template …` -- the MNI152 2009c head template of the head-frame step (`head-frame.ts`; added to this field in
  `import-job-main.ts`). It has nothing to do with naming; it is in this field only because the field was extended.

## What others use

- **Steve Pieper's scene of the Prisma scan** (April 2026): UKF tracking and TractCloud naming (Steve, 2026-10-21: "I
  have the results of UKF and TractCloud").
- **Lauren O'Donnell's group**: TractCloud is theirs; RapidParc is Bonn's, trained on their atlas. (Told to Lauren on
  2026-10-05, `email-odonnell-2026-10-05-pipeline-answers.md` in the workspace.)

## Why it was not transparent, and what changes

1. The naming code and the atlas table live in a folder named `tractcloud/`, and the table is in TractCloud's model file.
2. The record lists TractCloud's weights, which are loaded but not run.
3. Comments, test names and the README still say "named by TractCloud" (`module.ts`, `case-run.ts`, `tract-info.ts`,
   `README.md`), from before 2026-10-03.
4. The side decision's use of the atlas's average position from TractCloud's training data was stated nowhere outside
   the code.

Fixed with this page: the comments, test names and README (3). For Ron to decide (1, 2): move the naming code and the
atlas table out of `tractcloud/` into a folder of their own (`naming/`, with `atlas/` for the table), load the table
without TractCloud's weights, and record `atlas <hash>` instead of `tractcloud <hash>`, with the head template in a field
of its own (a new version of the record; older tract sets keep theirs).

*Drafted with Claude (Anthropic) for Ron Kikinis.*
