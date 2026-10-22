# How Albula names fiber tracts: which tool does what

Written 2026-10-22 after Ron found the record unclear ("It's not transparent to me"): a mail to Steve Pieper said our
tracts were named by TractCloud, and the first correction said "TractCloud's table". Every statement below was checked
against the code on that day; the file and line are given.

## In one paragraph

A whole-brain tractography (UKF two-tensor, `ukf.ts`) is **named by RapidParc's network**: it puts every streamline
into one of the fiber clusters of the O'Donnell group's atlas **ORG-800FC-100HCP** (800 clusters and an outlier twin
of each, 1,600 classes); **TractCloud's 42-tract table** (its `tract_mapping.py`, adopted by RapidParc) turns a cluster
into one of 42 tract names (or "Other") -- NOT the atlas's own tract list, which has 58 deep tracts and 198 superficial
clusters; **Albula** then decides the side (left or right). (Corrected 2026-10-22 afternoon: this page and the mail to
Steve first said "the atlas's own table".) **TractCloud's network is not
run.** TractCloud is a second network trained on the same atlas; Albula ported it first (2026-09-30) and replaced it
with RapidParc on 2026-10-03 (Ron: "2 yes").

## The three sources, and what each contributes

| what | from | used for | where in the code |
|---|---|---|---|
| **RapidParc's network** (Bisten, von Bornhaupt, Grün, Bauer, Rüber, Schultz; Imaging Neuroscience 2026; University of Bonn, BSD license) | `rapidparc/model/rapidparc.safetensors` (sha256 `6a158dad…`) | each streamline's fiber cluster, and how sure it is | `tractcloud/name-tracts.ts` (`nameAgainst`, the `opts.rapidParc` branch, `rapidParcGpu`) |
| **The clusters: ORG-800FC-100HCP** (the O'Donnell group's fiber clustering atlas: 800 clusters from 100 HCP subjects; Zhang, Wu, Norton, Rigolo, Rathi, Makris, O'Donnell; NeuroImage 2018; github.com/SlicerDMRI/ORG-Atlases) | what both networks were trained to give | the class each streamline is put in | the networks' outputs; the name `CLUSTERS_NAME` in `naming/table.ts` |
| **TractCloud's 42-tract table**: 1,600 classes → 42 tracts and "Other", with names, abbreviations and categories | `naming/tractcloud-42.json`, taken unchanged from TractCloud's converted release (SlicerDMRI/TractCloud `tract_mapping.py`; `naming/tractcloud-42.test.ts` keeps them identical); RapidParc's own copy (`mapping_from_800_800_to_43`, `int_to_label`) checked identical on 2026-10-03 | the tract name of each cluster | `naming/table.ts` `tractsOf` |
| **The training data's average position** (TractCloud's release, the ORG-800FC-100HCP subjects), used to center a brain | `naming/tractcloud-42.json` (`massCenter`) | deciding the SIDE: after centering, a streamline's mean left-right position says left or right | `naming/table.ts` `prepare`; `naming/name-tracts.ts` after the vote ("THE SIDE") |
| **TractCloud's network** (Xue, Chen, Zhang, Golby, Makris, Rathi, Cai, Zhang, O'Donnell; MICCAI 2023) | `tractcloud/model/weights.f32` (with `model.json`) | **nothing** in the app or the import job since 2026-10-03, and since 2026-10-22 not even loaded there; only validation runs that ask for it (`case-run.ts`, `labeler: "tractcloud"`) | `naming/name-tracts.ts`, the TractCloud branch |

Both networks give a cluster of the same atlas, so everything after the cluster is shared: the 40 mm cut (shorter
streamlines are not named), the table, the side, the vote over draws, the margin.

## Who calls what

- **The import-time job** (tract sets stored in the database): RapidParc, always (`import-job.ts` `makeTracts(...,
  labeler: RapidParcModel)`; `import-job-main.ts` loads `rapidparc.safetensors`).
- **The Diffusion module in the app** (Make tracts, Add lines): RapidParc, always; if its file is missing it stops with
  "the tract-naming model is missing" -- it never falls back to TractCloud (`module.ts`, `rapidParc()`).
- **Validation runs** (`case-run.ts`): RapidParc by default; TractCloud only when a run asks for it.

## Reading a tract set's record

Since 2026-10-22 a stored tract set's provenance says, in `rules`:

- `labeler`: `rapidparc 6a158dad6a0b; clusters ORG-800FC-100HCP; tract table TractCloud-42 017cd884c8b5` -- the network
  that named (its weights' fingerprint), the clusters it gives, the table that names them (its file's fingerprint);
- `headTemplate`: the MNI152 2009c head template of the head-frame step (`head-frame.ts`), in a field of its own;
- `code`: the fingerprint of the code that makes the tracts, without comments and file names (`code-fingerprint.ts`).

and, outside `rules`, `madeWith`: the commits of Albula's core and of this extension that ran -- said, not compared.

Before 2026-10-22 the same field read `rapidparc …, table …, tractcloud …, template …`: `table` was TractCloud's
`model.json` (the same table), `tractcloud` its network's weights -- loaded with the table but NOT run -- and `template`
the head template. The job reads such a record through `upgradeRules` (`import-job.ts`), which accepts it as today's
only where the network and the table are the same; `core` and `extension` were then part of `rules`.

## When stored tracts count as out of date

THE SERVER'S RUNS (`--only-missing`, how Albula runs the job, since Ron's decision of 2026-10-14: "for reproducibility
scientists would like to run a specific version of a pipeline on data"): a set is kept, whatever version made it, as long
as it was made from the scans the database holds now -- the reversed scan and the T1. A new set is made beside it only
when those differ; nothing is replaced or deleted. Since 2026-10-22 a set made WITH a reversed scan or T1 the job no
longer finds is not followed by one made without it: that case is refused and said ("NEVER REPLACED BY ONE MADE FROM
LESS", `import-job-main.ts`).

A RUN STARTED BY HAND without `--only-missing` compares everything that shapes the tracts: a rule number, the code
without comments, the naming network or table, the head template, SynthStrip's version, and the scans (`isCurrent`,
`import-job.ts`). A comment, a moved file or a commit elsewhere does not count. A code fingerprint that changed without a
change in the tracts is carried over only by an entry in `code-equivalences.json`, which says how the sameness was shown.

## What others use

- **Steve Pieper's scene of the Prisma scan** (April 2026): UKF tracking and TractCloud naming (Steve, 2026-10-21: "I
  have the results of UKF and TractCloud").
- **Lauren O'Donnell's group**: TractCloud is theirs; RapidParc is Bonn's, trained on their atlas. (Told to Lauren on
  2026-10-05, `email-odonnell-2026-10-05-pipeline-answers.md` in the workspace.)

## Why it was not transparent, and what was done (2026-10-22)

1. The naming code and the table lived in `tractcloud/`, the table inside TractCloud's model file. Now: `naming/`
   (`name-tracts.ts`, `table.ts`, `tractcloud-42.json`, `tract-colors.ts`); `tractcloud/` holds TractCloud's network only.
2. The record listed TractCloud's weights, loaded but not run, and the head template in the naming field. Now: the
   wording above; the app and the job load the table alone.
3. Comments, test names and the README said "named by TractCloud". Corrected.
4. The side decision's use of TractCloud's training data's average position was stated nowhere outside the code. Stated
   here and in `naming/table.ts`.

On the way the job's comparison of the commits of the core and of the extension was taken out (Ron, 2026-10-22: "Yes, as
proposed"). The reason given then -- that every commit had made the server remake every tract set of a database, 34 of
the test cases' 88 on 2026-10-21 -- WAS WRONG (the critic, 2026-10-22, `qa/2026-10-22-reversed-scan-regression.md` in
the workspace): the server's runs never compare commits or code; the remake of 2026-10-21 happened because the job no
longer found the reversed scan of converted cases (a bug of 2026-10-21, fixed the next day). The change matters only for
runs started by hand.

*Drafted with Claude (Anthropic) for Ron Kikinis.*
