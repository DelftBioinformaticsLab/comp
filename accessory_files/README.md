# accessory_files

Inputs and definitions the pipeline reads. The folder is git-ignored except for the files listed in `.gitignore`
(the ones the code reads); third-party annotations are downloaded into `annotations/` (sources in
`annotations/README.md`).

## TSS exclusion filter (tracked)

`model_hpc.py` removes a fixed set of the 19,376 biomart 10 kb TSS windows
before training; these `.npy` files hold the 0-based window indices to remove:

| file | contents | consumer |
|---|---|---|
| `final_tss_indices_to_remove_hg38.npy` | full filter (blacklist ∪ chrX ∪ overlapping-TSS ∪ high-variability) | `model_hpc.py` (default `--blacklist_file`) |
| `final_tss_indices_to_remove_hg19.npy` | hg19 twin (exact gene-name mapping onto `beds/biomart_10kb_hg19.bed`) | `model_hpc.py` for hg19 cohorts |
| `blacklist_and_std_outlier_and_chrX_indices.npy` | blacklist ∪ chrX ∪ high-variability (no overlapping-TSS set) | legacy option in `model_hpc.py` |
| `blacklist_and_std_outlier_and_chrX_indices_hg19.npy` | hg19 twin of the above | legacy |
| `blacklist_and_std_outlier_indices.npy` | blacklist set only (historical semantics) | `export_figS2.py` |

**These files are derived, not hand-made.** Regenerate them with:

```bash
python src/comp/derive_tss_filter.py
```

which needs `cris_manifest.csv` (tracked here; rebuilt by
`src/comp/build_cris_manifest.py`), the tracked BED files in `beds/`, and the
panel-of-normals feature matrices in `extracted_features/biomart_10kb/`
(cluster-side only). The script backs up whatever it overwrites — the
submitted-manuscript originals live in `submitted_backup/` (not tracked) — and
prints an index-level delta report. Full derivation rules and provenance notes
are in the docstring of `src/comp/derive_tss_filter.py`.

Note: the regenerated full filter removes 5,150 windows vs. 5,152 in the
submission (retained 14,226 vs 14,224). The submission's 72-window
high-variability subset was an unrecorded selection; the figS2.ipynb deep
dive showed it behaves like an SD ≥ 99.5th-percentile cutoff (as the
submitted figure legend stated), so the regenerated filter applies that rule
cleanly over the clean 50-participant panel of normals (70 windows). Of the 14,226
retained windows, `model_hpc.py` drops the columns that duplicate another window in
every sample, which leaves the 14,221 TSSs of the manuscript.

## Cohort manifests (tracked)

`cris_manifest.csv` — canonical Cristiano-cohort sample manifest (public EGA
IDs): role assignment (model / panel_of_normals / excluded), replicate
handling, task membership. Built by `src/comp/build_cris_manifest.py`.
It has one row per record of our FinaleDB retrieval (534 records). Five runs of Table S4 of Cristiano
et al. 2019 failed our preprocessing and are therefore not in it: CGPLH199 (healthy), CGCRC304
(colorectal cancer), CGPLPA168 (bile duct cancer) and CGPLLU198, CGPLLU202 (lung cancer).

`jiang_manifest.csv` + `jiang_sample_flow.csv` — the Jiang et al. 2015
cohort (FinaleDB IDs): 225 unique samples = 90 HCC vs 32 healthy + 67 chronic
hepatitis B + 36 HBV-associated cirrhosis, with an explicit `control_group`
column (the 135 non-HCC samples are not 135 healthy people). The raw FinaleDB
export `jiang_metadata.csv` has 249 rows because 20 IDs are repeated with
identical rows; the sample-flow table documents that and the 24 non-Jiang
accessions in `input/jiang_bam.txt`. Built by `src/comp/build_jiang_manifest.py`.

## Other tracked files

- `cris_exclude_deep_healthy.txt`: the 30 high-coverage healthy controls left out of every model
  (`src/comp/build_deep_healthy_exclusion.py`).
- `cris_loco_<type>_{exclude.txt,manifest.csv}`: the leave-one-cancer-type-out splits
  (`src/comp/build_loco_splits.py`).
- `cris_stage_vaf_coverage_ichor.csv` (+ `_aggregate`): stage, coverage and ichorCNA tumour fraction per sample
  (the tumour fraction column is written by `src/comp/collect_ichorcna_tf.py`).
- `cohort_protocols.csv`: the published protocol details of both cohorts (Supplementary Table S2).
- `tss_sets_membership.csv`: Hallmark and chromosome-arm membership of the TSSs (`src/comp/build_tss_set_matrices.py beds`).
- `cristiano_metadata.xlsx`, `esfahani_supplement.xlsx`, `EGAD00001007796-metadata/`: published supplementary
  tables and the EGA metadata of the Cristiano et al. data set.
- `hg38-blacklist.v2.bed` (ENCODE), `hg38_GCcorrection_ExclusionList.merged.sorted.bed` (GCparagon): exclusion lists.
- `gcfix_b0c7c0e_local_change.patch`: the one change to GCfix (top-level README, section 2).
