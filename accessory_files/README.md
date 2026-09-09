# accessory_files

This directory is gitignored as a whole (most contents are large, private, or
machine-local). The files below are **deliberately force-added** (`git add -f`)
because the pipeline cannot run without them.

## TSS exclusion filter (tracked)

`model_hpc.py` removes a fixed set of the 19,376 biomart 10 kb TSS windows
before training; these `.npy` files hold the 0-based window indices to remove:

| file | contents | consumer |
|---|---|---|
| `final_tss_indices_to_remove_hg38.npy` | full filter (blacklist ∪ chrX ∪ overlapping-TSS ∪ high-variability) | `model_hpc.py` (default `--blacklist_file`), `plot_aggregated_coverage.py` |
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

Note: the regenerated full filter removes 5,806 windows vs. 5,152 in the
submission. The submission's 72-window high-variability subset was an
unrecorded manual sub-selection; the regenerated filter applies the documented
95th-percentile rule cleanly over the 50-participant panel of normals.

## Cohort manifest (tracked)

`cris_manifest.csv` — canonical Cristiano-cohort sample manifest (public EGA
IDs): role assignment (model / panel_of_normals / excluded), replicate
handling, task membership. Built by `src/comp/build_cris_manifest.py`.

## Everything else (untracked)

Public metadata spreadsheets, blacklist BEDs downloadable from their sources
(ENCODE hg38-blacklist.v2, GCparagon exclusion list), and intermediate ID
lists.
